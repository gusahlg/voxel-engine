//! Per-frame render loop: `Renderer::draw_frame` and the stages it runs.
//! Split out of `mod.rs` so later work can touch the loop without opening
//! the renderer setup and teardown.

use ash::vk;

use crate::frame::DrawLists;
use crate::mesh::Pass;
use crate::skeleton::FrameSlot;

use super::buffers::{DrawIndexedIndirect, FRAMES_IN_FLIGHT, MESH_CONSUMER_STAGES};
use super::gpu_timer::GpuPass;
use super::pipeline;
use super::present::{HdrReadable, OverlayPresent};
use super::render_client::RenderReturn;
use super::scene_pass::RenderPass;
use super::shadow;
use super::timeline::{RenderSubmit, acquire_next_image};
use super::{
    Env, HdrSource, Renderer, SAMPLEABLE_DEPTH_REST_LAYOUT, color_range, depth_range,
    sampleable_depth_attachment_state,
};

/// Token returned by `acquire_slot` proving the slot is safe to render into
/// (its copy hazard is resolved).
struct SlotGuard(usize);

/// Byte offsets within a frame's packed immediate buffer.
#[derive(Clone, Copy)]
pub(super) struct ImmOffsets {
    pub(super) line: u64,
    pub(super) shadow: u64,
    pub(super) d2: u64,
    pub(super) d2_tex: u64,
}

/// Resolved mesh draw (one direction-run or whole mesh), pre-sort scratch.
/// Placement/style live in the persistent record SSBOs, reached through
/// `slot`; this carries only what the CPU sort/batch needs.
#[derive(Clone, Copy)]
pub(super) struct DrawEntry {
    buffer: vk::Buffer,
    pass: Pass,
    first: u32,
    count: u32,
    vertex_offset: i32,
    /// Mesh slot: the emitted command's `first_instance`, indexing the
    /// record/dyn SSBOs in the vertex shader.
    slot: u32,
    /// Squared distance to AABB center (monotonic; back-to-front sort key).
    dist2: f32,
}
/// Contiguous indirect commands sharing one buffer and pass.
#[derive(Clone, Copy)]
pub(super) struct DrawRun {
    pub(super) buffer: vk::Buffer,
    pub(super) pass: Pass,
    pub(super) first: u32,
    pub(super) count: u32,
}

/// Applies sub-pixel jitter to the view-proj matrix. Jitter only exists at
/// record time; the returned matrix is consumed immediately and never stored.
pub(super) fn jittered_clip(
    clean: glam::Mat4,
    jitter_px: glam::Vec2,
    extent: vk::Extent2D,
) -> glam::Mat4 {
    let ox = 2.0 * jitter_px.x / extent.width.max(1) as f32;
    let oy = -2.0 * jitter_px.y / extent.height.max(1) as f32;
    let t = glam::Mat4::from_cols(
        glam::Vec4::X,
        glam::Vec4::Y,
        glam::Vec4::Z,
        glam::Vec4::new(ox, oy, 0.0, 1.0),
    );
    t * clean
}

/// Get frame's sun direction, defaulting to up if absent.
/// Shadow-map content key: sun/eye-snap/occluders plus hashed avatar casters.
fn shadow_key(
    eye: glam::DVec3,
    sun: glam::DVec3,
    occluders: u64,
    lists: &DrawLists,
    cfg: &crate::skeleton::ShadowCfg,
) -> shadow::ShadowKey {
    use std::hash::{Hash, Hasher};
    let mut h = std::hash::DefaultHasher::new();
    bytemuck::cast_slice::<_, u8>(&lists.cube_verts).hash(&mut h);
    shadow::ShadowKey::of(eye, sun, occluders, h.finish(), cfg)
}

fn sun_dir(lists: &DrawLists) -> glam::DVec3 {
    lists
        .scene
        .as_ref()
        .map(|s| s.frame_uniforms)
        .map(|u| {
            glam::DVec3::new(
                u.sun_dir_elev[0] as f64,
                u.sun_dir_elev[1] as f64,
                u.sun_dir_elev[2] as f64,
            )
        })
        .filter(|d| d.length_squared() > 1e-6)
        .unwrap_or(glam::DVec3::Y)
}

impl Renderer {
    /// Records and submits one frame from the recorded draw lists, and
    /// presents it when the presentation engine can keep up (manual
    /// mailbox: frames that outrun presentation are rendered but dropped).
    ///
    /// Frame anatomy, top-down (each phase is its own helper below and its
    /// own [`crate::profile`] meter):
    /// 1. [`Self::wait_slot_and_reclaim`] — frame fence + deferred frees
    /// 2. [`Self::decide_present`]        — copy-fence check + acquire
    /// 3. [`Self::write_immediates`]      — pack cube/line/2D verts
    /// 4. [`Self::record_render`]         — barriers, rendering, draws
    /// 5. [`Self::submit_render`]         — render queue submit (fence)
    /// 6. [`Self::present`]               — copy submit + queue_present
    pub(crate) fn draw_frame(&mut self, lists: &DrawLists) {
        let size = self.size;
        if size.width == 0 || size.height == 0 {
            // Minimized: no rendering, but the game keeps running (remote
            // edits keep remeshing chunks), so uploads and frees must not
            // accumulate unboundedly until restore.
            unsafe { self.reclaim_while_idle() };
            return;
        }
        if self.needs_recreate {
            unsafe { self.apply_pending() };
            if self.swapchain.extent.width == 0 || self.swapchain.extent.height == 0 {
                return;
            }
        }

        let slot = self.slot;
        use crate::profile::{Meter, scope};
        crate::profile::count(crate::profile::Counter::Rendered);

        // Timed inside: the slot fence wait (wait tier) apart from the reclaim.
        self.wait_slot_and_reclaim(slot);

        let present_target;
        let guard = {
            // One scope; the blocking waits inside hand themselves to the wait
            // tier via `split`, so `acquire` stays pure CPU work.
            let mut p = scope(Meter::Acquire);
            present_target = self.decide_present(slot, &mut p);
            self.acquire_slot(slot, &mut p)
        };

        let offsets = {
            let _p = scope(Meter::Pack);
            let offsets = self.write_immediates(slot, lists);
            self.prepare_mesh_draws(slot, lists);
            offsets
        };

        // Per-frame UBO (set 0, binding 2). A 3D scene always carries lighting
        // (`Frame::begin_3d` takes it as a required `Lighting` argument, so
        // `Scene3D::frame_uniforms` is never optional). This `None` branch is
        // therefore reached ONLY by pure-2D frames, where the mesh shaders never
        // sample the block; the full-bright filler just keeps the binding live
        // and validated.
        {
            let mut u = lists
                .scene
                .as_ref()
                .map(|s| s.frame_uniforms)
                .unwrap_or_else(crate::skeleton::FrameUniformsGpu::full_bright);
            u.prepare_derived();
            // Debug-flat: claim the `extras` lane as [r, g, b, enabled] —
            // sRGB-encoded key channels + an enable flag. mesh3d.frag linearises rgb
            // (as it does every CPU colour) and outputs it flat while depth writes.
            // Overwriting `extras.x` (the stars gain) is safe: the lane's only
            // other consumer is sky.frag, and the app never draws the sky in the
            // debug-flat (TerrainKey) view.
            if let Some(c) = lists.debug_flat {
                u.extras = [
                    c.r as f32 / 255.0,
                    c.g as f32 / 255.0,
                    c.b as f32 / 255.0,
                    1.0,
                ];
            }
            self.ubo_ring.write(FrameSlot::new(slot), &u);
        }
        let (rs, hdr_readable) = {
            let _p = scope(Meter::Record);
            self.record_render(&guard, lists, offsets, present_target.is_some())
        };

        {
            let _p = scope(Meter::Submit);
            self.submit_render(rs, slot);
        }

        {
            let _p = scope(Meter::Present);
            // In wide-FOV the overlay was skipped in the scene pass; draw it here,
            // after the tonemap resample, so it stays crisp and unwarped.
            let overlay = OverlayPresent {
                d2_offset: offsets.d2,
                d2_count: lists.verts_2d.len() as u32,
                d2_tex_offset: offsets.d2_tex,
                d2_tex_count: lists.tex_verts_2d.len() as u32,
            };
            // Project the sun to presented uv for the tonemap godray march.
            // Computed here (not in the copy submit) so it rides the same camera +
            // frame-uniform snapshot the scene was drawn from. `project` returns a
            // strength-0 no-op when godrays are off, the sun is behind the camera,
            // or there is no 3D camera this frame.
            let godray = match lists.scene.as_ref() {
                Some(scene) => {
                    let cam = scene.camera;
                    let u = scene.frame_uniforms;
                    let (sun_dir, tint) = (
                        glam::Vec3::new(u.sun_dir_elev[0], u.sun_dir_elev[1], u.sun_dir_elev[2]),
                        [u.light[0], u.light[1], u.light[2]],
                    );
                    crate::camera::Godray::project(
                        // The tonemap shader samples single-sample depth; under
                        // MSAA that is the resolve target (`sampleable_depth`), so
                        // godrays are gated only on the feature flag now.
                        self.flags.godrays,
                        sun_dir,
                        tint,
                        &cam,
                        self.size.width as f32,
                        self.size.height as f32,
                        [
                            scene.jitter.0.x / self.render_extent.width as f32,
                            scene.jitter.0.y / self.render_extent.height as f32,
                        ],
                    )
                }
                None => crate::camera::Godray::OFF,
            };
            let warp_map = lists
                .scene
                .as_ref()
                .map_or(crate::camera::WarpMap::Identity, |s| s.warp_map);
            self.present(
                slot,
                present_target,
                warp_map,
                overlay,
                hdr_readable,
                godray,
            );
        }
        if self.vsync.current() {
            // Wait for the copy to pace at display refresh (a wait, not work).
            let _p = scope(Meter::WaitVsync);
            unsafe {
                self.timeline
                    .wait(&self.device.device, self.last_copy_value);
            }
        }

        self.slot = (self.slot + 1) % FRAMES_IN_FLIGHT as usize;
    }

    /// Tracks which offscreen slot the current copy is reading from.
    pub(super) fn track_copy(&mut self, slot: usize) {
        self.copy_slot = Some(slot);
    }

    /// Forgets any tracked copy hazard: the copy has been waited to
    /// completion, or the offscreen images it read no longer exist.
    pub(super) fn clear_copy(&mut self) {
        self.copy_slot = None;
    }

    /// The layout the scene-pass *attachment* (MS depth, or the single-sample
    /// depth when not multisampled) lives in *during* the pass. With the
    /// water-absorption path active it is `RENDERING_LOCAL_READ` — the one
    /// layout valid simultaneously as depth attachment and as the blend pass's
    /// input attachment (mid-pass transitions are illegal, so a single
    /// in-pass layout is the only coherent design). Every in-pass depth
    /// barrier and attachment info reads this ONE function, so the two
    /// configurations cannot drift apart.
    ///
    /// After `RenderPass::end` the *sampleable* single-sample image leaves this
    /// layout and rests in [`SAMPLEABLE_DEPTH_REST_LAYOUT`] until the next
    /// scene pass of this slot begins from `UNDEFINED`.
    pub(super) fn depth_pass_layout(&self) -> vk::ImageLayout {
        if self.pipelines.mesh3d_transparent_absorb.is_some() {
            vk::ImageLayout::RENDERING_LOCAL_READ_KHR
        } else {
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
        }
    }

    /// Layout and write scope of the single-sample depth image *during the
    /// scene pass* — the source of the rest-layout barrier in `RenderPass::end`.
    /// With MSAA this is a resolve attachment: Vulkan executes dynamic-rendering
    /// resolves at COLOR_ATTACHMENT_OUTPUT, even for depth. Treating it as an
    /// ordinary depth-test write leaves the resolve unordered on drivers that
    /// implement those stages independently. After that barrier the image rests
    /// in [`SAMPLEABLE_DEPTH_REST_LAYOUT`]; see that const for the contract.
    pub(super) fn sampleable_depth_attachment_state(
        &self,
    ) -> (vk::ImageLayout, vk::PipelineStageFlags2, vk::AccessFlags2) {
        sampleable_depth_attachment_state(self.targets.samples, self.depth_pass_layout())
    }

    /// Scene-pass → rest: sampleable depth becomes [`SAMPLEABLE_DEPTH_REST_LAYOUT`].
    ///
    /// Src is the attachment-write scope (depth tests, or COLOR_ATTACHMENT_OUTPUT
    /// for the MSAA SAMPLE_ZERO resolve). Dst covers every consumer that samples
    /// it without a further transition: VRS compute, TAA compute, and the
    /// tonemap present-copy fragment (godrays).
    pub(super) fn sampleable_depth_rest_barrier(&self, slot: usize) -> vk::ImageMemoryBarrier2<'_> {
        let (src_layout, src_stage, src_access) = self.sampleable_depth_attachment_state();
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            )
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(src_layout)
            .new_layout(SAMPLEABLE_DEPTH_REST_LAYOUT)
            .image(self.targets.sampleable_depth(slot).image())
            .subresource_range(depth_range())
    }

    /// Waits until the slot's last render has completed (GPU is done with its
    /// command buffer and immediate buffer), then reclaims retired GPU memory
    /// whose last possible use the timeline has reached. Publishes the
    /// one-cycle-late VRS mix and cull geometry gauges after the wait.
    fn wait_slot_and_reclaim(&mut self, slot: usize) {
        let device = &self.device.device;
        unsafe {
            {
                let _p = crate::profile::scope(crate::profile::Meter::Fence);
                self.timeline
                    .wait(device, self.slots[FrameSlot::new(slot)].render_value);
            }
            self.publish_vrs_mix(slot);
            self.publish_cull_stats(slot);
            // Nothing retired (the steady state): skip the two counter reads
            // and the queue drains, which would find nothing to reclaim.
            if !self.mesh_res.has_garbage()
                && self.retired_textures.is_empty()
                && !self.quad_ibo.has_garbage()
            {
                return;
            }
            let _p = crate::profile::scope(crate::profile::Meter::Reclaim);
            let current = self.timeline.counter(device);
            // Retired allocations return to the main-owned allocator freelist;
            // staging-block shrink happens main-side after it reclaims them.
            let ret = &self.ret;
            self.mesh_res
                .collect(current, &mut |a| drop(ret.send(RenderReturn::FreeAlloc(a))));
            self.retired_textures
                .collect(current, |mut tex| tex.destroy(device));
            // Superseded quad IBO buffers are render-owned raw buffers (not
            // allocator suballocations), so destroy them here rather than shipping
            // them back to main's freelist.
            self.quad_ibo.collect(device, current);
            // Staging submitted on a separate transfer queue is
            // reclaimed against the LANE's own timeline, not the render
            // one — a non-blocking probe (never waited: nothing here may
            // stall this reclaim pass on the transfer queue's progress).
            if let Some(transfer_current) = self.transfer_lane.counter(device) {
                self.mesh_res.collect_transfer(transfer_current, &mut |a| {
                    drop(ret.send(RenderReturn::FreeAlloc(a)))
                });
                self.quad_ibo.collect_transfer(device, transfer_current);
            }
        }
    }

    /// Last completed VRS histogram for this slot (2-frame delayed). Zeroed
    /// when VRS is off or the device has no attachment shading rate.
    fn publish_vrs_mix(&self, slot: usize) {
        if self.flags.vrs
            && let Some(vrs) = &self.targets.vrs
        {
            let [n1, n2, n4] = vrs.mix(slot);
            crate::profile::gauge(crate::profile::Gauge::Vrs1x1, n1 as u64);
            crate::profile::gauge(crate::profile::Gauge::Vrs2x2, n2 as u64);
            crate::profile::gauge(crate::profile::Gauge::Vrs4x4, n4 as u64);
        } else {
            crate::profile::gauge(crate::profile::Gauge::Vrs1x1, 0);
            crate::profile::gauge(crate::profile::Gauge::Vrs2x2, 0);
            crate::profile::gauge(crate::profile::Gauge::Vrs4x4, 0);
        }
    }

    /// Last completed cull geometry stats for this slot (2-frame delayed).
    /// `draws.full` / `tris.full` are camera group 0 (full-res opaque);
    /// `draws.lod` / `tris.lod` are group 2 (coarse LOD). Cutout (group 1) is
    /// accumulated on the GPU but not published here.
    fn publish_cull_stats(&self, slot: usize) {
        let [d0, i0, _d1, _i1, d2, i2] = self.cull.stats(slot);
        crate::profile::gauge(crate::profile::Gauge::DrawsFull, d0 as u64);
        crate::profile::gauge(crate::profile::Gauge::DrawsLod, d2 as u64);
        crate::profile::gauge(crate::profile::Gauge::TrisFull, u64::from(i0 / 3));
        crate::profile::gauge(crate::profile::Gauge::TrisLod, u64::from(i2 / 3));
    }

    /// Resolves the copy hazard on `slot` before it is rendered into: the
    /// in-flight present copy may still be reading this slot's offscreen
    /// image, which the render below overwrites. Rare (the copy usually
    /// retires well within the two-frame slot cycle) and sub-millisecond.
    /// Returns a guard proving the slot is safe to record into. `p` is the
    /// caller's `acquire` scope; the wait is split out of it into `copy`.
    fn acquire_slot(&mut self, slot: usize, p: &mut crate::profile::Guard) -> SlotGuard {
        if self.copy_slot == Some(slot) {
            let device = &self.device.device;
            p.split(crate::profile::Meter::WaitCopy);
            unsafe {
                self.timeline
                    .wait(device, self.slots[FrameSlot::new(slot)].copy_value)
            };
            p.split(crate::profile::Meter::Acquire);
            self.copy_slot = None;
        }
        SlotGuard(slot)
    }

    /// SUBOPTIMAL from acquire/present warrants a rebuild only when the
    /// swapchain no longer matches the window size. Some presentation stacks
    /// (Windows compositor states, Wine) report SUBOPTIMAL persistently even
    /// for a correctly sized swapchain; recreating on the flag alone then
    /// rebuilds the swapchain and render targets every frame, indefinitely.
    /// Genuine size changes still recreate via `on_resize`/OUT_OF_DATE.
    pub(super) fn recreate_if_stale(&mut self) {
        if self.swapchain.extent != self.size {
            self.needs_recreate = true;
        }
    }

    /// Present eligibility, decided before the render submit. Strict ordering:
    /// the previous copy's completion is probed first (non-blocking, so the
    /// mailbox drop never stalls), the acquire is only attempted once we know a
    /// copy can be submitted, and a successful acquire is ALWAYS followed by
    /// the copy + present in [`Self::present`] — never skipped.
    ///
    /// `p` is the caller's `acquire` scope: the blocking waits here (forced
    /// copy retire, vsync/forced drawable acquire) are split out of it into the
    /// wait tier, so `acquire` reports only the non-blocking work.
    fn decide_present(&mut self, slot: usize, p: &mut crate::profile::Guard) -> Option<u32> {
        use crate::profile::Meter;
        // With vsync off, throttle presents to refresh cadence to avoid blocking
        // on drawable availability; instead render frames in between unthrottled.
        // Present slightly ahead of the refresh interval so scheduling jitter
        // never pushes a present past the drawable's availability window.
        const PRESENT_THROTTLE: f32 = 0.9;
        // A pending capture makes this present MANDATORY: dropping it (throttle,
        // copy-in-flight mailbox skip, or a non-blocking acquire miss) would
        // silently discard the frame the caller asked to capture. So a forced
        // present ignores the throttle, WAITS for the prior copy to retire
        // instead of skipping, and blocks for a drawable.
        let force = self.pending_capture.is_some();
        let present_due = force
            || self.vsync.current()
            || self.last_present.elapsed() >= self.present_interval.mul_f32(PRESENT_THROTTLE);
        let mut present_target = None;
        let device = &self.device.device;
        unsafe {
            if force {
                // Wait out the prior copy rather than treating it as a drop.
                p.split(Meter::WaitCopy);
                self.timeline.wait(device, self.last_copy_value);
                p.split(Meter::Acquire);
            }
            // Skip present if previous copy still in flight (mailbox drop).
            let copy_ready =
                force || (present_due && self.timeline.probe(device).reached(self.last_copy_value));
            if copy_ready {
                // With vsync or a forced capture: wait for an image. Plain
                // vsync-off: never wait, allow drop.
                let timeout = if self.vsync.current() || force {
                    u64::MAX
                } else {
                    0
                };
                let blocking = timeout != 0;
                if blocking {
                    p.split(Meter::WaitVsync);
                }
                let acquired = acquire_next_image(
                    &self.swapchain.loader,
                    self.swapchain.swapchain,
                    timeout,
                    self.slots[FrameSlot::new(slot)].image_available,
                );
                if blocking {
                    p.split(Meter::Acquire);
                }
                match acquired {
                    Ok((image_index, suboptimal)) => {
                        if suboptimal {
                            self.recreate_if_stale();
                        }
                        present_target = Some(image_index);
                    }
                    // No image available; drop the present.
                    Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {}
                    // OUT_OF_DATE/SURFACE_LOST: environmental, recreate next frame.
                    // Other errors are unrecoverable.
                    Err(err) => match Env::classify(err) {
                        Some(Env::OutOfDate | Env::SurfaceLost) => self.needs_recreate = true,
                        _ => panic!("acquire_next_image failed: {err:?}"),
                    },
                }
            }
        }
        present_target
    }

    /// Packs frame immediates (cubes, lines, 2D) into host buffer and returns offsets.
    fn write_immediates(&mut self, slot: usize, lists: &DrawLists) -> ImmOffsets {
        let cube_bytes: &[u8] = bytemuck::cast_slice(&lists.cube_verts);
        let line_bytes: &[u8] = bytemuck::cast_slice(&lists.line_verts);
        let shadow_bytes: &[u8] = bytemuck::cast_slice(&lists.shadow_verts);
        let d2_bytes: &[u8] = bytemuck::cast_slice(&lists.verts_2d);
        let d2_tex_bytes: &[u8] = bytemuck::cast_slice(&lists.tex_verts_2d);
        let line = (cube_bytes.len() as u64).next_multiple_of(16);
        let shadow = (line + line_bytes.len() as u64).next_multiple_of(16);
        let d2 = (shadow + shadow_bytes.len() as u64).next_multiple_of(16);
        let d2_tex = (d2 + d2_bytes.len() as u64).next_multiple_of(16);
        let total = d2_tex + d2_tex_bytes.len() as u64;
        let imm = &mut self.slots[FrameSlot::new(slot)].imm;
        unsafe {
            imm.maintain(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                total,
            );
            if total > 0 {
                imm.write(0, cube_bytes);
                imm.write(line, line_bytes);
                imm.write(shadow, shadow_bytes);
                imm.write(d2, d2_bytes);
                imm.write(d2_tex, d2_tex_bytes);
            }
        }
        ImmOffsets {
            line,
            shadow,
            d2,
            d2_tex,
        }
    }

    /// Prepares this frame's two draw sources from the persistent state.
    ///
    /// - GPU cull (opaque/cutout/shadow): exact partitions from the arena live
    ///   counts, params (camera + cascade frusta, eye split), and the persistent
    ///   `visible_mask` as the dispatch's visibility input. The mask carries what
    ///   only the app knows (LOD selection, quadrant masks, occlusion): a
    ///   resident-but-hidden slot must not draw just because its record is live.
    ///   The cull shader frustum-tests every visible slot and appends
    ///   `first_instance = slot` commands the graphics side draws indirect-count.
    ///
    /// - CPU Blend re-source: transparency needs exact far→near ordering the GPU
    ///   cull does not provide, so Blend is the ONE pass still resolved CPU-side.
    ///   It is sourced from the SAME persistent state — records + arena directory
    ///   + `visible_mask`, NOT a per-frame draw list — by iterating the resident,
    ///   visible, Blend-pass slots, frustum-culling, sorting by distance, and
    ///   emitting whole-mesh indirect commands (also `first_instance = slot`, so
    ///   placement/style come from the record/dyn SSBOs, never rebuilt per frame).
    fn prepare_mesh_draws(&mut self, slot: usize, lists: &DrawLists) {
        use ash::vk::Handle;

        self.draw_scratch.clear();
        self.draw_commands.clear();
        self.draw_runs.clear();

        // Flush record/dyn patches into this slot's copies (post-fence, same
        // discipline as the HostBuffer maintains below).
        self.record_buffers = unsafe {
            self.records.flush(
                slot,
                &self.arena_dir,
                &self.mesh_res,
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
            )
        };

        // Cull emits shadow casters only when the shared map will actually be
        // rewritten this frame. A cache hit skips cascade `fit()` and sets
        // `shadow_enabled = 0` so invisible slots bail before the AABB load.
        // Far cascade radius follows full-res coverage (`lod_clip`); a render-
        // distance change snaps `ShadowKey` from `cfg.splits` and rebuilds once.
        let cfg = crate::skeleton::ShadowCfg::for_coverage(lists.lod_clip);
        let shadow_frusta = lists.scene.as_ref().and_then(|scene| {
            if self.flags.shadows {
                let key = shadow_key(
                    scene.eye,
                    sun_dir(lists),
                    self.records.occluder_rev(),
                    lists,
                    &cfg,
                );
                if !self.shadow_cache.prepare(Some((key, &cfg))) {
                    return None;
                }
                let sun = sun_dir(lists);
                Some(
                    [
                        crate::skeleton::Cascade::Near,
                        crate::skeleton::Cascade::Far,
                    ]
                    .map(|c| {
                        crate::camera::Frustum::from_view_proj(
                            &shadow::fit(scene.eye, sun, c, &cfg).view_proj.0,
                        )
                    }),
                )
            } else {
                self.shadow_cache.prepare(None);
                None
            }
        });

        // GPU cull prep: the persistent visibility mask IS the dispatch's
        // visibility input, grown to cover every live slot (zero = hidden).
        // The dispatch (and the mask it reads) stops at the directory's live
        // end, not the table length: a freed tail is all dead words.
        // Last frame's partition table is handed back so its allocation is
        // reused rather than rebuilt from scratch every frame.
        let recycled = self
            .cull_frame
            .take()
            .map_or_else(Vec::new, |f| f.partitions);
        self.cull_frame = if let Some(scene) = &lists.scene {
            let camera = crate::camera::Frustum::from_view_proj(&scene.view_proj);
            let eye = pipeline::EyeSplit::of(scene.eye);
            if let Some(records) = self.record_buffers {
                let slot_count = records.slots.min(self.arena_dir.live_end());
                let need = slot_count.div_ceil(32) as usize;
                if self.visible_mask.len() < need {
                    self.visible_mask.resize(need, 0);
                }
                unsafe {
                    self.cull.prepare(
                        slot,
                        &self.instance.instance,
                        &self.device.device,
                        self.device.physical,
                        &self.arena_dir,
                        records,
                        slot_count,
                        &camera,
                        shadow_frusta.as_ref(),
                        eye,
                        &self.visible_mask[..need],
                        recycled,
                    )
                }
            } else {
                None
            }
        } else {
            None
        };

        // CPU Blend re-source: walk the resident, visible, Blend-pass records.
        // The directory's Blend set is exactly that candidate list, so this is
        // O(transparent meshes), never a sweep of the whole slot table.
        if let Some(scene) = &lists.scene {
            let camera = crate::camera::Frustum::from_view_proj(&scene.view_proj);
            let eye = pipeline::EyeSplit::of(scene.eye);
            for &s in self.arena_dir.blend_slots() {
                // Arena word (0 = not resident) is the arena index + 1, giving
                // the vertex buffer without a residency-handle lookup. Gated
                // on `is_arrived` too: a budget-deferred copy is registered in
                // `arena_dir` (capacity/ref-count bookkeeping happens at
                // upload) before its bytes actually land — reading it here
                // early would source the CPU Blend draw from uninitialized
                // arena memory, same hazard `RecordTable::flush` guards for
                // the GPU cull path.
                let arena = self.arena_dir.arena_word(s as usize);
                if arena == 0 || !self.mesh_res.is_arrived(s) {
                    continue;
                }
                // The same persistent mask the GPU cull reads.
                let visible = self
                    .visible_mask
                    .get((s >> 5) as usize)
                    .is_some_and(|w| w & (1 << (s & 31)) != 0);
                if !visible {
                    continue;
                }
                let Some(rec) = self.records.record(s) else {
                    continue;
                };
                debug_assert_eq!(
                    rec.pass(),
                    Pass::Blend,
                    "Blend set holds a non-Blend record"
                );
                // Camera-relative placement reconstructed exactly as the vertex
                // shader does (integer block minus camera block, then the
                // fractional remainder), so the CPU sort/cull agrees with the GPU
                // draw. `detail_scale` decodes the BIASED detail field — never
                // decode `detail_pass` here by hand (it carries a to_gpu_bits offset).
                let scale = rec.detail_scale();
                let offset = glam::Vec3::new(
                    (rec.block[0] - eye.block[0]) as f32 - eye.frac[0] + rec.local_off[0],
                    (rec.block[1] - eye.block[1]) as f32 - eye.frac[1] + rec.local_off[1],
                    (rec.block[2] - eye.block[2]) as f32 - eye.frac[2] + rec.local_off[2],
                );
                let amin = glam::Vec3::from(rec.aabb_min);
                let amax = glam::Vec3::from(rec.aabb_max);
                if !camera.intersects_aabb(amin * scale + offset, amax * scale + offset) {
                    continue;
                }
                let center = offset + (amin + amax) * 0.5 * scale;
                let dist2 = (center - scene.cam_pos).length_squared();
                self.draw_scratch.push(DrawEntry {
                    buffer: self.arena_dir.arena_buffer((arena - 1) as usize),
                    pass: Pass::Blend,
                    first: 0,
                    count: rec.index_count,
                    vertex_offset: rec.vertex_offset,
                    slot: s,
                    dist2,
                });
            }
            // Blend far→near for correct back-to-front alpha compositing.
            self.draw_scratch.sort_unstable_by(|a, b| {
                b.dist2
                    .total_cmp(&a.dist2)
                    // Deterministic tiebreak; keeps equidistant same-arena draws batched.
                    .then_with(|| a.buffer.as_raw().cmp(&b.buffer.as_raw()))
            });

            for entry in &self.draw_scratch {
                let command_index = self.draw_commands.len() as u32;
                self.draw_commands.push(DrawIndexedIndirect {
                    index_count: entry.count,
                    instance_count: 1,
                    first_index: entry.first,
                    vertex_offset: entry.vertex_offset,
                    first_instance: entry.slot,
                });
                match self.draw_runs.last_mut() {
                    Some(run) if run.buffer == entry.buffer && run.pass == entry.pass => {
                        run.count += 1
                    }
                    _ => self.draw_runs.push(DrawRun {
                        buffer: entry.buffer,
                        pass: entry.pass,
                        first: command_index,
                        count: 1,
                    }),
                }
            }
        }
        let indirect_bytes: &[u8] = bytemuck::cast_slice(&self.draw_commands);
        unsafe {
            let indirect = &mut self.slots[FrameSlot::new(slot)].indirect;
            indirect.maintain(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                indirect_bytes.len() as u64,
            );
            if !indirect_bytes.is_empty() {
                indirect.write(0, indirect_bytes);
            }
        }
        crate::profile::gauge(
            crate::profile::Gauge::DrawsPacked,
            self.draw_commands.len() as u64,
        );
    }

    /// Records the command buffer: mesh copies, render pass, and transitions.
    fn record_render(
        &mut self,
        guard: &SlotGuard,
        lists: &DrawLists,
        offsets: ImmOffsets,
        will_present: bool,
    ) -> (RenderSubmit, HdrReadable) {
        let slot = guard.0;
        let cmd = self.slots[FrameSlot::new(slot)].cmd;
        // Read the prior render-pass GPU time for this slot before its queries
        // are reset below (the slot's fence was already waited this frame).
        let profiling = crate::profile::is_enabled();
        if profiling {
            let mut passes = [0.0f64; GpuPass::COUNT];
            if let Some((total, gap)) = unsafe {
                self.gpu_timer
                    .read_into(&self.device.device, slot, &mut passes)
            } {
                for pass in GpuPass::ALL {
                    crate::profile::add_ms(pass.meter(), passes[pass as usize]);
                }
                crate::profile::gpu_frame_ms(total);
                if let Some(gap) = gap {
                    crate::profile::gpu_gap_ms(gap, total);
                }
            }
        }
        // Begin render submission; this gets the timeline value to stamp mesh copies.
        let rs = self.timeline.begin_render(cmd);
        let done_at = rs.value();
        unsafe {
            let device = &self.device.device;
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("command buffer reset failed");
            device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .expect("begin command buffer failed");
            // Start timing before the staged copies so the whole buffer is
            // attributed (the `Copies` stamp closes this first span).
            if profiling {
                self.gpu_timer.begin(device, cmd, slot);
            }

            // Last frame's separate-queue copies: this submission is the first
            // that can draw them (see `MeshResidency::flush_copies`). This
            // frame's copies are flushed next and deferred one frame.
            let deferred = self.mesh_res.take_deferred_arrival(device, cmd);
            self.mesh_res.flush_copies(
                device,
                &mut self.transfer_lane,
                cmd,
                self.device.graphics_family,
                done_at,
            );
            // Slots whose copy just submitted: their arena word was gated to
            // 0 in every RecordTable copy until now (see `RecordTable::flush`);
            // re-mark them dirty so the NEXT prepare_mesh_draws re-reads
            // `is_arrived` and exposes the real word (this frame's cull dispatch
            // already ran, upstream of this flush — a one-frame-late reveal,
            // never early).
            let arrived = self.mesh_res.take_arrived();
            self.records.mark_arrived(&arrived);
            // Grow the shared quad IBO (if a bigger mesh arrived) before any
            // draw indexes it. Unified-memory uploads can draw the same frame,
            // so this copy keeps a same-frame wait (same-queue barrier, or the
            // lane wait folded into `pending_transfer_wait` below).
            let quad_wait = self.quad_ibo.ensure(
                &self.instance.instance,
                device,
                self.device.physical,
                &mut self.transfer_lane,
                cmd,
                self.device.graphics_family,
                done_at,
            );
            self.pending_transfer_wait = match (deferred, quad_wait) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            // Upload this slot's minimap texture (if its version is stale) on the
            // live frame command buffer, before the render pass begins.
            self.minimap.sync(device, cmd, slot);
            if profiling {
                self.gpu_timer.mark(device, cmd, slot, GpuPass::Copies);
            }
        }

        // GPU cull: emit this frame's opaque/cutout/shadow draw commands from
        // the persistent record set, BEFORE any pass that consumes them (the
        // shadow occluders below and the mesh passes). Outside any rendering
        // scope; its trailing barrier orders the writes against DRAW_INDIRECT.
        if lists.scene.is_some()
            && let Some(frame) = &self.cull_frame
            && let Some(records) = self.record_buffers
        {
            let _g = crate::profile::scope(crate::profile::Meter::Pack);
            unsafe {
                self.cull.record(
                    &self.device.device,
                    &self.device.push_descriptor,
                    cmd,
                    slot,
                    records,
                    frame,
                );
                if profiling {
                    self.gpu_timer
                        .mark(&self.device.device, cmd, slot, GpuPass::Cull);
                }
            }
        } else if profiling {
            // No dispatch this slot: CPU-zero so the one-cycle-late publish
            // does not report a stale histogram.
            self.cull.clear_stats_cpu(slot);
        }

        // Cascaded shadows: on a miss, fit both cascades, publish binding-3
        // uniforms, and render occluders into the *shared* map before the color
        // pass (it leaves the map in SHADER_READ_ONLY_OPTIMAL for mesh3d.frag).
        // Hits skip the producer (and skip `fit()`); the slot UBO is filled from
        // the cached block so sampling matches the resident depth.
        if let Some(scene) = &lists.scene {
            let cfg = crate::skeleton::ShadowCfg::for_coverage(lists.lod_clip);
            let sun = sun_dir(lists);
            let caster_verts = lists.cube_verts.len() as u32;
            let render = self.shadow_cache.pending_rebuild();
            if render {
                let _g = crate::profile::scope(crate::profile::Meter::RecShadow);
                // WAR: the other FIF slot may still be sampling the shared map.
                // Hits skip this wait (concurrent SHADER_READ_ONLY is legal).
                unsafe {
                    self.timeline
                        .wait(&self.device.device, self.last_render_value);
                }
                let cu = self.shadow_uniforms(scene.eye, sun, &cfg);
                self.shadow_cache.store_uniforms(cu);
                self.shadow.write_uniforms(slot, &cu);
                self.record_shadow_pass(cmd, slot, scene.eye, sun, &cfg, caster_verts);
                if profiling {
                    unsafe {
                        self.gpu_timer
                            .mark(&self.device.device, cmd, slot, GpuPass::ShadowMap)
                    };
                }
                if !self.flags.shadows {
                    self.shadow_cache.mark_lit_ready();
                }
            } else if let Some(cu) = self.shadow_cache.uniforms() {
                // Hit: copy cached matrices into this slot's UBO, no `fit()`.
                self.shadow.write_uniforms(slot, cu);
            }
        }

        // Cloud LUT: march (or zero) before the scene pass so the sky fragment
        // has a sampled image. Skipped when there is no sky.
        if self.flags.sky
            && lists.sky.is_some()
            && let Some(scene) = &lists.scene
        {
            self.record_sky_cloud_lut(cmd, slot, scene.frame_uniforms.anim[3]);
        }

        // Bind the rate image classified at the end of this slot's previous
        // use (two frames ago at 2FIF) — the same staleness the old
        // begin-of-frame classify accepted. First scene pass after
        // create/recreate skips VRS (`vrs_ready` is false); we still classify
        // at end so the next use is primed.
        let vrs_on = lists.scene.is_some() && self.flags.vrs && self.targets.vrs.is_some();
        let do_vrs = vrs_on && self.slots[FrameSlot::new(slot)].vrs_ready;
        let classify_vrs = vrs_on;

        let device = &self.device.device;
        let stamp = |p| {
            if profiling {
                unsafe { self.gpu_timer.mark(device, cmd, slot, p) };
            }
        };
        // A fresh scene render makes the offscreen the frame's HDR again; the
        // TAA pass overrides this if it runs (set before `begin` borrows self
        // shared for the whole pass).
        self.slots[FrameSlot::new(slot)].hdr_source = HdrSource::Offscreen;
        let pass = {
            let _g = crate::profile::scope(crate::profile::Meter::RecTransitions);
            unsafe { RenderPass::begin(self, cmd, slot, lists, offsets, do_vrs) }
        };
        if lists.scene.is_some() {
            use crate::profile::{Meter, scope};
            // Transparency forces an interleave: all opaque geometry (mesh runs
            // AND opaque debug cubes/lines) writes depth before any transparent
            // mesh run tests against it.
            unsafe {
                {
                    let _g = scope(Meter::RecMesh);
                    pass.record_mesh_indirect(Pass::Opaque);
                    // Cutout writes depth like opaque, so it belongs in the opaque
                    // prefix (before sky). Dormant until a block emits it.
                    pass.record_mesh_indirect(Pass::Cutout);
                }
                stamp(GpuPass::Opaque);
                // Sky fills the background (uncovered pixels) right after opaque
                // depth is laid down. It must precede the immediate debug
                // cubes/lines: the highlight lines are depth read-only (no depth
                // write), so a line silhouetted against the background leaves the
                // depth cleared there — drawing sky afterward would overpaint it.
                // Debug geometry and transparent water both composite over the sky.
                {
                    let _g = scope(Meter::RecSky);
                    if self.flags.sky {
                        pass.record_sky();
                    }
                }
                stamp(GpuPass::Sky);
                {
                    let _g = scope(Meter::RecImmediate);
                    pass.record_immediate_cubes();
                    stamp(GpuPass::Cubes);
                    pass.record_lines();
                    stamp(GpuPass::Lines);
                    // Contact shadows: translucent, blended over the opaque terrain
                    // depth just laid down, before transparent water.
                    pass.record_shadows();
                }
                stamp(GpuPass::Shadows);
                {
                    let _g = scope(Meter::RecMesh);
                    pass.record_mesh_indirect(Pass::Blend);
                }
                stamp(GpuPass::Transparent);
            }
        }
        // The offscreen HDR must reach SHADER_READ_ONLY before the tonemap
        // present copy samples it. `end` performs that COLOR_ATTACHMENT→
        // SHADER_READ barrier UNLESS a later offscreen writer runs after it: the
        // TAA resolve and exposure metering both write the offscreen *after*
        // `end`, so when either is active `end` must NOT transition (the barrier
        // would race their writes) — the deferred finalization below owns it
        // instead. With both disabled (the common path) `end` transitions.
        //
        // TAA keeps history every frame. Bloom and exposure metering feed only
        // the tonemap present-copy, so they run solely on frames that will
        // present (`decide_present` already ran; forced capture always presents).
        let taa = lists.scene.is_some() && self.flags.taa;
        let exposure_on = lists.scene.is_some() && self.flags.exposure;
        let run_exposure = exposure_on && will_present;
        let deferred = taa || run_exposure;
        // Overlay composited post-tonemap so warp/TAA don't affect the HUD.
        stamp(GpuPass::Overlay);
        // Finalize the offscreen to SHADER_READ_ONLY exactly once and obtain the
        // [`HdrReadable`] proof the tonemap present-copy requires. The branches
        // are exhaustive and each ends with the offscreen sampled when a present
        // will sample it: (a) not deferred → the render pass transitions (or we
        // skip the transition on an unpresented frame); (b) deferred + exposure
        // → metering owns the transition; (c) deferred + !exposure (TAA-on,
        // exposure-off/skipped) → TAA already left its output sampled.
        // Producing the proof only inside these paths is what makes "nobody
        // finalized the layout" fail to compile at `present` rather than trip the
        // validation layer (the exact bug from the exposure-default-off change).
        let readable: HdrReadable = {
            let _g = crate::profile::scope(crate::profile::Meter::RecTransitions);
            if deferred {
                unsafe { pass.end_deferred(classify_vrs) };
                // Close the end-rendering/MSAA-resolve span before the deferred
                // writers, so TAA and exposure report apart from it.
                stamp(GpuPass::Resolve);
                self.finish_vrs_classify(cmd, slot, classify_vrs, lists);
                // TAA resolve runs AFTER the HDR resolve and BEFORE exposure, so
                // exposure meters the stabilized image. It reads the current HDR +
                // reprojected history, writes the resolved HDR back, and leaves it
                // sampled for the exposure pass. Reprojection uses the un-jittered
                // view-proj; a false `flags.taa` never reaches here.
                if taa {
                    // `taa` is `lists.scene.is_some() && self.flags.taa` (above).
                    let scene = lists.scene.as_ref().expect("taa true implies a 3D scene");
                    self.record_taa_pass(
                        cmd,
                        FrameSlot::new(slot),
                        scene.view_proj,
                        scene.eye,
                        scene.jitter.0,
                    );
                    if profiling {
                        unsafe {
                            self.gpu_timer
                                .mark(&self.device.device, cmd, slot, GpuPass::Taa)
                        };
                    }
                }
                if run_exposure {
                    // Reduce the frame HDR to per-tile mean log2-luma, publish
                    // the smoothed exposure, and finalize the HDR in SHADER_READ.
                    let readable = self.record_exposure_pass(cmd, FrameSlot::new(slot));
                    if profiling {
                        unsafe {
                            self.gpu_timer
                                .mark(&self.device.device, cmd, slot, GpuPass::Exposure)
                        };
                    }
                    readable
                } else if self.slots[FrameSlot::new(slot)].hdr_source != HdrSource::Offscreen {
                    // TAA published its output as the frame HDR and already
                    // left it (and the offscreen) sampled: nothing to record.
                    HdrReadable::new(slot)
                } else if will_present {
                    let readable = unsafe { self.transition_offscreen_to_sampled(cmd, slot) };
                    // The finalize barrier belongs to the resolve span.
                    if profiling {
                        unsafe {
                            self.gpu_timer
                                .mark(&self.device.device, cmd, slot, GpuPass::Resolve)
                        };
                    }
                    readable
                } else {
                    HdrReadable::new(slot)
                }
            } else if will_present {
                // Common path (TAA + exposure both off): the render pass finalizes.
                let readable = unsafe { pass.end_sampled(classify_vrs) };
                // Close the resolve/finalize segment (MSAA resolve + transitions)
                // before bloom records, so the report splits them.
                if profiling {
                    unsafe {
                        self.gpu_timer
                            .mark(&self.device.device, cmd, slot, GpuPass::Resolve)
                    };
                }
                self.finish_vrs_classify(cmd, slot, classify_vrs, lists);
                readable
            } else {
                // Unpresented, no later HDR writer: skip the sampled transition;
                // the next begin discards the offscreen from UNDEFINED. Depth
                // still rests so the classifier (and a later present of a
                // different slot) can sample it.
                unsafe { pass.end_deferred(classify_vrs) };
                self.finish_vrs_classify(cmd, slot, classify_vrs, lists);
                HdrReadable::new(slot)
            }
        };
        // Bloom: threshold + downsample the finalized HDR into this slot's
        // mip chain; the tonemap present-copy composites it. Present-only — a
        // dropped mailbox frame never samples the pyramid. Forced capture always
        // presents, so it always gets a fresh chain.
        if will_present {
            self.record_bloom_pass(cmd, FrameSlot::new(slot));
        }
        // Close the tail: without this stamp the bloom work recorded above
        // ends after the last boundary and never reaches the report. (The
        // tonemap/present copy is timed on the copy command buffer; see
        // `submit_present_copy`.)
        if profiling {
            unsafe {
                self.gpu_timer
                    .mark(&self.device.device, cmd, slot, GpuPass::Bloom)
            };
            self.gpu_timer.finish(slot);
        }
        unsafe {
            self.device
                .device
                .end_command_buffer(cmd)
                .expect("end command buffer failed");
        }
        (rs, readable)
    }

    /// The image+view holding slot `slot`'s FINAL HDR (see `hdr_source`).
    pub(super) fn hdr_of(&self, slot: usize) -> (vk::Image, vk::ImageView) {
        match self.slots[FrameSlot::new(slot)].hdr_source {
            HdrSource::Offscreen => (
                self.targets.offscreen[slot].image(),
                self.targets.offscreen[slot].view(),
            ),
            HdrSource::TaaHistory(i) => self.taa.history_image(i),
        }
    }

    /// End-of-frame classify: this slot's just-written depth, for the next use
    /// of the slot (two frames later). Depth already rests in
    /// [`SAMPLEABLE_DEPTH_REST_LAYOUT`]; rate/history → GENERAL joined the
    /// post-scene barrier. Sets `vrs_ready` / `vrs_history` so the next scene
    /// pass of this slot can bind the rate image.
    fn finish_vrs_classify(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        classify: bool,
        lists: &DrawLists,
    ) {
        if !classify {
            return;
        }
        let scene = lists
            .scene
            .as_ref()
            .expect("classify_vrs implies a 3D scene");
        let focal_px = 0.5 * self.render_extent.height as f32 / scene.fovy_tan_half.max(1e-4);
        let d_threshold = crate::camera::Z_NEAR / focal_px;
        unsafe { self.record_vrs_generate(cmd, slot, d_threshold) };
        if crate::profile::is_enabled() {
            unsafe {
                self.gpu_timer
                    .mark(&self.device.device, cmd, slot, GpuPass::Vrs)
            };
        }
        self.slots[FrameSlot::new(slot)].vrs_ready = true;
        self.slots[FrameSlot::new(slot)].vrs_history = true;
    }

    /// Transitions slot `slot`'s offscreen HDR from `COLOR_ATTACHMENT_OPTIMAL` to
    /// `SHADER_READ_ONLY_OPTIMAL` for the tonemap present-copy, minting the
    /// [`HdrReadable`] proof. Used on the TAA-on/exposure-off path, where the TAA
    /// resolve left the offscreen in `COLOR_ATTACHMENT` and no metering pass owns
    /// the transition.
    unsafe fn transition_offscreen_to_sampled(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
    ) -> HdrReadable {
        let to_sampled = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.targets.offscreen[slot].image())
            .subresource_range(color_range())];
        unsafe {
            self.device.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        }
        HdrReadable::new(slot)
    }

    /// Submits the recorded command buffer and advances the timeline. Waits
    /// on the transfer lane at [`MESH_CONSUMER_STAGES`] when last frame's
    /// deferred mesh copies and/or this frame's `quad_ibo.ensure` submitted
    /// on a separate queue — a cross-queue dependency needs a semaphore wait;
    /// the in-command-buffer barrier used otherwise only orders work within
    /// one queue. The wait is vertex/index fetch (including the shadow
    /// cascades), not `ALL_COMMANDS`, so cull/clears/sky/post can overlap
    /// the copy.
    fn submit_render(&mut self, rs: RenderSubmit, slot: usize) {
        let extra_wait = self
            .pending_transfer_wait
            .take()
            .map(|value| (self.transfer_lane.semaphore(), value, MESH_CONSUMER_STAGES));
        let completion = unsafe {
            rs.submit(
                &self.device.device,
                self.device.graphics_queue,
                &self.timeline,
                extra_wait,
            )
        };
        self.slots[FrameSlot::new(slot)].render_value = completion.value();
        self.last_render_value = completion.value();
    }
}
