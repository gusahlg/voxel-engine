//! Per-frame render loop: `Renderer::draw_frame` and the stages it runs.
//! Split out of `mod.rs` so later work can touch the loop without opening
//! the renderer setup and teardown.

use ash::vk;

use crate::frame::DrawLists;
use crate::mesh::Pass;
use crate::skeleton::FrameSlot;

use super::buffers::{
    DrawIndexedIndirect, FRAMES_IN_FLIGHT, MESH_CONSUMER_STAGES, SUBMIT_BATCH_MAX,
};
use super::gpu_timer::{GpuPass, PipeStatPass};
use super::pipeline;
use super::present::{HdrReadable, OverlayPresent};
use super::render_client::RenderReturn;
use super::scene_pass::RenderPass;
use super::shadow;
use super::timeline::{RenderSubmit, TimelineValue, acquire_next_image};
use super::{
    Env, Renderer, SAMPLEABLE_DEPTH_REST_LAYOUT, depth_range, sampleable_depth_attachment_state,
    sampleable_depth_consumed,
};

/// Token returned by `acquire_slot` proving the slot is safe to render into
/// (its copy hazard is resolved).
struct SlotGuard(usize);

/// Slot whose `render_value` [`Renderer::wait_slot_and_reclaim`] waits before
/// recording into `slot`.
///
/// Uncapped: wait only the slot being reused, so all [`FRAMES_IN_FLIGHT`]
/// command buffers can sit on the GPU and hide submit latency. Vsync: wait the
/// slot that is two frames old so the effective depth stays 2 — a third
/// in-flight frame would add a full refresh of latency. Timeline values are
/// monotone, so that wait also proves `slot` itself is idle.
fn reclaim_wait_slot(slot: usize, vsync: bool) -> usize {
    if vsync {
        (slot + FRAMES_IN_FLIGHT as usize - 2) % FRAMES_IN_FLIGHT as usize
    } else {
        slot
    }
}

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

/// Recorded but not yet submitted: one entry per deferred unpresented frame.
pub(super) struct PendingSubmit {
    slot: usize,
    cmd: vk::CommandBuffer,
    extra_wait: Option<TimelineValue>,
    /// Value reserved by `begin_render`; the batch signals the last entry's.
    signal: TimelineValue,
}

/// Whether the just-recorded frame may join the pending batch (`Defer`) or
/// must be submitted this call (`Flush`). `pending_count` includes the current
/// frame. A presented / vsync-on frame always `Flush`es (the caller submits
/// any already-pending batch first, then this frame on its own).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubmitBatchAction {
    Defer,
    Flush,
}

fn submit_batch_action(
    uncapped: bool,
    present: bool,
    pending_count: usize,
    limit: usize,
) -> SubmitBatchAction {
    if !uncapped || present || pending_count >= limit {
        SubmitBatchAction::Flush
    } else {
        SubmitBatchAction::Defer
    }
}

/// True when `wait_slot` still has an unsubmitted command buffer. Waiting
/// then would block on a value that has not been queued (or, if `render_value`
/// was left at the slot's previous use, return immediately and reset a CB
/// still in the pending list).
fn pending_blocks_wait(pending_slots: impl IntoIterator<Item = usize>, wait_slot: usize) -> bool {
    pending_slots.into_iter().any(|s| s == wait_slot)
}

/// `VOXEL_SUBMIT_BATCH` (integer ≥ 1): command buffers per `vkQueueSubmit2`
/// for unpresented uncapped frames. Unset uses [`SUBMIT_BATCH_MAX`]. Clamped
/// to `1..=FRAMES_IN_FLIGHT-1` so the ring always has a free slot. Read once
/// at renderer creation.
pub(super) fn submit_batch_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let parsed = std::env::var("VOXEL_SUBMIT_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(SUBMIT_BATCH_MAX);
        parsed.clamp(1, FRAMES_IN_FLIGHT as usize - 1)
    })
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

/// Shadow-map content key: sun/eye-snap/occluders plus hashed avatar casters.
fn shadow_key(
    eye: glam::DVec3,
    sun: glam::DVec3,
    occluders: u64,
    lists: &DrawLists,
    cfg: &crate::skeleton::ShadowCfg,
) -> shadow::ShadowKey {
    shadow::ShadowKey::of(
        eye,
        sun,
        occluders,
        shadow::hash_casters(bytemuck::cast_slice(&lists.cube_verts)),
        cfg,
    )
}

/// Project the sun to presented uv for the spill-pass godray march.
fn project_godray(r: &Renderer, lists: &DrawLists) -> crate::camera::Godray {
    match lists.scene.as_ref() {
        Some(scene) => {
            let u = scene.frame_uniforms;
            crate::camera::Godray::project(
                r.flags.godrays,
                glam::Vec3::new(u.sun_dir_elev[0], u.sun_dir_elev[1], u.sun_dir_elev[2]),
                [u.light[0], u.light[1], u.light[2]],
                &scene.camera,
                r.size.width as f32,
                r.size.height as f32,
                [
                    scene.jitter.0.x / r.render_extent.width as f32,
                    scene.jitter.0.y / r.render_extent.height as f32,
                ],
            )
        }
        None => crate::camera::Godray::OFF,
    }
}

/// Get frame's sun direction, defaulting to up if absent.
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
    /// 5. [`Self::submit_or_defer`]       — render queue submit, or defer into
    ///    a pending batch (uncapped, unpresented frames only)
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

        if self.empty_submit > 0 {
            self.draw_empty_submit();
            return;
        }

        let slot = self.slot;
        use crate::profile::{Meter, scope};
        crate::profile::count(crate::profile::Counter::Rendered);

        let sun = sun_dir(lists);
        let camera_eye = lists.scene.as_ref().map(|scene| {
            (
                crate::camera::Frustum::from_view_proj(&scene.view_proj),
                pipeline::EyeSplit::of(scene.eye),
            )
        });

        // CPU Blend re-source: persistent render-thread state that cannot change
        // until the next command drain. Hoisted above the slot fence so it
        // overlaps the previous frame's GPU work.
        {
            let _p = scope(Meter::Pack);
            if let Some((camera, eye)) = &camera_eye {
                self.prepare_blend_draws(lists, camera, *eye);
            } else {
                self.draw_scratch.clear();
                self.draw_commands.clear();
                self.draw_runs.clear();
            }
        }

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
            let offsets = self.write_immediates(slot, lists, present_target.is_some());
            self.prepare_mesh_draws(slot, lists, sun, camera_eye.as_ref());
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
            // `prepare_derived` already ran in `Frame::begin_3d` (`gate_uniforms`)
            // or `full_bright`; do not redo it here.
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
            self.ubo_ring.write(
                FrameSlot::new(slot),
                &super::uniforms::FrameUniformsExt::derive(u),
            );
        }
        let warp_map = lists
            .scene
            .as_ref()
            .map_or(crate::camera::WarpMap::Identity, |s| s.warp_map);
        // Project the sun to presented uv for the spill-pass godray march.
        // Computed here (not in the copy submit) so it rides the same camera +
        // frame-uniform snapshot the scene was drawn from — and so the bloom
        // chain (same submit as the spill dispatch) sees the same values.
        // `project` returns a strength-0 no-op when godrays are off, the sun
        // is behind the camera, or there is no 3D camera this frame.
        let godray = project_godray(self, lists);
        let spill_live = self.flags.bloom || godray.strength > 0.0;
        let (rs, hdr_readable) = {
            let _p = scope(Meter::Record);
            self.record_render(
                &guard,
                lists,
                offsets,
                present_target.is_some(),
                warp_map,
                godray,
            )
        };

        {
            let _p = scope(Meter::Submit);
            self.submit_or_defer(
                rs,
                slot,
                present_target.is_some() || self.needs_recreate || self.pending_capture.is_some(),
            );
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
            let taa =
                lists
                    .scene
                    .as_ref()
                    .filter(|_| self.flags.taa)
                    .map(|s| super::taa::TaaPresent {
                        view_proj: s.view_proj,
                        eye: s.eye,
                        jitter: s.jitter.0,
                    });
            self.present(
                slot,
                present_target,
                warp_map,
                overlay,
                hdr_readable,
                spill_live,
                taa,
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

    /// Empty command-buffer submit used by `VOXEL_BENCH_EMPTY=K`: wait the slot,
    /// record K distinct empty primaries (begin/end only), one `vkQueueSubmit2`
    /// with K command buffers and the usual single timeline signal, skip present.
    /// Slot wait + timeline signal keep shutdown and FIF reuse intact.
    fn draw_empty_submit(&mut self) {
        let slot = self.slot;
        crate::profile::count(crate::profile::Counter::Rendered);
        self.wait_slot_and_reclaim(slot);
        let k = self.empty_submit.max(1) as usize;
        let primary = self.slots[FrameSlot::new(slot)].cmd;
        let extra_n = k - 1;
        let extra_base = slot * extra_n;
        let mut cmds = Vec::with_capacity(k);
        cmds.push(primary);
        if extra_n > 0 {
            cmds.extend_from_slice(&self.empty_extra[extra_base..extra_base + extra_n]);
        }
        unsafe {
            let device = &self.device.device;
            for &cmd in &cmds {
                device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .expect("command buffer reset failed");
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .expect("begin command buffer failed");
                device
                    .end_command_buffer(cmd)
                    .expect("end command buffer failed");
            }
        }
        let rs = self.timeline.begin_render(cmds[0]);
        {
            let _p = crate::profile::scope(crate::profile::Meter::Submit);
            let extra_wait = self
                .pending_transfer_wait
                .take()
                .map(|value| (self.transfer_lane.semaphore(), value, MESH_CONSUMER_STAGES));
            let completion = unsafe {
                rs.submit_bufs(
                    &self.device.device,
                    self.device.graphics_queue,
                    &self.timeline,
                    &cmds,
                    extra_wait,
                )
            };
            self.slots[FrameSlot::new(slot)].render_value = completion.value();
            self.last_render_value = completion.value();
            crate::profile::count(crate::profile::Counter::Submits);
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
    /// After `RenderPass::end`, if a later pass this frame samples it, the
    /// *sampleable* single-sample image leaves this layout and rests in
    /// [`SAMPLEABLE_DEPTH_REST_LAYOUT`]. The next scene pass of this slot
    /// begins from `UNDEFINED` either way.
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
    /// implement those stages independently. After that barrier (when issued)
    /// the image rests in [`SAMPLEABLE_DEPTH_REST_LAYOUT`]; see that const for
    /// the contract.
    pub(super) fn sampleable_depth_attachment_state(
        &self,
    ) -> (vk::ImageLayout, vk::PipelineStageFlags2, vk::AccessFlags2) {
        sampleable_depth_attachment_state(self.targets.samples, self.depth_pass_layout())
    }

    /// Scene-pass → rest: sampleable depth becomes [`SAMPLEABLE_DEPTH_REST_LAYOUT`].
    ///
    /// Issued only when [`sampleable_depth_consumed`] is true. Src is the
    /// attachment-write scope (depth tests, or COLOR_ATTACHMENT_OUTPUT for the
    /// MSAA SAMPLE_ZERO resolve). Dst covers every consumer that samples it
    /// without a further transition: Hi-Z compute, VRS compute, the quarter-res
    /// spill compute (godrays), and the present-time tonemap fragment (fused TAA).
    /// The present copy is a later submit that waits on the render timeline.
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
    ///
    /// With vsync on, waits [`reclaim_wait_slot`] (one slot earlier than the
    /// reuse target) so the effective in-flight depth stays 2.
    ///
    /// Flushes a still-pending batch that occupies `slot` before waiting:
    /// the wait must never target a timeline value that has not been submitted,
    /// and a deferred frame's host-visible slot (uniforms, cull buffers, query
    /// readback) must not be overwritten until its batch signals.
    fn wait_slot_and_reclaim(&mut self, slot: usize) {
        let vsync = self.vsync.current();
        let wait_slot = reclaim_wait_slot(slot, vsync);
        // Never wait on a timeline value that has not been submitted: if this
        // slot (or the vsync wait-slot) is still in the pending batch, flush
        // first. With `FRAMES_IN_FLIGHT >= SUBMIT_BATCH_MAX + 1` this is a
        // safety net, not the steady-state path.
        if pending_blocks_wait(self.pending_submits.iter().map(|p| p.slot), wait_slot)
            || pending_blocks_wait(self.pending_submits.iter().map(|p| p.slot), slot)
        {
            self.flush_pending_submits();
        }
        assert!(
            !pending_blocks_wait(self.pending_submits.iter().map(|p| p.slot), wait_slot),
            "wait_slot_and_reclaim must not wait on an unsubmitted frame"
        );
        let device = &self.device.device;
        unsafe {
            {
                let _p = crate::profile::scope(crate::profile::Meter::Fence);
                let value = self.slots[FrameSlot::new(wait_slot)].render_value;
                if vsync {
                    self.timeline.wait(device, value);
                } else {
                    const FENCE_SPIN_BUDGET: std::time::Duration =
                        std::time::Duration::from_micros(200);
                    self.timeline.wait_spin(device, value, FENCE_SPIN_BUDGET);
                }
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

    /// Last completed VRS histogram for this slot (`FRAMES_IN_FLIGHT`-frame delayed).
    /// Zeroed when VRS is off or the device has no attachment shading rate.
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

    /// Last completed cull geometry stats for this slot (`FRAMES_IN_FLIGHT`-frame delayed).
    /// `draws.full` / `tris.full` are camera group 0 (full-res opaque);
    /// `draws.cutout` / `tris.cutout` are group 1; `draws.lod` / `tris.lod`
    /// are group 2 (coarse LOD). Gauges no-op when profiling is off.
    fn publish_cull_stats(&self, slot: usize) {
        let [d0, i0, d1, i1, d2, i2, occ] = self.cull.stats(slot);
        crate::profile::gauge(crate::profile::Gauge::DrawsFull, d0 as u64);
        crate::profile::gauge(crate::profile::Gauge::DrawsCutout, d1 as u64);
        crate::profile::gauge(crate::profile::Gauge::DrawsLod, d2 as u64);
        crate::profile::gauge(crate::profile::Gauge::TrisFull, u64::from(i0 / 3));
        crate::profile::gauge(crate::profile::Gauge::TrisCutout, u64::from(i1 / 3));
        crate::profile::gauge(crate::profile::Gauge::TrisLod, u64::from(i2 / 3));
        crate::profile::gauge(crate::profile::Gauge::CulledOcc, occ as u64);
    }

    fn publish_pipe_stats(&mut self, slot: usize) {
        use crate::profile::Gauge;
        let Some((frag, prims_full)) =
            (unsafe { self.pipe_stats.read_into(&self.device.device, slot) })
        else {
            return;
        };
        crate::profile::gauge(Gauge::FragFull, frag[PipeStatPass::OpaqueFull as usize]);
        crate::profile::gauge(Gauge::FragLod, frag[PipeStatPass::OpaqueLod as usize]);
        crate::profile::gauge(Gauge::FragCutout, frag[PipeStatPass::Cutout as usize]);
        crate::profile::gauge(Gauge::FragBlend, frag[PipeStatPass::Transparent as usize]);
        crate::profile::gauge(Gauge::FragSky, frag[PipeStatPass::Sky as usize]);
        crate::profile::gauge(Gauge::PrimsFull, prims_full);
        crate::profile::overdraw_full(super::gpu_timer::overdraw_ratio(
            frag[PipeStatPass::OpaqueFull as usize],
            self.render_extent.width,
            self.render_extent.height,
        ));
    }

    /// Resolves the copy hazard on `slot` before it is rendered into: the
    /// in-flight present copy may still be reading this slot's offscreen
    /// image, which the render below overwrites. Rare (the copy usually
    /// retires well within the in-flight slot cycle) and sub-millisecond.
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
    ///
    /// 2D verts (`d2` / `d2_tex`) are read solely by the present overlay
    /// (`present.rs::record_overlay_present`). Skip those two writes when this
    /// frame will not present; a forced capture always presents. Offset math and
    /// `imm.maintain(total)` stay identical so the buffer layout is stable.
    fn write_immediates(
        &mut self,
        slot: usize,
        lists: &DrawLists,
        will_present: bool,
    ) -> ImmOffsets {
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
                if will_present {
                    imm.write(d2, d2_bytes);
                    imm.write(d2_tex, d2_tex_bytes);
                }
            }
        }
        ImmOffsets {
            line,
            shadow,
            d2,
            d2_tex,
        }
    }

    /// CPU Blend re-source: transparency needs exact far→near ordering the GPU
    /// cull does not provide, so Blend is the ONE pass still resolved CPU-side.
    /// Sourced from the same persistent records, arena directory, and
    /// `visible_mask` — not a per-frame draw list — by iterating resident,
    /// visible Blend-pass slots, frustum-culling, sorting by distance, and
    /// emitting whole-mesh indirect commands (`first_instance = slot` so
    /// placement/style come from the record/dyn SSBOs).
    ///
    /// Safe to run before the slot fence: everything it reads is render-thread
    /// state that cannot change until the next command drain (reclaim frees
    /// only already-retired allocations). The indirect buffer write stays in
    /// [`Self::prepare_mesh_draws`] after the wait.
    fn prepare_blend_draws(
        &mut self,
        lists: &DrawLists,
        camera: &crate::camera::Frustum,
        eye: pipeline::EyeSplit,
    ) {
        use ash::vk::Handle;

        self.draw_scratch.clear();
        self.draw_commands.clear();
        self.draw_runs.clear();

        // Walk the resident, visible, Blend-pass records. The directory's Blend
        // set is exactly that candidate list, so this is O(transparent meshes),
        // never a sweep of the whole slot table.
        let Some(scene) = &lists.scene else {
            return;
        };
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
                Some(run) if run.buffer == entry.buffer && run.pass == entry.pass => run.count += 1,
                _ => self.draw_runs.push(DrawRun {
                    buffer: entry.buffer,
                    pass: entry.pass,
                    first: command_index,
                    count: 1,
                }),
            }
        }
    }

    /// GPU-cull prep, shadow fits, and the Blend indirect write. Runs after the
    /// slot fence: flush and HostBuffer maintains must not race the previous
    /// use of this slot. `camera`/`eye` were computed before the wait.
    fn prepare_mesh_draws(
        &mut self,
        slot: usize,
        lists: &DrawLists,
        sun: glam::DVec3,
        camera_eye: Option<&(crate::camera::Frustum, pipeline::EyeSplit)>,
    ) {
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
            let rebuild = if self.flags.shadows {
                let key = shadow_key(scene.eye, sun, self.records.occluder_rev(), lists, &cfg);
                self.shadow_cache.prepare(Some((key, &cfg)))
            } else {
                self.shadow_cache.prepare(None)
            };
            if !rebuild {
                return None;
            }
            let fits = shadow::PerCascade::new(
                shadow::CASCADES.map(|c| shadow::fit(scene.eye, sun, c, &cfg)),
            );
            self.shadow_cache.store_fits(fits);
            self.flags.shadows.then(|| {
                shadow::CASCADES
                    .map(|c| crate::camera::Frustum::from_view_proj(&fits[c].view_proj.0))
            })
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
        self.cull_frame = if let Some((camera, eye)) = camera_eye {
            if let Some(records) = self.record_buffers {
                let slot_count = records.slots.min(self.arena_dir.live_end());
                let need = slot_count.div_ceil(32) as usize;
                if self.visible_mask.len() < need {
                    self.visible_mask.resize(need, 0);
                }
                let occ = self.occ_params(*eye);
                unsafe {
                    self.cull.prepare(
                        slot,
                        &self.instance.instance,
                        &self.device.device,
                        self.device.physical,
                        &mut self.arena_dir,
                        records,
                        self.records.records(),
                        |s| self.mesh_res.is_arrived(s),
                        slot_count,
                        camera,
                        shadow_frusta.as_ref(),
                        *eye,
                        lists.lod_clip,
                        lists.lod_clip_v,
                        &self.visible_mask[..need],
                        recycled,
                        occ,
                    )
                }
            } else {
                None
            }
        } else {
            None
        };
        self.records.clear_occ_new();

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
        warp_map: crate::camera::WarpMap,
        godray: crate::camera::Godray,
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
                // Combined `opaque` is the three group stamps, not a fourth
                // timestamp — keeps pre-split reports comparable.
                crate::profile::add_ms(
                    crate::profile::Meter::GpuOpaque,
                    GpuPass::opaque_ms(&passes),
                );
                crate::profile::gpu_frame_ms(total);
                if let Some(gap) = gap {
                    crate::profile::gpu_gap_ms(gap, total);
                }
            }
            self.publish_pipe_stats(slot);
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
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin command buffer failed");
            // Start timing before the staged copies so the whole buffer is
            // attributed (the `Copies` stamp closes this first span).
            if profiling {
                self.gpu_timer.begin(device, cmd, slot);
                self.pipe_stats.prepare(device, cmd, slot);
            }

            // Last frame's separate-queue copies: this submission is the first
            // that can draw them (see `MeshResidency::flush_copies`). This
            // frame's copies are flushed next and deferred one frame.
            let copies_pending = self.mesh_res.has_pending() || self.mesh_res.has_deferred();
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
            let minimap = self.minimap.sync(device, cmd, slot);
            if profiling {
                if copies_pending || quad_wait.is_some() || minimap {
                    self.gpu_timer.recorded(slot);
                }
                self.gpu_timer.mark(device, cmd, slot, GpuPass::Copies);
            }
        }

        // GPU cull: emit this frame's opaque/cutout/shadow draw commands from
        // the persistent record set, BEFORE any pass that consumes them (the
        // shadow occluders below and the mesh passes). Outside any rendering
        // scope; its trailing barrier orders the writes against DRAW_INDIRECT.
        if lists.scene.is_some() && self.cull_frame.is_some() && self.record_buffers.is_some() {
            let _g = crate::profile::scope(crate::profile::Meter::RecCull);
            let cpu = self.cull_frame.as_ref().expect("checked").cpu;
            let hiz = (!cpu).then(|| unsafe { self.hiz_sample_for_cull(cmd) });
            let frame = self.cull_frame.as_ref().expect("checked");
            let records = self.record_buffers.expect("checked");
            unsafe {
                let cull_gpu = match hiz {
                    None => false,
                    Some(hiz) => self.cull.record(
                        &self.device.device,
                        &self.device.push_descriptor,
                        cmd,
                        slot,
                        records,
                        frame,
                        hiz,
                    ),
                };
                if profiling {
                    if cull_gpu {
                        self.gpu_timer.recorded(slot);
                    }
                    self.gpu_timer
                        .mark(&self.device.device, cmd, slot, GpuPass::Cull);
                }
            }
        } else if profiling {
            // No dispatch this slot: CPU-zero so the one-cycle-late publish
            // does not report a stale histogram.
            self.cull.clear_stats_cpu(slot);
        }

        // Cascaded shadows: on a miss, `prepare_mesh_draws` already fitted both
        // cascades; publish binding-3 uniforms and render occluders into the
        // *shared* map before the color pass (it leaves the map in
        // SHADER_READ_ONLY_OPTIMAL for mesh3d.frag). Hits skip the producer
        // (and skip `fit()`); the slot UBO is filled from the cached block so
        // sampling matches the resident depth.
        if let Some(scene) = &lists.scene {
            let cfg = crate::skeleton::ShadowCfg::for_coverage(lists.lod_clip);
            let caster_verts = lists.cube_verts.len() as u32;
            let render = self.shadow_cache.pending_rebuild();
            if render {
                let _g = crate::profile::scope(crate::profile::Meter::RecShadow);
                // Previous frame's sampling of the shared map is ordered on the
                // GPU by the entry barrier of `record_shadow_pass`, whose first
                // synchronization scope covers every command earlier in
                // submission order on the graphics queue. `shadow.write_uniforms`
                // is per-slot and fence-waited, so no CPU-visible memory is shared.
                let fits = self
                    .shadow_cache
                    .fits()
                    .expect("pending rebuild stores fits in prepare_mesh_draws");
                let cu = self.shadow_uniforms(&fits, &cfg);
                self.shadow_cache.store_uniforms(cu);
                self.shadow
                    .write_uniforms(slot, &cu, self.shadow_cache.uniforms_gen());
                self.record_shadow_pass(cmd, slot, &fits, scene.eye, &cfg, caster_verts);
                if profiling {
                    unsafe {
                        self.gpu_timer.recorded(slot);
                        self.gpu_timer
                            .mark(&self.device.device, cmd, slot, GpuPass::ShadowMap)
                    };
                }
                if !self.flags.shadows {
                    self.shadow_cache.mark_lit_ready();
                }
            } else if let Some(cu) = self.shadow_cache.uniforms() {
                // Hit / shadows-off after prime: skip the BAR copy when this
                // slot already holds the current cascade generation.
                self.shadow
                    .write_uniforms(slot, cu, self.shadow_cache.uniforms_gen());
            }
        }

        // Cloud LUT: march (or zero) before the scene pass so the sky fragment
        // has a sampled image. Skipped when there is no sky.
        if self.flags.sky
            && lists.sky.is_some()
            && let Some(scene) = &lists.scene
        {
            self.record_sky_cloud_lut(
                cmd,
                slot,
                &scene.frame_uniforms,
                self.pending_capture.is_some(),
            );
        }

        // Bind the rate image classified at the end of this slot's previous
        // use (`FRAMES_IN_FLIGHT` frames ago) — the same staleness the old
        // begin-of-frame classify accepted. First scene pass after
        // create/recreate skips VRS (`vrs_ready` is false); we still classify
        // at end so the next use is primed.
        let vrs_on = lists.scene.is_some() && self.flags.vrs && self.targets.vrs.is_some();
        let do_vrs = vrs_on && self.slots[FrameSlot::new(slot)].vrs_ready;
        let classify_vrs = vrs_on;
        let build_hiz = lists.scene.is_some() && self.flags.occlusion;
        // Depth consumers after the scene pass, computed once: VRS classify,
        // Hi-Z reduce, spill/godrays (presented + bloom or a live march), fused TAA.
        let spill_live = self.flags.bloom || godray.strength > 0.0;
        let sample_depth = sampleable_depth_consumed(
            will_present,
            self.flags.taa,
            spill_live,
            classify_vrs,
            build_hiz,
        );
        // HDR colour is only read by bloom/exposure/spill/tonemap, all of which
        // run on presented frames. Minimap is a separate texture; screenshots
        // copy the swapchain after tonemap; VRS classify reads depth not colour.

        let device = &self.device.device;
        let stamp = |p| {
            if profiling {
                unsafe { self.gpu_timer.mark(device, cmd, slot, p) };
            }
        };
        let pass = {
            let _g = crate::profile::scope(crate::profile::Meter::RecTransitions);
            unsafe {
                RenderPass::begin(
                    self,
                    cmd,
                    slot,
                    lists,
                    offsets,
                    do_vrs,
                    sample_depth,
                    will_present,
                )
            }
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
                    // Each camera group stamps itself (OpaqueFull / OpaqueLod /
                    // Cutout) inside `record_group_indirect_count`.
                    pass.record_mesh_indirect(Pass::Cutout);
                }
                // Sky fills the background (uncovered pixels) right after opaque
                // depth is laid down. It must precede the immediate debug
                // cubes/lines: the highlight lines are depth read-only (no depth
                // write), so a line silhouetted against the background leaves the
                // depth cleared there — drawing sky afterward would overpaint it.
                // Debug geometry and transparent water both composite over the sky.
                {
                    let _g = scope(Meter::RecSky);
                    if profiling {
                        self.pipe_stats
                            .begin_pass(device, cmd, slot, PipeStatPass::Sky);
                    }
                    if self.flags.sky {
                        pass.record_sky();
                    }
                    if profiling {
                        self.pipe_stats
                            .end_pass(device, cmd, slot, PipeStatPass::Sky);
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
                    if profiling {
                        self.pipe_stats
                            .begin_pass(device, cmd, slot, PipeStatPass::Transparent);
                    }
                    pass.record_mesh_indirect(Pass::Blend);
                    if profiling {
                        self.pipe_stats
                            .end_pass(device, cmd, slot, PipeStatPass::Transparent);
                    }
                }
                stamp(GpuPass::Transparent);
            }
        }
        // The offscreen HDR must reach SHADER_READ_ONLY before the tonemap
        // present copy samples it. `end` performs that COLOR_ATTACHMENT→
        // SHADER_READ barrier UNLESS a later offscreen writer runs after it:
        // exposure metering writes after `end` (it owns the finalize). TAA no
        // longer writes the offscreen — it resolves in the present-time tonemap
        // — so TAA-on is the same finalize path as TAA-off. Bloom and exposure
        // metering feed only the tonemap present-copy, so they run solely on
        // frames that will present (`decide_present` already ran; forced capture
        // always presents).
        let run_exposure = lists.scene.is_some() && self.flags.exposure && will_present;
        // Overlay is drawn in the present copy (`GpuTonemap`); this scene-pass
        // stamp stays so the report still lists overlay, and accounts 0.
        stamp(GpuPass::Overlay);
        // Finalize the offscreen to SHADER_READ_ONLY exactly once and obtain the
        // [`HdrReadable`] proof the tonemap present-copy requires. The branches
        // are exhaustive: (a) exposure on + presenting → metering owns the
        // transition; (b) presenting without exposure → the render pass
        // finalizes; (c) unpresented → skip the sampled transition.
        let readable: HdrReadable = {
            let _g = crate::profile::scope(crate::profile::Meter::RecTransitions);
            if run_exposure {
                unsafe { pass.end_deferred(classify_vrs, build_hiz) };
                stamp(GpuPass::Resolve);
                self.finish_hiz(cmd, slot, build_hiz, lists);
                self.finish_vrs_classify(cmd, slot, classify_vrs, lists);
                // Reduce the (jittered, unresolved) frame HDR to per-tile mean
                // log2-luma, publish the smoothed exposure, and finalize the HDR
                // in SHADER_READ. Metering the unresolved image is acceptable:
                // it is spatial and low-frequency.
                let readable = self.record_exposure_pass(cmd, FrameSlot::new(slot));
                if profiling {
                    unsafe {
                        self.gpu_timer.recorded(slot);
                        self.gpu_timer
                            .mark(&self.device.device, cmd, slot, GpuPass::Exposure)
                    };
                }
                readable
            } else if will_present {
                let readable = unsafe { pass.end_sampled(classify_vrs, build_hiz) };
                if profiling {
                    unsafe {
                        self.gpu_timer
                            .mark(&self.device.device, cmd, slot, GpuPass::Resolve)
                    };
                }
                self.finish_hiz(cmd, slot, build_hiz, lists);
                self.finish_vrs_classify(cmd, slot, classify_vrs, lists);
                readable
            } else {
                // Unpresented, no later HDR writer: skip the sampled transition;
                // the next begin discards the offscreen from UNDEFINED. Depth
                // rests only when this frame samples it (classifier); a later
                // present uses a different slot's depth.
                unsafe { pass.end_deferred(classify_vrs, build_hiz) };
                stamp(GpuPass::Resolve);
                self.finish_hiz(cmd, slot, build_hiz, lists);
                self.finish_vrs_classify(cmd, slot, classify_vrs, lists);
                HdrReadable::new(slot)
            }
        };
        // Bloom pyramid + quarter-res spill (bloom composite + godrays). The
        // tonemap present-copy takes one bilinear tap of the spill. Present-only
        // — a dropped mailbox frame never samples either image. Forced capture
        // always presents, so it always gets a fresh chain. The GPU timer's
        // `Bloom` span covers this whole tail (pyramid + spill) when it ran.
        let bloom_work = if will_present {
            self.record_bloom_pass(cmd, FrameSlot::new(slot), warp_map, godray)
        } else {
            false
        };
        // Close the tail: without this stamp bloom/spill work recorded above
        // ends after the last boundary and never reaches the report. Empty
        // when bloom+godrays are off and the black/pyramid prime already ran.
        // (The tonemap/present copy is timed on the copy command buffer; see
        // `submit_present_copy`.)
        if profiling {
            if bloom_work {
                self.gpu_timer.recorded(slot);
            }
            unsafe {
                self.gpu_timer
                    .mark(&self.device.device, cmd, slot, GpuPass::Bloom)
            };
            self.gpu_timer.finish(slot);
            if lists.scene.is_some() {
                self.pipe_stats.finish(slot);
            }
        }
        unsafe {
            self.device
                .device
                .end_command_buffer(cmd)
                .expect("end command buffer failed");
        }
        (rs, readable)
    }

    /// The image+view holding slot `slot`'s HDR offscreen. TAA no longer
    /// rewrites this; the present-time tonemap resolves into a swapchain-sized
    /// history instead.
    pub(super) fn hdr_of(&self, slot: usize) -> (vk::Image, vk::ImageView) {
        (
            self.targets.offscreen[slot].image(),
            self.targets.offscreen[slot].view(),
        )
    }

    /// End-of-frame Hi-Z: this slot's just-written depth, for the next frame's
    /// cull compute. Depth already rests in [`SAMPLEABLE_DEPTH_REST_LAYOUT`];
    /// the pyramid → GENERAL joined the post-scene barrier. The reduce leaves
    /// it in `SHADER_READ_ONLY`.
    fn finish_hiz(&mut self, cmd: vk::CommandBuffer, slot: usize, build: bool, lists: &DrawLists) {
        if !build {
            return;
        }
        let scene = lists.scene.as_ref().expect("build_hiz implies a 3D scene");
        unsafe { self.record_hiz_generate(cmd, slot) };
        self.hiz_history = Some(super::hiz::HizHistory {
            slot,
            view_proj: scene.view_proj,
            eye: super::pipeline::EyeSplit::of(scene.eye),
        });
        if crate::profile::is_enabled() {
            unsafe {
                self.gpu_timer.recorded(slot);
                self.gpu_timer
                    .mark(&self.device.device, cmd, slot, GpuPass::HiZ)
            };
        }
    }

    /// End-of-frame classify: this slot's just-written depth, for the next use
    /// of the slot (`FRAMES_IN_FLIGHT` frames later). Depth already rests in
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
                self.gpu_timer.recorded(slot);
                self.gpu_timer
                    .mark(&self.device.device, cmd, slot, GpuPass::Vrs)
            };
        }
        self.slots[FrameSlot::new(slot)].vrs_ready = true;
        self.slots[FrameSlot::new(slot)].vrs_history = true;
    }

    /// Submits the recorded command buffer, or defers it into the pending
    /// batch. Uncapped unpresented frames join the batch until it reaches
    /// the runtime limit (`VOXEL_SUBMIT_BATCH`) or a flush condition. A
    /// presented / recreate / vsync-on frame flushes any pending batch first,
    /// then submits on its own (batch-size-1 semantics).
    ///
    /// Transfer-lane waits captured at record time ride the submit that
    /// actually queues the command buffer: a batch waits on the max value
    /// among its frames.
    fn submit_or_defer(&mut self, rs: RenderSubmit, slot: usize, present: bool) {
        let extra_wait = self.pending_transfer_wait.take();
        let uncapped = !self.vsync.current();
        // Presented / recreate / vsync: pending batch first, then this frame
        // alone. `submit_batch_action` returns Flush for these, so the second
        // flush submits the just-pushed frame as its own `vkQueueSubmit2`.
        if present || !uncapped {
            self.flush_pending_submits();
        }
        let (signal, cmd) = rs.into_parts();
        self.pending_submits.push(PendingSubmit {
            slot,
            cmd,
            extra_wait,
            signal,
        });
        if submit_batch_action(
            uncapped,
            present,
            self.pending_submits.len(),
            self.submit_batch_limit,
        ) == SubmitBatchAction::Flush
        {
            self.flush_pending_submits();
        }
    }

    /// One `vkQueueSubmit2` for every pending command buffer, in frame order,
    /// with a single timeline signal at the last frame's reserved value.
    /// Every slot in the batch records that value as `render_value`, so
    /// reclamation waits for the whole batch. No-op when the list is empty.
    pub(super) fn flush_pending_submits(&mut self) {
        if self.pending_submits.is_empty() {
            return;
        }
        let extra_wait = self
            .pending_submits
            .iter()
            .filter_map(|p| p.extra_wait)
            .max()
            .map(|value| (self.transfer_lane.semaphore(), value, MESH_CONSUMER_STAGES));
        let signal = self
            .pending_submits
            .last()
            .expect("non-empty pending batch")
            .signal;
        let cmds: Vec<vk::CommandBuffer> = self.pending_submits.iter().map(|p| p.cmd).collect();
        let completion = unsafe {
            self.timeline.submit_render(
                &self.device.device,
                self.device.graphics_queue,
                &cmds,
                signal,
                extra_wait,
            )
        };
        crate::profile::count(crate::profile::Counter::Submits);
        let done = completion.value();
        for p in self.pending_submits.drain(..) {
            self.slots[FrameSlot::new(p.slot)].render_value = done;
        }
        self.last_render_value = done;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FRAMES_IN_FLIGHT, SUBMIT_BATCH_MAX, SubmitBatchAction, pending_blocks_wait,
        reclaim_wait_slot, submit_batch_action,
    };

    #[test]
    fn reclaim_wait_slot_stays_at_two_deep_when_vsync() {
        for slot in 0..FRAMES_IN_FLIGHT as usize {
            assert_eq!(reclaim_wait_slot(slot, false), slot);
            let waited = reclaim_wait_slot(slot, true);
            // Two frames old: not the slot being reused (unless FIF == 2) and
            // not the just-submitted previous slot.
            let prev = (slot + FRAMES_IN_FLIGHT as usize - 1) % FRAMES_IN_FLIGHT as usize;
            assert_ne!(waited, prev, "vsync must not serialize to 1 in flight");
            assert_eq!(
                waited,
                (slot + FRAMES_IN_FLIGHT as usize - 2) % FRAMES_IN_FLIGHT as usize
            );
        }
    }

    #[test]
    fn submit_batch_action_defers_uncapped_unpresented_until_limit() {
        let limit = SUBMIT_BATCH_MAX;
        assert_eq!(
            submit_batch_action(true, false, 1, limit),
            SubmitBatchAction::Defer,
            "first unpresented frame of a batch of {limit} must defer"
        );
        assert_eq!(
            submit_batch_action(true, false, limit, limit),
            SubmitBatchAction::Flush,
            "reaching the limit must flush"
        );
        assert_eq!(
            submit_batch_action(true, false, limit + 1, limit),
            SubmitBatchAction::Flush
        );
    }

    #[test]
    fn submit_batch_action_flush_when_presented_or_capped_or_limit_one() {
        assert_eq!(
            submit_batch_action(true, true, 1, 2),
            SubmitBatchAction::Flush,
            "a presented frame never joins a batch"
        );
        assert_eq!(
            submit_batch_action(false, false, 1, 2),
            SubmitBatchAction::Flush,
            "vsync/capped is batch-size-1"
        );
        assert_eq!(
            submit_batch_action(true, false, 1, 1),
            SubmitBatchAction::Flush,
            "VOXEL_SUBMIT_BATCH=1 is today's per-frame submit"
        );
    }

    #[test]
    fn fif_covers_a_full_batch_plus_the_slot_being_recorded() {
        assert!(FRAMES_IN_FLIGHT as usize > SUBMIT_BATCH_MAX);
    }

    /// Walk the slot ring under the production defer/flush rules. The next
    /// slot to record must not still be in the pending list — otherwise
    /// `wait_slot_and_reclaim` would wait on a value that has not been
    /// submitted (or, worse, on the slot's previous already-signalled value
    /// and reset an unsubmitted command buffer).
    fn simulate_ring(uncapped: bool, limit: usize, present: impl Fn(usize) -> bool) {
        let fif = FRAMES_IN_FLIGHT as usize;
        let mut pending: Vec<usize> = Vec::new();
        let mut slot = 0usize;
        for i in 0..64 {
            let will_present = present(i);
            assert!(
                !pending_blocks_wait(pending.iter().copied(), slot),
                "slot {slot} still pending at reuse (frame {i}, pending {pending:?}); \
                 FRAMES_IN_FLIGHT must be SUBMIT_BATCH_MAX + 1"
            );
            if will_present || !uncapped {
                pending.clear();
            }
            pending.push(slot);
            if submit_batch_action(uncapped, will_present, pending.len(), limit)
                == SubmitBatchAction::Flush
            {
                pending.clear();
            }
            slot = (slot + 1) % fif;
        }
    }

    #[test]
    fn slot_ring_stays_available_under_uncapped_batching() {
        simulate_ring(true, SUBMIT_BATCH_MAX, |i| i % 5 == 0);
        simulate_ring(true, SUBMIT_BATCH_MAX, |_| false);
        simulate_ring(true, 1, |_| false);
        simulate_ring(false, SUBMIT_BATCH_MAX, |_| false);
        simulate_ring(true, SUBMIT_BATCH_MAX, |_| true);
    }

    #[test]
    fn pending_slot_is_not_available_until_flushed() {
        assert!(!pending_blocks_wait(std::iter::empty(), 0));
        assert!(pending_blocks_wait([0], 0));
        assert!(!pending_blocks_wait([0], 1));
        assert!(pending_blocks_wait([0, 1], 1));
        // The wait-path flush: once the blocking slot is submitted, reuse is
        // legal (the subsequent timeline wait covers the batch signal).
        let mut pending = vec![0, 1];
        assert!(pending_blocks_wait(pending.iter().copied(), 0));
        pending.clear();
        assert!(!pending_blocks_wait(pending.iter().copied(), 0));
    }

    #[test]
    fn a_pending_batch_as_large_as_the_ring_occupies_the_next_slot() {
        // Why FRAMES_IN_FLIGHT >= SUBMIT_BATCH_MAX + 1: a pending list of FIF
        // frames would occupy the slot about to be reused, and the wait path
        // would have to flush before waiting (the safety net in
        // `wait_slot_and_reclaim`). The extra slot keeps that path cold.
        let fif = FRAMES_IN_FLIGHT as usize;
        let pending: Vec<usize> = (0..fif).collect();
        assert!(
            pending_blocks_wait(pending.iter().copied(), 0),
            "reusing slot 0 while it is still pending is illegal"
        );
        let slack: Vec<usize> = (0..fif - 1).collect();
        assert!(
            !pending_blocks_wait(slack.iter().copied(), fif - 1),
            "a batch of FIF-1 leaves the next ring slot free"
        );
    }
}
