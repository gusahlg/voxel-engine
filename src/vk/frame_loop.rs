//! Per-frame render loop: `Renderer::draw_frame` and the stages it runs.
//! Split out of `mod.rs` so later work can touch the loop without opening
//! the renderer setup and teardown.

use ash::vk;

use crate::frame::DrawLists;
use crate::mesh::Pass;
use crate::skeleton::FrameSlot;

use super::block_textures::BLOCK_TEXTURE_CONSUMER_STAGES;
use super::buffers::{FRAMES_IN_FLIGHT, MESH_CONSUMER_STAGES};
use super::gpu_timer::{GpuPass, PipeStatPass};
use super::materials::MATERIAL_CONSUMER_STAGES;
use super::pipeline;
use super::present::{HdrReadable, OverlayPresent};
use super::scene_pass::RenderPass;
use super::timeline::{RenderSubmit, acquire_next_image};
use super::{Env, Renderer, SAMPLEABLE_DEPTH_REST_LAYOUT, sampleable_depth_consumed};

pub(crate) use super::draw_prep::{DrawEntry, DrawRun, ImmOffsets};
use super::submit::fold_transfer_wait;
pub(crate) use super::submit::{GpuBoundState, PendingSubmit, submit_batch_limit};

/// Token returned by `acquire_slot` proving the slot is safe to render into
/// (its copy hazard is resolved).
struct SlotGuard(usize);
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
        self.mesh_res.note_frame();
        crate::profile::gauge(
            crate::profile::Gauge::PoolAcquires,
            super::mesh_staging::take_acquire_count(),
        );
        crate::profile::gauge(
            crate::profile::Gauge::PoolAabbFallback,
            super::mesh_staging::take_aabb_fallback_count(),
        );
        crate::profile::gauge(crate::profile::Gauge::PoolCopies, 0);
        crate::profile::gauge(crate::profile::Gauge::PoolArrivalFrames, 0);

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
                crate::profile::gauge(crate::profile::Gauge::DrawsBlend, 0);
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
            self.ubo_ring
                .write_from_gpu(FrameSlot::new(slot), u, self.flags);
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
    /// Tracks which offscreen slot the current copy is reading from.
    pub(super) fn track_copy(&mut self, slot: usize) {
        self.copy_slot = Some(slot);
    }

    /// Forgets any tracked copy hazard: the copy has been waited to
    /// completion, or the offscreen images it read no longer exist.
    pub(super) fn clear_copy(&mut self) {
        self.copy_slot = None;
    }
    /// Last completed VRS histogram for this slot (`FRAMES_IN_FLIGHT`-frame delayed).
    /// Zeroed when VRS is off or the device has no attachment shading rate.
    pub(super) fn publish_vrs_mix(&self, slot: usize) {
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
    pub(super) fn publish_cull_stats(&self, slot: usize) {
        let [d0, i0, d1, i1, d2, i2] = self.cull.stats(slot);
        crate::profile::gauge(crate::profile::Gauge::DrawsFull, d0 as u64);
        crate::profile::gauge(crate::profile::Gauge::DrawsCutout, d1 as u64);
        crate::profile::gauge(crate::profile::Gauge::DrawsLod, d2 as u64);
        crate::profile::gauge(crate::profile::Gauge::TrisFull, u64::from(i0 / 3));
        crate::profile::gauge(crate::profile::Gauge::TrisCutout, u64::from(i1 / 3));
        crate::profile::gauge(crate::profile::Gauge::TrisLod, u64::from(i2 / 3));
        crate::profile::gauge(
            crate::profile::Gauge::Arenas,
            self.arena_dir.arena_count() as u64,
        );
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
        unsafe {
            self.gpu_timer.read_load(&self.device.device, slot);
        }
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
        // Overwrite of already-sampled layers: pending frames that still
        // sample SHADER_READ must be on the graphics queue before a
        // dedicated-family release (or a same-family extra wait).
        if self.block_textures.has_overwrite_pending() || self.materials.has_overwrite_pending() {
            self.flush_pending_submits();
        }
        // Begin render submission; this gets the timeline value to stamp mesh copies.
        let rs = self.timeline.begin_render(cmd);
        let done_at = rs.value();
        let copies_pending;
        let grew;
        let quad_wait_some;
        let tex_wait_some;
        let mat_wait_some;
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
            self.gpu_timer.begin_load(device, cmd, slot);
            if profiling {
                self.gpu_timer.begin(device, cmd, slot);
                self.pipe_stats.prepare(device, cmd, slot);
            }

            // Last frame's separate-queue copies: this submission is the first
            // that can draw them (see `MeshResidency::flush_copies`). This
            // frame's copies are flushed next and deferred one frame.
            copies_pending = self.mesh_res.has_pending() || self.mesh_res.has_deferred();
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
            let tex = self.block_textures.flush(
                &self.instance.instance,
                device,
                self.device.physical,
                &mut self.transfer_lane,
                cmd,
                self.device.graphics_queue,
                self.device.graphics_family,
                &self.timeline,
                self.last_render_value,
                done_at,
            );
            grew = tex.retire.is_some();
            if let Some((stamp, retired)) = tex.retire {
                self.retired_textures.push(stamp, retired);
            }
            let mat_wait = self.materials.flush(
                &self.instance.instance,
                device,
                self.device.physical,
                &mut self.transfer_lane,
                cmd,
                self.device.graphics_queue,
                self.device.graphics_family,
                &self.timeline,
                self.last_render_value,
                done_at,
            );
            self.pending_transfer_wait = fold_transfer_wait(
                fold_transfer_wait(
                    match (deferred, quad_wait) {
                        (Some(a), Some(b)) => Some((a.max(b), MESH_CONSUMER_STAGES)),
                        (Some(v), None) | (None, Some(v)) => Some((v, MESH_CONSUMER_STAGES)),
                        (None, None) => None,
                    },
                    tex.transfer_wait
                        .map(|v| (v, BLOCK_TEXTURE_CONSUMER_STAGES)),
                ),
                mat_wait.map(|v| (v, MATERIAL_CONSUMER_STAGES)),
            );
            quad_wait_some = quad_wait.is_some();
            tex_wait_some = tex.transfer_wait.is_some();
            mat_wait_some = mat_wait.is_some();
        }

        // Same-queue compute jobs: budgeted prefix before the scene.
        // Dedicated/async tiers already flushed in the render loop.
        // Empty queue → no commands (idle frames are untouched).
        self.record_compute_fallback(cmd, done_at);

        let minimap = unsafe { self.minimap.sync(&self.device.device, cmd, slot) };
        if profiling {
            if copies_pending || quad_wait_some || tex_wait_some || mat_wait_some || grew || minimap
            {
                self.gpu_timer.recorded(slot);
            }
            unsafe {
                self.gpu_timer
                    .mark(&self.device.device, cmd, slot, GpuPass::Copies);
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
            let _g = crate::profile::scope(crate::profile::Meter::RecCull);
            unsafe {
                let cull_gpu = self.cull.record(
                    &self.device.device,
                    &self.device.push_descriptor,
                    cmd,
                    slot,
                    records,
                    frame,
                );
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
        // Depth consumers after the scene pass, computed once: VRS classify,
        // spill/godrays (presented + bloom or a live march), fused TAA.
        let spill_live = self.flags.bloom || godray.strength > 0.0;
        // Absorb stores this frame's depth so the *next* frame can sample it.
        // Known here because `prepare_blend_draws` already filled `draw_runs`.
        let absorb_this_frame = self.pipelines.mesh3d_transparent_absorb.is_some()
            && self.draw_runs.iter().any(|run| run.pass == Pass::Blend);
        let sample_depth = sampleable_depth_consumed(
            will_present,
            self.flags.taa,
            spill_live,
            classify_vrs,
            absorb_this_frame,
        );
        // Lean opaque/LOD fragments: compile-time equivalent of every optional
        // lighting lane off and fog off. Chosen once per frame from flags.
        let mesh_lean = super::uniforms::mesh_lean(&self.flags);
        // HDR colour is only read by bloom/exposure/spill/tonemap, all of which
        // run on presented frames. Minimap is a separate texture; screenshots
        // copy the swapchain after tonemap; VRS classify reads depth not colour.

        let prev_valid = self.prev_depth.valid(slot, self.render_extent);
        let depth_sampled = self.prev_depth.begin_slot(slot);
        if absorb_this_frame && !prev_valid {
            self.ensure_prev_depth_dummy(cmd);
        }
        if absorb_this_frame && prev_valid {
            self.prev_depth
                .mark_sampled(super::PrevDepthTrack::prev_slot(slot));
        }

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
                    absorb_this_frame,
                    depth_sampled,
                    mesh_lean,
                )
            }
        };
        if lists.scene.is_none() {
            crate::profile::gauge(crate::profile::Gauge::CallsFull, 0);
            crate::profile::gauge(crate::profile::Gauge::CallsLod, 0);
            crate::profile::gauge(crate::profile::Gauge::CallsBlend, 0);
        }
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
                unsafe { pass.end_deferred(classify_vrs) };
                stamp(GpuPass::Resolve);
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
                let readable = unsafe { pass.end_sampled(classify_vrs) };
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
                // rests only when this frame samples it (classifier); a later
                // present uses a different slot's depth.
                unsafe { pass.end_deferred(classify_vrs) };
                stamp(GpuPass::Resolve);
                self.finish_vrs_classify(cmd, slot, classify_vrs, lists);
                HdrReadable::new(slot)
            }
        };
        self.prev_depth
            .finish(slot, sample_depth, self.render_extent);
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
                    .mark(&self.device.device, cmd, slot, GpuPass::Bloom);
                self.gpu_timer.finish(&self.device.device, cmd, slot);
            }
            if lists.scene.is_some() {
                self.pipe_stats.finish(slot);
            }
        }
        unsafe {
            self.gpu_timer.end_load(&self.device.device, cmd, slot);
            self.device
                .device
                .end_command_buffer(cmd)
                .expect("end command buffer failed");
        }
        (rs, readable)
    }

    /// Prime the 1×1 dummy depth to `SHADER_READ_ONLY_OPTIMAL` so binding 5 is
    /// a valid descriptor when previous-frame depth is missing. One transition
    /// from UNDEFINED; subsequent calls are no-ops.
    fn ensure_prev_depth_dummy(&mut self, cmd: vk::CommandBuffer) {
        if self.prev_depth_dummy.layout() == SAMPLEABLE_DEPTH_REST_LAYOUT {
            return;
        }
        self.prev_depth_dummy.transition(
            &self.device.device,
            cmd,
            super::image::LayoutUse::FragmentSampled,
        );
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
}
