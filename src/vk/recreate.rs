//! Swapchain-sized rebuilds: idle reclaim and applying pending vsync/MSAA/scale.
//! Split out of `mod.rs` so later work can touch recreation without opening
//! the frame loop.

use ash::vk;

use crate::skeleton::FrameSlot;

use super::buffers::{FRAMES_IN_FLIGHT, MESH_CONSUMER_STAGES};
use super::image::render_target_oom_message;
use super::pipeline::Pipelines;
use super::render_client::RenderReturn;
use super::swapchain::Swapchain;
use super::targets::RenderTargets;
use super::{Renderer, SampleCount, create_present_semaphores, scaled_extent};

impl Renderer {
    /// While no frames are being submitted (minimized window): waits out the
    /// in-flight fences, flushes any staged mesh copies with a standalone
    /// submit, and frees the whole retire queue.
    pub(super) unsafe fn reclaim_while_idle(&mut self) {
        if !self.mesh_res.has_pending() && !self.mesh_res.has_garbage() {
            return;
        }
        let device = &self.device.device;
        unsafe {
            // Wait for all in-flight submits to complete.
            self.timeline.wait(device, self.timeline.last_reserved());
            self.copy_slot = None;

            let had_pending = self.mesh_res.has_pending();
            if had_pending || self.mesh_res.has_deferred() {
                // Reuse slot 0's command buffer. Always real and valid: even
                // under a separate transfer queue, the `DedicatedFamily` tier
                // needs a real graphics-side command buffer to record its
                // ownership-transfer ACQUIRE barrier into (see
                // `MeshResidency::flush_copies`). Take the deferred arrival
                // after this flush so idle reclaim waits out this batch too
                // (live frames take *before* flush, one frame later).
                let cmd = self.slots[FrameSlot::new(0)].cmd;
                device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .expect("command buffer reset failed");
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device
                    .begin_command_buffer(cmd, &begin)
                    .expect("begin command buffer failed");
                if had_pending {
                    self.mesh_res.flush_copies(
                        device,
                        &mut self.transfer_lane,
                        cmd,
                        self.device.graphics_family,
                        self.last_render_value,
                    );
                }
                let transfer_wait = self.mesh_res.take_deferred_arrival(device, cmd);
                device
                    .end_command_buffer(cmd)
                    .expect("end command buffer failed");
                let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
                let wait_info = transfer_wait.map(|value| {
                    [vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.transfer_lane.semaphore())
                        .value(value.raw())
                        .stage_mask(MESH_CONSUMER_STAGES)]
                });
                let mut submit = vk::SubmitInfo2::default().command_buffer_infos(&cmd_info);
                if let Some(wait_info) = &wait_info {
                    submit = submit.wait_semaphore_infos(wait_info);
                }
                device
                    .queue_submit2(self.device.graphics_queue, &[submit], vk::Fence::null())
                    .expect("queue submit failed");
                // Waiting graphics idle here transitively proves the transfer
                // queue's copy completed too: this submission waited on its
                // semaphore before executing, so graphics cannot have
                // finished without that wait already being satisfied.
                device
                    .queue_wait_idle(self.device.graphics_queue)
                    .expect("queue wait failed");
                // Same reveal step as the live-frame path (see `draw_frame`):
                // without this, a mesh uploaded just before minimizing stays
                // gated at arena word 0 forever once the window is restored.
                let arrived = self.mesh_res.take_arrived();
                self.records.mark_arrived(&arrived);
            }

            // GPU idle + copies flushed: everything retired returns to main.
            let ret = &self.ret;
            self.mesh_res
                .collect_all(&mut |a| drop(ret.send(RenderReturn::FreeAlloc(a))));
        }
    }

    /// Applies pending vsync/MSAA changes and rebuilds swapchain-sized state.
    pub(super) unsafe fn apply_pending(&mut self) {
        unsafe {
            self.device
                .device
                .device_wait_idle()
                .expect("device_wait_idle failed");

            let size = self.size;
            if size.width == 0 || size.height == 0 {
                // Still minimized: can't rebuild swapchain yet.
                return;
            }

            // Commit pending changes; vsync must apply before swapchain rebuild.
            self.vsync.commit();

            let new_swapchain = Swapchain::new(
                &self.instance.instance,
                &self.device,
                &self.surface_loader,
                self.surface,
                size,
                self.vsync.effective(),
                self.swapchain.swapchain,
            );
            self.swapchain.destroy(&self.device.device);
            let format_changed = new_swapchain.format != self.swapchain.format;
            self.swapchain = new_swapchain;

            let prev_msaa = self.msaa.current();
            let prev_scale = self.render_scale.current();
            let prev_extent = self.render_extent;
            let prev_samples = self.targets.samples;
            self.msaa.commit();
            self.render_scale.commit();
            let requested_msaa = self.msaa.current();
            let requested_scale = self.render_scale.current();
            let requested_extent = scaled_extent(self.swapchain.extent, requested_scale);

            let mut replaced_targets = false;
            match RenderTargets::new(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                requested_extent,
                requested_msaa,
                self.device.fragment_shading_rate.as_ref(),
            ) {
                Ok(new_targets) => {
                    self.targets.destroy(&self.device.device);
                    self.targets = new_targets;
                    self.render_extent = requested_extent;
                    replaced_targets = true;
                }
                Err(err) => {
                    log::error!("{}", render_target_oom_message(&err));
                    let fallback_msaa = SampleCount::X1;
                    let fallback_scale = 1.0;
                    let fallback_extent = scaled_extent(self.swapchain.extent, fallback_scale);
                    let already_fallback = requested_msaa == fallback_msaa
                        && (requested_scale - fallback_scale).abs() <= f32::EPSILON;
                    let fallback_is_previous = prev_msaa == fallback_msaa
                        && (prev_scale - fallback_scale).abs() <= f32::EPSILON
                        && prev_extent.width == fallback_extent.width
                        && prev_extent.height == fallback_extent.height;
                    let fallback = if already_fallback || fallback_is_previous {
                        None
                    } else {
                        log::warn!("renderer: retrying render targets at MSAA 1 / render scale 1");
                        match RenderTargets::new(
                            &self.instance.instance,
                            &self.device.device,
                            self.device.physical,
                            fallback_extent,
                            fallback_msaa,
                            self.device.fragment_shading_rate.as_ref(),
                        ) {
                            Ok(new_targets) => Some(new_targets),
                            Err(err) => {
                                log::error!("{}", render_target_oom_message(&err));
                                None
                            }
                        }
                    };
                    if let Some(new_targets) = fallback {
                        self.targets.destroy(&self.device.device);
                        self.targets = new_targets;
                        self.msaa = super::Pending::new(fallback_msaa);
                        self.render_scale = super::Pending::new(fallback_scale);
                        self.render_extent = fallback_extent;
                        replaced_targets = true;
                    } else {
                        // Previous targets still exist: keep them and revert
                        // MSAA / scale so the next frame matches live GPU state.
                        self.msaa = super::Pending::new(prev_msaa);
                        self.render_scale = super::Pending::new(prev_scale);
                        self.render_extent = prev_extent;
                    }
                }
            }

            let memory_props = self
                .instance
                .instance
                .get_physical_device_memory_properties(self.device.physical);
            // Exposure's tile grid tracks the render extent: rebuild its GPU
            // resources in place (the published `ExposureShared` cell the main
            // thread holds is preserved, so `compose()` keeps reading it).
            if self.render_extent.width != prev_extent.width
                || self.render_extent.height != prev_extent.height
            {
                self.exposure
                    .recreate(&self.device.device, &memory_props, self.render_extent);
            }
            // History is swapchain-sized. Recreate rebuilds the images only
            // when that extent changed; a render-scale-only apply still
            // invalidates temporal state (new reconstruction kernel / sample
            // grid must not mix with the previous present's history).
            if let Err(err) =
                self.taa
                    .recreate(&self.device.device, &memory_props, self.swapchain.extent)
            {
                log::error!("{}", render_target_oom_message(&err));
            }

            // Offscreen images recreated; clear copy tracking. Always, because
            // the swapchain images themselves were replaced.
            self.clear_copy();
            if replaced_targets {
                // Depth and rate images recreated (layout UNDEFINED): skip VRS until
                // a classify at the end of the first post-recreate use primes the
                // rate image. Sampleable depth begins from UNDEFINED regardless.
                for slot in 0..FRAMES_IN_FLIGHT as usize {
                    let s = &mut self.slots[FrameSlot::new(slot)];
                    s.vrs_ready = false;
                    s.vrs_history = false;
                }
                // Shared shadow map is UNDEFINED after recreate: force a rewrite.
                self.shadow_cache.invalidate();
                // LUT images are UNDEFINED after recreate.
                self.sky_cloud.invalidate();
            }

            let samples_changed = self.targets.samples != prev_samples;
            if samples_changed || format_changed {
                self.pipelines.destroy(&self.device.device);
                self.pipelines = Pipelines::new(
                    &self.device.device,
                    self.pipeline_cache,
                    self.targets.color_format,
                    self.swapchain.format,
                    self.targets.depth_format,
                    self.targets.samples,
                    self.atlas.set_layout,
                    self.mesh3d_set_layout,
                    self.device.fragment_shading_rate.as_ref(),
                    self.device.dynamic_rendering_local_read,
                    self.device.independent_blend,
                );
            }

            for &sem in &self.present_semaphores {
                sem.destroy(&self.device.device);
            }
            self.present_semaphores =
                create_present_semaphores(&self.device.device, self.swapchain.images.len());

            self.needs_recreate = false;
        }
    }
}
