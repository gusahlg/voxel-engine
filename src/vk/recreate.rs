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
use super::targets::{RenderTargets, next_lower, walk_ladder};
use super::{Renderer, SampleCount, create_present_semaphores, scaled_extent};

/// Whether to keep live render targets or free them after the requested rung
/// fails at recreate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecreateOomStrategy {
    /// Previous targets still match the new size and the request was a
    /// higher-memory rung: keep them and revert MSAA / scale.
    Retain,
    /// Previous targets cannot serve (wrong size, or the failed request was
    /// already at or below them in memory): free first, then walk from the
    /// request.
    FreeFirst,
}

/// Decide retain vs free-first after the requested render-target rung fails.
///
/// `extent_changed` is true when the previous targets' pixel size does not
/// match the new swapchain at the previous scale. Requested vs previous is
/// first-order target memory: pixel count grows with `scale²`, MSAA
/// multiplies the sample-dependent planes.
fn recreate_oom_strategy(
    extent_changed: bool,
    requested_msaa: SampleCount,
    requested_scale: f32,
    previous_msaa: SampleCount,
    previous_scale: f32,
) -> RecreateOomStrategy {
    if extent_changed
        || rung_at_or_below(
            requested_msaa,
            requested_scale,
            previous_msaa,
            previous_scale,
        )
    {
        RecreateOomStrategy::FreeFirst
    } else {
        RecreateOomStrategy::Retain
    }
}

fn rung_at_or_below(
    requested_msaa: SampleCount,
    requested_scale: f32,
    previous_msaa: SampleCount,
    previous_scale: f32,
) -> bool {
    rung_memory_weight(requested_msaa, requested_scale)
        <= rung_memory_weight(previous_msaa, previous_scale)
}

fn rung_memory_weight(msaa: SampleCount, scale: f32) -> u64 {
    let cents = (scale * 100.0).round() as u64;
    u64::from(msaa.as_u32()) * cents * cents
}

impl Renderer {
    /// While no frames are being submitted (minimized window): waits out the
    /// in-flight fences, flushes any staged mesh copies with a standalone
    /// submit, and frees the whole retire queue.
    pub(super) unsafe fn reclaim_while_idle(&mut self) {
        self.flush_pending_submits();
        if !self.mesh_res.has_pending() && !self.mesh_res.has_garbage() {
            return;
        }
        let device = &self.device.device;
        unsafe {
            // Wait for all in-flight submits to complete. Pending batches
            // were flushed above so `last_reserved` is a submitted value.
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
        // Unsubmitted command buffers are invisible to `device_wait_idle`
        // and would keep sampling images this rebuild destroys.
        self.flush_pending_submits();
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
                    log::warn!("{}", render_target_oom_message(&err));
                    let at_prev_scale = scaled_extent(self.swapchain.extent, prev_scale);
                    let extent_changed = prev_extent.width != at_prev_scale.width
                        || prev_extent.height != at_prev_scale.height;
                    let strategy = recreate_oom_strategy(
                        extent_changed,
                        requested_msaa,
                        requested_scale,
                        prev_msaa,
                        prev_scale,
                    );
                    let found = {
                        let instance = &self.instance.instance;
                        let vk_device = &self.device.device;
                        let physical = self.device.physical;
                        let fsr = self.device.fragment_shading_rate.as_ref();
                        let swapchain_extent = self.swapchain.extent;
                        let try_alloc = |msaa: SampleCount, scale: f32| {
                            RenderTargets::new(
                                instance,
                                vk_device,
                                physical,
                                scaled_extent(swapchain_extent, scale),
                                msaa,
                                fsr,
                            )
                        };
                        let pack = |targets, msaa, scale| {
                            (targets, msaa, scale, scaled_extent(swapchain_extent, scale))
                        };
                        match strategy {
                            RecreateOomStrategy::FreeFirst => {
                                log::info!(
                                    "renderer: freeing previous render targets before walking the ladder (requested MSAA {} / scale {}, previous {} / {})",
                                    requested_msaa.as_u32(),
                                    requested_scale,
                                    prev_msaa.as_u32(),
                                    prev_scale,
                                );
                                // Device is already idle from the wait at the start
                                // of `apply_pending`.
                                self.targets.destroy(vk_device);
                                walk_ladder(requested_msaa, requested_scale, try_alloc)
                                    .map(|(targets, msaa, scale)| pack(targets, msaa, scale))
                            }
                            RecreateOomStrategy::Retain => {
                                log::info!(
                                    "renderer: retaining previous render targets (MSAA {} / scale {}); requested MSAA {} / scale {} did not fit",
                                    prev_msaa.as_u32(),
                                    prev_scale,
                                    requested_msaa.as_u32(),
                                    requested_scale,
                                );
                                next_lower(requested_msaa, requested_scale).and_then(
                                    |(msaa, scale)| {
                                        walk_ladder(msaa, scale, |msaa, scale| {
                                            let extent = scaled_extent(swapchain_extent, scale);
                                            // Live images already match this rung: do not
                                            // allocate a second copy, and do not walk below
                                            // a working config.
                                            if msaa == prev_msaa
                                                && (scale - prev_scale).abs() <= f32::EPSILON
                                                && prev_extent.width == extent.width
                                                && prev_extent.height == extent.height
                                            {
                                                return Ok(None);
                                            }
                                            try_alloc(msaa, scale).map(Some)
                                        })
                                        .and_then(
                                            |(targets, msaa, scale)| {
                                                targets.map(|t| pack(t, msaa, scale))
                                            },
                                        )
                                    },
                                )
                            }
                        }
                    };
                    if let Some((new_targets, msaa, scale, extent)) = found {
                        let fell_back = msaa != requested_msaa
                            || (scale - requested_scale).abs() > f32::EPSILON;
                        if fell_back {
                            log::warn!(
                                "renderer: render targets fell back to MSAA {} / render scale {} (requested {} / {})",
                                msaa.as_u32(),
                                scale,
                                requested_msaa.as_u32(),
                                requested_scale,
                            );
                        }
                        if matches!(strategy, RecreateOomStrategy::Retain) {
                            self.targets.destroy(&self.device.device);
                        }
                        self.targets = new_targets;
                        self.msaa = super::Pending::new(msaa);
                        self.render_scale = super::Pending::new(scale);
                        self.render_extent = extent;
                        replaced_targets = true;
                    } else {
                        match strategy {
                            RecreateOomStrategy::FreeFirst => {
                                panic!(
                                    "renderer: could not allocate any render-target rung after freeing previous targets (requested MSAA {} / scale {})",
                                    requested_msaa.as_u32(),
                                    requested_scale,
                                );
                            }
                            RecreateOomStrategy::Retain => {
                                // Previous targets still exist: keep them and revert
                                // MSAA / scale so the next frame matches live GPU state.
                                self.msaa = super::Pending::new(prev_msaa);
                                self.render_scale = super::Pending::new(prev_scale);
                                self.render_extent = prev_extent;
                            }
                        }
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
                // Depth images are new (UNDEFINED): previous-frame absorb
                // samples are invalid until a subsequent store.
                self.prev_depth.invalidate();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recreate_oom_strategy_retains_higher_request_at_same_extent() {
        assert_eq!(
            recreate_oom_strategy(false, SampleCount::X8, 2.0, SampleCount::X2, 2.0),
            RecreateOomStrategy::Retain,
        );
        assert_eq!(
            recreate_oom_strategy(false, SampleCount::X2, 2.0, SampleCount::X2, 1.0),
            RecreateOomStrategy::Retain,
        );
        assert_eq!(
            recreate_oom_strategy(false, SampleCount::X4, 1.0, SampleCount::X1, 1.0),
            RecreateOomStrategy::Retain,
        );
    }

    #[test]
    fn recreate_oom_strategy_frees_first_when_extent_changed() {
        // Fullscreen switch: previous MSAA 2 / 2.0 cannot serve the new size,
        // even though the request is a higher-memory rung.
        assert_eq!(
            recreate_oom_strategy(true, SampleCount::X8, 2.0, SampleCount::X2, 2.0),
            RecreateOomStrategy::FreeFirst,
        );
        assert_eq!(
            recreate_oom_strategy(true, SampleCount::X2, 2.0, SampleCount::X2, 2.0),
            RecreateOomStrategy::FreeFirst,
        );
        assert_eq!(
            recreate_oom_strategy(true, SampleCount::X1, 0.75, SampleCount::X2, 2.0),
            RecreateOomStrategy::FreeFirst,
        );
    }

    #[test]
    fn recreate_oom_strategy_frees_first_when_requested_at_or_below_previous() {
        // Same config (at), lower MSAA, lower scale: previous is holding the
        // memory the retry needs.
        assert_eq!(
            recreate_oom_strategy(false, SampleCount::X2, 2.0, SampleCount::X2, 2.0),
            RecreateOomStrategy::FreeFirst,
        );
        assert_eq!(
            recreate_oom_strategy(false, SampleCount::X1, 2.0, SampleCount::X2, 2.0),
            RecreateOomStrategy::FreeFirst,
        );
        assert_eq!(
            recreate_oom_strategy(false, SampleCount::X1, 0.75, SampleCount::X2, 2.0),
            RecreateOomStrategy::FreeFirst,
        );
    }
}
