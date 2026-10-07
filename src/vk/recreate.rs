//! Swapchain and render-target rebuilds.
//!
//! [`recreate_plan`] is pure and host-testable. [`Renderer::apply_pending`]
//! follows it: a same-size resize or a stale swapchain at an unchanged
//! extent does not rebuild render targets, and an empty plan returns without
//! idling the device. Split out of `mod.rs` so recreation can change without
//! opening the frame loop.

use ash::vk;

use crate::skeleton::FrameSlot;

use super::buffers::{FRAMES_IN_FLIGHT, MESH_CONSUMER_STAGES};
use super::image::render_target_oom_message;
use super::pipeline::Pipelines;
use super::render_client::RenderReturn;
use super::swapchain::{Swapchain, choose_present_mode, choose_swapchain_extent};
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

/// What [`Renderer::apply_pending`] will rebuild. Both false means the flags
/// are stale and the apply must not touch the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecreatePlan {
    swapchain: bool,
    targets: bool,
}

/// Which inputs differed. Logged with the plan; a scale change that rounds
/// to the same render extent is still reported, even if `targets` is false.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecreateFacts {
    stale: bool,
    resize: bool,
    present_mode: bool,
    msaa: bool,
    scale: bool,
}

impl RecreateFacts {
    fn label(self) -> String {
        let parts = [
            self.stale.then_some("stale"),
            self.resize.then_some("resize"),
            self.present_mode.then_some("present mode"),
            self.msaa.then_some("msaa"),
            self.scale.then_some("scale"),
        ];
        let mut out = String::new();
        for part in parts.into_iter().flatten() {
            if !out.is_empty() {
                out.push_str(", ");
            }
            out.push_str(part);
        }
        if out.is_empty() {
            "nothing".to_string()
        } else {
            out
        }
    }
}

/// Inputs to [`recreate_plan`]. Extents and present modes are plain values so
/// the decision can be tested without a device.
#[derive(Clone, Copy)]
struct RecreateInput {
    /// OUT_OF_DATE, SURFACE_LOST, or `recreate_if_stale`'s size mismatch.
    stale: bool,
    window: vk::Extent2D,
    swapchain_extent: vk::Extent2D,
    /// Extent [`choose_swapchain_extent`] would select. Pass
    /// `swapchain_extent` when the surface was not queried.
    next_swapchain_extent: vk::Extent2D,
    current_present_mode: vk::PresentModeKHR,
    /// Mode [`choose_present_mode`] would pick for the requested vsync.
    requested_present_mode: vk::PresentModeKHR,
    current_msaa: SampleCount,
    requested_msaa: SampleCount,
    current_scale: f32,
    requested_scale: f32,
    render_extent: vk::Extent2D,
}

/// Swapchain is rebuilt when it is stale, the window (or the surface-chosen
/// extent) differs from the live swapchain, or the present mode would change.
/// Render targets are rebuilt when the render extent that follows from the
/// extent the swapchain will have and the requested scale, or the requested
/// MSAA, differs from the live targets. Nothing else rebuilds them.
fn recreate_plan(input: RecreateInput) -> (RecreatePlan, RecreateFacts) {
    let present_mode = input.current_present_mode != input.requested_present_mode;
    let msaa = input.current_msaa != input.requested_msaa;
    let scale = (input.current_scale - input.requested_scale).abs() > f32::EPSILON;
    let resize = input.window != input.swapchain_extent
        || input.next_swapchain_extent != input.swapchain_extent;
    let swapchain = input.stale || resize || present_mode;
    // Targets follow the extent the swapchain will actually have. A
    // present-mode or same-size stale rebuild keeps the current extent.
    let base = if swapchain {
        input.next_swapchain_extent
    } else {
        input.swapchain_extent
    };
    let requested = scaled_extent(base, input.requested_scale);
    let targets = requested != input.render_extent || msaa;
    (
        RecreatePlan { swapchain, targets },
        RecreateFacts {
            stale: input.stale,
            resize,
            present_mode,
            msaa,
            scale,
        },
    )
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

    /// Reads the inputs [`recreate_plan`] needs. Queries the surface only
    /// when the swapchain might be rebuilt (stale, size, or vsync); MSAA and
    /// scale alone reuse the live extent and present mode.
    fn gather_recreate_input(&self) -> RecreateInput {
        let window = self.size;
        let swapchain_extent = self.swapchain.extent;
        let vsync_changed = self.vsync.effective() != self.vsync.current();
        let size_changed = window != swapchain_extent;
        let (next_swapchain_extent, requested_present_mode) = if self.swapchain_stale
            || size_changed
            || vsync_changed
        {
            let present_modes = unsafe {
                self.surface_loader
                    .get_physical_device_surface_present_modes(self.device.physical, self.surface)
                    .expect("Failed to get present modes")
            };
            let capabilities = unsafe {
                self.surface_loader
                    .get_physical_device_surface_capabilities(self.device.physical, self.surface)
                    .expect("Failed to get surface capabilities")
            };
            (
                choose_swapchain_extent(&capabilities, window),
                choose_present_mode(self.vsync.effective(), &present_modes),
            )
        } else {
            (swapchain_extent, self.swapchain.present_mode)
        };
        RecreateInput {
            stale: self.swapchain_stale,
            window,
            swapchain_extent,
            next_swapchain_extent,
            current_present_mode: self.swapchain.present_mode,
            requested_present_mode,
            current_msaa: self.msaa.current(),
            requested_msaa: self.msaa.effective(),
            current_scale: self.render_scale.current(),
            requested_scale: self.render_scale.effective(),
            render_extent: self.render_extent,
        }
    }

    /// Applies pending vsync / MSAA / render-scale changes, rebuilding only
    /// what [`recreate_plan`] asks for. An empty plan clears the flags and
    /// returns without flushing or idling the device.
    pub(super) unsafe fn apply_pending(&mut self) {
        // Both: a minimized window has no extent to build. Leave the flags
        // set so the restore's real size still applies. No idle — draw_frame
        // does not call this while minimized, and a 0-size apply must not
        // stall the device.
        if self.size.width == 0 || self.size.height == 0 {
            return;
        }

        let (plan, facts) = recreate_plan(self.gather_recreate_input());
        log::debug!(
            "recreate: swapchain={} targets={} ({})",
            plan.swapchain,
            plan.targets,
            facts.label(),
        );

        if !plan.swapchain && !plan.targets {
            // Neither: consume host state (a vsync toggle that selects the
            // same present mode, a scale that rounds to the same extent) and
            // stop. The GPU is untouched.
            self.vsync.commit();
            self.msaa.commit();
            self.render_scale.commit();
            self.needs_recreate = false;
            self.swapchain_stale = false;
            return;
        }

        // Both: unsubmitted command buffers are invisible to
        // `device_wait_idle` and would keep sampling images this rebuild
        // destroys.
        self.flush_pending_submits();
        unsafe {
            // Both: wait out in-flight frames before destroying images.
            self.device
                .device
                .device_wait_idle()
                .expect("device_wait_idle failed");

            // Swapchain: the present mode is chosen from the committed vsync.
            // Targets-only: still commit, so a FIFO-only toggle updates
            // pacing without recreating the swapchain.
            self.vsync.commit();

            // Swapchain: format is an input to pipeline rebuild below.
            let mut format_changed = false;
            let prev_swapchain_extent = self.swapchain.extent;
            if plan.swapchain {
                // Swapchain: images, views, and present mode.
                let new_swapchain = Swapchain::new(
                    &self.instance.instance,
                    &self.device,
                    &self.surface_loader,
                    self.surface,
                    self.size,
                    self.vsync.effective(),
                    self.swapchain.swapchain,
                );
                self.swapchain.destroy(&self.device.device);
                format_changed = new_swapchain.format != self.swapchain.format;
                self.swapchain = new_swapchain;
            }

            // Targets: previous rung, for the OOM ladder and for "did the
            // sample count / extent actually change".
            let prev_msaa = self.msaa.current();
            let prev_scale = self.render_scale.current();
            let prev_extent = self.render_extent;
            let prev_samples = self.targets.samples;
            // The created swapchain extent is authoritative. If the surface's
            // currentExtent disagreed with the plan, still rebuild targets.
            let projected = scaled_extent(self.swapchain.extent, self.render_scale.effective());
            let rebuild_targets = plan.targets
                || projected != self.render_extent
                || self.msaa.effective() != prev_msaa;

            let mut replaced_targets = false;
            if rebuild_targets {
                // Targets: commit the request, then the allocation ladder.
                self.msaa.commit();
                self.render_scale.commit();
                let requested_msaa = self.msaa.current();
                let requested_scale = self.render_scale.current();
                let requested_extent = scaled_extent(self.swapchain.extent, requested_scale);
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
            } else {
                // Targets kept (MSAA and render extent unchanged). Consume a
                // pending scale that rounded to the same extent.
                self.msaa.commit();
                self.render_scale.commit();
            }

            // Targets (extent) and swapchain (TAA history images) both allocate
            // from these properties. Cheap relative to the rebuilds above.
            let memory_props = self
                .instance
                .instance
                .get_physical_device_memory_properties(self.device.physical);
            // Targets: exposure's tile grid tracks the render extent. Unchanged
            // when the targets survive, so a swapchain-only apply skips this.
            // The published `ExposureShared` cell the main thread holds is
            // preserved, so `compose()` keeps reading it.
            if self.render_extent.width != prev_extent.width
                || self.render_extent.height != prev_extent.height
            {
                self.exposure
                    .recreate(&self.device.device, &memory_props, self.render_extent);
            }
            // Swapchain: history images are swapchain-sized. `TaaState::recreate`
            // rebuilds them only when the extent changed, but it always drops
            // temporal state, so a same-extent rebuild (vsync, OUT_OF_DATE)
            // must not call it.
            let swapchain_extent_changed = self.swapchain.extent != prev_swapchain_extent;
            if swapchain_extent_changed {
                if let Err(err) =
                    self.taa
                        .recreate(&self.device.device, &memory_props, self.swapchain.extent)
                {
                    log::error!("{}", render_target_oom_message(&err));
                }
            }
            // Targets: a render-scale change keeps the history images (the
            // swapchain extent did not change) and still invalidates temporal
            // state, so the new sample grid does not mix with the previous
            // present. `recreate` above already invalidated when the extent
            // changed. MSAA alone does not: history is the resolved image.
            let scale_changed = (self.render_scale.current() - prev_scale).abs() > f32::EPSILON;
            if scale_changed && !swapchain_extent_changed {
                self.taa.invalidate_history();
            }

            // Both: the idle above retired any in-flight copy. Drop the hazard
            // because swapchain images were replaced, or the offscreen targets
            // that copy read were.
            self.clear_copy();
            if replaced_targets {
                // Targets: depth and rate images were recreated (layout
                // UNDEFINED). Skip VRS until a classify at the end of the
                // first post-recreate use primes the rate image. Sampleable
                // and MS depth begin UNDEFINED regardless, with no classifier
                // read left to wait on. Not touched when the targets survive.
                for slot in 0..FRAMES_IN_FLIGHT as usize {
                    let s = &mut self.slots[FrameSlot::new(slot)];
                    s.vrs_ready = false;
                    s.vrs_history = false;
                    s.vrs_ms_depth_read = false;
                }
                // Targets: previous-frame depth samples are invalid until a
                // subsequent store.
                self.prev_depth.invalidate();
                // Targets: the shared shadow map is UNDEFINED.
                self.shadow_cache.invalidate();
                // Targets: sky LUT images are UNDEFINED.
                self.sky_cloud.invalidate();
            }

            // Swapchain: the present format changed. Targets: the sample count
            // changed. Either one rebuilds pipelines; a scale-only or
            // same-format swapchain rebuild does not.
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
                    self.device.sky_coarse_ok(self.targets.samples),
                    self.device.independent_blend,
                    self.device.vrs_depth_ms_ok(self.targets.samples),
                    self.device.shader_stats.as_ref(),
                );
            }

            if plan.swapchain {
                // Swapchain: one binary semaphore per swapchain image. The
                // count can change even at the same window size. Targets-only
                // keeps the existing set.
                for &sem in &self.present_semaphores {
                    sem.destroy(&self.device.device);
                }
                self.present_semaphores =
                    create_present_semaphores(&self.device.device, self.swapchain.images.len());
            }

            // Both: the request has been applied.
            self.needs_recreate = false;
            self.swapchain_stale = false;
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

    fn extent(width: u32, height: u32) -> vk::Extent2D {
        vk::Extent2D { width, height }
    }

    /// Live swapchain and targets agree: scale 1, MSAA 1, IMMEDIATE, not stale.
    fn steady(size: vk::Extent2D) -> RecreateInput {
        RecreateInput {
            stale: false,
            window: size,
            swapchain_extent: size,
            next_swapchain_extent: size,
            current_present_mode: vk::PresentModeKHR::IMMEDIATE,
            requested_present_mode: vk::PresentModeKHR::IMMEDIATE,
            current_msaa: SampleCount::X1,
            requested_msaa: SampleCount::X1,
            current_scale: 1.0,
            requested_scale: 1.0,
            render_extent: size,
        }
    }

    fn plan(input: RecreateInput) -> RecreatePlan {
        recreate_plan(input).0
    }

    #[test]
    fn recreate_plan_same_size_resize_is_nothing() {
        // winit/Wayland repeats Resized at the current extent. Nothing to do.
        let size = extent(3440, 1440);
        assert_eq!(
            plan(steady(size)),
            RecreatePlan {
                swapchain: false,
                targets: false,
            },
        );
    }

    #[test]
    fn recreate_plan_stale_at_same_size_is_swapchain_only() {
        // OUT_OF_DATE / SURFACE_LOST (or recreate_if_stale) with the extent,
        // present mode, MSAA, and scale unchanged.
        let mut input = steady(extent(3440, 1440));
        input.stale = true;
        assert_eq!(
            plan(input),
            RecreatePlan {
                swapchain: true,
                targets: false,
            },
        );
    }

    #[test]
    fn recreate_plan_vsync_change_is_swapchain_only() {
        // vsync off → on selects FIFO instead of IMMEDIATE. Targets stay.
        let mut input = steady(extent(3440, 1440));
        input.requested_present_mode = vk::PresentModeKHR::FIFO;
        assert_eq!(
            plan(input),
            RecreatePlan {
                swapchain: true,
                targets: false,
            },
        );
    }

    #[test]
    fn recreate_plan_msaa_change_is_targets_only() {
        let mut input = steady(extent(3440, 1440));
        input.requested_msaa = SampleCount::X2;
        assert_eq!(
            plan(input),
            RecreatePlan {
                swapchain: false,
                targets: true,
            },
        );
    }

    #[test]
    fn recreate_plan_scale_change_is_targets_only() {
        let mut input = steady(extent(3440, 1440));
        input.requested_scale = 0.5;
        assert_eq!(
            plan(input),
            RecreatePlan {
                swapchain: false,
                targets: true,
            },
        );
    }

    #[test]
    fn recreate_plan_size_change_is_both() {
        let original = extent(3440, 1440);
        let resized = extent(2560, 1440);
        let mut input = steady(original);
        input.window = resized;
        input.next_swapchain_extent = resized;
        assert_eq!(
            plan(input),
            RecreatePlan {
                swapchain: true,
                targets: true,
            },
        );
    }

    #[test]
    fn recreate_plan_size_change_then_back_before_apply_is_nothing() {
        // on_resize stores only the latest size. Away-and-back before apply
        // is the same inputs as a same-size event.
        let original = extent(3440, 1440);
        let away = extent(1920, 1080);
        let mut input = steady(original);
        input.window = away;
        input.next_swapchain_extent = away;
        assert_eq!(
            plan(input),
            RecreatePlan {
                swapchain: true,
                targets: true,
            },
        );
        input.window = original;
        input.next_swapchain_extent = original;
        assert_eq!(
            plan(input),
            RecreatePlan {
                swapchain: false,
                targets: false,
            },
        );
    }

    #[test]
    fn recreate_plan_stale_and_msaa_change_is_both() {
        let mut input = steady(extent(3440, 1440));
        input.stale = true;
        input.requested_msaa = SampleCount::X4;
        assert_eq!(
            plan(input),
            RecreatePlan {
                swapchain: true,
                targets: true,
            },
        );
    }
}
