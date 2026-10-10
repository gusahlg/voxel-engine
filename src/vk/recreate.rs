//! Swapchain and render-target rebuilds.
//!
//! [`recreate_plan`] is pure and host-testable. [`Renderer::apply_pending`]
//! follows it: a same-size resize or a stale swapchain at an unchanged
//! extent does not rebuild render targets, and an empty plan returns without
//! idling the device. Split out of `mod.rs` so recreation can change without
//! opening the frame loop.
//!
//! When the requested render targets do not fit, [`recreate_oom_strategy`]
//! and [`walk_after_oom`] (both pure) pick the rung;
//! [`Renderer::recover_targets_oom`] frees, allocates and installs it.

use ash::vk;

use crate::skeleton::FrameSlot;

use super::buffers::{FRAMES_IN_FLIGHT, MESH_CONSUMER_STAGES};
use super::image::{AllocError, render_target_oom_message};
use super::pipeline::Pipelines;
use super::render_client::RenderReturn;
use super::swapchain::{Swapchain, choose_present_mode, choose_swapchain_extent};
use super::targets::{RenderTargets, next_lower, walk_ladder};
use super::{Pending, Renderer, SampleCount, create_present_semaphores, scaled_extent};

/// One render-target configuration on the allocation ladder.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Rung {
    msaa: SampleCount,
    scale: f32,
}

/// Render scales within `f32::EPSILON` of each other are one rung.
fn same_scale(a: f32, b: f32) -> bool {
    (a - b).abs() <= f32::EPSILON
}

/// The render scale moved by more than `f32::EPSILON`. Not `!same_scale`: a
/// NaN scale counts as unchanged here, as in the comparisons this replaced.
fn scale_changed(a: f32, b: f32) -> bool {
    (a - b).abs() > f32::EPSILON
}

/// The rung that fit is below the request (logged as a fallback).
fn fell_back(requested: Rung, got: Rung) -> bool {
    got.msaa != requested.msaa || scale_changed(got.scale, requested.scale)
}

/// The applied render-target state before a rebuild: the rung the OOM ladder
/// retains or reverts to, and what exposure, TAA and the pipelines compare
/// against afterwards.
#[derive(Debug, Clone, Copy)]
struct LiveTargets {
    rung: Rung,
    extent: vk::Extent2D,
    samples: vk::SampleCountFlags,
}

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

/// Walks the render-target ladder after the `requested` rung failed.
///
/// [`FreeFirst`](RecreateOomStrategy::FreeFirst) (the live targets are
/// already freed): from the request, inclusive.
/// [`Retain`](RecreateOomStrategy::Retain): from the rung below the request,
/// and the walk stops without allocating at the rung the live images already
/// are (same MSAA, scale and extent): no second copy of them, and nothing
/// below a working config. `None` when nothing fit or the walk reached the
/// live rung. `extent_at` maps a scale to its render extent.
fn walk_after_oom<T>(
    strategy: RecreateOomStrategy,
    requested: Rung,
    live: Rung,
    live_extent: vk::Extent2D,
    extent_at: impl Fn(f32) -> vk::Extent2D,
    mut try_alloc: impl FnMut(SampleCount, f32) -> Result<T, AllocError>,
) -> Option<(T, Rung)> {
    match strategy {
        RecreateOomStrategy::FreeFirst => walk_ladder(requested.msaa, requested.scale, try_alloc)
            .map(|(targets, msaa, scale)| (targets, Rung { msaa, scale })),
        RecreateOomStrategy::Retain => {
            let (msaa, scale) = next_lower(requested.msaa, requested.scale)?;
            walk_ladder(msaa, scale, |msaa, scale| {
                if msaa == live.msaa
                    && same_scale(scale, live.scale)
                    && extent_at(scale) == live_extent
                {
                    return Ok(None);
                }
                try_alloc(msaa, scale).map(Some)
            })
            .and_then(|(targets, msaa, scale)| targets.map(|t| (t, Rung { msaa, scale })))
        }
    }
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
    let scale = scale_changed(input.current_scale, input.requested_scale);
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
        // Both: wait out in-flight frames before destroying images.
        unsafe {
            self.device
                .device
                .device_wait_idle()
                .expect("device_wait_idle failed");
        }

        // Swapchain: the present mode is chosen from the committed vsync.
        // Targets-only: still commit, so a FIFO-only toggle updates
        // pacing without recreating the swapchain.
        self.vsync.commit();

        let prev_swapchain_extent = self.swapchain.extent;
        // Swapchain: format is an input to pipeline rebuild below.
        let format_changed = if plan.swapchain {
            unsafe { self.rebuild_swapchain() }
        } else {
            false
        };

        // Targets: previous rung, for the OOM ladder and for "did the
        // sample count / extent actually change".
        let live = self.live_targets();
        let replaced_targets = unsafe { self.apply_targets(plan.targets, live) };

        self.resize_dependents(prev_swapchain_extent, live);

        // Both: the idle above retired any in-flight copy. Drop the hazard
        // because swapchain images were replaced, or the offscreen targets
        // that copy read were.
        self.clear_copy();
        if replaced_targets {
            self.invalidate_slot_targets();
        }

        // Swapchain: the present format changed. Targets: the sample count
        // changed. Either one rebuilds pipelines; a scale-only or
        // same-format swapchain rebuild does not.
        if self.targets.samples != live.samples || format_changed {
            unsafe { self.rebuild_pipelines() };
        }

        if plan.swapchain {
            unsafe { self.recreate_present_semaphores() };
        }

        // Both: the request has been applied.
        self.needs_recreate = false;
        self.swapchain_stale = false;
    }

    /// Swapchain: images, views, and present mode, from the committed vsync.
    /// Returns whether the surface format changed.
    ///
    /// # Safety
    /// The device is idle: the old swapchain is destroyed.
    unsafe fn rebuild_swapchain(&mut self) -> bool {
        let new_swapchain = Swapchain::new(
            &self.instance.instance,
            &self.device,
            &self.surface_loader,
            self.surface,
            self.size,
            self.vsync.effective(),
            self.swapchain.swapchain,
        );
        unsafe { self.swapchain.destroy(&self.device.device) };
        let format_changed = new_swapchain.format != self.swapchain.format;
        self.swapchain = new_swapchain;
        format_changed
    }

    fn live_targets(&self) -> LiveTargets {
        LiveTargets {
            rung: Rung {
                msaa: self.msaa.current(),
                scale: self.render_scale.current(),
            },
            extent: self.render_extent,
            samples: self.targets.samples,
        }
    }

    /// Render targets at `extent` and `msaa`: the requested rung and every
    /// ladder rung.
    fn create_targets(
        &self,
        extent: vk::Extent2D,
        msaa: SampleCount,
    ) -> Result<RenderTargets, AllocError> {
        RenderTargets::new(
            &self.instance.instance,
            &self.device.device,
            self.device.physical,
            extent,
            msaa,
            self.device.fragment_shading_rate.as_ref(),
        )
    }

    /// Targets: rebuilds them when the plan asks, or when the created
    /// swapchain or the requested MSAA disagrees with the live ones; a
    /// requested rung that does not fit goes to
    /// [`Self::recover_targets_oom`]. Commits the pending MSAA / scale either
    /// way. Returns whether the live targets were replaced.
    ///
    /// # Safety
    /// The device is idle: the live targets may be destroyed.
    unsafe fn apply_targets(&mut self, plan_targets: bool, live: LiveTargets) -> bool {
        // The created swapchain extent is authoritative. If the surface's
        // currentExtent disagreed with the plan, still rebuild targets.
        let projected = scaled_extent(self.swapchain.extent, self.render_scale.effective());
        let rebuild = plan_targets
            || projected != self.render_extent
            || self.msaa.effective() != live.rung.msaa;
        // Rebuild: commit the request, then the allocation ladder. Kept (MSAA
        // and render extent unchanged): consume a pending scale that rounded
        // to the same extent.
        self.msaa.commit();
        self.render_scale.commit();
        if !rebuild {
            return false;
        }
        let requested = Rung {
            msaa: self.msaa.current(),
            scale: self.render_scale.current(),
        };
        let requested_extent = scaled_extent(self.swapchain.extent, requested.scale);
        match self.create_targets(requested_extent, requested.msaa) {
            Ok(new_targets) => {
                unsafe { self.targets.destroy(&self.device.device) };
                self.targets = new_targets;
                self.render_extent = requested_extent;
                true
            }
            Err(err) => {
                log::warn!("{}", render_target_oom_message(&err));
                unsafe { self.recover_targets_oom(requested, live) }
            }
        }
    }

    /// The `requested` rung did not fit. Keeps the live targets or frees them
    /// first ([`recreate_oom_strategy`]), walks the ladder
    /// ([`walk_after_oom`]) and installs the rung that fit. When none did,
    /// retained targets stay and MSAA / scale revert to them; freed ones
    /// leave nothing to draw with, which panics. Returns whether the live
    /// targets were replaced.
    ///
    /// # Safety
    /// The device is idle: the live targets may be destroyed.
    unsafe fn recover_targets_oom(&mut self, requested: Rung, live: LiveTargets) -> bool {
        let at_live_scale = scaled_extent(self.swapchain.extent, live.rung.scale);
        let strategy = recreate_oom_strategy(
            live.extent != at_live_scale,
            requested.msaa,
            requested.scale,
            live.rung.msaa,
            live.rung.scale,
        );
        match strategy {
            RecreateOomStrategy::FreeFirst => {
                log::info!(
                    "renderer: freeing previous render targets before walking the ladder (requested MSAA {} / scale {}, previous {} / {})",
                    requested.msaa.as_u32(),
                    requested.scale,
                    live.rung.msaa.as_u32(),
                    live.rung.scale,
                );
                // Device is already idle from the wait at the start of
                // `apply_pending`.
                unsafe { self.targets.destroy(&self.device.device) };
            }
            RecreateOomStrategy::Retain => {
                log::info!(
                    "renderer: retaining previous render targets (MSAA {} / scale {}); requested MSAA {} / scale {} did not fit",
                    live.rung.msaa.as_u32(),
                    live.rung.scale,
                    requested.msaa.as_u32(),
                    requested.scale,
                );
            }
        }
        let swapchain_extent = self.swapchain.extent;
        let found = walk_after_oom(
            strategy,
            requested,
            live.rung,
            live.extent,
            |scale| scaled_extent(swapchain_extent, scale),
            |msaa, scale| self.create_targets(scaled_extent(swapchain_extent, scale), msaa),
        );
        match (found, strategy) {
            (Some((new_targets, rung)), _) => {
                if fell_back(requested, rung) {
                    log::warn!(
                        "renderer: render targets fell back to MSAA {} / render scale {} (requested {} / {})",
                        rung.msaa.as_u32(),
                        rung.scale,
                        requested.msaa.as_u32(),
                        requested.scale,
                    );
                }
                if strategy == RecreateOomStrategy::Retain {
                    unsafe { self.targets.destroy(&self.device.device) };
                }
                self.targets = new_targets;
                self.msaa = Pending::new(rung.msaa);
                self.render_scale = Pending::new(rung.scale);
                self.render_extent = scaled_extent(swapchain_extent, rung.scale);
                true
            }
            (None, RecreateOomStrategy::FreeFirst) => {
                panic!(
                    "renderer: could not allocate any render-target rung after freeing previous targets (requested MSAA {} / scale {})",
                    requested.msaa.as_u32(),
                    requested.scale,
                );
            }
            (None, RecreateOomStrategy::Retain) => {
                // Previous targets still exist: keep them and revert MSAA /
                // scale so the next frame matches live GPU state.
                self.msaa = Pending::new(live.rung.msaa);
                self.render_scale = Pending::new(live.rung.scale);
                self.render_extent = live.extent;
                false
            }
        }
    }

    /// Exposure and TAA after the swapchain / target rebuilds: exposure's
    /// tile grid follows the render extent, TAA's history images the
    /// swapchain extent, and a render-scale change drops TAA history.
    fn resize_dependents(&mut self, prev_swapchain_extent: vk::Extent2D, live: LiveTargets) {
        // Targets (extent) and swapchain (TAA history images) both allocate
        // from these properties. Cheap relative to the rebuilds above.
        let memory_props = unsafe {
            self.instance
                .instance
                .get_physical_device_memory_properties(self.device.physical)
        };
        // Targets: exposure's tile grid tracks the render extent. Unchanged
        // when the targets survive, so a swapchain-only apply skips this.
        // The published `ExposureShared` cell the main thread holds is
        // preserved, so `compose()` keeps reading it.
        if self.render_extent != live.extent {
            self.exposure
                .recreate(&self.device.device, &memory_props, self.render_extent);
        }
        // Swapchain: history images are swapchain-sized. `TaaState::recreate`
        // rebuilds them only when the extent changed, but it always drops
        // temporal state, so a same-extent rebuild (vsync, OUT_OF_DATE)
        // must not call it.
        let swapchain_extent_changed = self.swapchain.extent != prev_swapchain_extent;
        if swapchain_extent_changed
            && let Err(err) =
                self.taa
                    .recreate(&self.device.device, &memory_props, self.swapchain.extent)
        {
            log::error!("{}", render_target_oom_message(&err));
        }
        // Targets: a render-scale change keeps the history images (the
        // swapchain extent did not change) and still invalidates temporal
        // state, so the new sample grid does not mix with the previous
        // present. `recreate` above already invalidated when the extent
        // changed. MSAA alone does not: history is the resolved image.
        if scale_changed(self.render_scale.current(), live.rung.scale) && !swapchain_extent_changed
        {
            self.taa.invalidate_history();
        }
    }

    /// Targets: the depth and rate images were recreated (layout UNDEFINED).
    /// Skip VRS until a classify at the end of the first post-recreate use
    /// primes the rate image. Sampleable and MS depth begin UNDEFINED
    /// regardless, with no classifier read left to wait on. Not called when
    /// the targets survive.
    fn invalidate_slot_targets(&mut self) {
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

    /// Pipelines for the live targets' sample count and the swapchain format.
    ///
    /// # Safety
    /// The device is idle: the old pipelines are destroyed.
    unsafe fn rebuild_pipelines(&mut self) {
        unsafe { self.pipelines.destroy(&self.device.device) };
        self.pipelines = Pipelines::for_targets(
            &self.device,
            self.pipeline_cache,
            &self.targets,
            self.swapchain.format,
            self.atlas.set_layout,
            self.mesh3d_set_layout,
        );
    }

    /// Swapchain: one binary semaphore per swapchain image. The count can
    /// change even at the same window size. Targets-only keeps the existing
    /// set.
    ///
    /// # Safety
    /// The device is idle: the old semaphores are destroyed.
    unsafe fn recreate_present_semaphores(&mut self) {
        for &sem in &self.present_semaphores {
            unsafe { sem.destroy(&self.device.device) };
        }
        self.present_semaphores =
            create_present_semaphores(&self.device.device, self.swapchain.images.len());
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

    fn rung(msaa: SampleCount, scale: f32) -> Rung {
        Rung { msaa, scale }
    }

    fn oom() -> AllocError {
        AllocError::new(0, "test", vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
    }

    /// Render extent at `scale` of a 1000×500 swapchain.
    fn at(scale: f32) -> vk::Extent2D {
        scaled_extent(extent(1000, 500), scale)
    }

    #[test]
    fn walk_after_oom_free_first_walks_from_the_request() {
        // The live targets are already freed, so the walk starts at the
        // request and may allocate the live rung again.
        let mut tried = Vec::new();
        let got = walk_after_oom(
            RecreateOomStrategy::FreeFirst,
            rung(SampleCount::X4, 1.0),
            rung(SampleCount::X2, 1.0),
            at(1.0),
            at,
            |msaa, scale| {
                tried.push((msaa, scale));
                if msaa == SampleCount::X2 {
                    Ok("x2")
                } else {
                    Err(oom())
                }
            },
        );
        assert_eq!(got, Some(("x2", rung(SampleCount::X2, 1.0))));
        assert_eq!(tried, [(SampleCount::X4, 1.0), (SampleCount::X2, 1.0)]);
    }

    #[test]
    fn walk_after_oom_free_first_none_after_the_floor() {
        let mut tried = 0;
        let got = walk_after_oom(
            RecreateOomStrategy::FreeFirst,
            rung(SampleCount::X2, 0.5),
            rung(SampleCount::X2, 0.5),
            at(0.5),
            at,
            |_, _| {
                tried += 1;
                Err::<(), _>(oom())
            },
        );
        assert_eq!(got, None);
        assert_eq!(tried, 2, "X2 then X1 at the 0.5 floor");
    }

    #[test]
    fn walk_after_oom_retain_stops_at_the_live_rung() {
        // X8 over live X2: X4 is tried, X2 is the live images (no second
        // copy), X1 is below a working config.
        let mut tried = Vec::new();
        let got = walk_after_oom(
            RecreateOomStrategy::Retain,
            rung(SampleCount::X8, 1.0),
            rung(SampleCount::X2, 1.0),
            at(1.0),
            at,
            |msaa, scale| {
                tried.push((msaa, scale));
                Err::<(), _>(oom())
            },
        );
        assert_eq!(got, None);
        assert_eq!(tried, [(SampleCount::X4, 1.0)]);
    }

    #[test]
    fn walk_after_oom_retain_installs_a_rung_above_the_live_one() {
        let got = walk_after_oom(
            RecreateOomStrategy::Retain,
            rung(SampleCount::X8, 1.0),
            rung(SampleCount::X1, 1.0),
            at(1.0),
            at,
            |msaa, _| {
                if msaa == SampleCount::X2 {
                    Ok("x2")
                } else {
                    Err(oom())
                }
            },
        );
        assert_eq!(got, Some(("x2", rung(SampleCount::X2, 1.0))));
    }

    #[test]
    fn walk_after_oom_retain_lowers_scale_after_msaa() {
        // X1 at 2.0 over live X1 at 1.0: 1.75, 1.5 and 1.25 are tried, 1.0
        // is the live rung.
        let mut tried = Vec::new();
        let got = walk_after_oom(
            RecreateOomStrategy::Retain,
            rung(SampleCount::X1, 2.0),
            rung(SampleCount::X1, 1.0),
            at(1.0),
            at,
            |_, scale| {
                tried.push(scale);
                Err::<(), _>(oom())
            },
        );
        assert_eq!(got, None);
        assert_eq!(tried, [1.75, 1.5, 1.25]);
    }

    #[test]
    fn walk_after_oom_retain_allocates_the_live_rung_at_another_extent() {
        // Same MSAA and scale, but the live images are another size: they
        // cannot serve that rung, so it is allocated.
        let got = walk_after_oom(
            RecreateOomStrategy::Retain,
            rung(SampleCount::X4, 1.0),
            rung(SampleCount::X2, 1.0),
            extent(640, 480),
            at,
            |msaa, scale| Ok::<_, AllocError>((msaa, scale)),
        );
        assert_eq!(
            got,
            Some(((SampleCount::X2, 1.0), rung(SampleCount::X2, 1.0)))
        );
    }

    #[test]
    fn walk_after_oom_retain_below_the_floor_tries_nothing() {
        let mut tried = 0;
        let got = walk_after_oom(
            RecreateOomStrategy::Retain,
            rung(SampleCount::X1, 0.5),
            rung(SampleCount::X1, 0.5),
            at(0.5),
            at,
            |_, _| {
                tried += 1;
                Ok::<_, AllocError>(())
            },
        );
        assert_eq!(got, None);
        assert_eq!(tried, 0);
    }

    #[test]
    fn scale_comparisons_keep_their_epsilon_and_nan_reading() {
        assert!(same_scale(1.0, 1.0));
        assert!(same_scale(1.0, 1.0 + f32::EPSILON * 0.5));
        assert!(!same_scale(1.0, 1.25));
        assert!(scale_changed(1.0, 1.25));
        assert!(!scale_changed(0.75, 0.75));
        // NaN is neither the same scale nor a changed one.
        assert!(!same_scale(f32::NAN, 1.0));
        assert!(!scale_changed(f32::NAN, 1.0));
    }

    #[test]
    fn fell_back_when_msaa_or_scale_is_below_the_request() {
        let requested = rung(SampleCount::X4, 1.0);
        assert!(!fell_back(requested, requested));
        assert!(fell_back(requested, rung(SampleCount::X2, 1.0)));
        assert!(fell_back(requested, rung(SampleCount::X4, 0.75)));
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
