//! Timeline-semaphore sync primitives. Collapses the old per-slot render
//! fence + render_done semaphore + global copy fence into ONE monotonic
//! timeline. The only surviving binary semaphores are the two the WSI
//! mandates (acquire signal, present wait), modelled by `BinarySemaphore` so
//! a timeline handle can never reach acquire/present.
use std::time::{Duration, Instant};

use ash::vk;

// Monotonic timeline value (canonical counter for GPU sync).
pub use crate::rev::Rev as TimelineValue;

pub struct Timeline {
    sem: vk::Semaphore,
    next: u64,
}

impl Timeline {
    pub unsafe fn new(device: &ash::Device) -> Self {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let sem = unsafe {
            device
                .create_semaphore(&info, None)
                .expect("Failed to create timeline semaphore")
        };
        Self { sem, next: 0 }
    }

    fn reserve(&mut self) -> TimelineValue {
        self.next += 1;
        TimelineValue(self.next)
    }

    pub fn last_reserved(&self) -> TimelineValue {
        TimelineValue(self.next)
    }

    /// The underlying Vulkan semaphore handle.
    pub fn semaphore(&self) -> vk::Semaphore {
        self.sem
    }

    pub fn begin_render(&mut self, cmd: vk::CommandBuffer) -> RenderSubmit {
        RenderSubmit {
            value: self.reserve(),
            cmd,
        }
    }

    pub fn begin_copy(&mut self, cmd: vk::CommandBuffer) -> CopySubmit {
        CopySubmit {
            value: self.reserve(),
            cmd,
        }
    }

    pub unsafe fn counter(&self, device: &ash::Device) -> TimelineValue {
        let v = unsafe {
            device
                .get_semaphore_counter_value(self.sem)
                .expect("timeline counter query failed")
        };
        TimelineValue(v)
    }

    pub unsafe fn wait(&self, device: &ash::Device, value: TimelineValue) {
        let sems = [self.sem];
        let vals = [value.raw()];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&vals);
        unsafe {
            device
                .wait_semaphores(&info, u64::MAX)
                .expect("timeline wait failed");
        }
    }

    /// Poll the timeline for `budget`, then fall through to a blocking wait.
    /// Trades a spinning core for the kernel wake latency of `vkWaitSemaphores`.
    pub unsafe fn wait_spin(&self, device: &ash::Device, value: TimelineValue, budget: Duration) {
        let start = Instant::now();
        while start.elapsed() < budget {
            let v = unsafe {
                device
                    .get_semaphore_counter_value(self.sem)
                    .expect("timeline counter query failed")
            };
            if v >= value.raw() {
                return;
            }
            std::hint::spin_loop();
        }
        unsafe { self.wait(device, value) };
    }

    /// Non-blocking probe of the current timeline value.
    pub fn probe<'a>(&self, device: &'a ash::Device) -> TimelineProbe<'a> {
        TimelineProbe {
            sem: self.sem,
            device,
        }
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_semaphore(self.sem, None) };
    }
}

#[derive(Clone, Copy)]
pub struct TimelineProbe<'a> {
    sem: vk::Semaphore,
    device: &'a ash::Device,
}

impl TimelineProbe<'_> {
    pub unsafe fn reached(self, value: TimelineValue) -> bool {
        let v = unsafe {
            self.device
                .get_semaphore_counter_value(self.sem)
                .expect("timeline counter query failed")
        };
        v >= value.raw()
    }
}

/// Must be submitted to signal timeline.
#[must_use = "a reserved render submission must be submitted or the timeline stalls"]
pub struct RenderSubmit {
    value: TimelineValue,
    cmd: vk::CommandBuffer,
}

impl RenderSubmit {
    pub fn value(&self) -> TimelineValue {
        self.value
    }

    /// Submit with optional cross-queue dependency via `extra_wait`: the
    /// semaphore, the value, and the stages that first consume what the other
    /// queue produced (the wait's second synchronization scope). Pass
    /// `ALL_COMMANDS` unless the consumer set is known exactly.
    pub unsafe fn submit(
        self,
        device: &ash::Device,
        queue: vk::Queue,
        timeline: &Timeline,
        extra_wait: Option<(vk::Semaphore, TimelineValue, vk::PipelineStageFlags2)>,
    ) -> RenderCompletion {
        let cmd = self.cmd;
        // Transfer / compute lanes and graphics-queue copies (QFOT release,
        // overwrite) submit through here. Those queues — or those CBs — may
        // not support color-out / late-Z, so the signal stays `ALL_COMMANDS`.
        unsafe {
            self.submit_bufs(
                device,
                queue,
                timeline,
                &[cmd],
                extra_wait,
                vk::PipelineStageFlags2::ALL_COMMANDS,
            )
        }
    }

    /// One `vkQueueSubmit2` / one timeline signal, with `cmds.len()` command
    /// buffers (`VkCommandBufferSubmitInfo`s). Used by uncapped submit batching
    /// and `VOXEL_BENCH_EMPTY=K`. `signal_stages` is the first sync scope of
    /// the timeline signal: [`RENDER_SIGNAL_STAGES`] for a graphics render CB,
    /// `ALL_COMMANDS` on transfer/compute lanes.
    pub unsafe fn submit_bufs(
        self,
        device: &ash::Device,
        queue: vk::Queue,
        timeline: &Timeline,
        cmds: &[vk::CommandBuffer],
        extra_wait: Option<(vk::Semaphore, TimelineValue, vk::PipelineStageFlags2)>,
        signal_stages: vk::PipelineStageFlags2,
    ) -> RenderCompletion {
        let value = self.value;
        unsafe { timeline.submit_render(device, queue, cmds, value, extra_wait, signal_stages) }
    }

    /// Hold the reservation without submitting: the command buffer joins a
    /// pending batch and is submitted later with the batch's signal value.
    pub fn into_parts(self) -> (TimelineValue, vk::CommandBuffer) {
        (self.value, self.cmd)
    }
}

/// Stages that write slot-owned (or same-queue render-CB) resources.
/// A wait on the render timeline can retire the slot once these complete,
/// without draining idle pipe (`ALL_COMMANDS` / `BOTTOM_OF_PIPE`).
///
/// Only the graphics-queue **render** submit uses this mask. Transfer-lane
/// and compute-lane submits keep `ALL_COMMANDS`: those queue families do not
/// support color-out / late-Z (validation: stageMask vs queue flags).
///
/// Per-slot / same-queue writers on the render CB:
/// - `COLOR_ATTACHMENT_OUTPUT`: scene offscreen (and MSAA color / AVERAGE
///   resolve), sky, debug cubes/lines/contact shadows, color clears.
/// - `LATE_FRAGMENT_TESTS`: scene depth (early/late Z; late is last), shadow
///   cascade depth, depth clears. MSAA SAMPLE_ZERO resolve is color-out.
/// - `COMPUTE_SHADER`: GPU cull, sky-cloud LUT, VRS classify, exposure
///   reduce, bloom pyramid + spill, same-queue compute-lane jobs.
/// - `COPY`: same-queue mesh / quad-IBO / block-texture / material copies,
///   minimap upload, cull-stats and VRS-mix copies, compute-job readback.
/// - `TRANSFER`: `cmd_fill_buffer` (cull counts/stats, VRS mix) and
///   `cmd_clear_color_image` (hidden-cloud LUT, bloom-pyramid prime).
///   `TRANSFER` is `ALL_TRANSFER` (copy|blit|resolve|clear); `COPY` is listed
///   so the uploads are obvious in the mask.
pub(crate) const RENDER_SIGNAL_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw()
        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
        | vk::PipelineStageFlags2::COPY.as_raw()
        | vk::PipelineStageFlags2::TRANSFER.as_raw(),
);

impl Timeline {
    /// One `vkQueueSubmit2` with `cmds` (in order) and a single timeline signal
    /// at `signal`. Intermediate values reserved for earlier frames in a batch
    /// are holes: a wait for them succeeds once this higher value is signalled.
    ///
    /// `signal_stages` is the first sync scope of the timeline signal. Graphics
    /// render CBs pass [`RENDER_SIGNAL_STAGES`]; transfer/compute lanes pass
    /// `ALL_COMMANDS`.
    pub unsafe fn submit_render(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        cmds: &[vk::CommandBuffer],
        signal: TimelineValue,
        extra_wait: Option<(vk::Semaphore, TimelineValue, vk::PipelineStageFlags2)>,
        signal_stages: vk::PipelineStageFlags2,
    ) -> RenderCompletion {
        let signal_info = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.sem)
            .value(signal.raw())
            .stage_mask(signal_stages)];
        let cmd_infos: Vec<vk::CommandBufferSubmitInfo<'_>> = cmds
            .iter()
            .map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c))
            .collect();
        let waits = extra_wait.map(|(sem, value, stages)| {
            [vk::SemaphoreSubmitInfo::default()
                .semaphore(sem)
                .value(value.raw())
                .stage_mask(stages)]
        });
        let mut submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal_info);
        if let Some(waits) = &waits {
            submit = submit.wait_semaphore_infos(waits);
        }
        let submit = [submit];
        unsafe {
            device
                .queue_submit2(queue, &submit, vk::Fence::null())
                .expect("render submit failed");
        }
        RenderCompletion(signal)
    }
}

#[derive(Clone, Copy)]
pub struct RenderCompletion(TimelineValue);

impl RenderCompletion {
    pub fn value(self) -> TimelineValue {
        self.0
    }
    pub(crate) fn from_value(v: TimelineValue) -> Self {
        RenderCompletion(v)
    }
}

/// Copy submission with binary + timeline sync.
#[must_use = "a reserved copy submission must be submitted or the timeline stalls"]
pub struct CopySubmit {
    value: TimelineValue,
    cmd: vk::CommandBuffer,
}

impl CopySubmit {
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        self,
        device: &ash::Device,
        queue: vk::Queue,
        timeline: &Timeline,
        wait_image: BinarySemaphore,
        wait_render: RenderCompletion,
        signal_present: BinarySemaphore,
    ) -> TimelineValue {
        let waits = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(wait_image.0)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(timeline.sem)
                .value(wait_render.0.raw())
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        let signals = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(signal_present.0)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(timeline.sem)
                .value(self.value.raw())
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd)];
        let submit = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&cmds)
            .signal_semaphore_infos(&signals)];
        unsafe {
            device
                .queue_submit2(queue, &submit, vk::Fence::null())
                .expect("copy submit failed");
        }
        self.value
    }
}

/// Binary semaphore (type-distinct from timeline).
#[derive(Clone, Copy)]
pub struct BinarySemaphore(vk::Semaphore);

impl BinarySemaphore {
    pub unsafe fn new(device: &ash::Device) -> Self {
        let info = vk::SemaphoreCreateInfo::default();
        let sem = unsafe {
            device
                .create_semaphore(&info, None)
                .expect("Failed to create binary semaphore")
        };
        Self(sem)
    }

    pub unsafe fn destroy(self, device: &ash::Device) {
        unsafe { device.destroy_semaphore(self.0, None) };
    }
}

pub unsafe fn acquire_next_image(
    loader: &ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    timeout: u64,
    signal: BinarySemaphore,
) -> Result<(u32, bool), vk::Result> {
    unsafe { loader.acquire_next_image(swapchain, timeout, signal.0, vk::Fence::null()) }
}

pub unsafe fn queue_present(
    loader: &ash::khr::swapchain::Device,
    queue: vk::Queue,
    wait: BinarySemaphore,
    swapchain: vk::SwapchainKHR,
    image_index: u32,
) -> Result<bool, vk::Result> {
    let wait_semaphores = [wait.0];
    let swapchains = [swapchain];
    let image_indices = [image_index];
    let info = vk::PresentInfoKHR::default()
        .wait_semaphores(&wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);
    unsafe { loader.queue_present(queue, &info) }
}

#[cfg(test)]
impl TimelineValue {
    pub fn from_raw_for_test(n: u64) -> Self {
        TimelineValue(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_signal_stages_are_slot_writers_not_all_commands() {
        assert!(RENDER_SIGNAL_STAGES.contains(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT));
        assert!(RENDER_SIGNAL_STAGES.contains(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS));
        assert!(RENDER_SIGNAL_STAGES.contains(vk::PipelineStageFlags2::COMPUTE_SHADER));
        assert!(RENDER_SIGNAL_STAGES.contains(vk::PipelineStageFlags2::COPY));
        assert!(RENDER_SIGNAL_STAGES.contains(vk::PipelineStageFlags2::TRANSFER));
        assert!(!RENDER_SIGNAL_STAGES.contains(vk::PipelineStageFlags2::ALL_COMMANDS));
        assert!(!RENDER_SIGNAL_STAGES.contains(vk::PipelineStageFlags2::BOTTOM_OF_PIPE));
        assert_ne!(RENDER_SIGNAL_STAGES, vk::PipelineStageFlags2::ALL_COMMANDS);
    }
}
