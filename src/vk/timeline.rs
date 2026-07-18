//! Timeline-semaphore sync primitives. Collapses the old per-slot render
//! fence + render_done semaphore + global copy fence into ONE monotonic
//! timeline. The only surviving binary semaphores are the two the WSI
//! mandates (acquire signal, present wait), modelled by `BinarySemaphore` so
//! a timeline handle can never reach acquire/present.
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

    /// Submit with optional cross-queue dependency via `extra_wait`.
    pub unsafe fn submit(
        self,
        device: &ash::Device,
        queue: vk::Queue,
        timeline: &Timeline,
        extra_wait: Option<(vk::Semaphore, TimelineValue)>,
    ) -> RenderCompletion {
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline.sem)
            .value(self.value.raw())
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd)];
        let waits = extra_wait.map(|(sem, value)| {
            [vk::SemaphoreSubmitInfo::default()
                .semaphore(sem)
                .value(value.raw())
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)]
        });
        let mut submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmds)
            .signal_semaphore_infos(&signal);
        if let Some(waits) = &waits {
            submit = submit.wait_semaphore_infos(waits);
        }
        let submit = [submit];
        unsafe {
            device
                .queue_submit2(queue, &submit, vk::Fence::null())
                .expect("render submit failed");
        }
        RenderCompletion(self.value)
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
