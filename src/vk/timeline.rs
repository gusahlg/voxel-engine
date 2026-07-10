//! Timeline-semaphore sync primitives. Collapses the old per-slot render
//! fence + render_done semaphore + global copy fence into ONE monotonic
//! timeline. The only surviving binary semaphores are the two the WSI
//! mandates (acquire signal, present wait), modelled by `BinarySemaphore` so
//! a timeline handle can never reach acquire/present.
use ash::vk;

/// A point on the render timeline (opaque, created only by Timeline).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TimelineValue(u64);

impl TimelineValue {
    /// The initial timeline value.
    pub const START: TimelineValue = TimelineValue(0);
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// The timeline counter (strictly increasing by design).
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

    /// Reserve and return the next timeline value.
    fn reserve(&mut self) -> TimelineValue {
        self.next += 1;
        TimelineValue(self.next)
    }

    /// The highest value ever reserved (for idle/teardown drains).
    pub fn last_reserved(&self) -> TimelineValue {
        TimelineValue(self.next)
    }

    /// Begin a render submission with a reserved signal value.
    pub fn begin_render(&mut self, cmd: vk::CommandBuffer) -> RenderSubmit {
        RenderSubmit {
            value: self.reserve(),
            cmd,
        }
    }

    /// Begin a present-copy submission (mixed binary+timeline).
    pub fn begin_copy(&mut self, cmd: vk::CommandBuffer) -> CopySubmit {
        CopySubmit {
            value: self.reserve(),
            cmd,
        }
    }

    /// Non-blocking current GPU-side counter.
    pub unsafe fn counter(&self, device: &ash::Device) -> TimelineValue {
        let v = unsafe {
            device
                .get_semaphore_counter_value(self.sem)
                .expect("timeline counter query failed")
        };
        TimelineValue(v)
    }

    /// Blocking wait until the GPU reaches `value`.
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

    /// A non-blocking probe with NO wait method — hand to the mailbox path.
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

/// Non-blocking read of timeline progress (cannot stall).
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

/// A render submission in flight (must be submitted to signal the timeline).
#[must_use = "a reserved render submission must be submitted or the timeline stalls"]
pub struct RenderSubmit {
    value: TimelineValue,
    cmd: vk::CommandBuffer,
}

impl RenderSubmit {
    /// Get the value this render will signal.
    pub fn value(&self) -> TimelineValue {
        self.value
    }

    pub unsafe fn submit(
        self,
        device: &ash::Device,
        queue: vk::Queue,
        timeline: &Timeline,
    ) -> RenderCompletion {
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline.sem)
            .value(self.value.raw())
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd)];
        let submit = [vk::SubmitInfo2::default()
            .command_buffer_infos(&cmds)
            .signal_semaphore_infos(&signal)];
        unsafe {
            device
                .queue_submit2(queue, &submit, vk::Fence::null())
                .expect("render submit failed");
        }
        RenderCompletion(self.value)
    }
}

/// Handle to a completed render (tracked by timeline value).
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

/// A present-copy submission (mixes binary and timeline synchronization).
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
                .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(timeline.sem)
                .value(wait_render.0.raw())
                .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER),
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

/// A binary semaphore (distinct from the timeline semaphore).
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

/// Typed acquire: cannot be passed a timeline semaphore.
pub unsafe fn acquire_next_image(
    loader: &ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    timeout: u64,
    signal: BinarySemaphore,
) -> Result<(u32, bool), vk::Result> {
    unsafe { loader.acquire_next_image(swapchain, timeout, signal.0, vk::Fence::null()) }
}

/// Typed present: wait set is binary-only by type.
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
