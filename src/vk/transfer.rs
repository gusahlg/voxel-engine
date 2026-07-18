/// A dedicated queue for staging copies (or the graphics queue as fallback).
/// Three tiers in preference order: dedicated family, second queue in graphics family, or graphics queue.
/// Callers don't branch on availability; only sync behavior changes by tier.
use ash::vk;

use super::timeline::{Timeline, TimelineValue};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Tier {
    /// A queue family with `TRANSFER` set and `GRAPHICS` unset.
    DedicatedFamily,
    /// A second queue index within the graphics family.
    SecondQueueSameFamily,
    /// No spare queue; copies recorded inline on caller's command buffer.
    SameQueueFallback,
}

/// A batch of copies being recorded. Must be submitted or discarded.
#[must_use = "a begun transfer batch must be submitted or discarded, or the lane's command buffer is left recording"]
pub(crate) struct LaneRecording {
    cmd: vk::CommandBuffer,
}

impl LaneRecording {
    /// The buffer to record copies into.
    pub fn cmd(&self) -> vk::CommandBuffer {
        self.cmd
    }
}

/// Resources for tiers with a distinct queue (not SameQueueFallback).
struct LaneResources {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    timeline: Timeline,
}

pub(crate) struct TransferLane {
    tier: Tier,
    family: u32,
    queue: vk::Queue,
    resources: Option<LaneResources>,
}

impl TransferLane {
    /// Create from family/queue selection. No pool allocated under SameQueueFallback.
    pub unsafe fn new(device: &ash::Device, family: u32, queue: vk::Queue, tier: Tier) -> Self {
        let resources = (tier != Tier::SameQueueFallback).then(|| unsafe {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let pool = device
                .create_command_pool(&pool_info, None)
                .expect("Failed to create transfer command pool");
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = device
                .allocate_command_buffers(&alloc_info)
                .expect("Failed to allocate transfer command buffer")[0];
            LaneResources {
                pool,
                cmd,
                timeline: Timeline::new(device),
            }
        });
        Self {
            tier,
            family,
            queue,
            resources,
        }
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    pub fn family(&self) -> u32 {
        self.family
    }

    /// `true` when copies submit to a queue distinct from the graphics
    /// queue — the caller must order consumers with a timeline wait rather
    /// than an in-command-buffer barrier (which cannot scope across queues).
    pub fn is_separate_queue(&self) -> bool {
        self.resources.is_some()
    }

    /// `true` only for a genuinely disjoint queue family: the one case that
    /// needs `EXCLUSIVE` queue-family-ownership release/acquire barriers
    /// (same-family queues never separately own a resource in Vulkan's
    /// model, so `SecondQueueSameFamily`/`SameQueueFallback` need neither).
    pub fn needs_ownership_transfer(&self) -> bool {
        self.tier == Tier::DedicatedFamily
    }

    /// Begin recording a copy batch.
    pub unsafe fn begin(&mut self, device: &ash::Device) -> LaneRecording {
        let res = self
            .resources
            .as_mut()
            .expect("TransferLane::begin requires a separate queue");
        unsafe {
            device
                .reset_command_buffer(res.cmd, vk::CommandBufferResetFlags::empty())
                .expect("transfer command buffer reset failed");
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(res.cmd, &begin)
                .expect("begin transfer command buffer failed");
        }
        LaneRecording { cmd: res.cmd }
    }

    /// Submit the batch and return the timeline value to wait on.
    pub unsafe fn submit(&mut self, device: &ash::Device, batch: LaneRecording) -> TimelineValue {
        let res = self
            .resources
            .as_mut()
            .expect("TransferLane::submit requires a separate queue");
        unsafe {
            device
                .end_command_buffer(batch.cmd)
                .expect("end transfer command buffer failed");
        }
        let rs = res.timeline.begin_render(batch.cmd);
        let value = rs.value();
        let completion = unsafe { rs.submit(device, self.queue, &res.timeline, None) };
        debug_assert_eq!(completion.value(), value);
        value
    }

    /// End a batch without submitting (nothing to do).
    pub unsafe fn discard(&self, device: &ash::Device, batch: LaneRecording) {
        unsafe {
            device
                .end_command_buffer(batch.cmd)
                .expect("end empty transfer command buffer failed");
        }
    }

    /// The timeline semaphore for graphics submission's wait info.
    pub fn semaphore(&self) -> vk::Semaphore {
        self.resources
            .as_ref()
            .expect("TransferLane::semaphore requires a separate queue")
            .timeline
            .semaphore()
    }

    /// Wait until the queue reaches this timeline value.
    pub unsafe fn wait(&self, device: &ash::Device, value: TimelineValue) {
        let res = self
            .resources
            .as_ref()
            .expect("TransferLane::wait requires a separate queue");
        unsafe { res.timeline.wait(device, value) };
    }

    /// Non-blocking read of the lane's timeline progress. None if no separate queue.
    pub unsafe fn counter(&self, device: &ash::Device) -> Option<TimelineValue> {
        let res = self.resources.as_ref()?;
        Some(unsafe { res.timeline.counter(device) })
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if let Some(res) = self.resources.take() {
            unsafe {
                res.timeline.destroy(device);
                device.destroy_command_pool(res.pool, None);
            }
        }
    }
}
