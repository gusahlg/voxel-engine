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

/// Recycling state of one per-batch command buffer in a [`BatchRing`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BatchState {
    /// Never submitted, or its last batch was observed complete.
    Free,
    /// Handed out by `begin`, not yet submitted or discarded.
    Recording,
    /// Submitted; reusable once the lane timeline reaches the value.
    Pending(TimelineValue),
}

/// Timeline-recycled ring of per-batch command buffers. A single reused
/// buffer used to be reset at the next `begin` while its previous batch
/// could still be executing on the transfer queue (a reset in the PENDING
/// state — nothing waited on the lane between two flushes, and the quad-IBO
/// grow batches the same frame as a mesh batch). Each batch now records
/// into a buffer that is provably idle: `Free`, or `Pending(v)` with the
/// lane's counter at or past `v`. Pure bookkeeping (no Vulkan calls) so the
/// recycling rule is host-testable; the ring only ever grows (steady state
/// is two or three buffers) and the pool frees them all at destroy.
struct BatchRing<T> {
    entries: Vec<(T, BatchState)>,
}

impl<T: Copy + PartialEq> BatchRing<T> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Hands out a `Free` buffer (marking it `Recording`), if any.
    fn take_free(&mut self) -> Option<T> {
        let entry = self
            .entries
            .iter_mut()
            .find(|(_, state)| *state == BatchState::Free)?;
        entry.1 = BatchState::Recording;
        Some(entry.0)
    }

    /// Frees every `Pending` buffer whose batch the lane has completed.
    fn reclaim(&mut self, completed: TimelineValue) {
        for (_, state) in &mut self.entries {
            if let BatchState::Pending(value) = *state
                && value <= completed
            {
                *state = BatchState::Free;
            }
        }
    }

    /// Registers a freshly allocated buffer, already handed out.
    fn push_recording(&mut self, handle: T) {
        self.entries.push((handle, BatchState::Recording));
    }

    fn set_state(&mut self, handle: T, state: BatchState) {
        let entry = self
            .entries
            .iter_mut()
            .find(|(h, _)| *h == handle)
            .expect("batch handle belongs to this ring");
        debug_assert_eq!(entry.1, BatchState::Recording, "batch was not recording");
        entry.1 = state;
    }

    /// The batch on `handle` was submitted and signals `value` when done.
    fn submitted(&mut self, handle: T, value: TimelineValue) {
        self.set_state(handle, BatchState::Pending(value));
    }

    /// The batch on `handle` was ended without a submit: idle right away.
    #[allow(dead_code)] // used by `TransferLane::discard` (empty-batch path)
    fn discarded(&mut self, handle: T) {
        self.set_state(handle, BatchState::Free);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Resources for tiers with a distinct queue (not SameQueueFallback).
struct LaneResources {
    pool: vk::CommandPool,
    ring: BatchRing<vk::CommandBuffer>,
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
            LaneResources {
                pool,
                ring: BatchRing::new(),
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

    /// Begin recording a copy batch on an idle per-batch command buffer:
    /// a free one, else one whose batch the lane's counter proves complete
    /// (one non-blocking counter read), else a newly allocated one. Never
    /// resets a buffer that may still be pending on the queue.
    pub unsafe fn begin(&mut self, device: &ash::Device) -> LaneRecording {
        let res = self
            .resources
            .as_mut()
            .expect("TransferLane::begin requires a separate queue");
        let recycled = res.ring.take_free().or_else(|| {
            let completed = unsafe { res.timeline.counter(device) };
            res.ring.reclaim(completed);
            res.ring.take_free()
        });
        let cmd = match recycled {
            Some(cmd) => cmd,
            None => {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(res.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }
                    .expect("Failed to allocate transfer command buffer")[0];
                res.ring.push_recording(cmd);
                log::debug!("transfer lane grew to {} command buffers", res.ring.len());
                cmd
            }
        };
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("transfer command buffer reset failed");
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cmd, &begin)
                .expect("begin transfer command buffer failed");
        }
        LaneRecording { cmd }
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
        res.ring.submitted(batch.cmd, value);
        value
    }

    /// End a batch without submitting (nothing to do); its buffer is idle
    /// again immediately. Mesh copies no longer begin the lane speculatively,
    /// so production no longer hits an empty batch; tests still do.
    #[allow(dead_code)]
    pub unsafe fn discard(&mut self, device: &ash::Device, batch: LaneRecording) {
        let res = self
            .resources
            .as_mut()
            .expect("TransferLane::discard requires a separate queue");
        unsafe {
            device
                .end_command_buffer(batch.cmd)
                .expect("end empty transfer command buffer failed");
        }
        res.ring.discarded(batch.cmd);
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

#[cfg(test)]
mod tests {
    use super::{BatchRing, BatchState, TimelineValue};

    #[test]
    fn ring_recycles_only_completed_batches() {
        let mut ring: BatchRing<u32> = BatchRing::new();
        assert_eq!(ring.take_free(), None);
        ring.push_recording(1);
        ring.submitted(1, TimelineValue::from_raw_for_test(1));
        // Still pending: the counter has not reached its value.
        ring.reclaim(TimelineValue::from_raw_for_test(0));
        assert_eq!(ring.take_free(), None);
        // A second batch needs a second buffer while the first is in flight.
        ring.push_recording(2);
        ring.submitted(2, TimelineValue::from_raw_for_test(2));
        assert_eq!(ring.len(), 2);
        // Counter at 1: only the first buffer is idle again.
        ring.reclaim(TimelineValue::from_raw_for_test(1));
        assert_eq!(ring.take_free(), Some(1));
        assert_eq!(ring.take_free(), None);
        // Counter at 2: the second follows; the first is recording, not free.
        ring.reclaim(TimelineValue::from_raw_for_test(2));
        assert_eq!(ring.take_free(), Some(2));
        assert_eq!(ring.take_free(), None);
        assert_eq!(ring.len(), 2);
    }

    #[test]
    fn discarded_batch_is_idle_immediately() {
        let mut ring: BatchRing<u32> = BatchRing::new();
        ring.push_recording(7);
        ring.discarded(7);
        assert_eq!(ring.entries[0].1, BatchState::Free);
        assert_eq!(ring.take_free(), Some(7));
        assert_eq!(ring.entries[0].1, BatchState::Recording);
    }

    #[test]
    fn reclaim_is_idempotent_and_ignores_recording() {
        let mut ring: BatchRing<u32> = BatchRing::new();
        ring.push_recording(1);
        ring.reclaim(TimelineValue::from_raw_for_test(u64::MAX));
        assert_eq!(ring.entries[0].1, BatchState::Recording);
        ring.submitted(1, TimelineValue::from_raw_for_test(3));
        ring.reclaim(TimelineValue::from_raw_for_test(5));
        ring.reclaim(TimelineValue::from_raw_for_test(5));
        assert_eq!(ring.take_free(), Some(1));
    }
}
