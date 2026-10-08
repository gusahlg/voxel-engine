/// A dedicated queue for staging copies (or the graphics queue as fallback).
/// Three tiers in preference order: dedicated family, second queue in graphics family, or graphics queue.
/// Callers don't branch on availability; only sync behavior changes by tier.
use ash::vk;

use super::mesh_residency::CopyConsumer;
use super::retire::RetireQueue;
use super::timeline::{Timeline, TimelineValue};

/// Stage at which a lane batch waits for graphics before it overwrites data
/// earlier frames read: the copy's own stage, so the copy and the barriers
/// recorded before it chain on the wait.
pub(crate) const LANE_COPY_WAIT_STAGE: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::COPY;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Tier {
    /// A queue family with the specialized flag set (`TRANSFER` for the
    /// transfer lane, `COMPUTE` for the compute lane) and `GRAPHICS` unset.
    DedicatedFamily,
    /// A second (or third) queue index within the graphics family.
    SecondQueueSameFamily,
    /// No spare queue; work recorded inline on the caller's command buffer.
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

    pub(crate) fn from_cmd(cmd: vk::CommandBuffer) -> Self {
        Self { cmd }
    }
}

/// Recycling state of one per-batch command buffer in a [`BatchRing`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BatchState {
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
///
/// Shared with the compute lane, which has the same reset-while-pending trap.
pub(crate) struct BatchRing<T> {
    entries: Vec<(T, BatchState)>,
}

impl<T: Copy + PartialEq> BatchRing<T> {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Hands out a `Free` buffer (marking it `Recording`), if any.
    pub(crate) fn take_free(&mut self) -> Option<T> {
        let entry = self
            .entries
            .iter_mut()
            .find(|(_, state)| *state == BatchState::Free)?;
        entry.1 = BatchState::Recording;
        Some(entry.0)
    }

    /// Frees every `Pending` buffer whose batch the lane has completed.
    pub(crate) fn reclaim(&mut self, completed: TimelineValue) {
        for (_, state) in &mut self.entries {
            if let BatchState::Pending(value) = *state
                && value <= completed
            {
                *state = BatchState::Free;
            }
        }
    }

    /// Registers a freshly allocated buffer, already handed out.
    pub(crate) fn push_recording(&mut self, handle: T) {
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
    pub(crate) fn submitted(&mut self, handle: T, value: TimelineValue) {
        self.set_state(handle, BatchState::Pending(value));
    }

    /// The batch on `handle` was ended without a submit: idle right away.
    pub(crate) fn discarded(&mut self, handle: T) {
        self.set_state(handle, BatchState::Free);
    }

    pub(crate) fn len(&self) -> usize {
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
        unsafe { self.submit_after(device, batch, None) }
    }

    /// Submit the batch, optionally waiting on another timeline first (e.g.
    /// graphics sampling of layers this copy will overwrite).
    pub unsafe fn submit_after(
        &mut self,
        device: &ash::Device,
        batch: LaneRecording,
        extra_wait: Option<(vk::Semaphore, TimelineValue, vk::PipelineStageFlags2)>,
    ) -> TimelineValue {
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
        let completion = unsafe { rs.submit(device, self.queue, &res.timeline, extra_wait) };
        debug_assert_eq!(completion.value(), value);
        res.ring.submitted(batch.cmd, value);
        value
    }

    /// End a batch without submitting (nothing to do); its buffer is idle
    /// again immediately.
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

/// Per-frame arguments of the transfer-lane uploaders' `flush` (block
/// textures, the material table, far maps). Built by the frame loop after it
/// has submitted any batch an overwrite must follow and reserved this frame's
/// render value.
pub(crate) struct TransferCtx<'a> {
    pub instance: &'a ash::Instance,
    pub device: &'a ash::Device,
    pub physical: vk::PhysicalDevice,
    pub lane: &'a mut TransferLane,
    /// This frame's command buffer, on the graphics queue.
    pub graphics_cmd: vk::CommandBuffer,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
    pub graphics_timeline: &'a Timeline,
    /// Latest render value already submitted: a same-family lane overwrite
    /// waits for it before writing data earlier frames may still read.
    pub last_render_value: TimelineValue,
    /// Render value this frame signals.
    pub done_at: TimelineValue,
}

impl TransferCtx<'_> {
    pub fn memory_props(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical)
        }
    }
}

/// Staging buffers and ownership-release command buffers of one
/// transfer-lane uploader, each retired on the timeline of its last use.
pub(crate) struct UploadRetire {
    /// Graphics pool the release command buffers come from.
    command_pool: vk::CommandPool,
    /// Names the uploader in panic messages.
    label: &'static str,
    /// Staging read by the frame command buffer: freed once the frame's
    /// render value has signaled.
    staging: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
    /// Staging read by a separate-queue lane batch: freed on the lane's own
    /// timeline value.
    transfer: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
    /// Dedicated-family overwrite release: throwaway graphics timeline + CB.
    /// Not the render timeline — that would reserve a value after the frame
    /// already reserved its signal and submit out of order. Freed at the
    /// frame's render value.
    release_cmds: RetireQueue<(Timeline, vk::CommandBuffer)>,
}

impl UploadRetire {
    pub fn new(command_pool: vk::CommandPool, label: &'static str) -> Self {
        Self {
            command_pool,
            label,
            staging: RetireQueue::new(),
            transfer: RetireQueue::new(),
            release_cmds: RetireQueue::new(),
        }
    }

    pub fn has_garbage(&self) -> bool {
        !self.staging.is_empty() || !self.transfer.is_empty() || !self.release_cmds.is_empty()
    }

    /// Staging read by the frame command buffer that signals `done_at`.
    pub fn retire_on_graphics(
        &mut self,
        done_at: TimelineValue,
        staging: (vk::Buffer, vk::DeviceMemory),
    ) {
        self.staging.push(done_at, staging);
    }

    /// Staging read by the lane batch that signals `value` on the lane.
    pub fn retire_on_lane(
        &mut self,
        value: TimelineValue,
        staging: (vk::Buffer, vk::DeviceMemory),
    ) {
        self.transfer.push(value, staging);
    }

    /// Dedicated-family overwrite: record `release` (the graphics-side
    /// release of data earlier frames read) into a one-off graphics command
    /// buffer and submit it on a throwaway timeline, after the frames already
    /// submitted to the graphics queue. No host wait. Returns the wait the
    /// lane batch that acquires the data submits with. The command buffer
    /// and its timeline are freed once `done_at` has signaled.
    pub unsafe fn submit_release(
        &mut self,
        device: &ash::Device,
        graphics_queue: vk::Queue,
        done_at: TimelineValue,
        release: &vk::DependencyInfo<'_>,
    ) -> (vk::Semaphore, TimelineValue, vk::PipelineStageFlags2) {
        let label = self.label;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc)
                .unwrap_or_else(|e| panic!("allocate {label} release command buffer: {e:?}"))[0]
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .unwrap_or_else(|e| panic!("begin {label} release command buffer: {e:?}"));
            device.cmd_pipeline_barrier2(cmd, release);
            device
                .end_command_buffer(cmd)
                .unwrap_or_else(|e| panic!("end {label} release command buffer: {e:?}"));
        }
        let mut tmp = unsafe { Timeline::new(device) };
        let rs = tmp.begin_render(cmd);
        let completion = unsafe { rs.submit(device, graphics_queue, &tmp, None) };
        let sem = tmp.semaphore();
        let value = completion.value();
        self.release_cmds.push(done_at, (tmp, cmd));
        (sem, value, LANE_COPY_WAIT_STAGE)
    }

    /// Free graphics-side staging and release command buffers up to the
    /// completed render value `current`.
    pub unsafe fn collect(&mut self, device: &ash::Device, current: TimelineValue) {
        self.staging.collect(current, |staging| unsafe {
            destroy_staging(device, staging)
        });
        let pool = self.command_pool;
        self.release_cmds
            .collect(current, |(timeline, cmd)| unsafe {
                timeline.destroy(device);
                device.free_command_buffers(pool, &[cmd]);
            });
    }

    /// Free lane staging up to the lane's completed value `current`.
    pub unsafe fn collect_transfer(&mut self, device: &ash::Device, current: TimelineValue) {
        self.transfer.collect(current, |staging| unsafe {
            destroy_staging(device, staging)
        });
    }

    /// Free everything. The GPU must be idle.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.staging
                .collect_all(|staging| destroy_staging(device, staging));
            self.transfer
                .collect_all(|staging| destroy_staging(device, staging));
            let pool = self.command_pool;
            self.release_cmds.collect_all(|(timeline, cmd)| {
                timeline.destroy(device);
                device.free_command_buffers(pool, &[cmd]);
            });
        }
    }
}

/// Barriers of one staged copy into a storage-buffer range that graphics
/// reads at `consumer` (the material table, the far-map datum). `overwrite`:
/// earlier frames read the range, and the frame loop has submitted them
/// (`has_overwrite_pending`).
///
/// - `DedicatedFamily`: on an overwrite, graphics releases the range after
///   those reads and the lane acquires it, chained on its wait at
///   [`LANE_COPY_WAIT_STAGE`]. After the copy the lane releases the range and
///   the recording frame acquires it, chained on its lane wait at the
///   consumer stage. Both halves of each pair have the same families and range.
/// - `SecondQueueSameFamily`: no barriers. On an overwrite the lane waits for
///   the last submitted frame at [`LANE_COPY_WAIT_STAGE`]; the recording frame
///   waits on the lane.
/// - `SameQueueFallback`: in the frame command buffer, the copy follows the
///   earlier reads (overwrite only) and precedes this frame's reads.
#[derive(Default)]
pub(crate) struct BufferUploadBarriers {
    /// Graphics-side release, submitted before the lane batch.
    pub release: Option<vk::BufferMemoryBarrier2<'static>>,
    /// Where the copy is recorded, before it.
    pub before_copy: Option<vk::BufferMemoryBarrier2<'static>>,
    /// Where the copy is recorded, after it.
    pub after_copy: Option<vk::BufferMemoryBarrier2<'static>>,
    /// Graphics-side acquire in the recording frame.
    pub acquire: Option<vk::BufferMemoryBarrier2<'static>>,
}

/// `dst` is `(buffer, offset, size)`.
pub(crate) fn buffer_upload_barriers(
    dst: (vk::Buffer, u64, u64),
    consumer: CopyConsumer,
    overwrite: bool,
    tier: Tier,
    graphics_family: u32,
    lane_family: u32,
) -> BufferUploadBarriers {
    use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
    type B = vk::BufferMemoryBarrier2<'static>;
    let (buffer, offset, size) = dst;
    let range = B::default().buffer(buffer).offset(offset).size(size);
    // First -> second synchronization scope, each (stages, accesses).
    let scopes = |b: B, from: CopyConsumer, to: CopyConsumer| {
        b.src_stage_mask(from.0)
            .src_access_mask(from.1)
            .dst_stage_mask(to.0)
            .dst_access_mask(to.1)
    };
    let none = (S::NONE, A::NONE);
    let copy = (S::COPY, A::TRANSFER_WRITE);
    match tier {
        Tier::DedicatedFamily => {
            let to_lane = range
                .src_queue_family_index(graphics_family)
                .dst_queue_family_index(lane_family);
            let to_graphics = range
                .src_queue_family_index(lane_family)
                .dst_queue_family_index(graphics_family);
            let lane_wait = (LANE_COPY_WAIT_STAGE, A::NONE);
            let frame_wait = (consumer.0, A::NONE);
            BufferUploadBarriers {
                release: overwrite.then(|| scopes(to_lane, consumer, none)),
                before_copy: overwrite.then(|| scopes(to_lane, lane_wait, copy)),
                after_copy: Some(scopes(to_graphics, copy, none)),
                acquire: Some(scopes(to_graphics, frame_wait, consumer)),
            }
        }
        Tier::SecondQueueSameFamily => BufferUploadBarriers::default(),
        Tier::SameQueueFallback => BufferUploadBarriers {
            before_copy: overwrite.then(|| scopes(range, consumer, copy)),
            after_copy: Some(scopes(range, copy, consumer)),
            ..BufferUploadBarriers::default()
        },
    }
}

/// Copy `staging` into `dst` (`(buffer, offset, size)`), a storage-buffer
/// range graphics reads at `consumer`: on the lane, or in the frame command
/// buffer on `SameQueueFallback`. Barriers and `overwrite` as in
/// [`buffer_upload_barriers`]. Returns the lane value the frame must wait on
/// at the consumer stage. Staging retires on the timeline that reads it. No
/// host wait.
pub(crate) unsafe fn upload_buffer_range(
    ctx: &mut TransferCtx<'_>,
    retire: &mut UploadRetire,
    dst: (vk::Buffer, u64, u64),
    staging: (vk::Buffer, vk::DeviceMemory),
    consumer: CopyConsumer,
    overwrite: bool,
) -> Option<TimelineValue> {
    let device = ctx.device;
    let graphics_cmd = ctx.graphics_cmd;
    let done_at = ctx.done_at;
    let separate = ctx.lane.is_separate_queue();
    let barriers = buffer_upload_barriers(
        dst,
        consumer,
        overwrite,
        ctx.lane.tier(),
        ctx.graphics_family,
        ctx.lane.family(),
    );
    let extra_wait = if let Some(release) = barriers.release.as_ref() {
        let release =
            vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(release));
        Some(unsafe { retire.submit_release(device, ctx.graphics_queue, done_at, &release) })
    } else if separate && overwrite {
        Some((
            ctx.graphics_timeline.semaphore(),
            ctx.last_render_value,
            LANE_COPY_WAIT_STAGE,
        ))
    } else {
        None
    };
    let lane = &mut *ctx.lane;
    let lane_batch = separate.then(|| unsafe { lane.begin(device) });
    let record_cmd = lane_batch.as_ref().map_or(graphics_cmd, |b| b.cmd());
    let (buffer, offset, size) = dst;
    let region = vk::BufferCopy::default().dst_offset(offset).size(size);
    unsafe {
        cmd_buffer_barrier(device, record_cmd, barriers.before_copy.as_ref());
        device.cmd_copy_buffer(record_cmd, staging.0, buffer, &[region]);
        cmd_buffer_barrier(device, record_cmd, barriers.after_copy.as_ref());
    }
    let Some(lane_batch) = lane_batch else {
        retire.retire_on_graphics(done_at, staging);
        return None;
    };
    let value = unsafe { lane.submit_after(device, lane_batch, extra_wait) };
    unsafe { cmd_buffer_barrier(device, graphics_cmd, barriers.acquire.as_ref()) };
    retire.retire_on_lane(value, staging);
    Some(value)
}

pub(crate) unsafe fn cmd_buffer_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    barrier: Option<&vk::BufferMemoryBarrier2<'_>>,
) {
    if let Some(barrier) = barrier {
        let dep =
            vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(barrier));
        unsafe { device.cmd_pipeline_barrier2(cmd, &dep) };
    }
}

unsafe fn destroy_staging(device: &ash::Device, (buffer, memory): (vk::Buffer, vk::DeviceMemory)) {
    unsafe {
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
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

    mod buffer_upload {
        use super::super::{
            BufferUploadBarriers, LANE_COPY_WAIT_STAGE, Tier, buffer_upload_barriers,
        };
        use ash::vk::{self, AccessFlags2 as A, Handle, PipelineStageFlags2 as S};

        const GRAPHICS: u32 = 0;
        const LANE: u32 = 1;
        const STAGE: S = S::FRAGMENT_SHADER;
        const READS: A = A::SHADER_STORAGE_READ;

        fn build(tier: Tier, overwrite: bool) -> BufferUploadBarriers {
            let dst = (vk::Buffer::from_raw(7), 256, 1024);
            buffer_upload_barriers(dst, (STAGE, READS), overwrite, tier, GRAPHICS, LANE)
        }

        /// Families and range: what both halves of a transfer must agree on.
        fn key(b: Option<vk::BufferMemoryBarrier2<'_>>) -> (u32, u32, u64, u64, u64) {
            let b = b.expect("barrier present");
            let families = (b.src_queue_family_index, b.dst_queue_family_index);
            (families.0, families.1, b.buffer.as_raw(), b.offset, b.size)
        }

        fn stages(b: Option<vk::BufferMemoryBarrier2<'_>>) -> (S, A, S, A) {
            let b = b.expect("barrier present");
            (
                b.src_stage_mask,
                b.src_access_mask,
                b.dst_stage_mask,
                b.dst_access_mask,
            )
        }

        #[test]
        fn dedicated_family_halves_match_and_chain_on_the_waits() {
            let b = build(Tier::DedicatedFamily, true);
            assert_eq!(key(b.release), (GRAPHICS, LANE, 7, 256, 1024));
            assert_eq!(key(b.release), key(b.before_copy));
            assert_eq!(key(b.after_copy), (LANE, GRAPHICS, 7, 256, 1024));
            assert_eq!(key(b.after_copy), key(b.acquire));
            // Release after earlier reads; the lane acquire chains on the
            // lane's wait (`COPY`), the frame acquire on the frame's lane
            // wait (the consumer stage).
            assert_eq!(stages(b.release), (STAGE, READS, S::NONE, A::NONE));
            assert_eq!(LANE_COPY_WAIT_STAGE, S::COPY);
            let lane_acquire = (LANE_COPY_WAIT_STAGE, A::NONE, S::COPY, A::TRANSFER_WRITE);
            assert_eq!(stages(b.before_copy), lane_acquire);
            let lane_release = (S::COPY, A::TRANSFER_WRITE, S::NONE, A::NONE);
            assert_eq!(stages(b.after_copy), lane_release);
            assert_eq!(stages(b.acquire), (STAGE, A::NONE, STAGE, READS));
            // First upload: nothing to take from graphics.
            let first = build(Tier::DedicatedFamily, false);
            assert!(first.release.is_none() && first.before_copy.is_none());
            assert_eq!(stages(first.after_copy), lane_release);
            assert_eq!(key(first.after_copy), key(first.acquire));
            assert_eq!(stages(first.acquire), stages(b.acquire));
        }

        #[test]
        fn same_family_tiers_move_no_ownership() {
            for overwrite in [false, true] {
                // Ordered by the lane wait and the frame's wait on the lane.
                let second = build(Tier::SecondQueueSameFamily, overwrite);
                assert!(second.release.is_none() && second.before_copy.is_none());
                assert!(second.after_copy.is_none() && second.acquire.is_none());

                let same = build(Tier::SameQueueFallback, overwrite);
                assert!(same.release.is_none() && same.acquire.is_none());
                assert_eq!(same.before_copy.is_some(), overwrite);
                for b in same.before_copy.into_iter().chain(same.after_copy) {
                    assert_eq!(key(Some(b)), (0, 0, 7, 256, 1024));
                }
                let after = (S::COPY, A::TRANSFER_WRITE, STAGE, READS);
                assert_eq!(stages(same.after_copy), after);
                if overwrite {
                    let before = (STAGE, READS, S::COPY, A::TRANSFER_WRITE);
                    assert_eq!(stages(same.before_copy), before);
                }
            }
        }
    }
}
