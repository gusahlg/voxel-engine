use std::num::NonZeroU32;

use ash::vk;

use super::alloc::Allocation;
use super::mesh_resident::{CopySource, GpuResident};
use super::mesh_staging::{StagingLease, Stamp};
use super::retire::RetireQueue;
use super::timeline::TimelineValue;
use super::transfer::TransferLane;

/// Mesh-copy staging budget per frame; amortizes bursty uploads.
const TRANSFER_BUDGET_BYTES_PER_FRAME: u64 = 8 * 1024 * 1024;

/// A staged host→device copy owned by a not-yet-flushed [`GpuResident`].
/// The role of one staged-copy buffer barrier: the same-queue copy→draw
/// barrier, or the release/acquire halves of a queue-family ownership
/// transfer. Six call sites used to restate the stage/access pairings
/// field-by-field — the exact part a reviewer must get right — so the
/// pairings live here once and a site states only its buffer range, its
/// draw-side reads, and its role.
#[derive(Clone, Copy)]
pub(crate) enum CopyBarrier {
    /// Same queue: copy → vertex input, visible in this submission.
    Draw,
    /// QFOT release on the transfer queue: copy → nothing (ownership leaves).
    Release { src_family: u32, dst_family: u32 },
    /// QFOT acquire on graphics: nothing → vertex input (ownership arrives).
    Acquire { src_family: u32, dst_family: u32 },
}

/// Build one staged-copy barrier for `role` over `buffer[offset..offset+size]`;
/// `reads` is the draw-side access the data feeds (vertex+index for meshes,
/// index-only for the shared quad IBO). Ignored by `Release`, whose
/// destination half is the acquire's job.
pub(crate) fn copy_barrier(
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
    reads: vk::AccessFlags2,
    role: CopyBarrier,
) -> vk::BufferMemoryBarrier2<'static> {
    let barrier = vk::BufferMemoryBarrier2::default()
        .buffer(buffer)
        .offset(offset)
        .size(size);
    match role {
        CopyBarrier::Draw => barrier
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
            .dst_access_mask(reads),
        CopyBarrier::Release {
            src_family,
            dst_family,
        } => barrier
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family),
        CopyBarrier::Acquire {
            src_family,
            dst_family,
        } => barrier
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
            .dst_access_mask(reads)
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family),
    }
}

/// The draw-side reads a mesh buffer feeds (interleaved vertices + indices).
fn mesh_reads() -> vk::AccessFlags2 {
    vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::INDEX_READ
}

/// The stages at which uploaded mesh bytes (and the shared quad IBO) are
/// first consumed: fixed-function vertex/index fetch in the shadow cascades
/// and the scene passes (`VERTEX_INPUT` = `INDEX_INPUT | VERTEX_ATTRIBUTE_INPUT`
/// in synchronization2). Nothing samples or pulls mesh bytes from a shader
/// stage, so a cross-queue wait scoped here leaves cull, clears, the sky and
/// post free to start before the transfer queue signals.
pub(crate) const MESH_CONSUMER_STAGES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::VERTEX_INPUT;

/// One destination byte range written by a copy batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BufferRange {
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
}

/// Sorts by (buffer, offset) and merges ranges of the same buffer that touch
/// or overlap, so a burst of meshes landing in one arena needs one barrier per
/// contiguous run instead of one per mesh. Never widens past bytes the batch
/// actually wrote: on the dedicated-family tier these ranges become
/// ownership release/acquire pairs, and claiming bytes another live mesh in
/// the same arena is being drawn from would hand their ownership around
/// underneath those draws — wrong, not merely wasteful.
fn coalesce_ranges(mut ranges: Vec<BufferRange>) -> Vec<BufferRange> {
    use ash::vk::Handle;
    ranges.sort_unstable_by_key(|r| (r.buffer.as_raw(), r.offset));
    let mut merged: Vec<BufferRange> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match merged.last_mut() {
            Some(last) if last.buffer == r.buffer && last.offset + last.size >= r.offset => {
                last.size = last.size.max(r.offset + r.size - last.offset);
            }
            _ => merged.push(r),
        }
    }
    merged
}

/// The graphics-side half of a lane batch that has been submitted but not yet
/// consumed: the timeline value it signals and, on the dedicated-family tier,
/// the ownership-transfer ACQUIRE barriers over its destination ranges. Both
/// are applied by the NEXT graphics submission — see the hazard analysis in
/// [`MeshResidency::flush_copies`].
struct DeferredArrival {
    value: TimelineValue,
    acquires: Vec<vk::BufferMemoryBarrier2<'static>>,
}

/// Render-side residency mirror for meshes: keyed by the main-assigned slot,
/// with a generation mirror kept in sync from the ordered command stream,
/// guaranteeing correct handle-aliasing without cross-thread reads. Holds no
/// free-list or identity — that is [`HandleAllocator`]'s job.
pub(crate) struct MeshResidency {
    slots: Vec<Option<GpuResident>>,
    generations: Vec<NonZeroU32>,
    /// `(slot, frame)` queued at [`Self::apply_upload`]; `frame` is
    /// [`Self::frame`] at apply so arrival delay is host-testable.
    pending: Vec<(u32, u64)>,
    /// Device buffers and same-queue staging (render-Rev).
    retire: RetireQueue<Allocation>,
    /// Staging for separate transfer queue (lane-Rev).
    transfer_retire: RetireQueue<Allocation>,
    live: usize,
    /// Slots with just-submitted copies, ready to expose in arena word.
    arrived_since_flush: Vec<u32>,
    /// The last separate-queue batch's wait value + ACQUIRE barriers, owed
    /// to the next graphics submission (see [`Self::flush_copies`]).
    deferred: Option<DeferredArrival>,
    /// Bumped once per render-loop iteration ([`Self::note_frame`]) before
    /// flush, so a copy applied in the preceding drain has delay 1.
    frame: u64,
    /// Last flush's max pending-to-arrived delay in frames (pooled copies).
    last_pool_arrival_frames: u64,
}

impl MeshResidency {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            generations: Vec::new(),
            pending: Vec::new(),
            retire: RetireQueue::new(),
            transfer_retire: RetireQueue::new(),
            live: 0,
            arrived_since_flush: Vec::new(),
            deferred: None,
            frame: 0,
            last_pool_arrival_frames: 0,
        }
    }

    /// Advance the arrival-delay clock. Call once per render-loop iteration
    /// after the command drain (so apply sees the previous value) and before
    /// [`Self::flush_copies`].
    pub fn note_frame(&mut self) {
        self.frame = self.frame.saturating_add(1);
    }

    /// Check if slot's bytes are visible to the cull dispatch.
    pub fn is_arrived(&self, slot: u32) -> bool {
        self.slots
            .get(slot as usize)
            .and_then(|s| s.as_ref())
            .is_some_and(|r| r.arrived_at.is_some())
    }

    /// Drain arrived slots so RecordTable re-reads their arena words.
    pub fn take_arrived(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.arrived_since_flush)
    }

    fn ensure_slot(&mut self, i: usize) {
        if self.slots.len() <= i {
            self.slots.resize_with(i + 1, || None);
            self.generations.resize(i + 1, NonZeroU32::MIN);
        }
    }

    /// Installs a freshly-built resident at `slot`, updating the generation
    /// mirror. Queues its staging copy (if any) for the next flush.
    pub fn apply_upload(&mut self, slot: u32, generation: NonZeroU32, resident: GpuResident) {
        let i = slot as usize;
        self.ensure_slot(i);
        if resident.copy.is_some() {
            self.pending.push((slot, self.frame));
        }
        if self.slots[i].is_none() {
            self.live += 1;
        }
        self.slots[i] = Some(resident);
        self.generations[i] = generation;
    }

    /// Retires the resident at `slot` (gen-checked) past `done_at`. A no-op if
    /// the slot was already reused (the mirror generation moved on).
    pub fn apply_free(&mut self, slot: u32, generation: NonZeroU32, done_at: TimelineValue) {
        let i = slot as usize;
        if self.generations.get(i).copied() != Some(generation) {
            return;
        }
        if let Some(res) = self.slots.get_mut(i).and_then(Option::take) {
            self.retire.push(done_at, res.arena);
            if let Some(copy) = res.copy {
                match copy.source {
                    CopySource::Alloc(alloc) => self.retire.push(done_at, alloc),
                    CopySource::Pool(lease) => lease.stamp(Stamp::Render(done_at.raw())),
                }
            }
            self.live -= 1;
        }
    }

    /// Drains `self.pending` up to [`TRANSFER_BUDGET_BYTES_PER_FRAME`] (the
    /// first item is always serviced regardless of size, a forward-progress
    /// floor). A barrier cannot scope a cross-queue dependency, so only the
    /// `SameQueueFallback` tier gets an in-command-buffer barrier; separate-
    /// queue copies order via the lane's timeline instead.
    ///
    /// Copies are issued as one `vkCmdCopyBuffer` per (staging block, arena)
    /// pair and the barriers cover coalesced destination runs (see
    /// [`coalesce_ranges`]), so a burst of N meshes costs O(arenas) commands.
    /// Pooled copies share this path with per-mesh staging allocs: same lane
    /// batch, same coalesced barrier, same `arrived_at`. Regions stamp
    /// [`Stamp::Transfer`] on the separate-queue tier and [`Stamp::Render`]
    /// on the fallback tier.
    ///
    /// # Cross-queue hazard analysis (why the wait is deferred one frame)
    ///
    /// A batch submitted on the lane during frame N is never read by frame
    /// N's own graphics submission:
    /// - the GPU cull that emits mesh draws reads the arena word table, which
    ///   [`RecordTable::flush`] wrote *before* this call with the slot still
    ///   gated to 0 (`is_arrived` was false); [`Self::take_arrived`] then
    ///   marks the slot dirty so the word is revealed in frame N+1's table;
    /// - the CPU Blend walk is gated on `is_arrived` at the same point;
    /// - the shared quad IBO is grown by [`QuadIbo::ensure`] *after* this call
    ///   and, since a unified-memory upload can be drawn the same frame, keeps
    ///   its own same-frame wait.
    ///
    /// So the earliest consumer is frame N+1's vertex/index fetch. Frame N's
    /// submission therefore does not wait on the lane at all; the batch's
    /// value (and, on `DedicatedFamily`, its ACQUIRE barriers, which must
    /// execute after the release via that very semaphore wait) is stashed
    /// and applied by [`Self::take_deferred_arrival`] on frame N+1's command
    /// buffer, whose submission waits on the lane at
    /// [`MESH_CONSUMER_STAGES`]. Later frames are covered too: a queue wait's
    /// second scope includes every submission later in submission order. The
    /// benefit: the graphics queue never idles on a copy that was enqueued
    /// microseconds earlier, so a streaming burst no longer costs a frame
    /// spike. Destination ranges are only ever reused after their previous
    /// occupant's reads retired through the render timeline (the allocator
    /// sees a range back only past `RetireQueue::collect`), so the copy's
    /// write-after-read side needs no GPU-side ordering; the transfer queue
    /// takes ownership of such a range without an acquire, which Vulkan
    /// allows for `EXCLUSIVE` buffers at the cost of undefined prior
    /// contents — every byte is overwritten by the copy.
    ///
    /// `graphics_cmd` must be a real, valid (reset-and-begun) command buffer:
    /// the `SameQueueFallback` tier records its copies and barrier into it.
    pub unsafe fn flush_copies(
        &mut self,
        device: &ash::Device,
        lane: &mut TransferLane,
        graphics_cmd: vk::CommandBuffer,
        graphics_family: u32,
        render_done_at: TimelineValue,
    ) {
        if self.pending.is_empty() {
            return;
        }
        let _scope = crate::profile::scope(crate::profile::Meter::Upload);

        // Take first item unconditionally, then more while under budget.
        let mut budget = TRANSFER_BUDGET_BYTES_PER_FRAME;
        let mut take = 0;
        for (i, &(slot, _)) in self.pending.iter().enumerate() {
            let size = self
                .slots
                .get(slot as usize)
                .and_then(|s| s.as_ref())
                .and_then(|r| r.copy.as_ref())
                .map_or(0, |c| c.size);
            if i > 0 && size > budget {
                break;
            }
            budget = budget.saturating_sub(size);
            take = i + 1;
        }
        let remainder = self.pending.split_off(take);
        let batch = std::mem::replace(&mut self.pending, remainder);

        let separate_queue = lane.is_separate_queue();
        let needs_qfot = lane.needs_ownership_transfer();
        let lane_batch = separate_queue.then(|| unsafe { lane.begin(device) });
        let record_cmd = lane_batch.as_ref().map_or(graphics_cmd, |b| b.cmd());

        let mut bytes = 0u64;
        // Regions grouped per (staging block, destination arena) pair; the
        // destination ranges feed the coalesced barriers below.
        let mut copies: Vec<(vk::Buffer, vk::Buffer, Vec<vk::BufferCopy>)> = Vec::new();
        let mut written: Vec<BufferRange> = Vec::with_capacity(batch.len());
        let mut copied_slots: Vec<u32> = Vec::with_capacity(batch.len());
        let mut staging_allocs: Vec<Allocation> = Vec::with_capacity(batch.len());
        let mut staging_leases: Vec<StagingLease> = Vec::new();
        let mut pool_n = 0u64;
        let mut max_pool_delay = 0u64;
        for (slot, queued_at) in batch {
            let Some(res) = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()) else {
                continue;
            };
            let Some(copy) = res.copy.take() else {
                continue;
            };
            let region = vk::BufferCopy::default()
                .src_offset(copy.src_offset)
                .dst_offset(copy.dst_offset)
                .size(copy.size);
            match copies
                .iter_mut()
                .find(|(src, dst, _)| *src == copy.src_buffer && *dst == copy.dst_buffer)
            {
                Some((_, _, regions)) => regions.push(region),
                None => copies.push((copy.src_buffer, copy.dst_buffer, vec![region])),
            }
            written.push(BufferRange {
                buffer: copy.dst_buffer,
                offset: copy.dst_offset,
                size: copy.size,
            });
            bytes += copy.size;
            match copy.source {
                CopySource::Alloc(alloc) => staging_allocs.push(alloc),
                CopySource::Pool(lease) => {
                    staging_leases.push(lease);
                    pool_n += 1;
                    max_pool_delay = max_pool_delay.max(self.frame.saturating_sub(queued_at));
                }
            }
            copied_slots.push(slot);
        }

        if copied_slots.is_empty() {
            // Every batched resident was freed before this flush ran (nothing
            // actually copied), so there is no completion to hand back.
            if let Some(lane_batch) = lane_batch {
                unsafe { lane.discard(device, lane_batch) };
            }
            return;
        }
        crate::profile::gauge(crate::profile::Gauge::UploadBytes, bytes);

        unsafe {
            for (src, dst, regions) in &copies {
                device.cmd_copy_buffer(record_cmd, *src, *dst, regions);
            }
        }
        let written = coalesce_ranges(written);
        let barriers = |role: CopyBarrier| -> Vec<vk::BufferMemoryBarrier2<'static>> {
            written
                .iter()
                .map(|r| copy_barrier(r.buffer, r.offset, r.size, mesh_reads(), role))
                .collect()
        };

        let arrived_at = if let Some(lane_batch) = lane_batch {
            // Cross-queue: the RELEASE half rides the lane batch on the
            // dedicated-family tier; the ACQUIRE half and the semaphore wait
            // are deferred to the next graphics submission (see the hazard
            // analysis above). `SecondQueueSameFamily` needs neither: same
            // family, and the timeline signal/wait pair covers visibility.
            if needs_qfot {
                let release = barriers(CopyBarrier::Release {
                    src_family: lane.family(),
                    dst_family: graphics_family,
                });
                unsafe {
                    device.cmd_pipeline_barrier2(
                        record_cmd,
                        &vk::DependencyInfo::default().buffer_memory_barriers(&release),
                    );
                }
            }
            let value = unsafe { lane.submit(device, lane_batch) };
            let acquires = if needs_qfot {
                barriers(CopyBarrier::Acquire {
                    src_family: lane.family(),
                    dst_family: graphics_family,
                })
            } else {
                Vec::new()
            };
            self.defer_arrival(value, acquires);
            value
        } else {
            // Same queue: the barrier in graphics_cmd orders the copies ahead
            // of every later vertex fetch in submission order.
            let draw = barriers(CopyBarrier::Draw);
            unsafe {
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&draw),
                );
            }
            render_done_at
        };
        // Retire staging on its own timeline (separate queue) or render (fallback).
        let staging_queue = if separate_queue {
            &mut self.transfer_retire
        } else {
            &mut self.retire
        };
        for alloc in staging_allocs {
            staging_queue.push(arrived_at, alloc);
        }
        let pool_stamp = if separate_queue {
            Stamp::Transfer(arrived_at.raw())
        } else {
            Stamp::Render(arrived_at.raw())
        };
        for lease in staging_leases {
            lease.stamp(pool_stamp);
        }
        for slot in copied_slots {
            if let Some(res) = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()) {
                res.arrived_at = Some(arrived_at);
                self.arrived_since_flush.push(slot);
            }
        }
        self.last_pool_arrival_frames = max_pool_delay;
        crate::profile::gauge(crate::profile::Gauge::PoolCopies, pool_n);
        crate::profile::gauge(crate::profile::Gauge::PoolArrivalFrames, max_pool_delay);
    }

    /// Host-only flush of pending pooled copies: marks them arrived and
    /// stamps leases the way [`Self::flush_copies`] does on the separate-
    /// queue lane (`Stamp::Transfer`, graphics wait deferred one frame).
    /// Used to unit-test arrival latency without Vulkan.
    #[cfg(test)]
    pub(crate) fn complete_pooled_copies_for_test(&mut self, arrived_at: TimelineValue) {
        let batch = std::mem::take(&mut self.pending);
        let mut max_delay = 0u64;
        let mut n = 0u64;
        for (slot, queued_at) in batch {
            let Some(res) = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()) else {
                continue;
            };
            let Some(copy) = res.copy.take() else {
                continue;
            };
            match copy.source {
                CopySource::Pool(lease) => {
                    lease.stamp(Stamp::Transfer(arrived_at.raw()));
                    n += 1;
                    max_delay = max_delay.max(self.frame.saturating_sub(queued_at));
                }
                CopySource::Alloc(_) => {}
            }
            res.arrived_at = Some(arrived_at);
            self.arrived_since_flush.push(slot);
        }
        if n > 0 {
            self.defer_arrival(arrived_at, Vec::new());
        }
        self.last_pool_arrival_frames = max_delay;
        crate::profile::gauge(crate::profile::Gauge::PoolCopies, n);
        crate::profile::gauge(crate::profile::Gauge::PoolArrivalFrames, max_delay);
    }

    #[cfg(test)]
    pub(crate) fn last_pool_arrival_frames(&self) -> u64 {
        self.last_pool_arrival_frames
    }

    /// Stashes a submitted lane batch's graphics-side half. Folds into any
    /// batch still owed (two flushes between graphics submissions): the wait
    /// value is the max, the acquire lists concatenate.
    fn defer_arrival(
        &mut self,
        value: TimelineValue,
        acquires: Vec<vk::BufferMemoryBarrier2<'static>>,
    ) {
        match &mut self.deferred {
            Some(prev) => {
                prev.value = prev.value.max(value);
                prev.acquires.extend(acquires);
            }
            None => self.deferred = Some(DeferredArrival { value, acquires }),
        }
    }

    /// Applies the graphics-side half owed by earlier separate-queue batches:
    /// records their ownership-transfer ACQUIRE barriers (dedicated-family
    /// tier; empty otherwise) into `graphics_cmd` and returns the lane value
    /// the submission of `graphics_cmd` must wait on — at
    /// [`MESH_CONSUMER_STAGES`], which is also the acquires' destination
    /// stage, so the release→acquire ordering rides that same wait. Call
    /// before any pass is recorded into `graphics_cmd`. `None` when nothing
    /// is owed.
    pub unsafe fn take_deferred_arrival(
        &mut self,
        device: &ash::Device,
        graphics_cmd: vk::CommandBuffer,
    ) -> Option<TimelineValue> {
        let DeferredArrival { value, acquires } = self.deferred.take()?;
        if !acquires.is_empty() {
            unsafe {
                device.cmd_pipeline_barrier2(
                    graphics_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&acquires),
                );
            }
        }
        Some(value)
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// True when a separate-queue batch is owed a graphics wait + ACQUIRE.
    pub fn has_deferred(&self) -> bool {
        self.deferred.is_some()
    }

    pub fn has_garbage(&self) -> bool {
        !self.retire.is_empty() || !self.transfer_retire.is_empty()
    }

    /// Reclaim render-timeline allocations the GPU has passed.
    pub fn collect(&mut self, current: TimelineValue, recycle: &mut impl FnMut(Allocation)) {
        self.retire.collect(current, |alloc| recycle(alloc));
    }

    /// Reclaim transfer-timeline allocations (tighter bound than render).
    pub fn collect_transfer(
        &mut self,
        current: TimelineValue,
        recycle: &mut impl FnMut(Allocation),
    ) {
        self.transfer_retire
            .collect(current, |alloc| recycle(alloc));
    }

    /// Reclaims every retired allocation, both queues (GPU idle + copies
    /// flushed).
    pub fn collect_all(&mut self, recycle: &mut impl FnMut(Allocation)) {
        self.retire.collect_all(|alloc| recycle(alloc));
        self.transfer_retire.collect_all(|alloc| recycle(alloc));
    }

    /// Recycles every resident + retired allocation (GPU idle). Leaves the
    /// mirror empty.
    pub fn destroy_all(&mut self, recycle: &mut impl FnMut(Allocation)) {
        for slot in self.slots.iter_mut() {
            if let Some(res) = slot.take() {
                recycle(res.arena);
                if let Some(copy) = res.copy {
                    match copy.source {
                        CopySource::Alloc(alloc) => recycle(alloc),
                        CopySource::Pool(lease) => drop(lease),
                    }
                }
            }
        }
        self.retire.collect_all(|alloc| recycle(alloc));
        self.transfer_retire.collect_all(|alloc| recycle(alloc));
        self.pending.clear();
        self.live = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::super::timeline::TimelineValue;
    use ash::vk;
    use ash::vk::Handle;

    #[test]
    fn coalesce_merges_touching_runs_per_buffer_only() {
        use super::{BufferRange, coalesce_ranges};
        let a = vk::Buffer::from_raw(1);
        let b = vk::Buffer::from_raw(2);
        let r = |buffer, offset, size| BufferRange {
            buffer,
            offset,
            size,
        };
        // Out-of-order input; [0,256) + [256,512) + [512,520) touch; [1024,..)
        // is a separate run; buffer b's touching range must not merge into a.
        let merged = coalesce_ranges(vec![
            r(a, 512, 8),
            r(a, 0, 256),
            r(b, 520, 16),
            r(a, 1024, 100),
            r(a, 256, 256),
        ]);
        assert_eq!(merged, vec![r(a, 0, 520), r(a, 1024, 100), r(b, 520, 16)]);
        // Overlap keeps the farthest end; a gap of one byte stays split.
        let merged = coalesce_ranges(vec![r(a, 0, 100), r(a, 50, 100), r(a, 151, 1)]);
        assert_eq!(merged, vec![r(a, 0, 150), r(a, 151, 1)]);
        assert!(coalesce_ranges(Vec::new()).is_empty());
    }

    #[test]
    fn deferred_arrival_folds_until_taken() {
        use super::{DeferredArrival, MeshResidency};
        let mut res = MeshResidency::new();
        assert!(!res.has_deferred());
        res.defer_arrival(TimelineValue::from_raw_for_test(3), vec![]);
        res.defer_arrival(
            TimelineValue::from_raw_for_test(2),
            vec![vk::BufferMemoryBarrier2::default()],
        );
        assert!(res.has_deferred());
        let DeferredArrival { value, acquires } = res.deferred.take().expect("owed");
        assert_eq!(value, TimelineValue::from_raw_for_test(3));
        assert_eq!(acquires.len(), 1);
        assert!(!res.has_deferred());
    }

    #[test]
    fn pooled_mesh_arrives_one_frame_later_with_a_fake_timeline() {
        use super::super::alloc::Allocation;
        use super::super::mesh_resident::{CopySource, GpuResident, PendingCopy};
        use super::super::mesh_staging::MeshStagingPool;
        use super::MeshResidency;
        use std::num::NonZeroU32;

        let pool = MeshStagingPool::new_host(8);
        let staging = pool.stager().acquire(8).expect("host pool");
        let lease = staging.into_lease();

        let mut res = MeshResidency::new();
        res.apply_upload(
            0,
            NonZeroU32::MIN,
            GpuResident {
                buffer: vk::Buffer::null(),
                arena: Allocation::dummy(),
                copy: Some(PendingCopy {
                    src_buffer: vk::Buffer::null(),
                    src_offset: 0,
                    dst_buffer: vk::Buffer::null(),
                    dst_offset: 0,
                    size: 8,
                    source: CopySource::Pool(lease),
                }),
                arrived_at: None,
            },
        );
        assert!(!res.is_arrived(0), "pending copy is not arrived at apply");
        assert!(
            !res.has_deferred(),
            "the lane wait is not armed until flush"
        );

        // Drain then draw: note_frame then flush. Delay == 1; the graphics
        // wait is deferred (no same-frame wait).
        res.note_frame();
        let done = TimelineValue::from_raw_for_test(1);
        res.complete_pooled_copies_for_test(done);
        assert!(res.is_arrived(0));
        assert_eq!(res.last_pool_arrival_frames(), 1);
        assert_eq!(res.take_arrived(), vec![0]);
        assert!(
            res.has_deferred(),
            "lane wait is deferred to the next graphics submit"
        );

        // Stamp::Transfer(1): reclaim needs the transfer timeline.
        pool.reclaim(TimelineValue::START, None);
        assert!(
            pool.stager().acquire(8).is_none(),
            "region waits for the fake transfer timeline"
        );
        pool.reclaim(done, None);
        assert!(
            pool.stager().acquire(8).is_none(),
            "a render counter does not free a Transfer stamp"
        );
        pool.reclaim(TimelineValue::START, Some(done));
        assert!(
            pool.stager().acquire(8).is_some(),
            "reclaimed one deferred frame later, no GPU wait"
        );
    }
}
