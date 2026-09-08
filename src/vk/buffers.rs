/// GPU mesh registry and per-frame immediate-geometry buffers.
///
/// Meshes live in device-local memory suballocated from `GpuAllocator`
/// blocks: one allocation per mesh holding `[vertices][pad][indices]`. On
/// unified-memory devices uploads are direct memcpys; otherwise they go
/// through a staging allocation and a `cmd_copy_buffer` recorded at the next
/// frame's start (so a mesh uploaded mid-update is drawable the same frame).
/// Frees are deferred until the GPU provably finished the last frame that
/// could have referenced the mesh.
use ash::{khr, vk};
use glam::Vec3;

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};

use super::alloc::{Allocation, GpuAllocator, find_memory_type, try_find_memory_type};
use super::cull::ArenaDirectory;
use super::timeline::TimelineValue;
use super::transfer::TransferLane;
use crate::mesh::{Detail, MeshData, MeshHandle, Pass};

/// Mesh-copy staging budget per frame; amortizes bursty uploads.
const TRANSFER_BUDGET_BYTES_PER_FRAME: u64 = 8 * 1024 * 1024;

/// GPU storage-buffer offset alignment; the 256 half of `MESH_ALIGN`.
const GPU_OFFSET_ALIGN: u64 = 256;

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Suballocation alignment: must divide both GPU offset alignment (256) and vertex stride.
const MESH_ALIGN: u64 = {
    let stride = std::mem::size_of::<crate::mesh::MeshVertex>() as u64;
    stride / gcd(stride, GPU_OFFSET_ALIGN) * GPU_OFFSET_ALIGN
};
const _: () = {
    assert!(MESH_ALIGN % std::mem::size_of::<crate::mesh::MeshVertex>() as u64 == 0);
    assert!(MESH_ALIGN % GPU_OFFSET_ALIGN == 0);
};
pub const FRAMES_IN_FLIGHT: u64 = 2;

/// A deferred-reclaim queue: items stamped with their last possible GPU use.
/// [`collect`](Self::collect) only reclaims items the GPU has provably passed.
/// Allocator-agnostic: yields items for the caller to free.
pub struct RetireQueue<T> {
    entries: std::collections::VecDeque<(TimelineValue, T)>,
}

impl<T> RetireQueue<T> {
    pub fn new() -> Self {
        Self {
            entries: std::collections::VecDeque::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Retires `item`, stamped with the timeline value that last could
    /// reference it.
    pub fn push(&mut self, done_at: TimelineValue, item: T) {
        self.entries.push_back((done_at, item));
    }

    /// Drains entries whose GPU use has completed, calling `f` on each.
    pub fn collect(&mut self, current: TimelineValue, mut f: impl FnMut(T)) {
        while let Some((stamp, _)) = self.entries.front() {
            if *stamp > current {
                break;
            }
            let (_, item) = self.entries.pop_front().unwrap();
            f(item);
        }
    }

    /// Drains everything, calling `f` on each item.
    pub fn collect_all(&mut self, mut f: impl FnMut(T)) {
        for (_, item) in self.entries.drain(..) {
            f(item);
        }
    }
}

/// Vertex stride shared by the mesh pipelines (must divide [`MESH_ALIGN`]).
const VERTEX_STRIDE: u64 = std::mem::size_of::<crate::mesh::MeshVertex>() as u64;

/// Bridge letting [`HandleAllocator`] mint handles generically.
pub(crate) trait GpuHandle: Copy {
    fn from_parts(slot: u32, generation: NonZeroU32) -> Self;
    fn slot(self) -> u32;
    fn generation(self) -> NonZeroU32;
}

impl GpuHandle for MeshHandle {
    fn from_parts(slot: u32, generation: NonZeroU32) -> Self {
        MeshHandle { slot, generation }
    }
    fn slot(self) -> u32 {
        self.slot
    }
    fn generation(self) -> NonZeroU32 {
        self.generation
    }
}

/// Bumps a 1-based generation, skipping the reserved 0 niche on wrap so a
/// recycled slot never reuses a live handle's generation (and never hits 0).
fn bump_generation(g: NonZeroU32) -> NonZeroU32 {
    NonZeroU32::new(g.get().wrapping_add(1)).unwrap_or(NonZeroU32::MIN)
}

/// The single main-thread authority for handle identity + culling metadata.
/// Mints generational handles, recycles freed slots, and answers record-time
/// metadata lookups. Holds NO Vulkan resources — those live render-side in the
/// residency mirror ([`MeshResidency`]).
pub(crate) struct HandleAllocator<H: GpuHandle, M> {
    meta: Vec<Option<M>>,
    /// 1-based; bumped (never to 0) when a slot is freed.
    generations: Vec<NonZeroU32>,
    free: Vec<u32>,
    live: usize,
    _marker: std::marker::PhantomData<fn() -> H>,
}

impl<H: GpuHandle, M: Copy> HandleAllocator<H, M> {
    pub fn new() -> Self {
        Self {
            meta: Vec::new(),
            generations: Vec::new(),
            free: Vec::new(),
            live: 0,
            _marker: std::marker::PhantomData,
        }
    }

    #[cfg(test)]
    pub fn live_count(&self) -> usize {
        self.live
    }

    /// Assigns `meta` to a fresh or recycled slot and mints its handle.
    pub fn alloc_slot(&mut self, meta: M) -> H {
        self.live += 1;
        match self.free.pop() {
            Some(i) => {
                self.meta[i as usize] = Some(meta);
                H::from_parts(i, self.generations[i as usize])
            }
            None => {
                let i = self.meta.len() as u32;
                self.meta.push(Some(meta));
                self.generations.push(NonZeroU32::MIN);
                H::from_parts(i, NonZeroU32::MIN)
            }
        }
    }

    /// Frees `h`'s slot (gen-checked): bumps the generation and recycles the
    /// slot. Returns false for a stale or double free.
    pub fn free_slot(&mut self, h: H) -> bool {
        let slot = h.slot() as usize;
        let Some(generation) = self.generations.get_mut(slot) else {
            return false;
        };
        if *generation != h.generation() {
            return false;
        }
        if self.meta[slot].take().is_some() {
            *generation = bump_generation(*generation);
            self.free.push(h.slot());
            self.live -= 1;
            true
        } else {
            false
        }
    }

    /// Mutable metadata with generation check.
    pub fn meta_mut(&mut self, h: H) -> Option<&mut M> {
        if *self.generations.get(h.slot() as usize)? != h.generation() {
            return None;
        }
        self.meta.get_mut(h.slot() as usize)?.as_mut()
    }
}

pub(crate) type MeshHandles = HandleAllocator<MeshHandle, MeshMeta>;

/// Main-owned, `Send + Copy` culling/draw metadata for one mesh — NO Vulkan
/// handles. The record path reads this to frustum-cull and embeds the draw
/// params into the snapshot.
#[derive(Clone, Copy)]
pub(crate) struct MeshMeta {
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    /// Seven local (0-based) index boundaries into the shared quad IBO:
    /// `bounds[dir]..bounds[dir+1]` is direction `dir`'s range (cumulative
    /// `6*quads` in Normal order) and `bounds[0]..bounds[6]` the whole mesh.
    /// `bounds[0]` is always 0; always increasing (see [`build_mesh_resident`]).
    pub bounds: [u32; 7],
    /// First vertex (in vertices from block start); the command's `vertex_offset`.
    pub vertex_offset: i32,
    pub pass: Pass,
    /// GPU record placement on the main side.
    pub placement: PlacementState,
    /// GPU dyn lane cache; patched only on change.
    pub dyn_lane: DrawDyn,
}

/// Mesh placement sync strategy: pinned (immutable) or tracked (recovered and patched).
#[derive(Clone, Copy)]
pub(crate) enum PlacementState {
    /// Immutable at upload (terrain).
    Pinned,
    /// Recovered and patched on drift (movers).
    Tracked(Option<crate::mesh::MeshPlacement>),
}

impl MeshRecord {
    /// Decode pass bits from detail_pass.
    pub(crate) fn pass(&self) -> Pass {
        match (self.detail_pass >> 4) & 3 {
            0 => Pass::Opaque,
            1 => Pass::Cutout,
            _ => Pass::Blend,
        }
    }

    /// Decode per-draw scale from biased detail field.
    pub(crate) fn detail_scale(&self) -> f32 {
        Detail::from_gpu_bits((self.detail_pass & 0xF) as u8).scale()
    }

    /// Compose a GPU record from mesh metadata and placement.
    pub(crate) fn compose(meta: &MeshMeta, p: crate::mesh::MeshPlacement) -> Self {
        Self {
            block: p.block.to_array(),
            // Detail in bits 0..4, pass in bits 4..6.
            detail_pass: u32::from(p.detail.to_gpu_bits()) | ((meta.pass as u32) << 4),
            local_off: p.local_off.to_array(),
            _pad: 0,
            aabb_min: meta.aabb_min.to_array(),
            index_count: meta.bounds[6],
            aabb_max: meta.aabb_max.to_array(),
            vertex_offset: meta.vertex_offset,
        }
    }
}

/// A staged host→device copy owned by a not-yet-flushed [`GpuResident`].
/// The role of one staged-copy buffer barrier: the same-queue copy→draw
/// barrier, or the release/acquire halves of a queue-family ownership
/// transfer. Six call sites used to restate the stage/access pairings
/// field-by-field — the exact part a reviewer must get right — so the
/// pairings live here once and a site states only its buffer range, its
/// draw-side reads, and its role.
#[derive(Clone, Copy)]
enum CopyBarrier {
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
fn copy_barrier(
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

struct PendingCopy {
    staging: Allocation,
    dst_buffer: vk::Buffer,
    dst_offset: u64,
    size: u64,
}

/// Render-owned GPU residency for one mesh: the device buffer plus its
/// deferred staging copy. `Send` because [`Allocation`] is now `Send`.
pub(crate) struct GpuResident {
    alloc: Allocation,
    copy: Option<PendingCopy>,
    /// Timeline value ordering copy before reads; `None` while budget-deferred.
    arrived_at: Option<TimelineValue>,
}

impl GpuResident {
    /// Get the device buffer.
    pub fn buffer(&self) -> vk::Buffer {
        self.alloc.buffer
    }
}

/// Allocates a device buffer for `data`, writes/stages its bytes, and returns
/// the main-owned [`MeshMeta`] plus render-owned [`GpuResident`]. Main-thread
/// only: touches the allocator + persistent mapping, never the timeline.
/// `None` on empty data or OOM (the partial device alloc is freed on failure).
/// Stores vertices only; indices are the shared per-quad pattern in [`QuadIbo`].
pub(crate) unsafe fn build_mesh_resident(
    device: &ash::Device,
    allocator: &mut GpuAllocator,
    data: &MeshData,
) -> Option<(MeshMeta, GpuResident)> {
    let total_indices: usize = data.buckets.iter().map(Vec::len).sum();
    if total_indices == 0 || data.vertices.is_empty() {
        return None;
    }

    // Permuted pool holds the same vertices reordered by bucket then quad; every
    // vertex belongs to exactly one quad, so its length equals `data.vertices`.
    let vertex_bytes_len = data.vertices.len() * VERTEX_STRIDE as usize;
    let total = vertex_bytes_len as u64;

    let alloc = unsafe { allocator.alloc_device(device, total, MESH_ALIGN) }
        .map_err(|err| log::error!("mesh allocation failed: {err:?}"))
        .ok()?;

    let write_into = |dst: *mut u8| unsafe {
        let mut cursor = 0usize;
        for bucket in &data.buckets {
            debug_assert_eq!(bucket.len() % 6, 0, "each quad contributes 6 indices");
            for quad in bucket.chunks_exact(6) {
                let b = quad[0];
                debug_assert_eq!(
                    *quad,
                    [b, b + 1, b + 2, b, b + 2, b + 3],
                    "non-pattern quad indices break the shared-IBO permutation"
                );
                let verts: &[u8] = bytemuck::cast_slice(&data.vertices[b as usize..b as usize + 4]);
                std::ptr::copy_nonoverlapping(verts.as_ptr(), dst.add(cursor), verts.len());
                cursor += verts.len();
            }
        }
        debug_assert_eq!(
            cursor, vertex_bytes_len,
            "permutation must cover every vertex"
        );
    };

    let copy = if let Some(mapped) = alloc.mapped {
        // Unified memory: write straight into the device-local block.
        write_into(mapped.as_ptr());
        None
    } else {
        let staging = match unsafe { allocator.alloc_staging(device, total, 4) } {
            Ok(staging) => staging,
            Err(err) => {
                log::error!("staging allocation failed: {err:?}");
                unsafe { allocator.free(alloc) };
                return None;
            }
        };
        let mapped = staging
            .mapped
            .expect("staging memory is always host-visible");
        write_into(mapped.as_ptr());
        Some(PendingCopy {
            dst_buffer: alloc.buffer,
            dst_offset: alloc.offset,
            size: total,
            staging,
        })
    };

    let mut aabb_min = Vec3::splat(f32::INFINITY);
    let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);
    for v in &data.vertices {
        let p = Vec3::from_array(v.local_pos());
        aabb_min = aabb_min.min(p);
        aabb_max = aabb_max.max(p);
    }

    const _: () =
        assert!(MESH_ALIGN.is_multiple_of(VERTEX_STRIDE) && MESH_ALIGN.is_multiple_of(256));
    debug_assert_eq!(alloc.offset % VERTEX_STRIDE, 0);
    let vertex_offset = (alloc.offset / VERTEX_STRIDE) as i32;

    // Local, 0-based index boundaries into the shared quad IBO: `bounds[dir]` is
    // the cumulative `6*quads` before face `dir` (Normal order). The IBO's index
    // value at position `6j` is `4j`, and quad `j` sits at vertices `4j..4j+4`, so
    // adding the unchanged `vertex_offset` base reproduces the old vertex fetches.
    let mut bounds = [0u32; 7];
    for dir in 0..6 {
        bounds[dir + 1] = bounds[dir] + data.buckets[dir].len() as u32;
    }
    debug_assert_eq!(bounds[6], total_indices as u32);

    let meta = MeshMeta {
        aabb_min,
        aabb_max,
        bounds,
        vertex_offset,
        pass: data.pass,
        placement: PlacementState::Tracked(None),
        dyn_lane: DrawDyn::resting(),
    };
    // No staged copy (unified memory: already written above) is immediately
    // drawable; a staged copy gates drawability until `flush_copies` submits
    // it (see [`GpuResident::arrived_at`]).
    let arrived_at = copy.is_none().then_some(TimelineValue::START);
    Some((
        meta,
        GpuResident {
            alloc,
            copy,
            arrived_at,
        },
    ))
}

/// Render-side residency mirror for meshes: keyed by the main-assigned slot,
/// with a generation mirror kept in sync from the ordered command stream,
/// guaranteeing correct handle-aliasing without cross-thread reads. Holds no
/// free-list or identity — that is [`HandleAllocator`]'s job.
pub(crate) struct MeshResidency {
    slots: Vec<Option<GpuResident>>,
    generations: Vec<NonZeroU32>,
    pending: Vec<u32>,
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
        }
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
            self.pending.push(slot);
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
            self.retire.push(done_at, res.alloc);
            if let Some(copy) = res.copy {
                self.retire.push(done_at, copy.staging);
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
        for (i, &slot) in self.pending.iter().enumerate() {
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
        let mut staging: Vec<Allocation> = Vec::with_capacity(batch.len());
        for slot in batch {
            let Some(res) = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()) else {
                continue;
            };
            let Some(copy) = res.copy.take() else {
                continue;
            };
            let region = vk::BufferCopy::default()
                .src_offset(copy.staging.offset)
                .dst_offset(copy.dst_offset)
                .size(copy.size);
            match copies
                .iter_mut()
                .find(|(src, dst, _)| *src == copy.staging.buffer && *dst == copy.dst_buffer)
            {
                Some((_, _, regions)) => regions.push(region),
                None => copies.push((copy.staging.buffer, copy.dst_buffer, vec![region])),
            }
            written.push(BufferRange {
                buffer: copy.dst_buffer,
                offset: copy.dst_offset,
                size: copy.size,
            });
            bytes += copy.size;
            staging.push(copy.staging);
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
        for alloc in staging {
            staging_queue.push(arrived_at, alloc);
        }
        for slot in copied_slots {
            if let Some(res) = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()) {
                res.arrived_at = Some(arrived_at);
                self.arrived_since_flush.push(slot);
            }
        }
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
                recycle(res.alloc);
                if let Some(copy) = res.copy {
                    recycle(copy.staging);
                }
            }
        }
        self.retire.collect_all(|alloc| recycle(alloc));
        self.transfer_retire.collect_all(|alloc| recycle(alloc));
        self.pending.clear();
        self.live = 0;
    }
}

/// Smallest immediate-buffer capacity (also the floor the decay stops at).
const IMM_MIN_CAPACITY: u64 = 64 * 1024;
/// Decay window for capacity shrinking.
const IMM_SHRINK_WINDOW: u32 = 600;

/// Ceiling on [`HostBuffer`] bytes placed in a *small* BAR heap. Discrete GPUs
/// without ReBAR expose only a ~256 MiB `DEVICE_LOCAL | HOST_VISIBLE` window;
/// host buffers take a bounded slice of it and the rest stay in system memory.
/// ReBAR / unified heaps (>= [`SMALL_BAR_HEAP`]) are used without this cap.
const HOST_BAR_CAP: u64 = 64 << 20;
/// Same threshold [`super::alloc`] uses to tell a real unified/ReBAR heap from
/// a discrete GPU's small BAR window.
const SMALL_BAR_HEAP: u64 = 1 << 30;
/// Bytes currently charged against [`HOST_BAR_CAP`] (small-BAR devices only).
static HOST_BAR_BYTES: AtomicU64 = AtomicU64::new(0);

/// Host-visible, host-coherent — the property set every [`HostBuffer`] write
/// relies on (persistent mapping, no explicit flush).
const HOST_COHERENT: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);

fn heap_size(memory_props: &vk::PhysicalDeviceMemoryProperties, type_index: u32) -> u64 {
    let heap = memory_props.memory_types[type_index as usize].heap_index as usize;
    memory_props.memory_heaps[heap].size
}

/// Picks the memory type for a [`HostBuffer`] of `size` bytes: the first BAR
/// type (`DEVICE_LOCAL` on top of [`HOST_COHERENT`]) when that heap is large
/// (ReBAR / unified) or when `bar_used + size` stays under [`HOST_BAR_CAP`]
/// on a small BAR; else the first plain host-coherent type.
///
/// The `bool` is whether the pick *is* the BAR type (allocation failure then
/// falls back to system memory). Charging the cap is a separate decision at
/// allocate time: only small BAR heaps consume [`HOST_BAR_BYTES`].
fn host_buffer_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    size: u64,
    bar_used: u64,
) -> Option<(u32, bool)> {
    if let Some(i) = try_find_memory_type(
        memory_props,
        type_filter,
        HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_LOCAL,
    ) {
        let small = heap_size(memory_props, i) < SMALL_BAR_HEAP;
        if !small || bar_used.saturating_add(size) <= HOST_BAR_CAP {
            return Some((i, true));
        }
    }
    try_find_memory_type(memory_props, type_filter, HOST_COHERENT).map(|i| (i, false))
}

fn bar_charge(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_index: u32,
    size: u64,
) -> u64 {
    if heap_size(memory_props, type_index) < SMALL_BAR_HEAP {
        size
    } else {
        0
    }
}

/// A growable host-visible buffer written each frame, one per frame-in-flight.
/// Used for immediate geometry, offsets, and indirect commands.
pub struct HostBuffer {
    /// Null until first write; use [`Self::bound`] to obtain safely.
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    capacity: u64,
    /// Bytes this buffer holds against [`HOST_BAR_CAP`] (0 = system memory).
    bar_bytes: u64,
    usage: vk::BufferUsageFlags,
    /// Peak need in decay window.
    window_peak: u64,
    /// Frame count in decay window.
    window_frames: u32,
}

impl HostBuffer {
    /// Get the buffer handle, or `None` if unallocated.
    pub fn bound(&self) -> Option<vk::Buffer> {
        (self.buffer != vk::Buffer::null()).then_some(self.buffer)
    }

    pub fn new(usage: vk::BufferUsageFlags) -> Self {
        Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(),
            capacity: 0,
            bar_bytes: 0,
            usage,
            window_peak: 0,
            window_frames: 0,
        }
    }

    /// Maintain capacity and shrink if needed. Call after fence is waited.
    pub unsafe fn maintain(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        needed: u64,
    ) -> bool {
        let mut changed = false;
        if needed > self.window_peak {
            self.window_peak = needed;
        }
        self.window_frames += 1;
        if self.window_frames >= IMM_SHRINK_WINDOW {
            let peak = self.window_peak;
            self.window_frames = 0;
            self.window_peak = 0;
            if let Some(target) = shrink_capacity(self.capacity, peak) {
                unsafe {
                    self.destroy(device);
                    changed = true;
                    if target > 0 {
                        self.ensure_capacity(instance, device, physical, target);
                    }
                }
            }
        }
        if needed > 0 {
            changed |= unsafe { self.ensure_capacity(instance, device, physical, needed) };
        }
        changed
    }

    unsafe fn ensure_capacity(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        needed: u64,
    ) -> bool {
        if needed <= self.capacity {
            return false;
        }
        let new_capacity = needed.next_power_of_two().max(IMM_MIN_CAPACITY);
        unsafe {
            self.destroy(device);

            let memory_props = instance.get_physical_device_memory_properties(physical);
            let info = vk::BufferCreateInfo::default()
                .size(new_capacity)
                .usage(self.usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = device
                .create_buffer(&info, None)
                .expect("create host buffer");
            let req = device.get_buffer_memory_requirements(buffer);
            // BAR first (charged against the cap), then system memory. A BAR
            // allocation that the driver refuses anyway (the window is shared
            // with everything else) falls back the same way.
            let bar_used = HOST_BAR_BYTES.load(Ordering::Relaxed);
            let (type_index, is_bar) =
                host_buffer_memory_type(&memory_props, req.memory_type_bits, req.size, bar_used)
                    .expect("no HOST_VISIBLE | HOST_COHERENT memory type for a host buffer");
            let allocate = |type_index: u32| {
                device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(type_index),
                    None,
                )
            };
            let (memory, bar_bytes) = match allocate(type_index) {
                Ok(memory) => (
                    memory,
                    if is_bar {
                        bar_charge(&memory_props, type_index, req.size)
                    } else {
                        0
                    },
                ),
                Err(err) if is_bar => {
                    log::debug!(
                        "BAR host buffer allocation refused ({err:?}); using system memory"
                    );
                    let fallback =
                        try_find_memory_type(&memory_props, req.memory_type_bits, HOST_COHERENT)
                            .expect(
                                "no HOST_VISIBLE | HOST_COHERENT memory type for a host buffer",
                            );
                    (allocate(fallback).expect("allocate host buffer memory"), 0)
                }
                Err(err) => panic!("allocate host buffer memory: {err:?}"),
            };
            HOST_BAR_BYTES.fetch_add(bar_bytes, Ordering::Relaxed);
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind host buffer memory");
            let mapped = device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("Failed to map immediate buffer") as *mut u8;

            self.buffer = buffer;
            self.memory = memory;
            self.mapped = mapped;
            self.capacity = new_capacity;
            self.bar_bytes = bar_bytes;
        }
        true
    }

    pub unsafe fn write(&mut self, offset: u64, bytes: &[u8]) {
        assert!(
            offset
                .checked_add(bytes.len() as u64)
                .is_some_and(|end| end <= self.capacity)
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.mapped.add(offset as usize),
                bytes.len(),
            );
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.buffer != vk::Buffer::null() {
            unsafe {
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.memory, None);
            }
            HOST_BAR_BYTES.fetch_sub(self.bar_bytes, Ordering::Relaxed);
            self.buffer = vk::Buffer::null();
            self.memory = vk::DeviceMemory::null();
            self.mapped = std::ptr::null_mut();
            self.capacity = 0;
            self.bar_bytes = 0;
        }
    }
}

/// Engine-wide shared quad index buffer: the invariant per-quad pattern
/// `[4q, 4q+1, 4q+2, 4q, 4q+2, 4q+3]` stored once and grown on demand.
pub(crate) struct QuadIbo {
    /// `VK_NULL_HANDLE` until the first grow; read only through [`Self::bound`].
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Quads the current buffer can index; 0 until first allocation.
    capacity: u32,
    /// High-water quad count requested across all uploads (monotonic).
    required: u32,
    /// Superseded live buffers (render-Rev — read only by draws on the
    /// render timeline) and same-queue-fallback staging (render-Rev covers
    /// it too, since that copy rides the graphics cmd buffer).
    retire: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
    /// Staging for a pattern copy submitted on a SEPARATE transfer queue:
    /// stamped with the lane's OWN timeline value — see [`MeshResidency`]'s
    /// field of the same name for the full argument.
    transfer_retire: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
}

/// Initial capacity in quads.
const QUAD_IBO_MIN_QUADS: u32 = 1 << 16;
/// Six indices per quad — the fixed `quad()` pattern width.
const INDICES_PER_QUAD: u32 = 6;

impl QuadIbo {
    pub fn new() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            capacity: 0,
            required: 0,
            retire: RetireQueue::new(),
            transfer_retire: RetireQueue::new(),
        }
    }

    /// The device buffer, or `None` before the first grow. A recorded draw run
    /// implies a mesh was uploaded (which raised `required`), so [`Self::ensure`]
    /// has since allocated it — callers `.expect` it there.
    pub fn bound(&self) -> Option<vk::Buffer> {
        (self.buffer != vk::Buffer::null()).then_some(self.buffer)
    }

    /// Raises the required capacity to cover a newly-uploaded mesh's quad count.
    pub fn require(&mut self, quads: u32) {
        self.required = self.required.max(quads);
    }

    /// Grows the buffer to cover `required` quads if needed, staging the
    /// pattern via the transfer lane (mirrors `MeshResidency::flush_copies`'s
    /// tier/barrier handling) and retiring the old buffer past `done_at`.
    /// No-op (`None`) when the current buffer already suffices. Returns
    /// `Some(value)` when the pattern copy submitted on a separate queue:
    /// `graphics_cmd`'s submission must wait on the lane's semaphore for
    /// `value` before any draw indexes this buffer.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ensure(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        lane: &mut TransferLane,
        graphics_cmd: vk::CommandBuffer,
        graphics_family: u32,
        done_at: TimelineValue,
    ) -> Option<TimelineValue> {
        if self.required <= self.capacity {
            return None;
        }
        let new_capacity = self.required.next_power_of_two().max(QUAD_IBO_MIN_QUADS);
        let index_count = new_capacity as u64 * INDICES_PER_QUAD as u64;
        let size = index_count * std::mem::size_of::<u32>() as u64;
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        // Device-local destination for the pattern.
        let (buffer, memory) = unsafe {
            create_raw_buffer(
                device,
                &memory_props,
                size,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
        };

        // Host-visible staging: fill the pattern, copy, then retire it — a static
        // one-shot upload, so it need not linger like the per-slot HostBuffers do.
        let (staging, staging_mem) = unsafe {
            create_raw_buffer(
                device,
                &memory_props,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        };

        let separate_queue = lane.is_separate_queue();
        let needs_qfot = lane.needs_ownership_transfer();
        let lane_batch = separate_queue.then(|| unsafe { lane.begin(device) });
        let record_cmd = lane_batch.as_ref().map_or(graphics_cmd, |b| b.cmd());

        unsafe {
            let ptr = device
                .map_memory(staging_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("map quad IBO staging") as *mut u32;
            for q in 0..new_capacity {
                let b = q * 4;
                let base = ptr.add(q as usize * INDICES_PER_QUAD as usize);
                for (i, &v) in [b, b + 1, b + 2, b, b + 2, b + 3].iter().enumerate() {
                    base.add(i).write(v);
                }
            }
            device.unmap_memory(staging_mem);

            let region = vk::BufferCopy::default().size(size);
            device.cmd_copy_buffer(record_cmd, staging, buffer, &[region]);

            if !separate_queue {
                let barrier = [copy_barrier(
                    buffer,
                    0,
                    size,
                    vk::AccessFlags2::INDEX_READ,
                    CopyBarrier::Draw,
                )];
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&barrier),
                );
            } else if needs_qfot {
                let release = [copy_barrier(
                    buffer,
                    0,
                    size,
                    vk::AccessFlags2::INDEX_READ,
                    CopyBarrier::Release {
                        src_family: lane.family(),
                        dst_family: graphics_family,
                    },
                )];
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&release),
                );
            }
            // else: SecondQueueSameFamily — no barrier needed.
        }

        let arrived_at = if let Some(lane_batch) = lane_batch {
            let value = unsafe { lane.submit(device, lane_batch) };
            if needs_qfot {
                let acquire = [copy_barrier(
                    buffer,
                    0,
                    size,
                    vk::AccessFlags2::INDEX_READ,
                    CopyBarrier::Acquire {
                        src_family: lane.family(),
                        dst_family: graphics_family,
                    },
                )];
                unsafe {
                    device.cmd_pipeline_barrier2(
                        graphics_cmd,
                        &vk::DependencyInfo::default().buffer_memory_barriers(&acquire),
                    );
                }
            }
            Some(value)
        } else {
            None
        };

        // Retire old buffer on render timeline, staging on its own (or render).
        if self.capacity > 0 {
            self.retire.push(done_at, (self.buffer, self.memory));
        }
        match arrived_at {
            Some(value) => self.transfer_retire.push(value, (staging, staging_mem)),
            None => self.retire.push(done_at, (staging, staging_mem)),
        }
        self.buffer = buffer;
        self.memory = memory;
        self.capacity = new_capacity;

        arrived_at
    }

    /// True while a superseded buffer awaits its timeline value.
    pub fn has_garbage(&self) -> bool {
        !self.retire.is_empty() || !self.transfer_retire.is_empty()
    }

    /// Destroy render-timeline buffers the GPU has passed.
    pub unsafe fn collect(&mut self, device: &ash::Device, current: TimelineValue) {
        self.retire.collect(current, |(buffer, memory)| unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        });
    }

    /// Destroy transfer-timeline buffers the GPU has passed.
    pub unsafe fn collect_transfer(&mut self, device: &ash::Device, current: TimelineValue) {
        self.transfer_retire
            .collect(current, |(buffer, memory)| unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
    }

    /// Destroy all buffers.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.retire.collect_all(|(buffer, memory)| {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
            self.transfer_retire.collect_all(|(buffer, memory)| {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
            if self.buffer != vk::Buffer::null() {
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.memory, None);
                self.buffer = vk::Buffer::null();
            }
        }
    }
}

/// Create standalone buffer + memory (for one-off engine buffers).
unsafe fn create_raw_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    unsafe {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.create_buffer(&info, None).expect("create buffer");
        let req = device.get_buffer_memory_requirements(buffer);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(find_memory_type(
                memory_props,
                req.memory_type_bits,
                properties,
            ));
        let memory = device
            .allocate_memory(&alloc_info, None)
            .expect("allocate buffer memory");
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind buffer memory");
        (buffer, memory)
    }
}

/// Persistent per-mesh record, indexed by slot, mirrored in shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshRecord {
    pub block: [i32; 3],
    pub detail_pass: u32,
    pub local_off: [f32; 3],
    pub _pad: u32,
    pub aabb_min: [f32; 3],
    pub index_count: u32,
    pub aabb_max: [f32; 3],
    pub vertex_offset: i32,
}

// Stride must match the vertex shaders exactly; layout drift corrupts every draw.
const _: () = assert!(std::mem::size_of::<MeshRecord>() == 64);

/// Per-mesh dynamic style, patched on change.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawDyn {
    pub mode: u32,
    pub flat_rgba: u32,
}

const _: () = assert!(std::mem::size_of::<DrawDyn>() == 8);

impl DrawDyn {
    /// Rest state: plain textured.
    pub fn resting() -> Self {
        Self {
            mode: 0,
            flat_rgba: 0,
        }
    }
}

/// Persistent record store with deferred writes per frame.
pub(crate) struct RecordTable {
    records: Vec<MeshRecord>,
    dyns: Vec<DrawDyn>,
    gpu: [RecordCopy; FRAMES_IN_FLIGHT as usize],
    /// Incremented when drawable meshes change (signals shadow cache).
    occluder_rev: u64,
}

struct RecordCopy {
    records: HostBuffer,
    dyns: HostBuffer,
    arenas: HostBuffer,
    /// Slots to flush on next upload.
    dirty: Vec<u32>,
}

/// Flushed buffers for descriptor pushes.
#[derive(Clone, Copy)]
pub(crate) struct RecordBuffers {
    pub records: vk::Buffer,
    pub dyns: vk::Buffer,
    pub arenas: vk::Buffer,
    /// Table length in slots.
    pub slots: u32,
}

impl RecordTable {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            dyns: Vec::new(),
            gpu: std::array::from_fn(|_| RecordCopy {
                records: HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER),
                dyns: HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER),
                arenas: HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER),
                dirty: Vec::new(),
            }),
            occluder_rev: 0,
        }
    }

    /// The current occluder-set revision, read by the shadow cache each frame.
    pub(crate) fn occluder_rev(&self) -> u64 {
        self.occluder_rev
    }

    fn mark(&mut self, slot: u32) {
        for copy in &mut self.gpu {
            copy.dirty.push(slot);
        }
    }

    /// Install a freshly-uploaded mesh's record.
    pub fn install(&mut self, slot: u32, record: MeshRecord) {
        let n = slot as usize + 1;
        if self.records.len() < n {
            self.records.resize(n, bytemuck::Zeroable::zeroed());
            self.dyns.resize(n, DrawDyn::resting());
        }
        self.records[slot as usize] = record;
        self.dyns[slot as usize] = DrawDyn::resting();
        self.occluder_rev += 1;
        self.mark(slot);
    }

    /// Mark freed slot dirty to read its dead arena word.
    pub fn clear_arena(&mut self, slot: u32) {
        if (slot as usize) < self.records.len() {
            self.occluder_rev += 1;
            self.mark(slot);
        }
    }

    /// Mark arrived slots dirty to re-read their arena words.
    pub fn mark_arrived(&mut self, slots: &[u32]) {
        if !slots.is_empty() {
            self.occluder_rev += 1;
        }
        for &slot in slots {
            self.mark(slot);
        }
    }

    /// Reads a slot's record. `None` for a slot no mesh ever occupied.
    pub fn record(&self, slot: u32) -> Option<&MeshRecord> {
        self.records.get(slot as usize)
    }

    /// Replaces a mover's record (recomposed main-side); the dyn lane is
    /// untouched so a mover keeps its style.
    pub fn set_record(&mut self, slot: u32, record: MeshRecord) {
        let Some(rec) = self.records.get_mut(slot as usize) else {
            return;
        };
        *rec = record;
        self.occluder_rev += 1; // recomposed geometry moves the occluder
        self.mark(slot);
    }

    /// Patch the dynamic style.
    pub fn set_dyn(&mut self, slot: u32, dyn_lane: DrawDyn) {
        let Some(d) = self.dyns.get_mut(slot as usize) else {
            return;
        };
        *d = dyn_lane;
        self.mark(slot);
    }

    /// Flush slot's pending writes. Must run after fence is waited.
    pub unsafe fn flush(
        &mut self,
        slot: usize,
        dir: &ArenaDirectory,
        mesh_res: &MeshResidency,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Option<RecordBuffers> {
        let copy = &mut self.gpu[slot];
        let rec_bytes: &[u8] = bytemuck::cast_slice(&self.records);
        let dyn_bytes: &[u8] = bytemuck::cast_slice(&self.dyns);
        const ARENA: usize = std::mem::size_of::<u32>();
        let arena_len = (self.records.len() * ARENA) as u64;
        let arena_word = |s: usize| {
            if mesh_res.is_arrived(s as u32) {
                dir.arena_word(s)
            } else {
                0
            }
        };
        unsafe {
            let grew = copy
                .records
                .maintain(instance, device, physical, rec_bytes.len() as u64)
                | copy
                    .dyns
                    .maintain(instance, device, physical, dyn_bytes.len() as u64)
                | copy.arenas.maintain(instance, device, physical, arena_len);
            if rec_bytes.is_empty() {
                return None;
            }
            if grew {
                // Buffer reallocated; rewrite all contents.
                let arena_words: Vec<u32> = (0..self.records.len()).map(arena_word).collect();
                copy.records.write(0, rec_bytes);
                copy.dyns.write(0, dyn_bytes);
                copy.arenas.write(0, bytemuck::cast_slice(&arena_words));
            } else {
                const REC: usize = std::mem::size_of::<MeshRecord>();
                const DYN: usize = std::mem::size_of::<DrawDyn>();
                for &s in &copy.dirty {
                    let s = s as usize;
                    copy.records
                        .write((s * REC) as u64, &rec_bytes[s * REC..(s + 1) * REC]);
                    copy.dyns
                        .write((s * DYN) as u64, &dyn_bytes[s * DYN..(s + 1) * DYN]);
                    copy.arenas
                        .write((s * ARENA) as u64, &arena_word(s).to_ne_bytes());
                }
            }
        }
        copy.dirty.clear();
        Some(RecordBuffers {
            records: copy.records.bound()?,
            dyns: copy.dyns.bound()?,
            arenas: copy.arenas.bound()?,
            slots: self.records.len() as u32,
        })
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for copy in &mut self.gpu {
            unsafe {
                copy.records.destroy(device);
                copy.dyns.destroy(device);
                copy.arenas.destroy(device);
            }
        }
    }
}

/// Pod mirror of VkDrawIndexedIndirectCommand for HostBuffer writes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    /// Slot index in the SSBO.
    pub first_instance: u32,
}

// Layout must match `VkDrawIndexedIndirectCommand` field-for-field.
const _: () = {
    use ash::vk::DrawIndexedIndirectCommand as Ash;
    assert!(std::mem::size_of::<DrawIndexedIndirect>() == std::mem::size_of::<Ash>());
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, index_count)
            == std::mem::offset_of!(Ash, index_count)
    );
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, instance_count)
            == std::mem::offset_of!(Ash, instance_count)
    );
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, first_index)
            == std::mem::offset_of!(Ash, first_index)
    );
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, vertex_offset)
            == std::mem::offset_of!(Ash, vertex_offset)
    );
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, first_instance)
            == std::mem::offset_of!(Ash, first_instance)
    );
};

/// Create mesh3d push-descriptor set layout.
pub fn create_mesh3d_set_layout(device: &ash::Device, local_read: bool) -> vk::DescriptorSetLayout {
    let mut bindings = vec![
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
    ];
    if local_read {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
    }
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
        .bindings(&bindings);
    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .expect("Failed to create mesh3d set layout")
    }
}

/// Push mesh3d descriptors.
#[allow(clippy::too_many_arguments)]
pub fn push_mesh3d_descriptors(
    push: &khr::push_descriptor::Device,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    records: vk::Buffer,
    dyns: vk::Buffer,
    tex_sampler: vk::Sampler,
    tex_view: vk::ImageView,
    ubo: vk::Buffer,
    cascade_ubo: vk::Buffer,
    shadow_sampler: vk::Sampler,
    shadow_view: vk::ImageView,
) {
    let buffer_infos = [vk::DescriptorBufferInfo::default()
        .buffer(records)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let dyn_infos = [vk::DescriptorBufferInfo::default()
        .buffer(dyns)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let image_infos = [vk::DescriptorImageInfo::default()
        .sampler(tex_sampler)
        .image_view(tex_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let ubo_infos = [vk::DescriptorBufferInfo::default()
        .buffer(ubo)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let cascade_infos = [vk::DescriptorBufferInfo::default()
        .buffer(cascade_ubo)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let shadow_infos = [vk::DescriptorImageInfo::default()
        .sampler(shadow_sampler)
        .image_view(shadow_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&ubo_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&cascade_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&shadow_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&dyn_infos),
    ];
    unsafe {
        push.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &writes);
    }
}

/// Pushes only binding 5 (the scene depth as an input attachment) for the water
/// depth-absorption blend variant. Layered on top of an already-pushed 0-4 set
/// (same compatible layout, so the earlier writes stay live). The depth image is
/// the current depth attachment, so its descriptor layout matches the
/// attachment's `DEPTH_ATTACHMENT_OPTIMAL` (dynamic_rendering_local_read reads it
/// in place — the blend pipeline never writes depth).
pub fn push_depth_input_attachment(
    push: &khr::push_descriptor::Device,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    depth_view: vk::ImageView,
) {
    let image_infos = [vk::DescriptorImageInfo::default()
        .image_view(depth_view)
        // The whole scene pass runs depth in RENDERING_LOCAL_READ when the
        // absorb path is active (the only caller): the one layout valid as
        // BOTH depth attachment and input attachment, and the only truthful
        // value here (VUID-VkWriteDescriptorSet-descriptorType-04151).
        .image_layout(vk::ImageLayout::RENDERING_LOCAL_READ_KHR)];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_binding(5)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .image_info(&image_infos)];
    unsafe {
        push.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &writes);
    }
}

fn shrink_capacity(capacity: u64, peak: u64) -> Option<u64> {
    if capacity <= IMM_MIN_CAPACITY {
        return None; // already at (or below) the floor
    }
    if peak == 0 {
        return Some(0);
    }
    (capacity > peak.saturating_mul(4)).then(|| peak.saturating_mul(2))
}

#[cfg(test)]
mod tests {
    use super::super::timeline::TimelineValue;
    use super::{
        HOST_BAR_CAP, HOST_COHERENT, HandleAllocator, IMM_MIN_CAPACITY, RetireQueue,
        host_buffer_memory_type, shrink_capacity,
    };
    use crate::mesh::MeshHandle;
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
    fn mesh_handle_option_has_niche() {
        // NonZeroU32 generation gives Option<MeshHandle> a niche → 8 bytes, so
        // the streaming lane's millions of Option<MeshHandle> stay compact.
        assert_eq!(std::mem::size_of::<MeshHandle>(), 8);
        assert_eq!(
            std::mem::size_of::<Option<MeshHandle>>(),
            std::mem::size_of::<MeshHandle>()
        );
    }

    #[test]
    fn handle_allocator_reuses_slot_with_bumped_nonzero_generation() {
        // M = u32 stands in for MeshMeta: this exercises identity only.
        let mut a: HandleAllocator<MeshHandle, u32> = HandleAllocator::new();
        let h0 = a.alloc_slot(10);
        assert_eq!(h0.slot, 0);
        assert_eq!(h0.generation.get(), 1, "generations are 1-based");
        assert_eq!(a.meta_mut(h0).copied(), Some(10));
        assert_eq!(a.live_count(), 1);

        assert!(a.free_slot(h0));
        // A stale handle resolves to nothing after its slot is freed.
        assert_eq!(a.meta_mut(h0).copied(), None);
        // Double free is rejected (generation already moved on).
        assert!(!a.free_slot(h0));
        assert_eq!(a.live_count(), 0);

        // Realloc reuses slot 0 with a bumped, still-nonzero generation.
        let h1 = a.alloc_slot(20);
        assert_eq!(h1.slot, 0);
        assert_eq!(h1.generation.get(), 2);
        assert_eq!(a.meta_mut(h1).copied(), Some(20));
        // The old handle still doesn't alias the reused slot.
        assert_eq!(a.meta_mut(h0).copied(), None);
        assert_eq!(a.live_count(), 1);
    }

    #[test]
    fn retire_queue_reclaims_when_the_timeline_reaches_the_stamp() {
        // An entry stamped at value N reclaims once the timeline counter has
        // reached N (stamp <= current).
        let v = TimelineValue::from_raw_for_test;
        let mut q: RetireQueue<u32> = RetireQueue::new();
        assert!(q.is_empty());
        q.push(v(1), 100);
        q.push(v(2), 101);
        q.push(v(4), 103);
        assert!(!q.is_empty());

        // current = 0: nothing has completed yet.
        let mut freed = Vec::new();
        q.collect(v(0), |x| freed.push(x));
        assert_eq!(freed, Vec::<u32>::new());

        // current = 1: stamp 1 drains; stamp 2 stays.
        let mut freed = Vec::new();
        q.collect(v(1), |x| freed.push(x));
        assert_eq!(freed, vec![100]);

        // current = 3: stamp 2 (2 <= 3) drains; stamp 4 stays.
        let mut freed = Vec::new();
        q.collect(v(3), |x| freed.push(x));
        assert_eq!(freed, vec![101]);

        // collect_all drains the remainder (stamp 4, not yet reached).
        let mut freed = Vec::new();
        q.collect_all(|x| freed.push(x));
        assert_eq!(freed, vec![103]);
        assert!(q.is_empty());
    }

    fn props(
        types: &[(vk::MemoryPropertyFlags, u32)],
        heap_sizes: &[u64],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut p = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: types.len() as u32,
            memory_heap_count: heap_sizes.len() as u32,
            ..Default::default()
        };
        for (i, &(property_flags, heap_index)) in types.iter().enumerate() {
            p.memory_types[i] = vk::MemoryType {
                property_flags,
                heap_index,
            };
        }
        for (i, &size) in heap_sizes.iter().enumerate() {
            p.memory_heaps[i].size = size;
        }
        p
    }

    #[test]
    fn host_buffers_prefer_the_bar_type_until_the_cap_and_fall_back_to_system_memory() {
        // Discrete layout: device-local VRAM, system host-coherent, then a
        // small BAR window (no ReBAR).
        let bar = HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let discrete = props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (HOST_COHERENT, 1),
                (bar, 2),
            ],
            &[8 << 30, 16 << 30, 256 << 20],
        );
        let all = 0b111;
        assert_eq!(
            host_buffer_memory_type(&discrete, all, 1 << 20, 0),
            Some((2, true))
        );
        // At the cap the same request lands in system memory.
        assert_eq!(
            host_buffer_memory_type(&discrete, all, 1 << 20, HOST_BAR_CAP),
            Some((1, false))
        );
        assert_eq!(
            host_buffer_memory_type(&discrete, all, 1 << 20, HOST_BAR_CAP - (1 << 20)),
            Some((2, true))
        );
        // A request larger than the cap skips the small BAR even when unused.
        assert_eq!(
            host_buffer_memory_type(&discrete, all, HOST_BAR_CAP + 1, 0),
            Some((1, false))
        );
        // A type filter excluding the BAR type skips it regardless of headroom.
        assert_eq!(
            host_buffer_memory_type(&discrete, 0b011, 1 << 20, 0),
            Some((1, false))
        );
        // ReBAR: the BAR heap is the full VRAM window, so the cap does not apply.
        let rebar = props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (HOST_COHERENT, 1),
                (bar, 0),
            ],
            &[8 << 30, 16 << 30],
        );
        assert_eq!(
            host_buffer_memory_type(&rebar, all, 1 << 20, HOST_BAR_CAP),
            Some((2, true))
        );
        // No BAR type at all: plain host-coherent.
        let no_bar = props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (HOST_COHERENT, 1),
            ],
            &[8 << 30, 16 << 30],
        );
        assert_eq!(
            host_buffer_memory_type(&no_bar, 0b11, 4096, 0),
            Some((1, false))
        );
        // Device-local only (no host-visible type): nothing suitable.
        let dl = props(&[(vk::MemoryPropertyFlags::DEVICE_LOCAL, 0)], &[8 << 30]);
        assert_eq!(host_buffer_memory_type(&dl, 0b1, 4096, 0), None);
    }

    #[test]
    fn shrink_decay_rules() {
        // At or below the floor: never shrink, even when idle.
        assert_eq!(shrink_capacity(IMM_MIN_CAPACITY, 0), None);
        assert_eq!(shrink_capacity(0, 0), None);
        // A whole window with zero usage: destroy outright.
        assert_eq!(shrink_capacity(1 << 20, 0), Some(0));
        // Capacity within 4x of the mark: keep.
        assert_eq!(shrink_capacity(1 << 20, 1 << 18), None); // exactly 4x
        assert_eq!(shrink_capacity(1 << 20, (1 << 18) + 1), None);
        assert_eq!(shrink_capacity(1 << 20, 1 << 19), None);
        // Way oversized: recreate at 2x the mark.
        assert_eq!(shrink_capacity(1 << 20, (1 << 18) - 1), Some((1 << 19) - 2));
        assert_eq!(shrink_capacity(16 << 20, 100 << 10), Some(200 << 10));
        // The 2x target is always strictly below the old capacity.
        let target = shrink_capacity(16 << 20, 100 << 10).unwrap();
        assert!(target.next_power_of_two().max(IMM_MIN_CAPACITY) < 16 << 20);
    }

    /// Verify detail_pass encoding/decoding is consistent.
    #[test]
    fn compose_then_detail_scale_matches_placement_scale() {
        use super::{DrawDyn, MeshMeta, MeshRecord, PlacementState};
        use crate::mesh::{Detail, MeshPlacement};
        for k in -2..=13i8 {
            let detail = Detail(k);
            let meta = MeshMeta {
                aabb_min: glam::Vec3::ZERO,
                aabb_max: glam::Vec3::ONE,
                bounds: [0; 7],
                vertex_offset: 0,
                pass: crate::mesh::Pass::Opaque,
                placement: PlacementState::Pinned,
                dyn_lane: DrawDyn::resting(),
            };
            let p = MeshPlacement::terrain(glam::IVec3::ZERO, detail);
            let rec = MeshRecord::compose(&meta, p);
            assert_eq!(
                rec.detail_scale(),
                detail.scale(),
                "biased detail_pass must decode to the placement's scale (k={k})"
            );
            assert_eq!(rec.pass(), crate::mesh::Pass::Opaque, "pass bits intact");
        }
    }
}
