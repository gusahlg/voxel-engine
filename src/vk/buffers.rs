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

use super::alloc::{Allocation, GpuAllocator, find_memory_type};
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
    let barrier = vk::BufferMemoryBarrier2::default().buffer(buffer).offset(offset).size(size);
    match role {
        CopyBarrier::Draw => barrier
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
            .dst_access_mask(reads),
        CopyBarrier::Release { src_family, dst_family } => barrier
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family),
        CopyBarrier::Acquire { src_family, dst_family } => barrier
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
    /// queue copies order via the returned timeline value instead.
    ///
    /// Returns `Some(value)` when copies were submitted on the lane's own
    /// queue: the caller's render submission must wait on the lane's
    /// semaphore for `value` before touching the copied ranges. `graphics_cmd`
    /// must always be a real, valid (reset-and-begun) command buffer — even
    /// under a separate transfer queue, `DedicatedFamily` needs an ACQUIRE
    /// barrier recorded into it, and the caller is responsible for submitting
    /// `graphics_cmd` afterward (waiting on the returned value's semaphore
    /// when `Some`).
    pub unsafe fn flush_copies(
        &mut self,
        device: &ash::Device,
        lane: &mut TransferLane,
        graphics_cmd: vk::CommandBuffer,
        graphics_family: u32,
        render_done_at: TimelineValue,
    ) -> Option<TimelineValue> {
        if self.pending.is_empty() {
            return None;
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
        // Barrier handling depends on tier: same-queue scoped barrier,
        // or release/acquire for ownership transfer.
        let mut same_queue_barriers: Vec<vk::BufferMemoryBarrier2> = Vec::new();
        let mut release_barriers: Vec<vk::BufferMemoryBarrier2> = Vec::new();
        let mut acquire_barriers: Vec<vk::BufferMemoryBarrier2> = Vec::new();
        let mut copied_slots: Vec<u32> = Vec::with_capacity(batch.len());
        // Stamp value depends on tier (render-Rev or lane-Rev).
        let mut staging: Vec<Allocation> = Vec::with_capacity(batch.len());
        unsafe {
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
                device.cmd_copy_buffer(record_cmd, copy.staging.buffer, copy.dst_buffer, &[region]);
                bytes += copy.size;
                let (buffer, offset, size) = (copy.dst_buffer, copy.dst_offset, copy.size);
                if !separate_queue {
                    same_queue_barriers.push(copy_barrier(
                        buffer,
                        offset,
                        size,
                        mesh_reads(),
                        CopyBarrier::Draw,
                    ));
                } else if needs_qfot {
                    release_barriers.push(copy_barrier(
                        buffer,
                        offset,
                        size,
                        mesh_reads(),
                        CopyBarrier::Release {
                            src_family: lane.family(),
                            dst_family: graphics_family,
                        },
                    ));
                    acquire_barriers.push(copy_barrier(
                        buffer,
                        offset,
                        size,
                        mesh_reads(),
                        CopyBarrier::Acquire {
                            src_family: lane.family(),
                            dst_family: graphics_family,
                        },
                    ));
                }
                // else: SecondQueueSameFamily — no queue-family ownership
                // transfer (same family), and memory visibility is already
                // guaranteed by the timeline semaphore signal/wait pair
                // below; no barrier at all.
                staging.push(copy.staging);
                copied_slots.push(slot);
            }
        }

        if copied_slots.is_empty() {
            // Every batched resident was freed before this flush ran (nothing
            // actually copied), so there is no completion to hand back.
            if let Some(lane_batch) = lane_batch {
                unsafe { lane.discard(device, lane_batch) };
            }
            return None;
        }
        crate::profile::gauge(crate::profile::Gauge::UploadBytes, bytes);

        if !release_barriers.is_empty() {
            unsafe {
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&release_barriers),
                );
            }
        }

        let arrived_at = if let Some(lane_batch) = lane_batch {
            // Cross-queue: a timeline wait (folded into the caller's next
            // graphics submission) orders graphics against these copies; an
            // in-command-buffer barrier cannot scope a cross-queue
            // dependency. `graphics_cmd` still gets the ACQUIRE half of the
            // ownership-transfer pair when the tier needs one.
            let value = unsafe { lane.submit(device, lane_batch) };
            if !acquire_barriers.is_empty() {
                unsafe {
                    device.cmd_pipeline_barrier2(
                        graphics_cmd,
                        &vk::DependencyInfo::default().buffer_memory_barriers(&acquire_barriers),
                    );
                }
            }
            value
        } else {
            // Same queue: barrier in graphics_cmd orders the copies.
            unsafe {
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&same_queue_barriers),
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

        separate_queue.then_some(arrived_at)
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
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

/// A growable host-visible buffer written each frame, one per frame-in-flight.
/// Used for immediate geometry, offsets, and indirect commands.
pub struct HostBuffer {
    /// Null until first write; use [`Self::bound`] to obtain safely.
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    capacity: u64,
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
            let (buffer, memory) = create_raw_buffer(
                device,
                &memory_props,
                new_capacity,
                self.usage,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let mapped = device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("Failed to map immediate buffer") as *mut u8;

            self.buffer = buffer;
            self.memory = memory;
            self.mapped = mapped;
            self.capacity = new_capacity;
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
            self.buffer = vk::Buffer::null();
            self.memory = vk::DeviceMemory::null();
            self.mapped = std::ptr::null_mut();
            self.capacity = 0;
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
                let barrier =
                    [copy_barrier(buffer, 0, size, vk::AccessFlags2::INDEX_READ, CopyBarrier::Draw)];
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
                    CopyBarrier::Release { src_family: lane.family(), dst_family: graphics_family },
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
                    CopyBarrier::Acquire { src_family: lane.family(), dst_family: graphics_family },
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
        let arena_word =
            |s: usize| if mesh_res.is_arrived(s as u32) { dir.arena_word(s) } else { 0 };
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
pub fn create_mesh3d_set_layout(
    device: &ash::Device,
    local_read: bool,
) -> vk::DescriptorSetLayout {
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
    use super::{HandleAllocator, IMM_MIN_CAPACITY, RetireQueue, shrink_capacity};
    use crate::mesh::MeshHandle;

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
