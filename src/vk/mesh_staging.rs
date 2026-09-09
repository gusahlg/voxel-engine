//! Worker-thread mesh staging: a fixed host-visible pool carved by a lock-free
//! bump/ring, plus timeline-keyed reclaim.
//!
//! Game workers acquire a region, write vertices straight into the mapped
//! bytes, and hand the region to the main thread. Main copies those bytes into
//! a mesh arena block (host memcpy when the arena is mapped, otherwise one
//! `vkCmdCopyBuffer` through the transfer lane) and stamps the region so
//! reclaim can recycle it. Stale builds [`Drop`] the region; reclaim never
//! GPU-waits. The ring lives in cached system memory: it is transient staging,
//! not a vertex buffer.
//!
//! The ring is a FIFO (`reclaim` advances `tail` only past a contiguous ready
//! prefix) and therefore holds only **transient** data. A live mesh must never
//! pin a region: that stalls the tail, and after ~one pool of cumulative
//! uploads `acquire` returns `None` forever. Zero-copy residency (binding the
//! pool as the vertex buffer) would need a general-purpose allocator over the
//! pool (free-list, not FIFO) and is future work.
//!
//! Cost: uploads are many and small, so the ring is the only per-acquire work
//! on the worker. Reclaim, copies, and barriers stay batched on the render
//! thread (one pass / one transfer CB / one barrier per frame). Legacy
//! [`crate::mesh::MeshData`] uploads never touch the ring.

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

use super::alloc::try_find_memory_type;
use super::buffers::{HOST_BAR_BYTES, HOST_COHERENT};
use super::timeline::TimelineValue;
use crate::mesh::{FACE_UPLOAD_ORDER, MeshVertex};

/// Acquires since the last [`take_acquire_count`] (profiler).
static ACQUIRE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Regions that reached upload with no AABB recorded (fallback scan).
static AABB_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Swap-out the acquire counter. Called once per render frame.
pub(crate) fn take_acquire_count() -> u64 {
    ACQUIRE_COUNT.swap(0, Ordering::Relaxed)
}

/// Swap-out the AABB-fallback counter. Called once per render frame.
pub(crate) fn take_aabb_fallback_count() -> u64 {
    AABB_FALLBACK_COUNT.swap(0, Ordering::Relaxed)
}

/// Default host-visible mesh staging ring size (32 MiB).
///
/// A full chunk mesh is median ~5 KB / p95 ~12 KB / max ~28 KB; ~100 in-flight
/// chunks is ~0.5 MB, and 32 MB is ~3× the whole rd12 box. Override once at
/// renderer creation with `VOXEL_MESH_STAGING_MB` (integer MiB; `0` disables).
pub const MESH_STAGING_BYTES: u64 = 32 << 20;

const MESH_STAGING_ENV: &str = "VOXEL_MESH_STAGING_MB";

/// Ring size used at renderer creation. Reads `VOXEL_MESH_STAGING_MB` once.
pub(crate) fn mesh_staging_bytes() -> u64 {
    match std::env::var(MESH_STAGING_ENV) {
        Ok(s) => parse_mesh_staging_mb(&s).unwrap_or_else(|| {
            log::warn!(
                "invalid {MESH_STAGING_ENV}={s:?}; using {} MiB",
                MESH_STAGING_BYTES / (1 << 20)
            );
            MESH_STAGING_BYTES
        }),
        Err(_) => MESH_STAGING_BYTES,
    }
}

fn parse_mesh_staging_mb(s: &str) -> Option<u64> {
    let mb: u64 = s.trim().parse().ok()?;
    Some(mb.saturating_mul(1 << 20))
}

/// Cached system memory for the transient staging ring: prefer
/// `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` without `DEVICE_LOCAL` so
/// resident-mode memcpy (ring → mapped arena) is a cached load. Falls back
/// to any host-coherent type.
fn sysmem_staging_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
) -> Option<u32> {
    let n = memory_props.memory_type_count;
    let cached = (0..n).find(|&i| {
        if type_filter & (1 << i) == 0 {
            return false;
        }
        let flags = memory_props.memory_types[i as usize].property_flags;
        flags.contains(HOST_COHERENT)
            && flags.contains(vk::MemoryPropertyFlags::HOST_CACHED)
            && !flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    });
    cached.or_else(|| try_find_memory_type(memory_props, type_filter, HOST_COHERENT))
}

/// Vertex stride; regions are aligned to this.
const VERTEX_STRIDE: u64 = std::mem::size_of::<MeshVertex>() as u64;

/// A reserved byte range inside the pool. `len` is the aligned reservation
/// (may be larger than the acquire request); `generation` invalidates stale
/// [`MeshStaging`] drops after the range is recycled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StagingRegion {
    pub offset: u64,
    pub len: u64,
    pub generation: u64,
    /// Monotonic acquire head; keys [`StagingRing::live`] for O(log n) stamp.
    key: u64,
}

/// When a region may be reused. [`Held`] is still owned by a [`MeshStaging`];
/// the others become reusable once the matching timeline counter has reached
/// `value` (`0` = the next reclaim pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stamp {
    Held,
    Render(u64),
    /// Transfer-lane timeline. Pooled copies stamp this on the separate-queue
    /// tier and [`Self::Render`] on the same-queue fallback.
    Transfer(u64),
}

impl Stamp {
    fn ready(self, render: u64, transfer: Option<u64>) -> bool {
        match self {
            Stamp::Held => false,
            Stamp::Render(v) => v <= render,
            Stamp::Transfer(v) => transfer.is_some_and(|t| v <= t),
        }
    }
}

struct Live {
    region: StagingRegion,
    /// Monotonic span from this region's head value to the next (includes
    /// wrap padding so tail can skip the wasted tail sliver).
    span: u64,
    stamp: Stamp,
}

/// Lock-free bump/ring over a fixed capacity. Acquire CASes `head`; reclaim
/// (single logical consumer, once per frame) advances `tail` past a contiguous
/// prefix of ready regions. Out-of-order GPU completion does not un-hole the
/// ring: a later region waits for earlier ones, matching the test's reclaim
/// ordering.
pub(crate) struct StagingRing {
    capacity: u64,
    align: u64,
    /// Monotonic bump. Physical offset is `head % capacity`.
    head: AtomicU64,
    tail: AtomicU64,
    epoch: AtomicU64,
    live: Mutex<BTreeMap<u64, Live>>,
}

impl StagingRing {
    pub(crate) fn new(capacity: u64, align: u64) -> Self {
        let align = align.max(1);
        let capacity = (capacity / align) * align;
        Self {
            capacity,
            align,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            epoch: AtomicU64::new(1),
            live: Mutex::new(BTreeMap::new()),
        }
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.capacity
    }

    fn lock_live(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, Live>> {
        self.live.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Non-blocking. `None` when the request is empty, larger than the pool,
    /// or the ring has no contiguous free run of the aligned size.
    pub(crate) fn acquire(&self, bytes: usize) -> Option<StagingRegion> {
        if self.capacity == 0 || bytes == 0 {
            return None;
        }
        let size = (bytes as u64).next_multiple_of(self.align);
        if size > self.capacity {
            return None;
        }
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let (offset, new_head, span) = place(head, tail, size, self.capacity)?;
            if self
                .head
                .compare_exchange_weak(head, new_head, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let generation = self.epoch.fetch_add(1, Ordering::Relaxed);
            let region = StagingRegion {
                offset,
                len: size,
                generation,
                key: head,
            };
            self.lock_live().insert(
                head,
                Live {
                    region,
                    span,
                    stamp: Stamp::Held,
                },
            );
            if crate::profile::is_enabled() {
                ACQUIRE_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            return Some(region);
        }
    }

    /// Stamp a held region. A generation mismatch (stale drop after recycle)
    /// is a no-op.
    pub(crate) fn stamp(&self, region: StagingRegion, stamp: Stamp) {
        let mut live = self.lock_live();
        let Some(entry) = live.get_mut(&region.key) else {
            return;
        };
        if entry.region.generation == region.generation && entry.stamp == Stamp::Held {
            entry.stamp = stamp;
        }
    }

    /// Release without a GPU submit: reusable at the next reclaim pass.
    pub(crate) fn release(&self, region: StagingRegion) {
        self.stamp(region, Stamp::Render(0));
    }

    /// Submit: reusable once the render (or test) timeline reaches `value`.
    #[cfg(test)]
    pub(crate) fn submit(&self, region: StagingRegion, value: u64) {
        self.stamp(region, Stamp::Render(value));
    }

    /// Advance `tail` through the ready FIFO prefix. One call per frame.
    pub(crate) fn reclaim(&self, render: u64, transfer: Option<u64>) {
        let mut live = self.lock_live();
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let Some((&mono, entry)) = live.first_key_value() else {
                break;
            };
            // A CAS-then-insert race can leave a gap at `tail`; wait for it.
            if mono != tail {
                break;
            }
            if !entry.stamp.ready(render, transfer) {
                break;
            }
            let span = entry.span;
            live.pop_first();
            self.tail.store(tail + span, Ordering::Release);
        }
    }
}

/// Place `size` bytes at `head` in a circular buffer of `cap`. Returns
/// `(physical_offset, new_head, span)` where `span = new_head - head` includes
/// wrap padding. `head == tail` is empty; full is `head - tail == cap`.
fn place(head: u64, tail: u64, size: u64, cap: u64) -> Option<(u64, u64, u64)> {
    if size == 0 || size > cap || cap == 0 {
        return None;
    }
    let used = head.saturating_sub(tail);
    if used >= cap {
        return None;
    }
    let phys = head % cap;
    if phys + size <= cap {
        let span = size;
        if used + span > cap {
            return None;
        }
        Some((phys, head + span, span))
    } else {
        let pad = cap - phys;
        let span = pad + size;
        if used + span > cap {
            return None;
        }
        // Wrapping onto [0, size) is only legal while live bytes still sit in
        // this lap at [tail_phys, phys): [0, tail_phys) is the free prefix.
        let same_lap = head / cap == tail / cap;
        let tail_phys = tail % cap;
        if !same_lap || size > tail_phys {
            return None;
        }
        Some((0, head + span, span))
    }
}

/// Writes per-[`crate::mesh::Normal`] vertex slices into `dst` in
/// [`FACE_UPLOAD_ORDER`]. Returns bytes written.
pub(crate) fn write_dir_vertices(dst: &mut [u8], dir_slices: [&[MeshVertex]; 6]) -> usize {
    let mut cursor = 0usize;
    for &dir in &FACE_UPLOAD_ORDER {
        let bytes: &[u8] = bytemuck::cast_slice(dir_slices[dir]);
        let end = cursor + bytes.len();
        dst[cursor..end].copy_from_slice(bytes);
        cursor = end;
    }
    cursor
}

/// Expand `aabb` by the 5-bit local positions of `verts` (same decode as
/// [`crate::mesh::MeshData::quad`]).
fn expand_aabb(aabb: &mut Option<([f32; 3], [f32; 3])>, verts: &[MeshVertex]) {
    for v in verts {
        let p = v.local_pos();
        match aabb {
            None => *aabb = Some((p, p)),
            Some((min, max)) => {
                for i in 0..3 {
                    min[i] = min[i].min(p[i]);
                    max[i] = max[i].max(p[i]);
                }
            }
        }
    }
}

/// Typed cursor over a staging region's mapped bytes.
pub struct MeshVertexWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
    aabb: &'a mut Option<([f32; 3], [f32; 3])>,
}

impl MeshVertexWriter<'_> {
    /// Appends `verts` and tracks their 5-bit local-position AABB; returns
    /// `false` if they would not fit.
    pub fn write(&mut self, verts: &[MeshVertex]) -> bool {
        let n = std::mem::size_of_val(verts);
        let Some(dst) = self.buf.get_mut(self.pos..self.pos + n) else {
            return false;
        };
        dst.copy_from_slice(bytemuck::cast_slice(verts));
        self.pos += n;
        expand_aabb(self.aabb, verts);
        true
    }

    /// Appends one quad and tracks the AABB of its 5-bit local positions,
    /// mirroring [`crate::mesh::MeshData::quad`]. Returns `false` if the
    /// quad would not fit.
    ///
    /// Unlike `MeshData::quad`, vertices are written sequentially (the
    /// caller is responsible for GPU upload order); only the AABB decode
    /// is shared.
    pub fn quad(&mut self, corners: [MeshVertex; 4]) -> bool {
        self.write(&corners)
    }

    /// Bytes written so far.
    pub fn bytes_written(&self) -> usize {
        self.pos
    }
}

/// Cheap `Clone` handle to the process-wide mesh staging pool. `Send + Sync`;
/// workers call [`Self::acquire`].
#[derive(Clone)]
pub struct MeshStager {
    pool: Arc<MeshStagingPool>,
}

impl MeshStager {
    /// Non-blocking acquire of a host-mapped region sized in bytes. `None` if
    /// the pool is exhausted (the job retries next frame).
    pub fn acquire(&self, bytes: usize) -> Option<MeshStaging> {
        MeshStagingPool::acquire(&self.pool, bytes)
    }

    /// Ring capacity in bytes (aligned down to the vertex stride).
    pub fn capacity_bytes(&self) -> u64 {
        self.pool.ring.capacity()
    }
}

/// A mapped staging region. Write with [`Self::bytes`], [`Self::write_vertices`],
/// or [`Self::vertex_writer`]; [`Drop`] (and `Engine::release_mesh_staging`)
/// returns it to the pool's reclaim list without a GPU wait.
///
/// [`Self::write_vertices`] and [`MeshVertexWriter::quad`] track the mesh AABB
/// as they write. Callers that fill [`Self::bytes`] raw must call
/// [`Self::set_aabb`] so install does not scan the region.
#[must_use = "dropping a MeshStaging releases the region; pass it to upload_mesh_staged to keep the bytes"]
pub struct MeshStaging {
    pool: Arc<MeshStagingPool>,
    region: StagingRegion,
    requested: usize,
    /// Raw pointer so this type is `!Sync` (exclusive writer) while `Send`.
    ptr: *mut u8,
    consumed: bool,
    /// Inclusive min/max of 5-bit local positions, as `f32`. `None` until a
    /// write path or [`Self::set_aabb`] records them.
    aabb: Option<([f32; 3], [f32; 3])>,
}

// SAFETY: the mapped range is exclusively owned by this region until release;
// workers send the region to main after writing, never aliasing it.
unsafe impl Send for MeshStaging {}

impl MeshStaging {
    /// Mapped bytes of the acquire request (not the aligned reservation).
    ///
    /// Filling this slice raw does not track an AABB; call [`Self::set_aabb`]
    /// before [`crate::Engine::upload_mesh_staged`]. Prefer
    /// [`Self::write_vertices`] or [`Self::vertex_writer`] when the source is
    /// typed vertices.
    pub fn bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.requested) }
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.requested) }
    }

    pub(crate) fn requested_bytes(&self) -> usize {
        self.requested
    }

    /// Concatenate per-direction vertex slices in GPU upload order and track
    /// the AABB of their 5-bit local positions (same decode as
    /// [`crate::mesh::MeshData::quad`]).
    pub fn write_vertices(&mut self, dir_slices: [&[MeshVertex]; 6]) {
        write_dir_vertices(self.bytes(), dir_slices);
        for slice in dir_slices {
            expand_aabb(&mut self.aabb, slice);
        }
    }

    /// Sequential typed writer over [`Self::bytes`].
    ///
    /// [`MeshVertexWriter::quad`] and [`MeshVertexWriter::write`] track the
    /// AABB as they write.
    pub fn vertex_writer(&mut self) -> MeshVertexWriter<'_> {
        MeshVertexWriter {
            buf: unsafe { std::slice::from_raw_parts_mut(self.ptr, self.requested) },
            pos: 0,
            aabb: &mut self.aabb,
        }
    }

    /// Record the mesh AABB for a region filled through [`Self::bytes`].
    ///
    /// `min` and `max` are the inclusive 5-bit local positions (chunk-local
    /// `0..=16`), converted the same way [`crate::mesh::MeshData::quad`]
    /// converts [`crate::mesh::MeshVertex::local_pos`]. Callers that use
    /// [`Self::write_vertices`] or [`MeshVertexWriter::quad`] do not need
    /// this: those paths track the bounds as they write.
    pub fn set_aabb(&mut self, min: [u8; 3], max: [u8; 3]) {
        self.aabb = Some((
            [min[0] as f32, min[1] as f32, min[2] as f32],
            [max[0] as f32, max[1] as f32, max[2] as f32],
        ));
    }

    /// AABB recorded by a write path or [`Self::set_aabb`]. `None` if the
    /// region was filled through [`Self::bytes`] without `set_aabb`.
    pub(crate) fn recorded_aabb(&self) -> Option<([f32; 3], [f32; 3])> {
        self.aabb
    }

    /// AABB for install: the recorded bounds, or a scan of `vertex_bytes`
    /// from this (cached) region. The scan is a documented fallback for
    /// callers that filled [`Self::bytes`] without [`Self::set_aabb`]; it is
    /// counted in the `pool.aabb` profiler gauge.
    pub(crate) fn aabb_for_upload(&self, vertex_bytes: usize) -> ([f32; 3], [f32; 3]) {
        if let Some(aabb) = self.recorded_aabb() {
            return aabb;
        }
        AABB_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
        let verts: &[MeshVertex] = bytemuck::cast_slice(&self.as_bytes()[..vertex_bytes]);
        let mut aabb_min = [f32::INFINITY; 3];
        let mut aabb_max = [f32::NEG_INFINITY; 3];
        for v in verts {
            let p = v.local_pos();
            for i in 0..3 {
                aabb_min[i] = aabb_min[i].min(p[i]);
                aabb_max[i] = aabb_max[i].max(p[i]);
            }
        }
        (aabb_min, aabb_max)
    }

    /// Explicit release; same as drop.
    pub fn release(self) {
        drop(self);
    }

    pub(crate) fn into_lease(mut self) -> StagingLease {
        self.consumed = true;
        StagingLease {
            pool: Arc::clone(&self.pool),
            region: self.region,
        }
    }
}

impl Drop for MeshStaging {
    fn drop(&mut self) {
        if !self.consumed {
            self.pool.ring.release(self.region);
        }
    }
}

/// Pool-owned region that outlives [`MeshStaging`]: the source of a pending
/// host memcpy or `vkCmdCopyBuffer`. Always stamped before the mesh is kept.
pub(crate) struct StagingLease {
    pool: Arc<MeshStagingPool>,
    region: StagingRegion,
}

impl StagingLease {
    pub(crate) fn offset(&self) -> u64 {
        self.region.offset
    }

    pub(crate) fn stamp(self, stamp: Stamp) {
        self.pool.ring.stamp(self.region, stamp);
        std::mem::forget(self);
    }
}

impl Drop for StagingLease {
    fn drop(&mut self) {
        self.pool.ring.release(self.region);
    }
}

/// Fixed host-visible mesh staging buffer and its bump/ring.
pub(crate) struct MeshStagingPool {
    ring: StagingRing,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: Option<NonNull<u8>>,
    bar_bytes: u64,
    destroyed: AtomicBool,
    /// Host-only backing for unit tests (the pointer in `mapped` aliases this).
    #[allow(dead_code)]
    _pin: Option<Box<[u8]>>,
}

// SAFETY: the persistent mapping is process-wide; disjoint regions are written
// by the workers that acquired them, and GPU reads start only after submit.
unsafe impl Send for MeshStagingPool {}
unsafe impl Sync for MeshStagingPool {}

impl MeshStagingPool {
    /// Allocate the pool buffer in host-visible *system* memory.
    ///
    /// Prefers `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` so resident-mode
    /// memcpy (ring → mapped arena) is a cached load; falls back to any
    /// host-coherent type. The ring is transient staging, never a vertex
    /// buffer. `size == 0` disables the pool (acquire always returns `None`).
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        size: u64,
    ) -> Arc<Self> {
        if size == 0 {
            return Arc::new(Self::disabled());
        }
        let size = size.next_multiple_of(VERTEX_STRIDE).max(VERTEX_STRIDE);
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        // TRANSFER_SRC only: the ring is never bound as a vertex buffer.
        let usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            device
                .create_buffer(&info, None)
                .expect("create mesh staging buffer")
        };
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let type_index = sysmem_staging_type(&memory_props, req.memory_type_bits)
            .expect("no HOST_VISIBLE | HOST_COHERENT memory type for mesh staging");
        let memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(type_index),
                    None,
                )
                .expect("allocate mesh staging memory")
        };
        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind mesh staging memory");
        }
        let mapped = unsafe {
            device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("map mesh staging buffer") as *mut u8
        };
        let flags = memory_props.memory_types[type_index as usize].property_flags;
        let cached = flags.contains(vk::MemoryPropertyFlags::HOST_CACHED);
        log::info!(
            "mesh staging pool: {} MiB, memory type {type_index} cached={cached} ({flags:?})",
            size / (1024 * 1024)
        );
        Arc::new(Self {
            ring: StagingRing::new(size, VERTEX_STRIDE),
            buffer,
            memory,
            mapped: NonNull::new(mapped),
            bar_bytes: 0,
            destroyed: AtomicBool::new(false),
            _pin: None,
        })
    }

    fn disabled() -> Self {
        Self {
            ring: StagingRing::new(0, VERTEX_STRIDE),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: None,
            bar_bytes: 0,
            destroyed: AtomicBool::new(true),
            _pin: None,
        }
    }

    pub(crate) fn stager(self: &Arc<Self>) -> MeshStager {
        MeshStager {
            pool: Arc::clone(self),
        }
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn acquire(pool: &Arc<Self>, bytes: usize) -> Option<MeshStaging> {
        let mapped = pool.mapped?;
        let region = pool.ring.acquire(bytes)?;
        Some(MeshStaging {
            pool: Arc::clone(pool),
            region,
            requested: bytes,
            ptr: unsafe { mapped.as_ptr().add(region.offset as usize) },
            consumed: false,
            aabb: None,
        })
    }

    /// One reclaim pass: regions whose stamp the given timelines have passed
    /// become reusable. `transfer` is `None` on the same-queue fallback tier.
    pub(crate) fn reclaim(&self, render: TimelineValue, transfer: Option<TimelineValue>) {
        self.ring
            .reclaim(render.raw(), transfer.map(TimelineValue::raw));
    }

    /// Destroy the Vulkan buffer. Safe to call once; GPU must be idle.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        if self.destroyed.swap(true, Ordering::AcqRel) {
            return;
        }
        if self.buffer != vk::Buffer::null() {
            unsafe {
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.memory, None);
            }
            HOST_BAR_BYTES.fetch_sub(self.bar_bytes, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
impl MeshStagingPool {
    pub(crate) fn new_host(capacity: usize) -> Arc<Self> {
        let mut pin = vec![0u8; capacity].into_boxed_slice();
        let mapped = NonNull::new(pin.as_mut_ptr());
        Arc::new(Self {
            ring: StagingRing::new(capacity as u64, VERTEX_STRIDE),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped,
            bar_bytes: 0,
            destroyed: AtomicBool::new(true),
            _pin: Some(pin),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Ao, Light, MeshData, MeshVertex, Normal, Pass};

    fn ring(cap: u64) -> StagingRing {
        StagingRing::new(cap, VERTEX_STRIDE)
    }

    #[test]
    fn place_fits_then_rejects_when_full() {
        assert_eq!(place(0, 0, 8, 64), Some((0, 8, 8)));
        assert_eq!(place(0, 0, 64, 64), Some((0, 64, 64)));
        assert_eq!(place(64, 0, 8, 64), None, "full: head - tail == cap");
        assert_eq!(place(0, 0, 0, 64), None);
        assert_eq!(place(0, 0, 72, 64), None);
    }

    #[test]
    fn place_wraps_onto_freed_prefix() {
        // Live [24, 56); free [56, 64) U [0, 24). A 16-byte alloc wraps to 0.
        assert_eq!(place(56, 24, 16, 64), Some((0, 80, 24)));
        // Already wrapped (head=72 → phys 8): live [24, 64) U [0, 8), hole [8, 24).
        assert_eq!(place(72, 24, 16, 64), Some((8, 88, 16)));
        assert_eq!(place(72, 24, 24, 64), None, "hole is only 16 bytes");
        // Not enough free prefix to land a wrap from phys 56 onto [0, 16).
        assert_eq!(place(56, 8, 16, 64), None);
    }

    #[test]
    fn acquire_until_none_release_then_wrap() {
        let r = ring(64);
        let mut held = Vec::new();
        loop {
            match r.acquire(8) {
                Some(region) => held.push(region),
                None => break,
            }
        }
        assert_eq!(held.len(), 8);
        assert!(r.acquire(8).is_none());
        for region in &held {
            assert_eq!(region.offset % VERTEX_STRIDE, 0);
            assert_eq!(region.len % VERTEX_STRIDE, 0);
        }
        // Free the first half; reclaim makes [0, 32) reusable.
        for region in held.drain(..4) {
            r.release(region);
        }
        r.reclaim(0, None);
        // Head is at 64 (phys 0); the next 32-byte alloc wraps onto offset 0.
        let wrapped = r.acquire(32).expect("wrap-around after reclaim");
        assert_eq!(wrapped.offset, 0);
        assert_eq!(wrapped.len, 32);
        assert!(r.acquire(8).is_none(), "ring is full again");
        // Alignment: a 1-byte request still occupies a vertex-stride unit.
        r.release(wrapped);
        for region in held {
            r.release(region);
        }
        r.reclaim(0, None);
        let tiny = r.acquire(1).unwrap();
        assert_eq!(tiny.offset % VERTEX_STRIDE, 0);
        assert_eq!(tiny.len, VERTEX_STRIDE);
    }

    #[test]
    fn reclaim_is_ordered_by_timeline_value() {
        let r = ring(16);
        let a = r.acquire(8).unwrap();
        let b = r.acquire(8).unwrap();
        r.submit(a, 5);
        r.submit(b, 3);
        // b completed first, but a is the FIFO head and is stamped later.
        r.reclaim(3, None);
        assert!(r.acquire(8).is_none(), "a still occupies the tail");
        r.reclaim(4, None);
        assert!(r.acquire(8).is_none());
        r.reclaim(5, None);
        // Both drain in order once the head's stamp is reached.
        let c = r.acquire(16).unwrap();
        assert_eq!(c.offset, 0);
    }

    #[test]
    fn resident_mode_submit_reclaims_at_render_zero() {
        // Vertex-resident installs stamp Render(0); one reclaim at 0 must free
        // the region. A FIFO that pinned live meshes would fail the 1000-upload
        // loop below after ~one pool of cumulative traffic.
        let r = ring(64);
        let a = r.acquire(64).unwrap();
        r.submit(a, 0);
        assert!(r.acquire(8).is_none(), "submitted region occupies the ring");
        r.reclaim(0, None);
        let b = r
            .acquire(64)
            .expect("Render(0) region reclaimed at render value 0");
        assert_eq!(b.offset, 0);
        r.release(b);
        r.reclaim(0, None);

        let r = ring(MESH_STAGING_BYTES);
        const CHUNK: usize = 32 << 10;
        for i in 0..1000 {
            let region = r
                .acquire(CHUNK)
                .unwrap_or_else(|| panic!("ring exhausted at upload {i}"));
            r.submit(region, 0);
            if (i + 1) % 8 == 0 {
                r.reclaim(0, None);
            }
        }
    }

    #[test]
    fn submit_release_sequence_with_fake_timeline() {
        let r = ring(16);
        let a = r.acquire(16).unwrap();
        r.submit(a, 1);
        r.reclaim(0, None);
        assert!(
            r.acquire(8).is_none(),
            "submitted region waits for timeline"
        );
        r.reclaim(1, None);
        let b = r.acquire(16).unwrap();
        assert_eq!(b.offset, 0);
        // Explicit release (stale build): next reclaim, no GPU wait.
        r.release(b);
        r.reclaim(0, None);
        assert!(r.acquire(16).is_some());
    }

    #[test]
    fn stale_generation_does_not_release_recycled_region() {
        let r = ring(8);
        let a = r.acquire(8).unwrap();
        r.release(a);
        r.reclaim(0, None);
        let b = r.acquire(8).unwrap();
        assert_eq!(b.offset, a.offset);
        assert_ne!(b.generation, a.generation);
        r.release(a);
        r.reclaim(0, None);
        assert!(
            r.acquire(8).is_none(),
            "stale drop must not free the live tenant"
        );
        r.release(b);
        r.reclaim(0, None);
        assert!(r.acquire(8).is_some());
    }

    #[test]
    fn drop_and_submit_go_through_the_pool_handle() {
        let pool = MeshStagingPool::new_host(32);
        let stager = pool.stager();
        {
            let mut s = stager.acquire(32).unwrap();
            s.bytes()[0] = 7;
        }
        pool.reclaim(TimelineValue::START, None);
        // Drop+reclaim must free the only region so a full-size acquire works.
        let staging = stager.acquire(32).expect("drop released the region");
        let lease = staging.into_lease();
        lease.stamp(Stamp::Render(2));
        pool.reclaim(TimelineValue::from_raw_for_test(1), None);
        assert!(
            stager.acquire(8).is_none(),
            "submitted region waits for the fake timeline"
        );
        pool.reclaim(TimelineValue::from_raw_for_test(2), None);
        assert!(stager.acquire(32).is_some());
    }

    #[test]
    fn parse_mesh_staging_mb_env() {
        assert_eq!(parse_mesh_staging_mb("32"), Some(32 << 20));
        assert_eq!(parse_mesh_staging_mb(" 0 "), Some(0));
        assert_eq!(parse_mesh_staging_mb("1"), Some(1 << 20));
        assert_eq!(parse_mesh_staging_mb(""), None);
        assert_eq!(parse_mesh_staging_mb("nope"), None);
    }

    fn memory_props(
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
    fn sysmem_staging_skips_bar_types() {
        let bar = HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let discrete = memory_props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (HOST_COHERENT, 1),
                (bar, 2),
            ],
            &[8 << 30, 16 << 30, 256 << 20],
        );
        assert_eq!(sysmem_staging_type(&discrete, 0b111), Some(1));
        // ReBAR: still prefer the non-DEVICE_LOCAL host type.
        let rebar = memory_props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (HOST_COHERENT, 1),
                (bar, 0),
            ],
            &[8 << 30, 16 << 30],
        );
        assert_eq!(sysmem_staging_type(&rebar, 0b111), Some(1));
        // Unified-only (no sysmem type): fall back to the host-coherent BAR type.
        let unified = memory_props(&[(bar, 0)], &[8 << 30]);
        assert_eq!(sysmem_staging_type(&unified, 0b1), Some(0));
    }

    #[test]
    fn sysmem_staging_prefers_cached_system_memory() {
        let cached = HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED;
        let uncached = HOST_COHERENT;
        let bar = HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let discrete = memory_props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (uncached, 1),
                (cached, 1),
                (bar, 2),
            ],
            &[8 << 30, 16 << 30, 16 << 30, 256 << 20],
        );
        assert_eq!(sysmem_staging_type(&discrete, 0b1111), Some(2));
        // Cached BAR must not beat uncached sysmem: prefer any HOST_COHERENT
        // only when no cached sysmem type exists.
        let cached_bar = cached | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let no_cached_sys = memory_props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (uncached, 1),
                (cached_bar, 0),
            ],
            &[8 << 30, 16 << 30],
        );
        assert_eq!(sysmem_staging_type(&no_cached_sys, 0b111), Some(1));
    }

    #[test]
    fn transfer_stamp_uses_the_transfer_timeline_not_render() {
        let r = ring(16);
        let a = r.acquire(16).unwrap();
        r.stamp(a, Stamp::Transfer(1));
        // A high render value must not free a Transfer stamp.
        r.reclaim(100, None);
        assert!(
            r.acquire(8).is_none(),
            "Transfer stamp is not ready without a transfer counter"
        );
        r.reclaim(100, Some(0));
        assert!(
            r.acquire(8).is_none(),
            "transfer counter 0 has not reached 1"
        );
        r.reclaim(0, Some(1));
        let b = r
            .acquire(16)
            .expect("Transfer(1) reclaims at transfer value 1");
        assert_eq!(b.offset, 0);
    }

    #[test]
    fn pooled_copy_arrival_latency_is_one_frame_on_a_fake_timeline() {
        // Mirrors the render loop: apply_upload (Held) in the command drain,
        // then note_frame + flush (Stamp::Transfer of the lane submit) in
        // draw. The graphics wait is deferred one frame; reclaim waits for
        // the transfer counter — no same-frame GPU wait on the host.
        let r = ring(16);
        let region = r.acquire(16).unwrap();
        let queued_frame = 0u64;
        assert_eq!(region.offset, 0);

        // Frame 1: flush stamps the transfer timeline; region still live.
        let frame = queued_frame + 1;
        assert_eq!(
            frame - queued_frame,
            1,
            "flushed the iteration it was applied"
        );
        r.stamp(region, Stamp::Transfer(frame));
        r.reclaim(0, None);
        assert!(
            r.acquire(8).is_none(),
            "Transfer(1) region waits for the fake transfer timeline"
        );

        r.reclaim(frame, None);
        assert!(
            r.acquire(8).is_none(),
            "a high render value does not free a Transfer stamp"
        );
        r.reclaim(0, Some(frame));
        let recycled = r
            .acquire(16)
            .expect("pooled region reclaimed one frame later");
        assert_eq!(recycled.offset, 0);
    }

    #[test]
    fn stager_reports_capacity() {
        let pool = MeshStagingPool::new_host(64);
        assert_eq!(pool.stager().capacity_bytes(), 64);
        let disabled = Arc::new(MeshStagingPool::disabled());
        assert_eq!(disabled.stager().capacity_bytes(), 0);
    }

    #[test]
    fn stager_is_send_sync_staging_is_send() {
        fn send<T: Send>() {}
        fn send_sync<T: Send + Sync>() {}
        send_sync::<MeshStager>();
        send::<MeshStaging>();
        send_sync::<Arc<MeshStagingPool>>();
    }

    fn tagged_quad(normal: Normal, tag: u8, pos: [u8; 3]) -> [MeshVertex; 4] {
        std::array::from_fn(|i| {
            MeshVertex::new(
                [pos[0] + i as u8, pos[1], pos[2]],
                normal,
                u16::from(tag),
                Ao::NONE,
                Light::FULL,
                false,
            )
        })
    }

    #[test]
    fn write_vertices_and_quad_writer_track_aabb_like_meshdata() {
        let mut data = MeshData::new(Pass::Opaque);
        let a = tagged_quad(Normal::NegY, 1, [1, 2, 3]);
        let b = tagged_quad(Normal::PosX, 2, [4, 5, 6]);
        data.quad(a);
        data.quad(b);

        let n = data.vertex_bytes().max(64);
        let pool = MeshStagingPool::new_host(n * 2);
        let stager = pool.stager();
        let mut staging = stager.acquire(data.vertex_bytes()).unwrap();
        assert!(staging.recorded_aabb().is_none());
        staging.write_vertices(std::array::from_fn(|i| data.vertices[i].as_slice()));
        assert_eq!(staging.recorded_aabb(), Some(data.aabb()));

        let mut writer_staging = stager.acquire(data.vertex_bytes()).unwrap();
        {
            let mut w = writer_staging.vertex_writer();
            assert!(w.quad(a));
            assert!(w.quad(b));
        }
        assert_eq!(writer_staging.recorded_aabb(), Some(data.aabb()));
    }

    #[test]
    fn set_aabb_records_bounds_for_a_raw_bytes_fill() {
        let pool = MeshStagingPool::new_host(32);
        let mut staging = pool.stager().acquire(8).unwrap();
        staging.bytes()[0] = 1;
        assert!(staging.recorded_aabb().is_none());
        staging.set_aabb([1, 2, 3], [4, 5, 6]);
        assert_eq!(
            staging.recorded_aabb(),
            Some(([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]))
        );
    }

    #[test]
    fn aabb_for_upload_scans_when_unrecorded() {
        let mut data = MeshData::new(Pass::Opaque);
        data.quad(tagged_quad(Normal::PosY, 0, [2, 3, 4]));
        let pool = MeshStagingPool::new_host(data.vertex_bytes().max(32));
        let mut staging = pool.stager().acquire(data.vertex_bytes()).unwrap();
        staging
            .bytes()
            .copy_from_slice(bytemuck::cast_slice(&data.vertices()));
        let _ = take_aabb_fallback_count();
        let got = staging.aabb_for_upload(data.vertex_bytes());
        assert_eq!(got, data.aabb());
        assert_eq!(take_aabb_fallback_count(), 1);
        // A recorded AABB must not scan.
        staging.set_aabb([0, 0, 0], [1, 1, 1]);
        let recorded = staging.aabb_for_upload(data.vertex_bytes());
        assert_eq!(recorded, ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        assert_eq!(take_aabb_fallback_count(), 0);
    }

    #[test]
    fn write_vertices_matches_face_upload_order() {
        fn tagged(normal: Normal, tag: u8) -> [MeshVertex; 4] {
            std::array::from_fn(|i| {
                MeshVertex::new(
                    [i as u8, tag, 0],
                    normal,
                    u16::from(tag),
                    Ao::NONE,
                    Light::FULL,
                    false,
                )
            })
        }
        let mut data = MeshData::new(Pass::Opaque);
        data.quad(tagged(Normal::NegY, 1));
        data.quad(tagged(Normal::PosX, 2));
        data.quad(tagged(Normal::PosX, 3));
        let want = data.vertices();
        let mut buf = vec![0u8; data.vertex_bytes()];
        let n = write_dir_vertices(
            &mut buf,
            std::array::from_fn(|i| data.vertices[i].as_slice()),
        );
        assert_eq!(n, data.vertex_bytes());
        let got: &[MeshVertex] = bytemuck::cast_slice(&buf);
        assert_eq!(got, want.as_slice());
    }

    #[test]
    fn concurrent_acquire_returns_disjoint_regions() {
        let r = Arc::new(ring(1 << 16));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let r = Arc::clone(&r);
                std::thread::spawn(move || {
                    let mut got = Vec::new();
                    for _ in 0..64 {
                        if let Some(region) = r.acquire(32) {
                            got.push(region);
                        }
                    }
                    got
                })
            })
            .collect();
        let mut all = Vec::new();
        for t in threads {
            all.extend(t.join().unwrap());
        }
        all.sort_by_key(|r| r.offset);
        for pair in all.windows(2) {
            assert!(
                pair[0].offset + pair[0].len <= pair[1].offset,
                "overlapping regions {pair:?}"
            );
            assert_ne!(pair[0].generation, pair[1].generation);
        }
    }
}
