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
use super::timeline::TimelineValue;
use crate::frame::MeshDraw;
use crate::mesh::{MeshData, MeshHandle, Pass};

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

    /// Gen-checked metadata; the record path reads this to frustum-cull.
    pub fn meta(&self, h: H) -> Option<M> {
        if *self.generations.get(h.slot() as usize)? != h.generation() {
            return None;
        }
        self.meta[h.slot() as usize]
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
}

impl MeshMeta {
    pub fn aabb(&self) -> (Vec3, Vec3) {
        (self.aabb_min, self.aabb_max)
    }
}

/// A staged host→device copy owned by a not-yet-flushed [`GpuResident`].
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
        debug_assert_eq!(cursor, vertex_bytes_len, "permutation must cover every vertex");
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
    };
    Some((meta, GpuResident { alloc, copy }))
}

/// Render-side residency mirror for meshes: keyed by the main-assigned slot,
/// with a generation mirror kept in sync from the ordered command stream,
/// guaranteeing correct handle-aliasing without cross-thread reads. Holds no
/// free-list or identity — that is [`HandleAllocator`]'s job.
pub(crate) struct MeshResidency {
    slots: Vec<Option<GpuResident>>,
    generations: Vec<NonZeroU32>,
    pending: Vec<u32>,
    retire: RetireQueue<Allocation>,
    live: usize,
}

impl MeshResidency {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            generations: Vec::new(),
            pending: Vec::new(),
            retire: RetireQueue::new(),
            live: 0,
        }
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

    /// Gen-checked buffer resolve for a recorded draw. `None` (Option-skip) when
    /// the mirror generation moved on — a stale snapshot referencing a
    /// since-freed/realloc'd slot (the transient-hole case).
    pub fn resolve(&self, d: &MeshDraw) -> Option<vk::Buffer> {
        let i = d.slot as usize;
        if *self.generations.get(i)? != d.generation {
            return None;
        }
        Some(self.slots.get(i)?.as_ref()?.alloc.buffer)
    }

    /// Records staged uploads into `cmd`. Returns true if a barrier guarding
    /// transfer -> vertex/index reads was emitted.
    pub unsafe fn flush_copies(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        done_at: TimelineValue,
    ) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        let mut any = false;
        unsafe {
            for slot in std::mem::take(&mut self.pending) {
                let Some(res) = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()) else {
                    continue;
                };
                let Some(copy) = res.copy.take() else { continue };
                let region = vk::BufferCopy::default()
                    .src_offset(copy.staging.offset)
                    .dst_offset(copy.dst_offset)
                    .size(copy.size);
                device.cmd_copy_buffer(cmd, copy.staging.buffer, copy.dst_buffer, &[region]);
                self.retire.push(done_at, copy.staging);
                any = true;
            }
            if any {
                let barrier = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
                    .dst_access_mask(
                        vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::INDEX_READ,
                    )];
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&barrier),
                );
            }
        }
        any
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn has_garbage(&self) -> bool {
        !self.retire.is_empty()
    }

    /// Reclaims retired allocations the GPU has passed, handing each to `recycle`
    /// (which returns it to the main-owned allocator freelist).
    pub fn collect(&mut self, current: TimelineValue, recycle: &mut impl FnMut(Allocation)) {
        self.retire.collect(current, |alloc| recycle(alloc));
    }

    /// Reclaims every retired allocation (GPU idle + copies flushed).
    pub fn collect_all(&mut self, recycle: &mut impl FnMut(Allocation)) {
        self.retire.collect_all(|alloc| recycle(alloc));
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
        self.pending.clear();
        self.live = 0;
    }
}

/// Smallest immediate-buffer capacity (also the floor the decay stops at).
const IMM_MIN_CAPACITY: u64 = 64 * 1024;
/// Frames per decay window: capacity shrinks only when it stayed > 4x the
/// window's high-water mark for this many consecutive frames.
const IMM_SHRINK_WINDOW: u32 = 600;

/// A growable host-visible buffer written each frame, one per frame-in-flight.
/// Used for immediate geometry, offsets, and indirect commands.
pub struct HostBuffer {
    /// Private: `VK_NULL_HANDLE` until the first non-empty write, so it must
    /// never be handed out raw. Consumers obtain it through [`Self::bound`],
    /// which yields `None` while unallocated — turning "bound/pushed a null
    /// buffer" (a validation error + potential GPU hang) into a `None` the call
    /// site is forced to handle.
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    capacity: u64,
    usage: vk::BufferUsageFlags,
    /// Largest `needed` seen in the current decay window.
    window_peak: u64,
    /// Frames elapsed in the current decay window.
    window_frames: u32,
}

impl HostBuffer {
    /// The device buffer handle, or `None` if nothing has been written yet (the
    /// handle is still `VK_NULL_HANDLE`). This is the ONLY way to read the
    /// handle: binding it as a vertex/index buffer or pushing it as a descriptor
    /// requires a valid handle, so routing every such use through this `Option`
    /// makes an empty-frame null bind a compile-visible case rather than a
    /// runtime validation error in a pass far from this buffer.
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

    /// Ensures capacity and shrinks oversized buffers when needed.
    /// Must be called after the frame fence is waited (GPU idle).
    /// Returns `true` if the buffer handle changed.
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

            let info = vk::BufferCreateInfo::default()
                .size(new_capacity)
                .usage(self.usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = device
                .create_buffer(&info, None)
                .expect("Failed to create immediate buffer");
            let requirements = device.get_buffer_memory_requirements(buffer);
            let memory_props = instance.get_physical_device_memory_properties(physical);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(find_memory_type(
                    &memory_props,
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                ));
            let memory = device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate immediate buffer memory");
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("Failed to bind immediate buffer memory");
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
    /// Superseded buffers, freed once the timeline passes their last use.
    retire: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
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

    /// Grows the buffer to cover `required` quads if needed, staging the pattern
    /// on `cmd` (ordered before any index read by a barrier) and retiring the old
    /// buffer past `done_at`. No-op when the current buffer already suffices.
    pub unsafe fn ensure(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        cmd: vk::CommandBuffer,
        done_at: TimelineValue,
    ) {
        if self.required <= self.capacity {
            return;
        }
        let new_capacity = self.required.next_power_of_two().max(QUAD_IBO_MIN_QUADS);
        let index_count = new_capacity as u64 * INDICES_PER_QUAD as u64;
        let size = index_count * std::mem::size_of::<u32>() as u64;
        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(physical) };

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
            device.cmd_copy_buffer(cmd, staging, buffer, &[region]);
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
                .dst_access_mask(vk::AccessFlags2::INDEX_READ)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&barrier),
            );
        }

        if self.capacity > 0 {
            self.retire.push(done_at, (self.buffer, self.memory));
        }
        self.retire.push(done_at, (staging, staging_mem));
        self.buffer = buffer;
        self.memory = memory;
        self.capacity = new_capacity;
    }

    /// Destroys superseded buffers the GPU has provably finished reading.
    pub unsafe fn collect(&mut self, device: &ash::Device, current: TimelineValue) {
        self.retire.collect(current, |(buffer, memory)| unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        });
    }

    /// Destroys the live buffer and every retired one (GPU idle, at teardown).
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.retire.collect_all(|(buffer, memory)| {
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

/// Creates a standalone buffer + bound memory of the given usage/properties.
/// For one-off engine buffers outside the suballocator (the quad IBO); mesh
/// storage still goes through [`GpuAllocator`].
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
            .memory_type_index(find_memory_type(memory_props, req.memory_type_bits, properties));
        let memory = device
            .allocate_memory(&alloc_info, None)
            .expect("allocate buffer memory");
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind buffer memory");
        (buffer, memory)
    }
}

/// Per-draw translation and scale. Naming `scale` (not `w`) avoids silent zeroing.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawOffset {
    pub offset: [f32; 3],
    pub scale: f32,
}

/// `VkDrawIndexedIndirectCommand` as a Pod struct so a frame's command array
/// is one `cast_slice` write into the indirect [`HostBuffer`]. ash's
/// `vk::DrawIndexedIndirectCommand` is not `bytemuck::Pod`, so we define this
/// mirror; the `const _` below pins its layout to ash's at compile time.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    /// Slot index in the offsets SSBO (instance_count is always 1).
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

/// 3D pipeline push-descriptor set: binding 0 = offsets SSBO (vertex),
/// binding 1 = texture array (fragment), binding 2 = per-frame UBO (both),
/// binding 3 = cascade UBO, binding 4 = shadow map. When `local_read` is set,
/// binding 5 = the scene depth as an input attachment (fragment), read by the
/// water-absorption blend variant; the opaque/sky pipelines share this layout
/// but never touch binding 5 (push descriptors bind only what a shader uses).
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

/// Push-descriptor call for mesh3d set 0 bindings 0-4.
#[allow(clippy::too_many_arguments)]
pub fn push_mesh3d_descriptors(
    push: &khr::push_descriptor::Device,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    offsets: vk::Buffer,
    tex_sampler: vk::Sampler,
    tex_view: vk::ImageView,
    ubo: vk::Buffer,
    cascade_ubo: vk::Buffer,
    shadow_sampler: vk::Sampler,
    shadow_view: vk::ImageView,
) {
    let buffer_infos = [vk::DescriptorBufferInfo::default()
        .buffer(offsets)
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
        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)];
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
        assert_eq!(a.meta(h0), Some(10));
        assert_eq!(a.live_count(), 1);

        assert!(a.free_slot(h0));
        // A stale handle resolves to nothing after its slot is freed.
        assert_eq!(a.meta(h0), None);
        // Double free is rejected (generation already moved on).
        assert!(!a.free_slot(h0));
        assert_eq!(a.live_count(), 0);

        // Realloc reuses slot 0 with a bumped, still-nonzero generation.
        let h1 = a.alloc_slot(20);
        assert_eq!(h1.slot, 0);
        assert_eq!(h1.generation.get(), 2);
        assert_eq!(a.meta(h1), Some(20));
        // The old handle still doesn't alias the reused slot.
        assert_eq!(a.meta(h0), None);
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
}
