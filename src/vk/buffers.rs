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

use super::alloc::{Allocation, GpuAllocator, find_memory_type};
use super::timeline::TimelineValue;
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

/// Suballocation alignment: LCM of GPU offset alignment (256) and vertex stride.
/// Aligns all vertex data to stride boundaries and allows binding at offset 0.
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

pub struct GpuMesh {
    alloc: Allocation,
    /// Seven absolute first-index boundaries: `bounds[dir]..bounds[dir+1]` is
    /// direction `dir`'s index range (in the same index space as the old
    /// `first_index`), and `bounds[0]..bounds[6]` is the whole mesh. A
    /// zero-length bucket has `bounds[dir] == bounds[dir+1]`. Monotonic
    /// non-decreasing by construction (see [`MeshRegistry::upload`]).
    bounds: [u32; 7],
    /// This mesh's first vertex (in vertices from block start).
    /// Used as the indirect command's `vertex_offset`.
    vertex_offset: i32,
    pass: Pass,
    aabb_min: Vec3,
    aabb_max: Vec3,
}

impl GpuMesh {
    pub fn buffer(&self) -> vk::Buffer {
        self.alloc.buffer
    }

    pub fn pass(&self) -> Pass {
        self.pass
    }

    pub fn vertex_offset(&self) -> i32 {
        self.vertex_offset
    }

    pub fn aabb(&self) -> (Vec3, Vec3) {
        (self.aabb_min, self.aabb_max)
    }

    /// Index range covering directions `[start, end)` (both in `0..=6`) — one
    /// coalesced run of adjacent buckets. `all_range()` is `run_range(0, 6)`.
    pub fn run_range(&self, start: usize, end: usize) -> std::ops::Range<u32> {
        debug_assert!(start <= end && end <= 6);
        self.bounds[start]..self.bounds[end]
    }

    /// Index range covering the whole mesh (all six directions coalesced).
    pub fn all_range(&self) -> std::ops::Range<u32> {
        self.run_range(0, 6)
    }
}

struct PendingCopy {
    staging: Allocation,
    dst_buffer: vk::Buffer,
    dst_offset: u64,
    size: u64,
}

pub struct MeshRegistry {
    slots: Vec<Option<GpuMesh>>,
    generations: Vec<u32>,
    free_slots: Vec<u32>,
    pending: Vec<PendingCopy>,
    retire: RetireQueue<Allocation>,
    pub live_count: usize,
}

impl MeshRegistry {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            generations: Vec::new(),
            free_slots: Vec::new(),
            pending: Vec::new(),
            retire: RetireQueue::new(),
            live_count: 0,
        }
    }

    /// Uploads mesh data.
    pub unsafe fn upload(
        &mut self,
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        data: &MeshData,
    ) -> Option<MeshHandle> {
        let total_indices: usize = data.buckets.iter().map(Vec::len).sum();
        if total_indices == 0 || data.vertices.is_empty() {
            return None;
        }

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&data.vertices);
        // Index data starts 4-byte aligned right after the vertices.
        let index_start = (vertex_bytes.len() as u64).next_multiple_of(4);
        let index_bytes_len = total_indices * std::mem::size_of::<u32>();
        let total = index_start + index_bytes_len as u64;

        let alloc = unsafe { allocator.alloc_device(device, total, MESH_ALIGN) }
            .map_err(|err| log::error!("mesh allocation failed: {err:?}"))
            .ok()?;

        // Writes vertices at 0, then the six index buckets concatenated in
        // `Normal` order starting at `index_start` (one memcpy per bucket).
        let write_into = |dst: *mut u8| unsafe {
            std::ptr::copy_nonoverlapping(vertex_bytes.as_ptr(), dst, vertex_bytes.len());
            let mut cursor = index_start as usize;
            for bucket in &data.buckets {
                let bytes: &[u8] = bytemuck::cast_slice(bucket);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.add(cursor), bytes.len());
                cursor += bytes.len();
            }
        };

        if let Some(mapped) = alloc.mapped {
            // Unified memory: write straight into the device-local block.
            write_into(mapped.as_ptr());
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
            self.pending.push(PendingCopy {
                dst_buffer: alloc.buffer,
                dst_offset: alloc.offset,
                size: total,
                staging,
            });
        }

        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);
        for v in &data.vertices {
            let p = Vec3::from_array(v.local_pos());
            aabb_min = aabb_min.min(p);
            aabb_max = aabb_max.max(p);
        }

        // All meshes share the same index and vertex buffer bindings.
        const _: () =
            assert!(MESH_ALIGN.is_multiple_of(VERTEX_STRIDE) && MESH_ALIGN.is_multiple_of(256));
        debug_assert_eq!(alloc.offset % VERTEX_STRIDE, 0);
        debug_assert_eq!((alloc.offset + index_start) % 4, 0);
        let vertex_offset = (alloc.offset / VERTEX_STRIDE) as i32;
        let first_index = ((alloc.offset + index_start) / 4) as u32;

        // Absolute first-index boundaries: bounds[0] = first_index, then a
        // running sum of bucket lengths. Monotonic by construction.
        let mut bounds = [first_index; 7];
        for dir in 0..6 {
            bounds[dir + 1] = bounds[dir] + data.buckets[dir].len() as u32;
        }
        debug_assert!(bounds.windows(2).all(|w| w[0] <= w[1]));
        debug_assert_eq!(bounds[6], first_index + total_indices as u32);

        let mesh = GpuMesh {
            alloc,
            bounds,
            vertex_offset,
            pass: data.pass,
            aabb_min,
            aabb_max,
        };

        let index = match self.free_slots.pop() {
            Some(i) => {
                self.slots[i as usize] = Some(mesh);
                i
            }
            None => {
                self.slots.push(Some(mesh));
                self.generations.push(0);
                (self.slots.len() - 1) as u32
            }
        };
        self.live_count += 1;

        Some(MeshHandle {
            index,
            generation: self.generations[index as usize],
        })
    }

    pub fn get(&self, handle: MeshHandle) -> Option<&GpuMesh> {
        if *self.generations.get(handle.index as usize)? != handle.generation {
            return None;
        }
        self.slots[handle.index as usize].as_ref()
    }

    pub fn free(&mut self, handle: MeshHandle, done_at: TimelineValue) {
        let Some(generation) = self.generations.get_mut(handle.index as usize) else {
            return;
        };
        if *generation != handle.generation {
            return;
        }
        if let Some(mesh) = self.slots[handle.index as usize].take() {
            *generation = generation.wrapping_add(1);
            self.free_slots.push(handle.index);
            self.retire.push(done_at, mesh.alloc);
            self.live_count -= 1;
        }
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
        unsafe {
            for copy in &self.pending {
                let region = vk::BufferCopy::default()
                    .src_offset(copy.staging.offset)
                    .dst_offset(copy.dst_offset)
                    .size(copy.size);
                device.cmd_copy_buffer(cmd, copy.staging.buffer, copy.dst_buffer, &[region]);
            }
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
        for copy in self.pending.drain(..) {
            self.retire.push(done_at, copy.staging);
        }
        true
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn has_garbage(&self) -> bool {
        !self.retire.is_empty()
    }

    /// Frees every retired allocation (GPU must be idle and copies flushed).
    pub unsafe fn collect_all(&mut self, allocator: &mut GpuAllocator) {
        self.retire
            .collect_all(|alloc| unsafe { allocator.free(alloc) });
    }

    /// Frees retired allocations based on the timeline's current value.
    pub unsafe fn collect(&mut self, allocator: &mut GpuAllocator, current: TimelineValue) {
        self.retire
            .collect(current, |alloc| unsafe { allocator.free(alloc) });
    }

    /// Frees all allocations (GPU must be idle).
    pub unsafe fn destroy_all(&mut self, allocator: &mut GpuAllocator) {
        unsafe {
            for slot in self.slots.iter_mut() {
                if let Some(mesh) = slot.take() {
                    allocator.free(mesh.alloc);
                }
            }
            self.retire.collect_all(|alloc| allocator.free(alloc));
            for copy in self.pending.drain(..) {
                allocator.free(copy.staging);
            }
        }
        self.live_count = 0;
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
    pub buffer: vk::Buffer,
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

/// One per-draw offsets-SSBO element: a camera-relative translation plus a
/// uniform scale, read per-vertex in the mesh vertex shader as
/// `world = local * scale + offset`. Naming `scale` (vs a bare `[f32; 4]` w)
/// keeps it from silently defaulting to zero. 16 bytes, `std430`-compatible.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawOffset {
    pub offset: [f32; 3],
    pub scale: f32,
}

/// `VkDrawIndexedIndirectCommand` as a Pod struct so a frame's command array
/// is one `cast_slice` write into the indirect [`HostBuffer`]. ash's
/// `vk::DrawIndexedIndirectCommand` is not `bytemuck::Pod`, hence this mirror;
/// the `const _` below pins its layout to ash's at compile time.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    /// Doubles as the draw's slot in the offsets SSBO: the vertex shader
    /// reads `draw_offsets[InstanceIndex]` and InstanceIndex starts at
    /// `first_instance` (instance_count is always 1).
    pub first_instance: u32,
}

// Layout must match `VkDrawIndexedIndirectCommand` field-for-field so the
// struct can be memcpy'd straight into the indirect buffer. Checked against
// ash's own type at compile time (stronger than a size-only runtime test:
// this also catches a transposed field).
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

/// Single push-descriptor set layout for the 3D mesh pipeline: binding 0 =
/// per-draw offsets SSBO (vertex stage), binding 1 = block texture array
/// (fragment stage). Both live in one set because Vulkan permits at most one
/// push-descriptor set per pipeline layout. No pool or set: the current
/// frame's buffer and texture are pushed at record time.
pub fn create_mesh3d_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let bindings = [
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
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
        .bindings(&bindings);
    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .expect("Failed to create mesh3d set layout")
    }
}

/// Pushes the per-draw offsets SSBO (binding 0) and block texture array
/// (binding 1) into set 0 of the bound 3D layout, in one call.
pub fn push_mesh3d_descriptors(
    push: &khr::push_descriptor::Device,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    offsets: vk::Buffer,
    tex_sampler: vk::Sampler,
    tex_view: vk::ImageView,
) {
    let buffer_infos = [vk::DescriptorBufferInfo::default()
        .buffer(offsets)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let image_infos = [vk::DescriptorImageInfo::default()
        .sampler(tex_sampler)
        .image_view(tex_view)
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
    ];
    unsafe {
        push.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &writes);
    }
}

/// Shrink a buffer if it's oversized relative to usage.
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
    use super::{IMM_MIN_CAPACITY, RetireQueue, shrink_capacity};

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
