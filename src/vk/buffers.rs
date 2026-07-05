/// GPU mesh registry and per-frame immediate-geometry buffers.
///
/// Meshes live in device-local memory suballocated from `GpuAllocator`
/// blocks: one allocation per mesh holding `[vertices][pad][indices]`. On
/// unified-memory devices uploads are direct memcpys; otherwise they go
/// through a staging allocation and a `cmd_copy_buffer` recorded at the next
/// frame's start (so a mesh uploaded mid-update is drawable the same frame).
/// Frees are deferred until the GPU provably finished the last frame that
/// could have referenced the mesh.
use ash::vk;
use glam::Vec3;

use super::alloc::{Allocation, GpuAllocator};
use super::targets::find_memory_type;
use crate::mesh::{MeshData, MeshHandle};

/// Suballocation alignment: lcm(256, 24) so every mesh's vertex data starts
/// at a whole multiple of the 24-byte vertex stride. That makes
/// `vertex_offset = byte_offset / 24` exact, which is what lets the renderer
/// bind each arena buffer ONCE at offset 0 and address meshes purely through
/// `VkDrawIndexedIndirectCommand::vertexOffset` (batched indirect draws).
/// Costs ~384 B average padding per mesh — negligible against multi-KB
/// chunk meshes.
const MESH_ALIGN: u64 = 768;
pub const FRAMES_IN_FLIGHT: u64 = 2;

/// Vertex stride shared by the mesh pipelines (must divide [`MESH_ALIGN`]).
const VERTEX_STRIDE: u64 = std::mem::size_of::<crate::mesh::Vertex>() as u64;

pub struct GpuMesh {
    alloc: Allocation,
    pub index_count: u32,
    /// Absolute u32 index into the block buffer (index buffer bound at 0).
    pub first_index: u32,
    /// This mesh's first vertex, in vertices, from the start of the block
    /// buffer (vertex buffer bound at 0; exact because MESH_ALIGN is a
    /// multiple of the vertex stride). Goes into the indirect command's
    /// `vertex_offset`.
    pub vertex_offset: i32,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
}

impl GpuMesh {
    pub fn buffer(&self) -> vk::Buffer {
        self.alloc.buffer
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
    retire: std::collections::VecDeque<(u64, Allocation)>,
    pub live_count: usize,
}

impl MeshRegistry {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            generations: Vec::new(),
            free_slots: Vec::new(),
            pending: Vec::new(),
            retire: std::collections::VecDeque::new(),
            live_count: 0,
        }
    }

    /// Uploads mesh data; `frame_no` is the frame about to be submitted.
    pub unsafe fn upload(
        &mut self,
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        data: &MeshData,
        frame_no: u64,
    ) -> Option<MeshHandle> {
        if data.indices.is_empty() || data.vertices.is_empty() {
            return None;
        }

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&data.vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&data.indices);
        // Index data starts 4-byte aligned right after the vertices.
        let index_start = (vertex_bytes.len() as u64).next_multiple_of(4);
        let total = index_start + index_bytes.len() as u64;

        let alloc = unsafe { allocator.alloc_device(device, total, MESH_ALIGN) }
            .map_err(|err| log::error!("mesh allocation failed: {err:?}"))
            .ok()?;

        if let Some(mapped) = alloc.mapped {
            // Unified memory: write straight into the device-local block.
            unsafe {
                let dst = mapped.as_ptr();
                std::ptr::copy_nonoverlapping(vertex_bytes.as_ptr(), dst, vertex_bytes.len());
                std::ptr::copy_nonoverlapping(
                    index_bytes.as_ptr(),
                    dst.add(index_start as usize),
                    index_bytes.len(),
                );
            }
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
            unsafe {
                let dst = mapped.as_ptr();
                std::ptr::copy_nonoverlapping(vertex_bytes.as_ptr(), dst, vertex_bytes.len());
                std::ptr::copy_nonoverlapping(
                    index_bytes.as_ptr(),
                    dst.add(index_start as usize),
                    index_bytes.len(),
                );
            }
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
            let p = Vec3::from_array(v.pos);
            aabb_min = aabb_min.min(p);
            aabb_max = aabb_max.max(p);
        }

        // Suballocation offsets are 768-aligned = a multiple of BOTH the
        // 4-byte index size and the 24-byte vertex stride, so every mesh in
        // a block shares one index-buffer bind at 0 (absolute first_index)
        // AND one vertex-buffer bind at 0 (exact vertex_offset in vertices).
        const _: () =
            assert!(MESH_ALIGN.is_multiple_of(VERTEX_STRIDE) && MESH_ALIGN.is_multiple_of(256));
        debug_assert_eq!(alloc.offset % VERTEX_STRIDE, 0);
        debug_assert_eq!((alloc.offset + index_start) % 4, 0);
        let vertex_offset = (alloc.offset / VERTEX_STRIDE) as i32;
        let first_index = ((alloc.offset + index_start) / 4) as u32;

        let mesh = GpuMesh {
            alloc,
            index_count: data.indices.len() as u32,
            first_index,
            vertex_offset,
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
        let _ = frame_no;

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

    pub fn free(&mut self, handle: MeshHandle, frame_no: u64) {
        let Some(generation) = self.generations.get_mut(handle.index as usize) else {
            return;
        };
        if *generation != handle.generation {
            return;
        }
        if let Some(mesh) = self.slots[handle.index as usize].take() {
            *generation = generation.wrapping_add(1);
            self.free_slots.push(handle.index);
            self.retire.push_back((frame_no, mesh.alloc));
            self.live_count -= 1;
        }
    }

    /// Records staged uploads into `cmd`. Returns true if a barrier guarding
    /// transfer -> vertex/index reads was emitted.
    pub unsafe fn flush_copies(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_no: u64,
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
            self.retire.push_back((frame_no, copy.staging));
        }
        true
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn has_garbage(&self) -> bool {
        !self.retire.is_empty()
    }

    /// Frees every retired allocation. Only valid when the GPU is provably
    /// idle for all our submissions AND no pending copy still targets a
    /// retired region (i.e. after flushing copies).
    pub unsafe fn collect_all(&mut self, allocator: &mut GpuAllocator) {
        for (_, alloc) in self.retire.drain(..) {
            unsafe { allocator.free(alloc) };
        }
    }

    /// Frees retired allocations whose last possible GPU use has completed.
    /// Call after waiting the frame slot's fence.
    pub unsafe fn collect(&mut self, allocator: &mut GpuAllocator, frame_no: u64) {
        while let Some((stamp, _)) = self.retire.front() {
            if stamp + FRAMES_IN_FLIGHT > frame_no {
                break;
            }
            let (_, alloc) = self.retire.pop_front().unwrap();
            unsafe { allocator.free(alloc) };
        }
    }

    /// Returns every allocation to the allocator. GPU must be idle.
    pub unsafe fn destroy_all(&mut self, allocator: &mut GpuAllocator) {
        unsafe {
            for slot in self.slots.iter_mut() {
                if let Some(mesh) = slot.take() {
                    allocator.free(mesh.alloc);
                }
            }
            for (_, alloc) in self.retire.drain(..) {
                allocator.free(alloc);
            }
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

/// A growable host-visible buffer written fresh every frame, one per
/// frame-in-flight: immediate geometry (VERTEX_BUFFER), per-draw offsets
/// (STORAGE_BUFFER), indirect commands (INDIRECT_BUFFER). Growing (or the
/// decay shrink) destroys the old buffer immediately — safe because the
/// owning frame slot's fence has already been waited when the buffer is
/// written.
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

    /// Per-frame capacity guarantee plus a gentle decay, so a one-off burst
    /// (a menu full of text) doesn't pin a huge buffer forever: when the
    /// capacity exceeded 4x the high-water mark of the last
    /// [`IMM_SHRINK_WINDOW`] frames, the buffer is recreated at 2x that mark.
    /// Steady-state cost: one compare, one increment, one compare.
    ///
    /// Must be called at the point where the owning frame slot's fence has
    /// just been waited (the GPU no longer reads this buffer), because both
    /// growth and decay destroy the old buffer immediately.
    ///
    /// Returns `true` when the `buffer` handle changed (created, recreated,
    /// or destroyed) so callers holding descriptors pointing at it can
    /// rewrite them. Handle-value comparison would be wrong here: Vulkan may
    /// reuse a destroyed handle's value for the next buffer.
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
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(find_memory_type(
                    instance,
                    physical,
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
        debug_assert!(offset + bytes.len() as u64 <= self.capacity);
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

/// `VkDrawIndexedIndirectCommand` as a Pod struct (identical 20-byte layout,
/// asserted in tests) so a frame's command array is one `cast_slice` write
/// into the indirect [`HostBuffer`].
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

/// Descriptor machinery for the per-draw offsets SSBO: one set layout
/// (binding 0, storage buffer, vertex stage) and one set per frame-in-flight
/// slot, each pointing at that slot's offsets [`HostBuffer`]. Sets are
/// rewritten by [`write_offsets_descriptor`] whenever the underlying buffer
/// is recreated (growth/decay) — safe at the maintain() point because the
/// slot's fence was just waited, so no pending command buffer uses the set.
pub fn create_offsets_descriptors(
    device: &ash::Device,
    count: u32,
) -> (
    vk::DescriptorSetLayout,
    vk::DescriptorPool,
    Vec<vk::DescriptorSet>,
) {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .expect("Failed to create offsets set layout")
    };

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(count)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(count)
        .pool_sizes(&pool_sizes);
    let pool = unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .expect("Failed to create offsets descriptor pool")
    };

    let layouts = vec![set_layout; count as usize];
    let set_alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe {
        device
            .allocate_descriptor_sets(&set_alloc)
            .expect("Failed to allocate offsets descriptor sets")
    };

    (set_layout, pool, sets)
}

/// Points `set` at `buffer` (whole range). Only call when no in-flight
/// command buffer references the set (post-fence, same rule as the buffer
/// recreation that triggers this).
pub fn write_offsets_descriptor(device: &ash::Device, set: vk::DescriptorSet, buffer: vk::Buffer) {
    let buffer_infos = [vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&buffer_infos)];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

/// Decay rule for [`HostBuffer::maintain`]: given the current capacity and
/// the window's high-water mark, returns the capacity to recreate at
/// (0 = destroy only; the buffer went unused all window), or `None` to keep
/// the buffer as is.
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
    use super::{DrawIndexedIndirect, IMM_MIN_CAPACITY, MESH_ALIGN, shrink_capacity};

    /// The struct is memcpy'd into an INDIRECT_BUFFER: its layout must match
    /// VkDrawIndexedIndirectCommand exactly (20 bytes, field-for-field).
    #[test]
    fn indirect_command_matches_vulkan_layout() {
        assert_eq!(std::mem::size_of::<DrawIndexedIndirect>(), 20);
        assert_eq!(std::mem::offset_of!(DrawIndexedIndirect, index_count), 0);
        assert_eq!(std::mem::offset_of!(DrawIndexedIndirect, instance_count), 4);
        assert_eq!(std::mem::offset_of!(DrawIndexedIndirect, first_index), 8);
        assert_eq!(std::mem::offset_of!(DrawIndexedIndirect, vertex_offset), 12);
        assert_eq!(
            std::mem::offset_of!(DrawIndexedIndirect, first_instance),
            16
        );
        let reference = ash::vk::DrawIndexedIndirectCommand::default();
        assert_eq!(
            std::mem::size_of_val(&reference),
            std::mem::size_of::<DrawIndexedIndirect>()
        );
    }

    /// MESH_ALIGN must keep suballocated vertex data stride-aligned (exact
    /// indirect vertex_offset) and index data 4-aligned (absolute
    /// first_index).
    #[test]
    fn mesh_align_is_lcm_of_stride_and_256() {
        assert_eq!(
            MESH_ALIGN % std::mem::size_of::<crate::mesh::Vertex>() as u64,
            0
        );
        assert_eq!(MESH_ALIGN % 256, 0);
        assert_eq!(MESH_ALIGN, 768);
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
