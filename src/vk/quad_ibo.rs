use ash::vk;

use super::alloc::find_memory_type;
use super::mesh_residency::{CopyBarrier, copy_barrier};
use super::retire::RetireQueue;
use super::timeline::TimelineValue;
use super::transfer::TransferLane;

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
