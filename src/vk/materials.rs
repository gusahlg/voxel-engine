//! Fixed-capacity device-local SSBO of [`MaterialDesc`] (16384 × 16 B).
//!
//! Init uploads the array-layer default table (blocking, like the placeholder
//! block texture). Runtime [`Self::queue_set`] / [`Self::queue_append`] copy
//! through the transfer lane with no idle wait; the sampled buffer is never
//! reallocated.

use ash::vk;

use super::alloc::create_buffer;
use super::buffers::RetireQueue;
use super::mesh_residency::{CopyBarrier, copy_barrier};
use super::timeline::{Timeline, TimelineValue};
use super::transfer::TransferLane;
use crate::material::{
    MATERIAL_DESC_CAPACITY, MaterialDesc, default_material_table, write_material_append,
    write_material_set,
};

/// Stages that first read the material table (mesh3d fragment).
pub(crate) const MATERIAL_CONSUMER_STAGES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::FRAGMENT_SHADER;

const ENTRY_BYTES: u64 = size_of::<MaterialDesc>() as u64;
const TABLE_BYTES: u64 = MATERIAL_DESC_CAPACITY as u64 * ENTRY_BYTES;

fn material_reads() -> vk::AccessFlags2 {
    vk::AccessFlags2::SHADER_STORAGE_READ
}

/// Shader-storage copy barrier: same pairings as [`copy_barrier`] but the
/// consumer is the fragment shader, not vertex input.
fn material_copy_barrier(
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
    role: CopyBarrier,
) -> vk::BufferMemoryBarrier2<'static> {
    let b = copy_barrier(buffer, offset, size, material_reads(), role);
    match role {
        CopyBarrier::Draw | CopyBarrier::Acquire { .. } => b
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(material_reads()),
        CopyBarrier::Release { .. } => b,
    }
}

pub(crate) struct MaterialTable {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    cpu: Box<[MaterialDesc]>,
    used: usize,
    pending_lo: u32,
    pending_hi: u32,
    command_pool: vk::CommandPool,
    staging_retire: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
    transfer_retire: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
    release_cmds: RetireQueue<(Timeline, vk::CommandBuffer)>,
}

impl MaterialTable {
    /// Device-local 16384-entry table, filled with [`MaterialDesc::ARRAY_LAYER`].
    /// Blocks until the copy completes (init-time, like the default block
    /// texture).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        command_pool: vk::CommandPool,
        lane: &mut TransferLane,
    ) -> Self {
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let (buffer, memory) = create_buffer(
            device,
            &memory_props,
            TABLE_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "material desc SSBO",
        );
        let cpu = default_material_table();
        let mut table = Self {
            buffer,
            memory,
            cpu,
            used: 0,
            pending_lo: 0,
            pending_hi: MATERIAL_DESC_CAPACITY as u32,
            command_pool,
            staging_retire: RetireQueue::new(),
            transfer_retire: RetireQueue::new(),
            release_cmds: RetireQueue::new(),
        };
        unsafe {
            table.upload_blocking(
                instance,
                device,
                physical,
                graphics_queue,
                graphics_family,
                command_pool,
                lane,
            );
        }
        table.pending_lo = 0;
        table.pending_hi = 0;
        table
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn used(&self) -> usize {
        self.used
    }

    /// Pending writes to a table the GPU may already be sampling.
    pub fn has_overwrite_pending(&self) -> bool {
        self.pending_lo < self.pending_hi
    }

    pub fn has_garbage(&self) -> bool {
        !self.staging_retire.is_empty()
            || !self.transfer_retire.is_empty()
            || !self.release_cmds.is_empty()
    }

    /// Replace the whole table (index = layer id). Unused tail returns to
    /// [`MaterialDesc::ARRAY_LAYER`].
    pub fn queue_set(&mut self, descs: &[MaterialDesc]) {
        if descs.len() > MATERIAL_DESC_CAPACITY {
            log::error!(
                "set_material_descs: {} entries exceeds the 14-bit layer cap of {MATERIAL_DESC_CAPACITY}; truncating",
                descs.len()
            );
        }
        let old_used = self.used;
        self.used = write_material_set(&mut self.cpu, descs);
        // Prefix plus any previously-used tail that must go back to default.
        self.mark_pending(0, old_used.max(self.used) as u32);
    }

    /// Append at the current used count. Excess past capacity is dropped.
    pub fn queue_append(&mut self, descs: &[MaterialDesc]) {
        if descs.is_empty() {
            return;
        }
        let room = MATERIAL_DESC_CAPACITY.saturating_sub(self.used);
        if descs.len() > room {
            log::error!(
                "append_material_descs: {} entries would exceed the 14-bit layer cap of {MATERIAL_DESC_CAPACITY}; truncating",
                self.used + descs.len()
            );
        }
        let start = self.used as u32;
        self.used = write_material_append(&mut self.cpu, self.used, descs);
        self.mark_pending(start, self.used as u32);
    }

    fn mark_pending(&mut self, lo: u32, hi: u32) {
        if lo >= hi {
            return;
        }
        if self.pending_lo >= self.pending_hi {
            self.pending_lo = lo;
            self.pending_hi = hi;
        } else {
            self.pending_lo = self.pending_lo.min(lo);
            self.pending_hi = self.pending_hi.max(hi);
        }
    }

    /// Record a pending range copy on the transfer lane (or the frame
    /// command buffer on `SameQueueFallback`). No host wait.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn flush(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        lane: &mut TransferLane,
        graphics_cmd: vk::CommandBuffer,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        graphics_timeline: &Timeline,
        last_render_value: TimelineValue,
        done_at: TimelineValue,
    ) -> Option<TimelineValue> {
        if self.pending_lo >= self.pending_hi {
            return None;
        }
        let lo = self.pending_lo;
        let hi = self.pending_hi;
        self.pending_lo = 0;
        self.pending_hi = 0;

        let offset = u64::from(lo) * ENTRY_BYTES;
        let size = u64::from(hi - lo) * ENTRY_BYTES;
        let bytes = bytemuck::cast_slice(&self.cpu[lo as usize..hi as usize]);
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let (staging, staging_mem) = fill_staging(device, &memory_props, bytes);

        let separate_queue = lane.is_separate_queue();
        let needs_qfot = lane.needs_ownership_transfer();
        let buffer = self.buffer;

        let extra_wait = if needs_qfot {
            Some(unsafe {
                self.submit_overwrite_release(
                    device,
                    graphics_queue,
                    graphics_family,
                    lane.family(),
                    done_at,
                    offset,
                    size,
                )
            })
        } else if separate_queue {
            Some((
                graphics_timeline.semaphore(),
                last_render_value,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
            ))
        } else {
            None
        };

        let lane_batch = separate_queue.then(|| unsafe { lane.begin(device) });
        let record_cmd = lane_batch.as_ref().map_or(graphics_cmd, |b| b.cmd());

        unsafe {
            if !separate_queue {
                let to_dst = [vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .buffer(buffer)
                    .offset(offset)
                    .size(size)];
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&to_dst),
                );
            } else if needs_qfot {
                let acquire = [vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_queue_family_index(graphics_family)
                    .dst_queue_family_index(lane.family())
                    .buffer(buffer)
                    .offset(offset)
                    .size(size)];
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&acquire),
                );
            }
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(offset)
                .size(size);
            device.cmd_copy_buffer(record_cmd, staging, buffer, &[region]);
        }

        if let Some(lane_batch) = lane_batch {
            let release = needs_qfot.then(|| {
                material_copy_barrier(
                    buffer,
                    offset,
                    size,
                    CopyBarrier::Release {
                        src_family: lane.family(),
                        dst_family: graphics_family,
                    },
                )
            });
            unsafe {
                if let Some(release) = release.as_ref() {
                    device.cmd_pipeline_barrier2(
                        record_cmd,
                        &vk::DependencyInfo::default()
                            .buffer_memory_barriers(std::slice::from_ref(release)),
                    );
                }
            }
            let value = unsafe { lane.submit_after(device, lane_batch, extra_wait) };
            if needs_qfot {
                let acquire = [material_copy_barrier(
                    buffer,
                    offset,
                    size,
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
            self.transfer_retire.push(value, (staging, staging_mem));
            Some(value)
        } else {
            let to_shader = [material_copy_barrier(
                buffer,
                offset,
                size,
                CopyBarrier::Draw,
            )];
            unsafe {
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&to_shader),
                );
            }
            self.staging_retire.push(done_at, (staging, staging_mem));
            None
        }
    }

    /// Dedicated-family overwrite: release the sampled range on graphics so
    /// the transfer queue can acquire it. Submitted (not host-waited) after
    /// in-flight frames that read the table.
    #[allow(clippy::too_many_arguments)]
    unsafe fn submit_overwrite_release(
        &mut self,
        device: &ash::Device,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        transfer_family: u32,
        done_at: TimelineValue,
        offset: u64,
        size: u64,
    ) -> (vk::Semaphore, TimelineValue, vk::PipelineStageFlags2) {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc)
                .expect("Failed to allocate material-desc release command buffer")[0]
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .expect("Failed to begin material-desc release command buffer");
            let release = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .src_queue_family_index(graphics_family)
                .dst_queue_family_index(transfer_family)
                .buffer(self.buffer)
                .offset(offset)
                .size(size)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&release),
            );
            device
                .end_command_buffer(cmd)
                .expect("Failed to end material-desc release command buffer");
        }
        let mut tmp = unsafe { Timeline::new(device) };
        let rs = tmp.begin_render(cmd);
        let completion = unsafe { rs.submit(device, graphics_queue, &tmp, None) };
        let sem = tmp.semaphore();
        let value = completion.value();
        self.release_cmds.push(done_at, (tmp, cmd));
        (sem, value, vk::PipelineStageFlags2::COPY)
    }

    /// Init-time full-table copy; host-waits the transfer (OnceBeforeUse).
    #[allow(clippy::too_many_arguments)]
    unsafe fn upload_blocking(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        command_pool: vk::CommandPool,
        lane: &mut TransferLane,
    ) {
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let bytes: &[u8] = bytemuck::cast_slice(&self.cpu);
        let (staging, staging_mem) = fill_staging(device, &memory_props, bytes);
        let separate_queue = lane.is_separate_queue();
        let needs_qfot = lane.needs_ownership_transfer();
        let buffer = self.buffer;

        enum Batch {
            Lane(super::transfer::LaneRecording),
            OneShot(vk::CommandBuffer),
        }
        impl Batch {
            fn cmd(&self) -> vk::CommandBuffer {
                match self {
                    Batch::Lane(b) => b.cmd(),
                    Batch::OneShot(c) => *c,
                }
            }
        }
        let batch = if separate_queue {
            Batch::Lane(unsafe { lane.begin(device) })
        } else {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe {
                device
                    .allocate_command_buffers(&alloc)
                    .expect("Failed to allocate material-desc upload command buffer")[0]
            };
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                device
                    .begin_command_buffer(cmd, &begin)
                    .expect("Failed to begin material-desc upload command buffer");
            }
            Batch::OneShot(cmd)
        };
        let copy_cmd = batch.cmd();
        unsafe {
            let region = vk::BufferCopy::default().size(TABLE_BYTES);
            device.cmd_copy_buffer(copy_cmd, staging, buffer, &[region]);
            if needs_qfot {
                let release = [material_copy_barrier(
                    buffer,
                    0,
                    TABLE_BYTES,
                    CopyBarrier::Release {
                        src_family: lane.family(),
                        dst_family: graphics_family,
                    },
                )];
                device.cmd_pipeline_barrier2(
                    copy_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&release),
                );
            } else if !separate_queue {
                let to_shader = [material_copy_barrier(
                    buffer,
                    0,
                    TABLE_BYTES,
                    CopyBarrier::Draw,
                )];
                device.cmd_pipeline_barrier2(
                    copy_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&to_shader),
                );
            }
        }
        match batch {
            Batch::Lane(batch) => {
                let value = unsafe { lane.submit(device, batch) };
                if needs_qfot {
                    let alloc = vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1);
                    let acquire_cmd = unsafe {
                        device
                            .allocate_command_buffers(&alloc)
                            .expect("Failed to allocate material-desc acquire command buffer")[0]
                    };
                    let begin = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    unsafe {
                        device
                            .begin_command_buffer(acquire_cmd, &begin)
                            .expect("Failed to begin material-desc acquire command buffer");
                        let acquire = [material_copy_barrier(
                            buffer,
                            0,
                            TABLE_BYTES,
                            CopyBarrier::Acquire {
                                src_family: lane.family(),
                                dst_family: graphics_family,
                            },
                        )];
                        device.cmd_pipeline_barrier2(
                            acquire_cmd,
                            &vk::DependencyInfo::default().buffer_memory_barriers(&acquire),
                        );
                        device
                            .end_command_buffer(acquire_cmd)
                            .expect("Failed to end material-desc acquire command buffer");
                    }
                    let mut done = unsafe { Timeline::new(device) };
                    let rs = done.begin_render(acquire_cmd);
                    let completion = unsafe {
                        rs.submit(
                            device,
                            graphics_queue,
                            &done,
                            Some((
                                lane.semaphore(),
                                value,
                                vk::PipelineStageFlags2::ALL_COMMANDS,
                            )),
                        )
                    };
                    unsafe { done.wait(device, completion.value()) };
                    unsafe { done.destroy(device) };
                    unsafe { device.free_command_buffers(command_pool, &[acquire_cmd]) };
                } else {
                    unsafe { lane.wait(device, value) };
                }
            }
            Batch::OneShot(copy_cmd) => unsafe {
                device
                    .end_command_buffer(copy_cmd)
                    .expect("Failed to end material-desc upload command buffer");
                let mut done = Timeline::new(device);
                let rs = done.begin_render(copy_cmd);
                let completion = rs.submit(device, graphics_queue, &done, None);
                done.wait(device, completion.value());
                done.destroy(device);
                device.free_command_buffers(command_pool, &[copy_cmd]);
            },
        }
        unsafe {
            device.destroy_buffer(staging, None);
            device.free_memory(staging_mem, None);
        }
    }

    pub unsafe fn collect(&mut self, device: &ash::Device, current: TimelineValue) {
        self.staging_retire
            .collect(current, |(buffer, memory)| unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
        let pool = self.command_pool;
        self.release_cmds
            .collect(current, |(timeline, cmd)| unsafe {
                timeline.destroy(device);
                device.free_command_buffers(pool, &[cmd]);
            });
    }

    pub unsafe fn collect_transfer(&mut self, device: &ash::Device, current: TimelineValue) {
        self.transfer_retire
            .collect(current, |(buffer, memory)| unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.staging_retire.collect_all(|(buffer, memory)| {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
            self.transfer_retire.collect_all(|(buffer, memory)| {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
            let pool = self.command_pool;
            self.release_cmds.collect_all(|(timeline, cmd)| {
                timeline.destroy(device);
                device.free_command_buffers(pool, &[cmd]);
            });
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn fill_staging(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    bytes: &[u8],
) -> (vk::Buffer, vk::DeviceMemory) {
    let size = (bytes.len() as u64).max(1);
    let (staging, memory) = create_buffer(
        device,
        memory_props,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        "material desc staging",
    );
    unsafe {
        let ptr = device
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            .expect("Failed to map material-desc staging memory");
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
        device.unmap_memory(memory);
    }
    (staging, memory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::MATERIAL_FLAG_PROCEDURAL;

    #[test]
    fn table_bytes_are_256_kib() {
        assert_eq!(TABLE_BYTES, 256 * 1024);
        assert_eq!(ENTRY_BYTES, 16);
    }

    #[test]
    fn queue_set_marks_prefix_and_resets_used() {
        let mut cpu = default_material_table();
        let d = MaterialDesc {
            flags: MATERIAL_FLAG_PROCEDURAL,
            ..MaterialDesc::ARRAY_LAYER
        };
        let used = write_material_set(&mut cpu, &[d, d, d]);
        assert_eq!(used, 3);
        let used = write_material_set(&mut cpu, &[d]);
        assert_eq!(used, 1);
        assert_eq!(cpu[1], MaterialDesc::ARRAY_LAYER);
    }
}
