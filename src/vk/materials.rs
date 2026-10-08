//! Fixed-capacity device-local SSBO of [`MaterialDesc`] (16384 × 16 B).
//!
//! Init uploads the array-layer default table (blocking, like the placeholder
//! block texture). Runtime [`Self::queue_set`] / [`Self::queue_append`] copy
//! through the transfer lane with no idle wait; the sampled buffer is never
//! reallocated.

use ash::vk;

use super::alloc::{create_buffer, create_filled_staging};
use super::mesh_residency::CopyConsumer;
use super::timeline::{Timeline, TimelineValue};
use super::transfer::{
    TransferCtx, TransferLane, UploadRetire, buffer_upload_barriers, cmd_buffer_barrier,
    upload_buffer_range,
};
use crate::material::{
    MATERIAL_DESC_CAPACITY, MaterialDesc, default_material_table, write_material_append,
    write_material_set,
};

/// Stages that first read the material table (mesh3d fragment).
pub(crate) const MATERIAL_CONSUMER_STAGES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::FRAGMENT_SHADER;

/// Copy-barrier consumer of the table: fragment-shader storage reads.
const MATERIAL_CONSUMER: CopyConsumer = (
    MATERIAL_CONSUMER_STAGES,
    vk::AccessFlags2::SHADER_STORAGE_READ,
);

const ENTRY_BYTES: u64 = size_of::<MaterialDesc>() as u64;
const TABLE_BYTES: u64 = MATERIAL_DESC_CAPACITY as u64 * ENTRY_BYTES;

pub(crate) struct MaterialTable {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    cpu: Box<[MaterialDesc]>,
    used: usize,
    pending_lo: u32,
    pending_hi: u32,
    retire: UploadRetire,
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
            retire: UploadRetire::new(command_pool, "material-desc"),
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
        self.retire.has_garbage()
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
    pub unsafe fn flush(&mut self, ctx: &mut TransferCtx<'_>) -> Option<TimelineValue> {
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
        let staging = create_filled_staging(
            ctx.device,
            &ctx.memory_props(),
            bytes,
            "material desc staging",
        );
        // Earlier frames read the table, and the frame loop has submitted
        // them (`has_overwrite_pending`): every flush is an overwrite.
        let dst = (self.buffer, offset, size);
        unsafe { upload_buffer_range(ctx, &mut self.retire, dst, staging, MATERIAL_CONSUMER, true) }
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
        let (staging, staging_mem) =
            create_filled_staging(device, &memory_props, bytes, "material desc staging");
        let separate_queue = lane.is_separate_queue();
        let buffer = self.buffer;
        // A first upload: graphics has not read the table yet.
        let barriers = buffer_upload_barriers(
            (buffer, 0, TABLE_BYTES),
            MATERIAL_CONSUMER,
            false,
            lane.tier(),
            graphics_family,
            lane.family(),
        );

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
            cmd_buffer_barrier(device, copy_cmd, barriers.after_copy.as_ref());
        }
        match batch {
            Batch::Lane(batch) => {
                let value = unsafe { lane.submit(device, batch) };
                if let Some(acquire) = barriers.acquire.as_ref() {
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
                        // Chains on the `ALL_COMMANDS` lane wait below.
                        cmd_buffer_barrier(device, acquire_cmd, Some(acquire));
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
        unsafe { self.retire.collect(device, current) };
    }

    pub unsafe fn collect_transfer(&mut self, device: &ash::Device, current: TimelineValue) {
        unsafe { self.retire.collect_transfer(device, current) };
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.retire.destroy(device);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::MATERIAL_FLAG_PROCEDURAL;

    #[test]
    fn copy_consumer_is_fragment_storage_reads() {
        assert_eq!(
            MATERIAL_CONSUMER,
            (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )
        );
    }

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
