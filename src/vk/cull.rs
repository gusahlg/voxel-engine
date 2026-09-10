//! GPU draw-command emission: cull compute shader + indirect-count.
//! One dispatch per frame frustum-tests each mesh and appends commands
//! per-(camera-group, arena, distance-bucket) and per-(cascade, arena) for
//! shadows. Blend uses CPU path; immediates untouched. A small live camera-group
//! count skips the dispatch and writes the same commands on the host.
//!
//! Camera groups (bucketed): full-res Opaque, Cutout, coarse-LOD Opaque
//! (`scale > 1`). The LOD split exists so full-res opaque draws bind a
//! fragment module with no `discard` (early depth write) while only the LOD
//! partition pays for the slab clip. Coarse-LOD meshes whose camera-relative
//! AABB lies entirely inside that slab are not emitted (every fragment would
//! be discarded). Shadow Near/Far stay unbucketed and reuse the full-res
//! Opaque live count.

use ash::vk;

use super::alloc::{GpuCpuReadback, find_memory_type};
use super::buffers::{FRAMES_IN_FLIGHT, HostBuffer, MeshRecord, RecordBuffers};
use super::cull_math::{
    CpuCullScratch, FLAG_STATS, STATS_BYTES, STATS_COUNT, WORKGROUP, cpu_cull_into, cpu_cull_max,
};
use super::pass;
use crate::camera::Frustum;

pub(crate) use super::arena::{ArenaDirectory, MeshAabb};
pub(crate) use super::cull_math::{
    BUCKETS, CMD_STRIDE, Group, MAX_FACE_RUNS, PartitionGpu, camera_part, group_indirect_calls,
    shadow_part,
};

const SLOTS: usize = FRAMES_IN_FLIGHT as usize;

static CULL_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cull.comp.spv"));
static CULL_COMP_WAVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cull_wave.comp.spv"));

/// GPU CullParams struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullParamsGpu {
    cam_planes: [[f32; 4]; 5],
    shadow_planes: [[f32; 4]; 10],
    cam_block: [i32; 3],
    slot_count: u32,
    cam_frac: [f32; 3],
    arena_count: u32,
    shadow_enabled: u32,
    flags: u32,
    clip: f32,
    clip_v: f32,
}
// std140: clip/clip_v occupy the former pad tail (same 288-byte size).
const _: () = assert!(size_of::<CullParamsGpu>() == 288);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, cam_planes) == 0);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, shadow_planes) == 80);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, cam_block) == 240);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, slot_count) == 252);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, cam_frac) == 256);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, arena_count) == 268);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, shadow_enabled) == 272);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, flags) == 276);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, clip) == 280);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, clip_v) == 284);

/// Device-local grow-only buffer for GPU scratch.
struct DeviceBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: u64,
    usage: vk::BufferUsageFlags,
}

impl DeviceBuffer {
    fn new(usage: vk::BufferUsageFlags) -> Self {
        Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            capacity: 0,
            usage,
        }
    }

    fn bound(&self) -> Option<vk::Buffer> {
        (self.buffer != vk::Buffer::null()).then_some(self.buffer)
    }

    /// Grow to at least `needed` bytes. Safe after fence is waited.
    unsafe fn ensure(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        needed: u64,
    ) {
        if needed <= self.capacity {
            return;
        }
        unsafe {
            self.destroy(device);
            let capacity = needed.next_power_of_two().max(4096);
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(capacity)
                        .usage(self.usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .expect("create cull buffer");
            let reqs = device.get_buffer_memory_requirements(buffer);
            let memory_props = instance.get_physical_device_memory_properties(physical);
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(find_memory_type(
                            &memory_props,
                            reqs.memory_type_bits,
                            vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        )),
                    None,
                )
                .expect("allocate cull buffer memory");
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind cull buffer memory");
            self.buffer = buffer;
            self.memory = memory;
            self.capacity = capacity;
        }
    }

    unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.buffer != vk::Buffer::null() {
            unsafe {
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.memory, None);
            }
            self.buffer = vk::Buffer::null();
            self.memory = vk::DeviceMemory::null();
            self.capacity = 0;
        }
    }
}

/// Per-frame cull result (commands, counts, partition table).
pub(crate) struct CullFrame {
    pub commands: vk::Buffer,
    pub counts: vk::Buffer,
    pub partitions: Vec<PartitionGpu>,
    pub arena_count: usize,
    pub slot_count: u32,
    /// Host-written commands/counts; [`CullState::record`] is a no-op.
    pub cpu: bool,
}

pub(crate) struct CullState {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    params: [HostBuffer; SLOTS],
    parts: [HostBuffer; SLOTS],
    visible: [HostBuffer; SLOTS],
    commands: [DeviceBuffer; SLOTS],
    counts: [DeviceBuffer; SLOTS],
    cpu_cmds: [HostBuffer; SLOTS],
    cpu_counts: [HostBuffer; SLOTS],
    stats: [StatsReadback; SLOTS],
    /// Recycled partition table when [`Self::prepare`] returns `None`, so a
    /// frame with nothing to cull does not drop last frame's allocation.
    spare_parts: Vec<PartitionGpu>,
    /// CPU-side command/count staging reused across frames (capacity retained).
    cpu_scratch: CpuCullScratch,
    /// Per-direction face-run culling. On by default; follows
    /// [`crate::Engine::set_cull_faces`].
    face_cull: bool,
}

/// Host-visible copy of the per-slot geometry histogram, fence-safe to read
/// after the slot's timeline wait. Same mechanism as the VRS mix buffer.
struct StatsReadback(GpuCpuReadback);

impl std::ops::Deref for StatsReadback {
    type Target = GpuCpuReadback;
    fn deref(&self) -> &GpuCpuReadback {
        &self.0
    }
}

impl CullState {
    pub fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        cache: vk::PipelineCache,
        wave_atomics: bool,
    ) -> Self {
        // Bindings match cull.comp.slang.
        let storage = |binding: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let bindings = [
            storage(0),
            storage(1),
            storage(2),
            storage(3),
            storage(4),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            storage(6),
            storage(7),
        ];
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(&bindings),
                    None,
                )
                .expect("create cull set layout")
        };
        let set_layouts = [set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(4)];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .expect("create cull pipeline layout")
        };
        let bytes = if wave_atomics {
            CULL_COMP_WAVE
        } else {
            CULL_COMP
        };
        let pipeline = pass::compute_pipeline(
            device,
            cache,
            layout,
            bytes,
            if wave_atomics { "cull-wave" } else { "cull" },
        );
        Self {
            set_layout,
            layout,
            pipeline,
            params: std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::UNIFORM_BUFFER)),
            parts: std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER)),
            visible: std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER)),
            commands: std::array::from_fn(|_| {
                DeviceBuffer::new(
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                )
            }),
            counts: std::array::from_fn(|_| {
                DeviceBuffer::new(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
            }),
            cpu_cmds: std::array::from_fn(|_| {
                HostBuffer::new(vk::BufferUsageFlags::INDIRECT_BUFFER)
            }),
            cpu_counts: std::array::from_fn(|_| {
                HostBuffer::new(vk::BufferUsageFlags::INDIRECT_BUFFER)
            }),
            stats: std::array::from_fn(|_| StatsReadback::new(device, memory_props)),
            spare_parts: Vec::new(),
            cpu_scratch: CpuCullScratch::default(),
            face_cull: true,
        }
    }

    /// Applied between frames; [`Self::prepare`] snapshots the value so a
    /// toggle cannot size partitions for one run count and advertise the other.
    pub fn set_face_cull(&mut self, on: bool) {
        self.face_cull = on;
    }

    /// Last completed histogram for `slot`: `[draws0, idx0, draws1, idx1, draws2, idx2]`.
    pub fn stats(&self, slot: usize) -> [u32; STATS_COUNT] {
        unsafe {
            let p = self.stats[slot].mapped;
            std::array::from_fn(|i| *p.add(i))
        }
    }

    /// CPU-zero the mapped histogram so a skipped cull publishes zeros.
    pub fn clear_stats_cpu(&self, slot: usize) {
        unsafe { std::ptr::write_bytes(self.stats[slot].mapped, 0, STATS_COUNT) };
    }

    /// Prepare buffers and params for cull dispatch. Safe after fence is waited.
    ///
    /// `slot_count` bounds the dispatch (the caller trims it to the directory's
    /// live end); `visible` must cover it. `partitions` is last frame's table
    /// handed back for reuse (its contents are discarded). `clip` / `clip_v`
    /// are the full-res slab extents (`DrawLists::lod_clip`, `lod_clip_v`);
    /// 0 disables, matching the mesh3d push constants.
    ///
    /// When the directory's camera-group live count is at most `CPU_CULL_MAX`
    /// (or `VOXEL_CPU_CULL_MAX`), commands and counts are written to host-visible
    /// buffers here and no compute work is recorded.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare(
        &mut self,
        slot: usize,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        dir: &mut ArenaDirectory,
        records: RecordBuffers,
        host_records: &[MeshRecord],
        is_arrived: impl Fn(u32) -> bool,
        slot_count: u32,
        camera: &Frustum,
        shadow: Option<&[Frustum; 2]>,
        eye: super::pipeline::EyeSplit,
        clip: f32,
        clip_v: f32,
        visible: &[u32],
        mut partitions: Vec<PartitionGpu>,
    ) -> Option<CullFrame> {
        debug_assert!(slot_count <= records.slots, "slot_count exceeds the table");
        debug_assert!(visible.len() >= slot_count.div_ceil(32) as usize);
        if partitions.capacity() == 0 {
            partitions = std::mem::take(&mut self.spare_parts);
        }
        // One snapshot for both partition capacity and CullParams.flags so a
        // mid-frame toggle cannot size runs for one value and advertise the other.
        let face_cull = self.face_cull;
        let runs_per_mesh = if face_cull { MAX_FACE_RUNS } else { 1 };
        let total = dir.partitions_into(&mut partitions, runs_per_mesh, Some(eye));
        if partitions.is_empty() || total == 0 {
            self.spare_parts = partitions;
            return None;
        }
        if dir.camera_live() <= cpu_cull_max() {
            let stats_hist = cpu_cull_into(
                host_records,
                dir,
                is_arrived,
                visible,
                &partitions,
                camera,
                shadow,
                eye,
                slot_count,
                clip,
                clip_v,
                face_cull,
                &mut self.cpu_scratch,
            );
            unsafe {
                let cb = &mut self.cpu_cmds[slot];
                cb.maintain(instance, device, physical, u64::from(total) * CMD_STRIDE);
                let nb = &mut self.cpu_counts[slot];
                nb.maintain(instance, device, physical, (partitions.len() * 4) as u64);
                // One contiguous copy per live partition into write-combined
                // memory; never a scattered per-slot store.
                for (i, p) in partitions.iter().enumerate() {
                    let src = &self.cpu_scratch.part_cmds[i];
                    if src.is_empty() {
                        continue;
                    }
                    cb.write(u64::from(p.offset) * CMD_STRIDE, bytemuck::cast_slice(src));
                }
                nb.write(0, bytemuck::cast_slice(&self.cpu_scratch.counts));
                std::ptr::copy_nonoverlapping(
                    stats_hist.as_ptr(),
                    self.stats[slot].mapped,
                    STATS_COUNT,
                );
            }
            return Some(CullFrame {
                commands: self.cpu_cmds[slot].bound()?,
                counts: self.cpu_counts[slot].bound()?,
                partitions,
                arena_count: dir.arena_count(),
                slot_count,
                cpu: true,
            });
        }
        let mut params = CullParamsGpu {
            cam_planes: camera.planes().map(|p| p.to_array()),
            shadow_planes: [[0.0; 4]; 10],
            cam_block: eye.block,
            slot_count,
            cam_frac: eye.frac,
            arena_count: dir.arena_count() as u32,
            shadow_enabled: shadow.is_some() as u32,
            flags: u32::from(face_cull),
            clip,
            clip_v,
        };
        if let Some(frusta) = shadow {
            for (c, f) in frusta.iter().enumerate() {
                for (p, plane) in f.planes().iter().enumerate() {
                    params.shadow_planes[c * 5 + p] = plane.to_array();
                }
            }
        }
        let part_bytes: &[u8] = bytemuck::cast_slice(&partitions);
        unsafe {
            let pb = &mut self.params[slot];
            pb.maintain(
                instance,
                device,
                physical,
                size_of::<CullParamsGpu>() as u64,
            );
            pb.write(0, bytemuck::bytes_of(&params));
            let tb = &mut self.parts[slot];
            tb.maintain(instance, device, physical, part_bytes.len() as u64);
            tb.write(0, part_bytes);
            let vis_bytes: &[u8] = bytemuck::cast_slice(visible);
            let vb = &mut self.visible[slot];
            vb.maintain(instance, device, physical, vis_bytes.len() as u64);
            vb.write(0, vis_bytes);
            self.commands[slot].ensure(instance, device, physical, u64::from(total) * CMD_STRIDE);
            self.counts[slot].ensure(instance, device, physical, (partitions.len() * 4) as u64);
        }
        Some(CullFrame {
            commands: self.commands[slot].bound()?,
            counts: self.counts[slot].bound()?,
            partitions,
            arena_count: dir.arena_count(),
            slot_count,
            cpu: false,
        })
    }

    /// Record cull dispatch (zeros counts, executes cull, fences writes).
    /// Geometry stats fill/atomics/copy run only while profiling.
    /// A CPU-culled frame records no compute work: commands and counts were
    /// written to host-coherent memory in [`Self::prepare`] after the slot wait.
    /// Returns whether this recorded GPU commands (fill/dispatch/barrier).
    pub unsafe fn record(
        &self,
        device: &ash::Device,
        push: &ash::khr::push_descriptor::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        records: RecordBuffers,
        frame: &CullFrame,
    ) -> bool {
        if frame.cpu {
            return false;
        }
        let stats = crate::profile::is_enabled();
        unsafe {
            device.cmd_fill_buffer(cmd, frame.counts, 0, vk::WHOLE_SIZE, 0);
            if stats {
                device.cmd_fill_buffer(cmd, self.stats[slot].gpu, 0, STATS_BYTES, 0);
            }
            // CLEAR / TRANSFER_WRITE → COMPUTE / SHADER_STORAGE_{READ,WRITE}
            // covers the counts fill and, when profiling, the stats fill.
            let to_compute = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                )];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_compute),
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            let info = |buffer: vk::Buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            };
            // Binding order matches cull.comp.slang.
            let infos = [
                info(records.records),
                info(records.arenas),
                info(
                    self.parts[slot]
                        .bound()
                        .expect("partitions were just written"),
                ),
                info(frame.commands),
                info(frame.counts),
                info(self.params[slot].bound().expect("params were just written")),
                info(
                    self.visible[slot]
                        .bound()
                        .expect("visibility was just written"),
                ),
                info(self.stats[slot].gpu),
            ];
            let writes: [vk::WriteDescriptorSet; 8] = std::array::from_fn(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_type(if i == 5 {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    })
                    .buffer_info(std::slice::from_ref(&infos[i]))
            });
            push.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &writes,
            );
            let flags = if stats { FLAG_STATS } else { 0 };
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&flags),
            );
            device.cmd_dispatch(cmd, frame.slot_count.div_ceil(WORKGROUP), 1, 1);

            // COMPUTE / SHADER_STORAGE_WRITE → DRAW_INDIRECT / INDIRECT_COMMAND_READ,
            // and when profiling also COPY / TRANSFER_READ for the stats copy.
            let mut dst_stage = vk::PipelineStageFlags2::DRAW_INDIRECT;
            let mut dst_access = vk::AccessFlags2::INDIRECT_COMMAND_READ;
            if stats {
                dst_stage |= vk::PipelineStageFlags2::COPY;
                dst_access |= vk::AccessFlags2::TRANSFER_READ;
            }
            let to_draws = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_draws),
            );
            if stats {
                device.cmd_copy_buffer(
                    cmd,
                    self.stats[slot].gpu,
                    self.stats[slot].cpu,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: STATS_BYTES,
                    }],
                );
                // COPY / TRANSFER_WRITE → HOST / HOST_READ. Mapped read is after
                // this slot's timeline wait (one cycle later).
                let copy_to_host = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&copy_to_host),
                );
            }
        }
        true
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            for b in self
                .params
                .iter_mut()
                .chain(&mut self.parts)
                .chain(&mut self.visible)
                .chain(&mut self.cpu_cmds)
                .chain(&mut self.cpu_counts)
            {
                b.destroy(device);
            }
            for b in self.commands.iter_mut().chain(&mut self.counts) {
                b.destroy(device);
            }
            for s in &self.stats {
                s.destroy(device);
            }
        }
    }
}

impl StatsReadback {
    fn new(device: &ash::Device, memory_props: &vk::PhysicalDeviceMemoryProperties) -> Self {
        Self(GpuCpuReadback::new(
            device,
            memory_props,
            STATS_COUNT,
            "cull stats buffer",
            "cull stats readback",
        ))
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { self.0.destroy(device) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_histogram_is_two_u32s_per_camera_group() {
        assert_eq!(STATS_COUNT, 6);
        assert_eq!(STATS_BYTES, 24);
        assert_eq!(FLAG_STATS, 1);
    }

    #[test]
    fn cull_params_std140_tail_is_slab_extents() {
        assert_eq!(size_of::<CullParamsGpu>(), 288);
        assert_eq!(std::mem::offset_of!(CullParamsGpu, flags), 276);
        assert_eq!(std::mem::offset_of!(CullParamsGpu, clip), 280);
        assert_eq!(std::mem::offset_of!(CullParamsGpu, clip_v), 284);
    }
}
