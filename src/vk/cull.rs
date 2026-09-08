//! GPU draw-command emission: cull compute shader + indirect-count.
//! One dispatch per frame frustum-tests each mesh and appends commands
//! per-(pass, arena, distance-bucket) for camera groups and per-(cascade, arena)
//! for shadows. Blend uses CPU path; immediates untouched.

use std::num::NonZeroU32;

use ash::vk;

use super::alloc::find_memory_type;
use super::buffers::{FRAMES_IN_FLIGHT, HostBuffer, RecordBuffers};
use super::pass;
use crate::camera::Frustum;
use crate::mesh::Pass;

const SLOTS: usize = FRAMES_IN_FLIGHT as usize;
/// Emission groups: Opaque, Cutout, ShadowNear, ShadowFar.
pub(crate) const GROUPS: usize = 4;
pub(crate) const CAMERA_GROUPS: usize = 2;
pub(crate) const SHADOW_GROUPS: usize = 2;
/// Front-to-back buckets on camera groups only (shadows stay unbucketed).
pub(crate) const BUCKETS: usize = crate::genconst::CULL_DISTANCE_BUCKETS as usize;
/// Size of VkDrawIndexedIndirectCommand.
pub(crate) const CMD_STRIDE: u64 = 20;
const WORKGROUP: u32 = crate::genconst::CULL_WORKGROUP;

static CULL_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cull.comp.spv"));
static CULL_COMP_WAVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cull_wave.comp.spv"));

const _: () = assert!(crate::genconst::CULL_DISTANCE_BUCKETS == 4);
const _: () = assert!(GROUPS == CAMERA_GROUPS + SHADOW_GROUPS);

/// Camera-distance bucket of an AABB centre, matching `cull.comp.slang`.
#[cfg(test)]
fn distance_bucket(dist: f32) -> u32 {
    let mut b = 0u32;
    if dist >= crate::genconst::CULL_BUCKET_SPLIT_0 {
        b = 1;
    }
    if dist >= crate::genconst::CULL_BUCKET_SPLIT_1 {
        b = 2;
    }
    if dist >= crate::genconst::CULL_BUCKET_SPLIT_2 {
        b = 3;
    }
    b.min(crate::genconst::CULL_DISTANCE_BUCKETS - 1)
}

/// Partition index for a camera (pass, arena, bucket) triple.
pub(crate) fn camera_part(group: usize, arena: usize, bucket: usize, arena_count: usize) -> usize {
    debug_assert!(group < CAMERA_GROUPS);
    debug_assert!(bucket < BUCKETS);
    (group * arena_count + arena) * BUCKETS + bucket
}

/// Partition index for a shadow (cascade, arena) pair. Cascades are unbucketed.
pub(crate) fn shadow_part(cascade: usize, arena: usize, arena_count: usize) -> usize {
    debug_assert!(cascade < SHADOW_GROUPS);
    CAMERA_GROUPS * arena_count * BUCKETS + cascade * arena_count + arena
}

/// Partition table length for `arena_count` live arena rows.
pub(crate) fn partition_count(arena_count: usize) -> usize {
    arena_count * (CAMERA_GROUPS * BUCKETS + SHADOW_GROUPS)
}

/// GPU Partition struct.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PartitionGpu {
    pub offset: u32,
    pub capacity: u32,
}
const _: () = assert!(size_of::<PartitionGpu>() == 8);
const _: () = assert!(std::mem::offset_of!(PartitionGpu, offset) == 0);
const _: () = assert!(std::mem::offset_of!(PartitionGpu, capacity) == 4);

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
    _pad: [u32; 3],
}
// Padding ensures alignment matches shader layout.
const _: () = assert!(size_of::<CullParamsGpu>() == 288);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, cam_planes) == 0);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, shadow_planes) == 80);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, cam_block) == 240);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, slot_count) == 252);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, cam_frac) == 256);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, arena_count) == 268);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, shadow_enabled) == 272);

/// Arena registry with live counts per (arena, pass).
pub(crate) struct ArenaDirectory {
    buffers: Vec<vk::Buffer>,
    /// Live [Opaque, Cutout] counts per arena (shadow reuses Opaque).
    live: Vec<[u32; 2]>,
    /// Reference count per arena; zero = reusable.
    refs: Vec<u32>,
    /// Slot placement (arena, pass, gen) for free decrement.
    slots: Vec<Option<(u32, Pass, NonZeroU32)>>,
    /// The registered Blend slots (unordered). The CPU Blend re-source walks
    /// exactly this set, so its cost scales with the transparent meshes, not
    /// with the whole slot table.
    blend: Vec<u32>,
    /// Position + 1 of a slot in `blend` (0 = not a Blend slot); O(1) removal.
    blend_pos: Vec<u32>,
    /// One past the highest registered slot: the cull dispatch and its
    /// visibility mask stop here rather than at the table's high-water mark
    /// when the tail has been freed.
    live_end: u32,
}

impl ArenaDirectory {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            live: Vec::new(),
            refs: Vec::new(),
            slots: Vec::new(),
            blend: Vec::new(),
            blend_pos: Vec::new(),
            live_end: 0,
        }
    }

    /// Registers an upload: interns the arena block (reusing a drained row),
    /// bumps its counts, and returns the arena index for the slot's word.
    pub fn note_upload(
        &mut self,
        slot: u32,
        generation: NonZeroU32,
        buffer: vk::Buffer,
        pass: Pass,
    ) -> u32 {
        let hit = (0..self.buffers.len())
            .find(|&i| self.refs[i] > 0 && self.buffers[i] == buffer)
            .or_else(|| {
                let reuse = self.refs.iter().position(|&r| r == 0);
                if let Some(i) = reuse {
                    self.buffers[i] = buffer;
                    debug_assert_eq!(self.live[i], [0; 2], "drained row kept live counts");
                }
                reuse
            });
        let arena = match hit {
            Some(i) => i as u32,
            None => {
                self.buffers.push(buffer);
                self.live.push([0; 2]);
                self.refs.push(0);
                (self.buffers.len() - 1) as u32
            }
        };
        self.refs[arena as usize] += 1;
        if let Some(lane) = group_lane(pass) {
            self.live[arena as usize][lane] += 1;
        }
        let n = slot as usize + 1;
        if self.slots.len() < n {
            self.slots.resize(n, None);
        }
        self.slots[slot as usize] = Some((arena, pass, generation));
        self.set_blend(slot, pass == Pass::Blend);
        self.live_end = self.live_end.max(slot + 1);
        arena
    }

    /// Adds `slot` to (or removes it from) the Blend set; idempotent either way.
    fn set_blend(&mut self, slot: u32, on: bool) {
        let i = slot as usize;
        if self.blend_pos.len() <= i {
            self.blend_pos.resize(i + 1, 0);
        }
        let pos = self.blend_pos[i];
        if on && pos == 0 {
            self.blend.push(slot);
            self.blend_pos[i] = self.blend.len() as u32;
        } else if !on && pos != 0 {
            let at = (pos - 1) as usize;
            self.blend.swap_remove(at);
            self.blend_pos[i] = 0;
            if let Some(&moved) = self.blend.get(at) {
                self.blend_pos[moved as usize] = pos;
            }
        }
    }

    /// The registered Blend slots, in no particular order.
    pub fn blend_slots(&self) -> &[u32] {
        &self.blend
    }

    /// One past the highest registered slot (0 when nothing is registered):
    /// every slot at or beyond it has a dead arena word.
    pub fn live_end(&self) -> u32 {
        self.live_end
    }

    /// Register a free with generation check.
    pub fn note_free(&mut self, slot: u32, generation: NonZeroU32) {
        let Some(Some((arena, pass, stored_gen))) =
            self.slots.get_mut(slot as usize).map(Option::take)
        else {
            return;
        };
        if stored_gen != generation {
            // Stale free for a newer generation; restore slot.
            self.slots[slot as usize] = Some((arena, pass, stored_gen));
            return;
        }
        self.refs[arena as usize] -= 1;
        if let Some(lane) = group_lane(pass) {
            self.live[arena as usize][lane] -= 1;
        }
        self.set_blend(slot, false);
        if slot + 1 == self.live_end {
            // The tail died: retreat to the next registered slot. Amortised
            // O(1) — each dead slot is stepped over once per retreat.
            self.live_end = self.slots[..slot as usize]
                .iter()
                .rposition(Option::is_some)
                .map_or(0, |i| i as u32 + 1);
        }
    }

    pub fn arena_buffer(&self, arena: usize) -> vk::Buffer {
        self.buffers[arena]
    }

    /// Get arena word for cull shader (0 = dead, else arena+1).
    pub fn arena_word(&self, slot: usize) -> u32 {
        self.slots
            .get(slot)
            .copied()
            .flatten()
            .map_or(0, |(a, _, _)| a + 1)
    }

    pub fn arena_count(&self) -> usize {
        self.buffers.len()
    }

    /// Fills `parts` with the group-major partition table, reusing its
    /// allocation, and returns the total command count.
    ///
    /// Camera groups (Opaque, Cutout) emit K distance buckets per arena, each
    /// sized to the full live count (worst case: every mesh lands in one
    /// bucket). Shadow groups (Near, Far) are unbucketed and reuse the Opaque
    /// live count — a caster may land in both cascades.
    fn partitions_into(&self, parts: &mut Vec<PartitionGpu>) -> u32 {
        let a = self.live.len();
        parts.clear();
        parts.reserve(partition_count(a));
        let mut offset = 0u32;
        for group in 0..CAMERA_GROUPS {
            for arena in 0..a {
                let capacity = self.live[arena][group];
                for _bucket in 0..BUCKETS {
                    parts.push(PartitionGpu { offset, capacity });
                    offset += capacity;
                }
            }
        }
        for _cascade in 0..SHADOW_GROUPS {
            for arena in 0..a {
                let capacity = self.live[arena][0];
                parts.push(PartitionGpu { offset, capacity });
                offset += capacity;
            }
        }
        offset
    }

    /// Get group-major partition table (group, arena pairs).
    #[cfg(test)]
    fn partitions(&self) -> (Vec<PartitionGpu>, u32) {
        let mut parts = Vec::new();
        let total = self.partitions_into(&mut parts);
        (parts, total)
    }
}

/// Get live-count lane for pass (Blend returns None).
fn group_lane(pass: Pass) -> Option<usize> {
    match pass {
        Pass::Opaque => Some(0),
        Pass::Cutout => Some(1),
        Pass::Blend => None,
    }
}

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
    /// Recycled partition table when [`Self::prepare`] returns `None`, so a
    /// frame with nothing to cull does not drop last frame's allocation.
    spare_parts: Vec<PartitionGpu>,
}

impl CullState {
    pub fn new(device: &ash::Device, cache: vk::PipelineCache, wave_atomics: bool) -> Self {
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
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
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
            spare_parts: Vec::new(),
        }
    }

    /// Prepare buffers and params for cull dispatch. Safe after fence is waited.
    ///
    /// `slot_count` bounds the dispatch (the caller trims it to the directory's
    /// live end); `visible` must cover it. `partitions` is last frame's table
    /// handed back for reuse (its contents are discarded).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare(
        &mut self,
        slot: usize,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        dir: &ArenaDirectory,
        records: RecordBuffers,
        slot_count: u32,
        camera: &Frustum,
        shadow: Option<&[Frustum; 2]>,
        eye: super::pipeline::EyeSplit,
        visible: &[u32],
        mut partitions: Vec<PartitionGpu>,
    ) -> Option<CullFrame> {
        debug_assert!(slot_count <= records.slots, "slot_count exceeds the table");
        debug_assert!(visible.len() >= slot_count.div_ceil(32) as usize);
        if partitions.capacity() == 0 {
            partitions = std::mem::take(&mut self.spare_parts);
        }
        let total = dir.partitions_into(&mut partitions);
        if partitions.is_empty() || total == 0 {
            self.spare_parts = partitions;
            return None;
        }
        let mut params = CullParamsGpu {
            cam_planes: camera.planes().map(|p| p.to_array()),
            shadow_planes: [[0.0; 4]; 10],
            cam_block: eye.block,
            slot_count,
            cam_frac: eye.frac,
            arena_count: dir.arena_count() as u32,
            shadow_enabled: shadow.is_some() as u32,
            _pad: [0; 3],
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
        })
    }

    /// Record cull dispatch (zeros counts, executes cull, fences writes).
    pub unsafe fn record(
        &self,
        device: &ash::Device,
        push: &ash::khr::push_descriptor::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        records: RecordBuffers,
        frame: &CullFrame,
    ) {
        unsafe {
            device.cmd_fill_buffer(cmd, frame.counts, 0, vk::WHOLE_SIZE, 0);
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
            ];
            let writes: [vk::WriteDescriptorSet; 7] = std::array::from_fn(|i| {
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
            device.cmd_dispatch(cmd, frame.slot_count.div_ceil(WORKGROUP), 1, 1);

            // Fence writes for DRAW_INDIRECT consumers.
            let to_draws = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_draws),
            );
        }
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
            {
                b.destroy(device);
            }
            for b in self.commands.iter_mut().chain(&mut self.counts) {
                b.destroy(device);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle;

    use super::*;

    fn buf(raw: u64) -> vk::Buffer {
        vk::Buffer::from_raw(raw)
    }

    fn genr(v: u32) -> NonZeroU32 {
        NonZeroU32::new(v).unwrap()
    }
    const G1: NonZeroU32 = NonZeroU32::new(1).unwrap();

    #[test]
    fn empty_directory_has_no_partitions() {
        let dir = ArenaDirectory::new();
        let (parts, total) = dir.partitions();
        assert!(parts.is_empty());
        assert_eq!(total, 0);
    }

    #[test]
    fn single_arena_single_opaque_upload_produces_exact_partition() {
        let mut dir = ArenaDirectory::new();
        let arena = dir.note_upload(0, G1, buf(1), Pass::Opaque);
        assert_eq!(arena, 0);
        assert_eq!(dir.arena_count(), 1);
        let (parts, total) = dir.partitions();
        // 2 camera groups * K buckets + 2 shadow groups, one arena.
        assert_eq!(parts.len(), partition_count(1));
        for bucket in 0..BUCKETS {
            assert_eq!(
                parts[camera_part(0, 0, bucket, 1)],
                PartitionGpu {
                    offset: bucket as u32,
                    capacity: 1
                }
            );
            assert_eq!(
                parts[camera_part(1, 0, bucket, 1)].capacity,
                0,
                "cutout buckets stay empty"
            );
        }
        // Shadow Near/Far reuse Opaque's live count, unbucketed.
        assert_eq!(
            parts[shadow_part(0, 0, 1)],
            PartitionGpu {
                offset: BUCKETS as u32,
                capacity: 1
            }
        );
        assert_eq!(
            parts[shadow_part(1, 0, 1)],
            PartitionGpu {
                offset: BUCKETS as u32 + 1,
                capacity: 1
            }
        );
        // K camera slots + 2 shadow slots.
        assert_eq!(total, BUCKETS as u32 + 2);
    }

    #[test]
    fn blend_uploads_do_not_occupy_a_cull_lane() {
        // Blend never reaches the GPU cull (CPU-sorted path); its records still
        // register a reference (for reuse bookkeeping) but no live count.
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Blend);
        let (parts, total) = dir.partitions();
        assert!(parts.iter().all(|p| p.capacity == 0));
        assert_eq!(total, 0);
    }

    #[test]
    fn partitions_are_group_major_offsets_accumulate_across_arenas() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque); // arena 0: 1 opaque
        dir.note_upload(1, G1, buf(1), Pass::Opaque); // arena 0: 2 opaque (same buffer)
        dir.note_upload(2, G1, buf(2), Pass::Cutout); // arena 1: 1 cutout
        assert_eq!(dir.arena_count(), 2);
        let (parts, total) = dir.partitions();
        assert_eq!(parts.len(), partition_count(2));
        // Opaque a0: K buckets, each capacity 2, offsets 0,2,4,6.
        for bucket in 0..BUCKETS {
            let p = parts[camera_part(0, 0, bucket, 2)];
            assert_eq!(p.capacity, 2);
            assert_eq!(p.offset, (bucket * 2) as u32);
        }
        // Opaque a1: empty.
        for bucket in 0..BUCKETS {
            assert_eq!(parts[camera_part(0, 1, bucket, 2)].capacity, 0);
        }
        // Cutout a0 empty, a1 capacity 1 across K buckets.
        for bucket in 0..BUCKETS {
            assert_eq!(parts[camera_part(1, 0, bucket, 2)].capacity, 0);
            assert_eq!(parts[camera_part(1, 1, bucket, 2)].capacity, 1);
        }
        // Shadows unbucketed, reuse Opaque live counts (a0=2, a1=0).
        assert_eq!(parts[shadow_part(0, 0, 2)].capacity, 2);
        assert_eq!(parts[shadow_part(0, 1, 2)].capacity, 0);
        assert_eq!(parts[shadow_part(1, 0, 2)].capacity, 2);
        assert_eq!(parts[shadow_part(1, 1, 2)].capacity, 0);
        // Opaque: K*2, Cutout: K*1, ShadowNear: 2, ShadowFar: 2.
        assert_eq!(total, (BUCKETS * 2 + BUCKETS + 2 + 2) as u32);
    }

    #[test]
    fn note_free_decrements_live_count_and_capacity_shrinks() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque);
        dir.note_upload(1, G1, buf(1), Pass::Opaque);
        dir.note_free(0, G1);
        let (parts, total) = dir.partitions();
        assert_eq!(parts[camera_part(0, 0, 0, 1)].capacity, 1);
        // K camera slots + 2 shadow slots, each sized off the remaining live count.
        assert_eq!(total, BUCKETS as u32 + 2);
    }

    #[test]
    fn note_free_on_last_reference_drains_the_arena_row_for_reuse() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque);
        dir.note_free(0, G1);
        assert_eq!(dir.arena_count(), 1); // row kept, but refs == 0 now
        // A fresh upload reuses the drained row instead of growing the table.
        let arena = dir.note_upload(1, G1, buf(2), Pass::Cutout);
        assert_eq!(arena, 0, "drained row should be reused, not appended");
        assert_eq!(dir.arena_count(), 1);
    }

    #[test]
    fn note_upload_matches_a_still_live_buffer_instead_of_reusing_a_drained_row() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque);
        // A second upload to the SAME live buffer must hit the existing row, not
        // mint a new one (this is how one arena block accrues multiple meshes).
        let arena = dir.note_upload(1, G1, buf(1), Pass::Cutout);
        assert_eq!(arena, 0);
        assert_eq!(dir.arena_count(), 1);
        let (parts, _) = dir.partitions();
        assert_eq!(parts[camera_part(0, 0, 0, 1)].capacity, 1); // Opaque
        assert_eq!(parts[camera_part(1, 0, 0, 1)].capacity, 1); // Cutout
    }

    #[test]
    fn note_free_with_stale_generation_is_a_no_op() {
        // A slot freed then immediately re-uploaded (new generation) must not
        // have a late/duplicate free for the OLD generation decrement its
        // still-live count out from under it.
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque);
        dir.note_free(0, genr(1)); // real free: drains the slot
        dir.note_upload(0, genr(2), buf(2), Pass::Cutout); // reused, new generation
        dir.note_free(0, genr(1)); // stale duplicate: must be ignored
        let (parts, total) = dir.partitions();
        assert_eq!(
            parts[camera_part(1, 0, 0, 1)].capacity,
            1,
            "Cutout slot must still be live"
        );
        assert_eq!(
            total, BUCKETS as u32,
            "stale free must not have drained the reused slot"
        );
    }

    #[test]
    fn blend_set_tracks_uploads_and_frees_only_for_blend_slots() {
        let mut dir = ArenaDirectory::new();
        assert!(dir.blend_slots().is_empty());
        dir.note_upload(3, G1, buf(1), Pass::Blend);
        dir.note_upload(7, G1, buf(1), Pass::Opaque);
        dir.note_upload(9, G1, buf(2), Pass::Blend);
        dir.note_upload(12, G1, buf(2), Pass::Blend);
        let mut got = dir.blend_slots().to_vec();
        got.sort_unstable();
        assert_eq!(got, [3, 9, 12]);
        // Removing from the middle (swap_remove) keeps the moved slot findable.
        dir.note_free(9, G1);
        let mut got = dir.blend_slots().to_vec();
        got.sort_unstable();
        assert_eq!(got, [3, 12]);
        dir.note_free(12, G1);
        assert_eq!(dir.blend_slots(), [3]);
        // A stale free never touches the set; a real one drains it.
        dir.note_free(3, genr(2));
        assert_eq!(dir.blend_slots(), [3]);
        dir.note_free(3, G1);
        assert!(dir.blend_slots().is_empty());
        // Re-registering a drained slot as Blend re-adds it exactly once.
        dir.note_upload(3, genr(2), buf(1), Pass::Blend);
        dir.note_upload(3, genr(2), buf(1), Pass::Blend);
        assert_eq!(dir.blend_slots(), [3]);
        // Re-registering it as Opaque (without a free in between) removes it.
        dir.note_upload(3, genr(3), buf(1), Pass::Opaque);
        assert!(dir.blend_slots().is_empty());
    }

    #[test]
    fn partitions_into_reuses_the_callers_allocation() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque);
        let mut parts = Vec::with_capacity(64);
        let ptr = parts.as_ptr();
        let total = dir.partitions_into(&mut parts);
        assert_eq!(parts.as_ptr(), ptr);
        assert_eq!(parts.len(), partition_count(1));
        // K camera (opaque) slots + 2 unbucketed shadow slots.
        assert_eq!(total, BUCKETS as u32 + 2);
    }

    #[test]
    fn live_end_follows_the_highest_registered_slot() {
        let mut dir = ArenaDirectory::new();
        assert_eq!(dir.live_end(), 0);
        dir.note_upload(4, G1, buf(1), Pass::Opaque);
        assert_eq!(dir.live_end(), 5);
        dir.note_upload(40, G1, buf(1), Pass::Blend);
        dir.note_upload(20, G1, buf(1), Pass::Cutout);
        assert_eq!(dir.live_end(), 41);
        // Freeing below the top leaves it; freeing the top retreats past the
        // dead gap to the next registered slot.
        dir.note_free(20, G1);
        assert_eq!(dir.live_end(), 41);
        dir.note_free(40, G1);
        assert_eq!(dir.live_end(), 5);
        // A stale free of the top is ignored.
        dir.note_free(4, genr(2));
        assert_eq!(dir.live_end(), 5);
        dir.note_free(4, G1);
        assert_eq!(dir.live_end(), 0);
    }

    #[test]
    fn group_lane_maps_camera_passes_and_excludes_blend() {
        assert_eq!(group_lane(Pass::Opaque), Some(0));
        assert_eq!(group_lane(Pass::Cutout), Some(1));
        assert_eq!(group_lane(Pass::Blend), None);
    }

    #[test]
    fn distance_bucket_splits_match_genconst_edges() {
        assert_eq!(distance_bucket(0.0), 0);
        assert_eq!(
            distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_0 - 0.01),
            0
        );
        assert_eq!(distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_0), 1);
        assert_eq!(
            distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_1 - 0.01),
            1
        );
        assert_eq!(distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_1), 2);
        assert_eq!(
            distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_2 - 0.01),
            2
        );
        assert_eq!(distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_2), 3);
        assert_eq!(distance_bucket(1.0e6), 3);
    }

    #[test]
    fn camera_buckets_are_k_wide_shadows_are_unbucketed() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque);
        let (parts, _) = dir.partitions();
        assert_eq!(GROUPS, 4, "Opaque, Cutout, ShadowNear, ShadowFar");
        assert_eq!(parts.len(), CAMERA_GROUPS * BUCKETS + SHADOW_GROUPS);
        // Adjacent camera buckets of the same arena share capacity but not offset.
        let b0 = parts[camera_part(0, 0, 0, 1)];
        let b1 = parts[camera_part(0, 0, 1, 1)];
        assert_eq!(b0.capacity, b1.capacity);
        assert_eq!(b1.offset, b0.offset + b0.capacity);
        // Shadow groups occupy one partition per arena, not K.
        assert_eq!(
            shadow_part(1, 0, 1) - shadow_part(0, 0, 1),
            1,
            "cascades are adjacent unbucketed partitions"
        );
    }
}
