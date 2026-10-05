//! Datum storage and albedo cubes for [`crate::FarShape::Mapped`].
//!
//! One device-local buffer holds eight headers plus every datum. Cube images
//! are created on `set_map` (the [`crate::FarMapError::OutOfMemory`] path) and
//! filled later: face bytes and mip blits are recorded into the frame command
//! buffer, which is the queue that can blit. The datum copy follows the
//! block-texture / material transfer lane (timeline wait, queue-family
//! ownership when the families differ, staging retired on that timeline).
//! Nothing here waits on the GPU.

use ash::vk;

use super::alloc::{create_buffer, find_memory_type};
use super::buffers::RetireQueue;
use super::image::allocate_and_bind_image;
use super::mesh_residency::{CopyBarrier, copy_barrier};
use super::timeline::{Timeline, TimelineValue};
use super::transfer::TransferLane;
use crate::far_body::{FarMapError, MAX_FAR_MAPS};

/// Stages that first read the datum buffer and the albedo cubes (sky fragment).
pub(crate) const FAR_MAP_CONSUMER_STAGES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::FRAGMENT_SHADER;

const MAX_G: usize = 65;
/// Floats reserved for one map, including the padding past a smaller `g`.
const SLOT_FLOATS: usize = 6 * MAX_G * MAX_G;
const HEADER_UINTS: usize = MAX_FAR_MAPS * 8;
const BUFFER_UINTS: usize = HEADER_UINTS + MAX_FAR_MAPS * SLOT_FLOATS;

struct GpuCube {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    mips: u32,
    size: u32,
    /// Recorded transition out of `UNDEFINED`. The next commands in that
    /// command buffer see `SHADER_READ_ONLY`.
    cleared: bool,
}

impl GpuCube {
    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

struct Slot {
    g: u32,
    min_off: f32,
    max_off: f32,
    albedo_size: u32,
    landed: u32,
    generation: u32,
    cube: Option<GpuCube>,
}

struct PendingFace {
    id: u8,
    face: u32,
    generation: u32,
    bytes: Box<[u8]>,
}

pub(crate) struct FarMaps {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Header (8 × 8 uints) then `MAX_FAR_MAPS` datum slots. Floats are stored
    /// as their bits. Uploaded whole when `buffer_dirty`.
    words: Vec<u32>,
    buffer_dirty: bool,
    /// `true` after the first copy has handed the buffer to graphics, so the
    /// next overwrite has to order against fragment reads.
    buffer_on_graphics: bool,
    sampler: vk::Sampler,
    dummy: GpuCube,
    slots: [Slot; MAX_FAR_MAPS],
    pending_faces: Vec<PendingFace>,
    pending_retire: Vec<GpuCube>,
    image_retire: RetireQueue<GpuCube>,
    staging_retire: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
    transfer_retire: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
    /// Graphics-pool command buffers used for a dedicated-family release.
    release_cmds: RetireQueue<(Timeline, vk::CommandBuffer)>,
    command_pool: vk::CommandPool,
}

impl FarMaps {
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
    ) -> Self {
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let fmt = unsafe {
            instance.get_physical_device_format_properties(physical, vk::Format::R8G8B8A8_SRGB)
        };
        let need = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::BLIT_SRC
            | vk::FormatFeatureFlags::BLIT_DST
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        assert!(
            fmt.optimal_tiling_features.contains(need),
            "R8G8B8A8_SRGB cannot blit far-map cubes: {:?}",
            fmt.optimal_tiling_features
        );
        let (buffer, memory) = create_buffer(
            device,
            &memory_props,
            (BUFFER_UINTS * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "far-map datum",
        );
        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .min_lod(0.0)
                        .max_lod(vk::LOD_CLAMP_NONE),
                    None,
                )
                .expect("create far-map albedo sampler")
        };
        let dummy = unsafe { create_cube(device, &memory_props, 1).expect("far-map dummy cube") };
        let slot = || Slot {
            g: 0,
            min_off: 0.0,
            max_off: 0.0,
            albedo_size: 0,
            landed: 0,
            generation: 1,
            cube: None,
        };
        Self {
            buffer,
            memory,
            words: vec![0; BUFFER_UINTS],
            buffer_dirty: true,
            buffer_on_graphics: false,
            sampler,
            dummy,
            slots: std::array::from_fn(|_| slot()),
            pending_faces: Vec::new(),
            pending_retire: Vec::new(),
            image_retire: RetireQueue::new(),
            staging_retire: RetireQueue::new(),
            transfer_retire: RetireQueue::new(),
            release_cmds: RetireQueue::new(),
            command_pool,
        }
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub(crate) fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Cube view for `id`, or the 1×1 dummy when that slot has no albedo.
    pub(crate) fn view(&self, id: usize) -> vk::ImageView {
        self.slots
            .get(id)
            .and_then(|s| s.cube.as_ref())
            .map(|c| c.view)
            .unwrap_or(self.dummy.view)
    }

    /// Maximum datum offset of each map, for the hi-radius cone. An empty
    /// slot reports 0.
    pub(crate) fn max_offsets(&self) -> [f32; MAX_FAR_MAPS] {
        let mut out = [0.0; MAX_FAR_MAPS];
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.g >= 2 {
                out[i] = slot.max_off;
            }
        }
        out
    }

    pub(crate) fn has_garbage(&self) -> bool {
        !self.image_retire.is_empty()
            || !self.staging_retire.is_empty()
            || !self.transfer_retire.is_empty()
            || !self.release_cmds.is_empty()
    }

    /// A datum overwrite of a buffer graphics has already read, or a face
    /// blit onto a cube a previous frame may still be sampling. The frame
    /// loop submits pending batches first so `last_render_value` covers them.
    pub(crate) fn has_overwrite_pending(&self) -> bool {
        if self.buffer_dirty && self.buffer_on_graphics {
            return true;
        }
        self.pending_faces.iter().any(|face| {
            self.slots.get(face.id as usize).is_some_and(|slot| {
                slot.generation == face.generation
                    && slot.cube.as_ref().is_some_and(|cube| cube.cleared)
            })
        })
    }

    /// Install `datum` and, when `albedo_size > 0`, a fresh cube. The previous
    /// cube is retired at the next flush. Landed bits start clear, so faces
    /// must be uploaded again. No GPU wait.
    pub(crate) fn set_map(
        &mut self,
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        id: u8,
        g: u32,
        datum: &[f32],
        albedo_size: u32,
    ) -> Result<(), FarMapError> {
        if id as usize >= MAX_FAR_MAPS {
            return Err(FarMapError::BadId);
        }
        let slot_i = id as usize;
        let cube = if albedo_size == 0 {
            None
        } else {
            Some(unsafe { create_cube(device, memory_props, albedo_size)? })
        };
        let slot = &mut self.slots[slot_i];
        if let Some(old) = slot.cube.take() {
            self.pending_retire.push(old);
        }
        slot.generation = slot.generation.wrapping_add(1);
        slot.albedo_size = albedo_size;
        slot.landed = 0;
        slot.g = g;
        slot.cube = cube;
        let base = HEADER_UINTS + slot_i * SLOT_FLOATS;
        for w in &mut self.words[base..base + SLOT_FLOATS] {
            *w = 0;
        }
        let mut min_off = f32::INFINITY;
        let mut max_off = f32::NEG_INFINITY;
        let n = datum.len().min(SLOT_FLOATS);
        for (i, v) in datum.iter().take(n).enumerate() {
            let v = if v.is_finite() { *v } else { 0.0 };
            min_off = min_off.min(v);
            max_off = max_off.max(v);
            self.words[base + i] = v.to_bits();
        }
        if !min_off.is_finite() {
            min_off = 0.0;
            max_off = 0.0;
        }
        slot.min_off = min_off;
        slot.max_off = max_off;
        self.write_header(slot_i);
        self.buffer_dirty = true;
        Ok(())
    }

    /// Queue one face. Dropped at flush if `generation` is stale or the cube
    /// is gone. Does not wait.
    pub(crate) fn queue_face(&mut self, id: u8, face: u32, bytes: Box<[u8]>) {
        let Some(slot) = self.slots.get(id as usize) else {
            return;
        };
        self.pending_faces.push(PendingFace {
            id,
            face,
            generation: slot.generation,
            bytes,
        });
    }

    /// Drop the slot. The cube is retired at the next flush; the header
    /// reports an empty map (`g == 0`) so the shader's datum read is zero.
    pub(crate) fn clear(&mut self, id: u8) {
        let Some(slot) = self.slots.get_mut(id as usize) else {
            return;
        };
        if let Some(old) = slot.cube.take() {
            self.pending_retire.push(old);
        }
        slot.generation = slot.generation.wrapping_add(1);
        slot.g = 0;
        slot.min_off = 0.0;
        slot.max_off = 0.0;
        slot.albedo_size = 0;
        slot.landed = 0;
        let base = HEADER_UINTS + id as usize * SLOT_FLOATS;
        for w in &mut self.words[base..base + SLOT_FLOATS] {
            *w = 0;
        }
        self.write_header(id as usize);
        self.buffer_dirty = true;
    }

    fn write_header(&mut self, id: usize) {
        let slot = &self.slots[id];
        let b = id * 8;
        self.words[b] = slot.g;
        self.words[b + 1] = (id * SLOT_FLOATS) as u32;
        self.words[b + 2] = slot.min_off.to_bits();
        self.words[b + 3] = slot.max_off.to_bits();
        self.words[b + 4] = slot.landed;
        self.words[b + 5] = slot.albedo_size;
        self.words[b + 6] = 0;
        self.words[b + 7] = 0;
    }

    /// Record pending clears, face blits, and the datum copy. Returns the
    /// transfer-lane value the graphics submit must wait on, if the copy left
    /// this command buffer.
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
        unsafe {
            if !self.dummy.cleared {
                clear_cube(device, graphics_cmd, &self.dummy);
                self.dummy.cleared = true;
            }
            for slot in &mut self.slots {
                if let Some(cube) = slot.cube.as_mut() {
                    if !cube.cleared {
                        clear_cube(device, graphics_cmd, cube);
                        cube.cleared = true;
                    }
                }
            }
            let face_staging = self.record_faces(instance, device, physical, graphics_cmd);
            if let Some((buffer, memory)) = face_staging {
                self.staging_retire.push(done_at, (buffer, memory));
            }
            let wait = if self.buffer_dirty {
                self.upload_buffer(
                    instance,
                    device,
                    physical,
                    lane,
                    graphics_cmd,
                    graphics_queue,
                    graphics_family,
                    graphics_timeline,
                    last_render_value,
                    done_at,
                )
            } else {
                None
            };
            for cube in self.pending_retire.drain(..) {
                self.image_retire.push(done_at, cube);
            }
            wait
        }
    }

    /// Copy each current face into mip 0 and blit the chain, on `cmd` (the
    /// graphics buffer: a transfer-only queue cannot blit). Sets the landed
    /// bit only for a face whose commands were recorded. Returns the staging
    /// buffer, retired by the caller at `done_at`.
    unsafe fn record_faces(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        cmd: vk::CommandBuffer,
    ) -> Option<(vk::Buffer, vk::DeviceMemory)> {
        let faces = std::mem::take(&mut self.pending_faces);
        if faces.is_empty() {
            return None;
        }
        let mut packed: Vec<u8> = Vec::new();
        struct Job {
            id: usize,
            face: u32,
            offset: u64,
        }
        let mut jobs: Vec<Job> = Vec::new();
        for face in faces {
            let id = face.id as usize;
            let Some(slot) = self.slots.get(id) else {
                continue;
            };
            if slot.generation != face.generation || face.face >= 6 {
                continue;
            }
            let Some(cube) = slot.cube.as_ref() else {
                continue;
            };
            if !cube.cleared {
                continue;
            }
            let expect = cube.size as usize * cube.size as usize * 4;
            if face.bytes.len() != expect {
                continue;
            }
            let offset = packed.len() as u64;
            packed.extend_from_slice(&face.bytes);
            jobs.push(Job {
                id,
                face: face.face,
                offset,
            });
        }
        if jobs.is_empty() {
            return None;
        }
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let (staging, staging_mem) = unsafe { fill_staging(device, &memory_props, &packed) };
        for job in &jobs {
            let (image, mips, size) = {
                let cube = self.slots[job.id]
                    .cube
                    .as_ref()
                    .expect("face job has a cube");
                (cube.image, cube.mips, cube.size)
            };
            unsafe {
                blit_face(
                    device, cmd, image, mips, size, job.face, staging, job.offset,
                )
            };
            self.slots[job.id].landed |= 1 << job.face;
            self.write_header(job.id);
            self.buffer_dirty = true;
        }
        Some((staging, staging_mem))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn upload_buffer(
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
        self.buffer_dirty = false;
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let bytes: &[u8] = bytemuck::cast_slice(&self.words);
        let (staging, staging_mem) = unsafe { fill_staging(device, &memory_props, bytes) };
        let separate = lane.is_separate_queue();
        let qfot = lane.needs_ownership_transfer();
        let buffer = self.buffer;
        let size = bytes.len() as u64;
        let reads = vk::AccessFlags2::SHADER_STORAGE_READ;

        let extra_wait = if qfot && self.buffer_on_graphics {
            Some(unsafe {
                self.submit_release(
                    device,
                    graphics_queue,
                    graphics_family,
                    lane.family(),
                    done_at,
                    size,
                )
            })
        } else if separate && self.buffer_on_graphics {
            Some((
                graphics_timeline.semaphore(),
                last_render_value,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
            ))
        } else {
            None
        };

        let lane_batch = separate.then(|| unsafe { lane.begin(device) });
        let record_cmd = lane_batch.as_ref().map_or(graphics_cmd, |b| b.cmd());
        unsafe {
            if !separate && self.buffer_on_graphics {
                let to_copy = [vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .src_access_mask(reads)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .buffer(buffer)
                    .size(size)];
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&to_copy),
                );
            } else if qfot && self.buffer_on_graphics {
                let acquire = [vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_queue_family_index(graphics_family)
                    .dst_queue_family_index(lane.family())
                    .buffer(buffer)
                    .size(size)];
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&acquire),
                );
            }
            device.cmd_copy_buffer(
                record_cmd,
                staging,
                buffer,
                &[vk::BufferCopy::default().size(size)],
            );
        }

        let arrived = if let Some(lane_batch) = lane_batch {
            if qfot {
                let release = [datum_barrier(
                    buffer,
                    size,
                    CopyBarrier::Release {
                        src_family: lane.family(),
                        dst_family: graphics_family,
                    },
                )];
                unsafe {
                    device.cmd_pipeline_barrier2(
                        record_cmd,
                        &vk::DependencyInfo::default().buffer_memory_barriers(&release),
                    );
                }
            }
            let value = unsafe { lane.submit_after(device, lane_batch, extra_wait) };
            if qfot {
                let acquire = [datum_barrier(
                    buffer,
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
            let to_shader = [datum_barrier(buffer, size, CopyBarrier::Draw)];
            unsafe {
                device.cmd_pipeline_barrier2(
                    graphics_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&to_shader),
                );
            }
            self.staging_retire.push(done_at, (staging, staging_mem));
            None
        };
        self.buffer_on_graphics = true;
        arrived
    }

    unsafe fn submit_release(
        &mut self,
        device: &ash::Device,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        transfer_family: u32,
        done_at: TimelineValue,
        size: u64,
    ) -> (vk::Semaphore, TimelineValue, vk::PipelineStageFlags2) {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc)
                .expect("far-map release command buffer")[0]
        };
        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin far-map release");
            let release = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .src_queue_family_index(graphics_family)
                .dst_queue_family_index(transfer_family)
                .buffer(self.buffer)
                .size(size)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&release),
            );
            device.end_command_buffer(cmd).expect("end far-map release");
        }
        let mut tmp = unsafe { Timeline::new(device) };
        let rs = tmp.begin_render(cmd);
        let completion = unsafe { rs.submit(device, graphics_queue, &tmp, None) };
        let sem = tmp.semaphore();
        let value = completion.value();
        self.release_cmds.push(done_at, (tmp, cmd));
        (sem, value, vk::PipelineStageFlags2::COPY)
    }

    pub unsafe fn collect(&mut self, device: &ash::Device, current: TimelineValue) {
        unsafe {
            self.staging_retire.collect(current, |(buffer, memory)| {
                destroy_staging(device, buffer, memory)
            });
            self.image_retire
                .collect(current, |cube| cube.destroy(device));
            let pool = self.command_pool;
            self.release_cmds.collect(current, |(timeline, cmd)| {
                timeline.destroy(device);
                device.free_command_buffers(pool, &[cmd]);
            });
        }
    }

    pub unsafe fn collect_transfer(&mut self, device: &ash::Device, current: TimelineValue) {
        unsafe {
            self.transfer_retire.collect(current, |(buffer, memory)| {
                destroy_staging(device, buffer, memory)
            });
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.staging_retire
                .collect_all(|(buffer, memory)| destroy_staging(device, buffer, memory));
            self.transfer_retire
                .collect_all(|(buffer, memory)| destroy_staging(device, buffer, memory));
            self.image_retire.collect_all(|cube| cube.destroy(device));
            for cube in self.pending_retire.drain(..) {
                cube.destroy(device);
            }
            let pool = self.command_pool;
            self.release_cmds.collect_all(|(timeline, cmd)| {
                timeline.destroy(device);
                device.free_command_buffers(pool, &[cmd]);
            });
            self.dummy.destroy(device);
            for slot in &mut self.slots {
                if let Some(cube) = slot.cube.take() {
                    cube.destroy(device);
                }
            }
            device.destroy_sampler(self.sampler, None);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn datum_barrier(
    buffer: vk::Buffer,
    size: u64,
    role: CopyBarrier,
) -> vk::BufferMemoryBarrier2<'static> {
    let barrier = copy_barrier(buffer, 0, size, vk::AccessFlags2::SHADER_STORAGE_READ, role);
    match role {
        CopyBarrier::Draw | CopyBarrier::Acquire { .. } => barrier
            .dst_stage_mask(FAR_MAP_CONSUMER_STAGES)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ),
        CopyBarrier::Release { .. } => barrier,
    }
}

fn color_range(
    base_mip: u32,
    mips: u32,
    base_layer: u32,
    layers: u32,
) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: base_mip,
        level_count: mips,
        base_array_layer: base_layer,
        layer_count: layers,
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    range: vk::ImageSubresourceRange,
) {
    let barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(range)];
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&barrier),
        );
    }
}

/// Cube of `size`² faces, full mip chain, still `UNDEFINED`. `Err` only when
/// the image memory cannot be allocated; the image is destroyed on that path.
unsafe fn create_cube(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u32,
) -> Result<GpuCube, FarMapError> {
    let mips = size.trailing_zeros() + 1;
    let info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_SRGB)
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        })
        .mip_levels(mips)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe {
        device
            .create_image(&info, None)
            .expect("create far-map albedo cube")
    };
    let memory = match allocate_and_bind_image(device, memory_props, image, "far-map albedo cube") {
        Ok(memory) => memory,
        Err(_) => {
            unsafe { device.destroy_image(image, None) };
            return Err(FarMapError::OutOfMemory);
        }
    };
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::CUBE)
        .format(vk::Format::R8G8B8A8_SRGB)
        .subresource_range(color_range(0, mips, 0, 6));
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .expect("create far-map albedo cube view")
    };
    Ok(GpuCube {
        image,
        memory,
        view,
        mips,
        size,
        cleared: false,
    })
}

/// Every mip of every face: `UNDEFINED` → cleared black → `SHADER_READ`.
/// A later face upload may sample a face it did not write, so none may stay
/// `UNDEFINED`.
unsafe fn clear_cube(device: &ash::Device, cmd: vk::CommandBuffer, cube: &GpuCube) {
    let range = color_range(0, cube.mips, 0, 6);
    unsafe {
        image_barrier(
            device,
            cmd,
            cube.image,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::CLEAR,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            range,
        );
        device.cmd_clear_color_image(
            cmd,
            cube.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue { float32: [0.0; 4] },
            &[range],
        );
        image_barrier(
            device,
            cmd,
            cube.image,
            vk::PipelineStageFlags2::CLEAR,
            vk::AccessFlags2::TRANSFER_WRITE,
            FAR_MAP_CONSUMER_STAGES,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            range,
        );
    }
}

/// Copy mip 0 of `face` from `staging` and blit the rest of the chain.
/// The cube is `SHADER_READ` on entry (after [`clear_cube`]) and on return.
#[allow(clippy::too_many_arguments)]
unsafe fn blit_face(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    mips: u32,
    size: u32,
    face: u32,
    staging: vk::Buffer,
    offset: u64,
) {
    let mip0 = color_range(0, 1, face, 1);
    unsafe {
        image_barrier(
            device,
            cmd,
            image,
            FAR_MAP_CONSUMER_STAGES,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip0,
        );
        let region = vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: face,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            });
        device.cmd_copy_buffer_to_image(
            cmd,
            staging,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
        let mut src_w = size;
        for mip in 0..mips.saturating_sub(1) {
            let dst_w = (src_w / 2).max(1);
            let src_range = color_range(mip, 1, face, 1);
            let dst_range = color_range(mip + 1, 1, face, 1);
            image_barrier(
                device,
                cmd,
                image,
                vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_range,
            );
            image_barrier(
                device,
                cmd,
                image,
                FAR_MAP_CONSUMER_STAGES,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                dst_range,
            );
            let blit = vk::ImageBlit::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip,
                    base_array_layer: face,
                    layer_count: 1,
                })
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src_w as i32,
                        y: src_w as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip + 1,
                    base_array_layer: face,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst_w as i32,
                        y: dst_w as i32,
                        z: 1,
                    },
                ]);
            device.cmd_blit_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );
            image_barrier(
                device,
                cmd,
                image,
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_READ,
                FAR_MAP_CONSUMER_STAGES,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_range,
            );
            src_w = dst_w;
        }
        let last = color_range(mips.saturating_sub(1), 1, face, 1);
        image_barrier(
            device,
            cmd,
            image,
            vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_WRITE,
            FAR_MAP_CONSUMER_STAGES,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            last,
        );
    }
}

unsafe fn fill_staging(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    bytes: &[u8],
) -> (vk::Buffer, vk::DeviceMemory) {
    let size = bytes.len() as vk::DeviceSize;
    let info = vk::BufferCreateInfo::default()
        .size(size.max(1))
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    unsafe {
        let staging = device
            .create_buffer(&info, None)
            .expect("create far-map staging buffer");
        let req = device.get_buffer_memory_requirements(staging);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(find_memory_type(
                memory_props,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
        let memory = device
            .allocate_memory(&alloc, None)
            .expect("allocate far-map staging memory");
        device
            .bind_buffer_memory(staging, memory, 0)
            .expect("bind far-map staging memory");
        if !bytes.is_empty() {
            let ptr = device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .expect("map far-map staging memory");
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            device.unmap_memory(memory);
        }
        (staging, memory)
    }
}

unsafe fn destroy_staging(device: &ash::Device, buffer: vk::Buffer, memory: vk::DeviceMemory) {
    unsafe {
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
    }
}
