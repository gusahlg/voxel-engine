//! Variable-rate shading: per-slot rate image, history, and classifier.

use ash::vk;

use super::alloc::{find_memory_type, try_find_memory_type};
use super::buffers::FRAMES_IN_FLIGHT;
use super::device::FragmentShadingRate;
use super::image::{AllocError, ImageDesc, ImageResource, create_image_array, image_purpose};
use super::{SAMPLEABLE_DEPTH_REST_LAYOUT, color_range};
use crate::skeleton::FrameSlot;

const SLOTS: usize = FRAMES_IN_FLIGHT as usize;

pub(crate) const FLAG_ALLOW_4X4: u32 = 1 << 0;
pub(crate) const FLAG_USE_HISTORY: u32 = 1 << 1;
pub(crate) const FLAG_WRITE_MIX: u32 = 1 << 2;

const MIX_COUNT: usize = 3;
pub(crate) const MIX_BYTES: u64 = (MIX_COUNT * size_of::<u32>()) as u64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VrsPush {
    pub d_threshold: f32,
    pub texel_w: u32,
    pub texel_h: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub depth_w: u32,
    pub depth_h: u32,
    pub flags: u32,
}

pub(crate) struct RateAttachment {
    pub view: vk::ImageView,
    pub texel_size: vk::Extent2D,
}

/// Host-visible copy of the per-slot tile-mix histogram, fence-safe to read
/// after the slot's timeline wait.
struct MixReadback {
    gpu: vk::Buffer,
    gpu_memory: vk::DeviceMemory,
    cpu: vk::Buffer,
    cpu_memory: vk::DeviceMemory,
    mapped: *mut u32,
}

pub(crate) struct Vrs {
    /// Texel size for VRS attachment; shared by all pipelines.
    pub texel_size: vk::Extent2D,
    tiles: vk::Extent2D,
    images: [ImageResource; SLOTS],
    history: [ImageResource; SLOTS],
    mix: [MixReadback; SLOTS],
}

impl Vrs {
    pub fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        fsr: &FragmentShadingRate,
        render_extent: vk::Extent2D,
    ) -> Result<Vrs, AllocError> {
        let texel_size = fsr.texel_size;
        let tiles = vk::Extent2D {
            width: render_extent.width.div_ceil(texel_size.width).max(1),
            height: render_extent.height.div_ceil(texel_size.height).max(1),
        };
        let rate_purpose = image_purpose("VRS rate", tiles, vk::SampleCountFlags::TYPE_1);
        let history_purpose = image_purpose("VRS history", tiles, vk::SampleCountFlags::TYPE_1);
        let images = create_image_array(device, || {
            create_r8_image(
                device,
                memory_props,
                tiles,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
                &rate_purpose,
            )
        })?;
        let history = match create_image_array(device, || {
            create_r8_image(
                device,
                memory_props,
                tiles,
                vk::ImageUsageFlags::STORAGE,
                &history_purpose,
            )
        }) {
            Ok(history) => history,
            Err(err) => {
                for img in &images {
                    unsafe { img.destroy(device) };
                }
                return Err(err);
            }
        };
        let mix = std::array::from_fn(|_| MixReadback::new(device, memory_props));
        Ok(Vrs {
            texel_size,
            tiles,
            images,
            history,
            mix,
        })
    }

    pub fn tiles(&self) -> vk::Extent2D {
        self.tiles
    }

    pub fn view(&self, slot: usize) -> vk::ImageView {
        self.images[slot].view()
    }

    pub fn image(&self, slot: usize) -> vk::Image {
        self.images[slot].image()
    }

    pub fn history_view(&self, slot: usize) -> vk::ImageView {
        self.history[slot].view()
    }

    pub fn history_image(&self, slot: usize) -> vk::Image {
        self.history[slot].image()
    }

    pub fn mix_gpu(&self, slot: usize) -> vk::Buffer {
        self.mix[slot].gpu
    }

    pub fn mix_cpu(&self, slot: usize) -> vk::Buffer {
        self.mix[slot].cpu
    }

    /// Last completed histogram for `slot`: `[1x1, 2x2, 4x4]`.
    pub fn mix(&self, slot: usize) -> [u32; MIX_COUNT] {
        unsafe {
            let p = self.mix[slot].mapped;
            [*p, *p.add(1), *p.add(2)]
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for img in &self.images {
                img.destroy(device);
            }
            for img in &self.history {
                img.destroy(device);
            }
            for mix in &self.mix {
                mix.destroy(device);
            }
        }
    }
}

impl MixReadback {
    fn new(device: &ash::Device, memory_props: &vk::PhysicalDeviceMemoryProperties) -> Self {
        let gpu = create_buffer(
            device,
            memory_props,
            MIX_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let (cpu, cpu_memory, mapped) = create_mapped_buffer(
            device,
            memory_props,
            MIX_BYTES,
            vk::BufferUsageFlags::TRANSFER_DST,
        );
        Self {
            gpu: gpu.0,
            gpu_memory: gpu.1,
            cpu,
            cpu_memory,
            mapped,
        }
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.cpu_memory);
            device.destroy_buffer(self.cpu, None);
            device.free_memory(self.cpu_memory, None);
            device.destroy_buffer(self.gpu, None);
            device.free_memory(self.gpu_memory, None);
        }
    }
}

fn create_r8_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    tiles: vk::Extent2D,
    usage: vk::ImageUsageFlags,
    purpose: &str,
) -> Result<ImageResource, AllocError> {
    ImageResource::create(
        device,
        memory_props,
        &ImageDesc {
            extent: tiles,
            format: vk::Format::R8_UINT,
            usage,
            layers: 1,
            aspect: vk::ImageAspectFlags::COLOR,
            samples: vk::SampleCountFlags::TYPE_1,
        },
        purpose,
    )
}

fn create_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    props: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .expect("create VRS mix buffer")
    };
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(find_memory_type(
                        memory_props,
                        reqs.memory_type_bits,
                        props,
                    )),
                None,
            )
            .expect("allocate VRS mix buffer")
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind VRS mix buffer");
    }
    (buffer, memory)
}

fn create_mapped_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> (vk::Buffer, vk::DeviceMemory, *mut u32) {
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .expect("create VRS mix readback")
    };
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let cached = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::HOST_CACHED;
    let plain = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let type_index = try_find_memory_type(memory_props, reqs.memory_type_bits, cached)
        .unwrap_or_else(|| find_memory_type(memory_props, reqs.memory_type_bits, plain));
    let memory = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(type_index),
                None,
            )
            .expect("allocate VRS mix readback")
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind VRS mix readback");
    }
    let mapped = unsafe {
        device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .expect("map VRS mix readback") as *mut u32
    };
    unsafe { std::ptr::write_bytes(mapped, 0, MIX_COUNT) };
    (buffer, memory, mapped)
}

impl super::Renderer {
    /// Rate-attachment view/texel size for the scene-pass begin. The image is
    /// still in GENERAL from the previous classify of this slot; the caller
    /// joins [`Self::vrs_rate_to_attachment_barrier`] into the begin batch.
    pub(super) fn vrs_rate_attachment(&self, slot: usize) -> RateAttachment {
        let vrs = self.targets.vrs.as_ref().expect("use_vrs implies vrs");
        RateAttachment {
            view: vrs.view(slot),
            texel_size: vrs.texel_size,
        }
    }

    /// GENERAL → `FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL`.
    ///
    /// Src: last classify of this slot (`COMPUTE_SHADER` / `SHADER_STORAGE_WRITE`).
    /// The slot timeline wait makes that submit complete; this barrier is the
    /// layout transition plus queue-side availability for the shading-rate
    /// attachment read. Dst: `FRAGMENT_SHADING_RATE_ATTACHMENT_KHR` /
    /// `FRAGMENT_SHADING_RATE_ATTACHMENT_READ`. Joins the scene-pass begin batch.
    pub(super) fn vrs_rate_to_attachment_barrier(
        &self,
        slot: usize,
    ) -> vk::ImageMemoryBarrier2<'_> {
        let vrs = self.targets.vrs.as_ref().expect("use_vrs implies vrs");
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR)
            .dst_access_mask(vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR)
            .image(vrs.image(slot))
            .subresource_range(color_range())
    }

    /// Rate image → GENERAL for the end-of-frame classify.
    ///
    /// If this slot bound the rate image as a shading-rate attachment (`vrs_ready`),
    /// src is the FSR read (`FRAGMENT_SHADING_RATE_ATTACHMENT_KHR` /
    /// `FRAGMENT_SHADING_RATE_ATTACHMENT_READ`) and old layout is FSR_OPTIMAL.
    /// First classify after create/recreate: image is still UNDEFINED, src NONE.
    /// Dst: `COMPUTE_SHADER` / `SHADER_STORAGE_WRITE`. Contents are rewritten, so
    /// UNDEFINED would also be a valid old layout on the FSR path; we keep FSR
    /// so the execution dependency on the attachment read is explicit.
    pub(super) fn vrs_rate_to_general_barrier(&self, slot: usize) -> vk::ImageMemoryBarrier2<'_> {
        let vrs = self.targets.vrs.as_ref().expect("classify_vrs implies vrs");
        let used = self.slots[FrameSlot::new(slot)].vrs_ready;
        if used {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR)
                .src_access_mask(vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(vrs.image(slot))
                .subresource_range(color_range())
        } else {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(vrs.image(slot))
                .subresource_range(color_range())
        }
    }

    /// History image → GENERAL for the end-of-frame classify.
    ///
    /// With history: src is the previous classify write (`COMPUTE_SHADER` /
    /// `SHADER_STORAGE_WRITE`), old GENERAL. First classify: UNDEFINED, src NONE.
    /// Dst: `COMPUTE_SHADER` / `SHADER_STORAGE_READ | SHADER_STORAGE_WRITE`.
    pub(super) fn vrs_history_to_general_barrier(
        &self,
        slot: usize,
    ) -> vk::ImageMemoryBarrier2<'_> {
        let vrs = self.targets.vrs.as_ref().expect("classify_vrs implies vrs");
        let use_history = self.slots[FrameSlot::new(slot)].vrs_history;
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(if use_history {
                vk::PipelineStageFlags2::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags2::NONE
            })
            .src_access_mask(if use_history {
                vk::AccessFlags2::SHADER_STORAGE_WRITE
            } else {
                vk::AccessFlags2::NONE
            })
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
            .old_layout(if use_history {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            })
            .new_layout(vk::ImageLayout::GENERAL)
            .image(vrs.history_image(slot))
            .subresource_range(color_range())
    }

    /// Zeros the mix histogram. Returns whether a CLEAR→COMPUTE memory barrier
    /// must join the upcoming pipeline barrier (only when profiling).
    pub(super) unsafe fn record_vrs_mix_fill(&self, cmd: vk::CommandBuffer, slot: usize) -> bool {
        if !crate::profile::is_enabled() {
            return false;
        }
        let vrs = self.targets.vrs.as_ref().expect("classify_vrs implies vrs");
        unsafe {
            self.device
                .device
                .cmd_fill_buffer(cmd, vrs.mix_gpu(slot), 0, MIX_BYTES, 0);
        }
        true
    }

    /// CLEAR / TRANSFER_WRITE → COMPUTE / SHADER_STORAGE_{READ,WRITE} for the
    /// mix fill. Only issued when [`Self::record_vrs_mix_fill`] returned true.
    pub(super) fn vrs_mix_fill_memory_barrier() -> vk::MemoryBarrier2<'static> {
        vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
    }

    /// Dispatch the classifier. Depth already rests in
    /// [`SAMPLEABLE_DEPTH_REST_LAYOUT`]; rate + history are already GENERAL
    /// (joined into the post-scene barrier). The rate image stays GENERAL
    /// until the next scene-pass begin of this slot. Mix fill/copy/host
    /// barriers run only while profiling.
    ///
    /// Only called when classifying, so both `vrs` and `vrs_compute` are
    /// present. `cmd` must be recording, outside a render pass.
    pub(super) unsafe fn record_vrs_generate(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        d_threshold: f32,
    ) {
        let device = &self.device.device;
        let vrs = self.targets.vrs.as_ref().expect("classify_vrs implies vrs");
        let compute = self
            .pipelines
            .vrs_compute
            .as_ref()
            .expect("classify_vrs implies vrs_compute");
        // Sampleable depth is already in SAMPLEABLE_DEPTH_REST_LAYOUT (the
        // post-scene rest barrier). MSAA: this is the single-sample resolve;
        // single-sampled it is the depth image itself.
        let depth = self.targets.sampleable_depth(slot);
        let tiles = vrs.tiles();
        let use_history = self.slots[FrameSlot::new(slot)].vrs_history;
        let allow_4x4 = self
            .device
            .fragment_shading_rate
            .as_ref()
            .is_some_and(|f| f.allow_4x4(self.targets.samples));
        let write_mix = crate::profile::is_enabled();
        let mut flags = 0u32;
        if allow_4x4 {
            flags |= FLAG_ALLOW_4X4;
        }
        if use_history {
            flags |= FLAG_USE_HISTORY;
        }
        if write_mix {
            flags |= FLAG_WRITE_MIX;
        }
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, compute.pipeline);
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(compute.depth_sampler)
                .image_view(depth.view())
                .image_layout(SAMPLEABLE_DEPTH_REST_LAYOUT)];
            let rate_info = [vk::DescriptorImageInfo::default()
                .image_view(vrs.view(slot))
                .image_layout(vk::ImageLayout::GENERAL)];
            let history_info = [vk::DescriptorImageInfo::default()
                .image_view(vrs.history_view(slot))
                .image_layout(vk::ImageLayout::GENERAL)];
            let mix_info = [vk::DescriptorBufferInfo::default()
                .buffer(vrs.mix_gpu(slot))
                .offset(0)
                .range(MIX_BYTES)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&rate_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&history_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&mix_info),
            ];
            self.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                compute.layout,
                0,
                &writes,
            );

            let push = VrsPush {
                d_threshold,
                texel_w: vrs.texel_size.width,
                texel_h: vrs.texel_size.height,
                tiles_x: tiles.width,
                tiles_y: tiles.height,
                depth_w: self.render_extent.width,
                depth_h: self.render_extent.height,
                flags,
            };
            device.cmd_push_constants(
                cmd,
                compute.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(cmd, tiles.width.div_ceil(8), tiles.height.div_ceil(8), 1);

            if write_mix {
                // COMPUTE / SHADER_STORAGE_WRITE → COPY / TRANSFER_READ.
                let mix_to_copy = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&mix_to_copy),
                );
                device.cmd_copy_buffer(
                    cmd,
                    vrs.mix_gpu(slot),
                    vrs.mix_cpu(slot),
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: MIX_BYTES,
                    }],
                );
                // COPY / TRANSFER_WRITE → HOST / HOST_READ. The mapped read
                // happens after this slot's timeline wait (one cycle later).
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKY_DEPTH: f32 = 1.0e-6;
    const RATE_1X1: u32 = 0;
    const RATE_2X2: u32 = (1 << 2) | 1;
    const RATE_4X4: u32 = (2 << 2) | 2;

    fn classify_tile(dmin: f32, dmax: f32, d_threshold: f32, allow_4x4: bool) -> u32 {
        if dmax < SKY_DEPTH {
            return if allow_4x4 { RATE_4X4 } else { RATE_2X2 };
        }
        let far = dmax < d_threshold;
        let flat = (dmax - dmin) < (d_threshold * 0.5);
        if far && flat { RATE_2X2 } else { RATE_1X1 }
    }

    fn conservative_rate(raw: u32, prev_raw: u32, neighbor_full: bool) -> u32 {
        if raw == RATE_1X1 || prev_raw == RATE_1X1 || neighbor_full {
            RATE_1X1
        } else {
            raw
        }
    }

    #[test]
    fn sky_is_coarse_and_uses_4x4_when_advertised() {
        assert_eq!(classify_tile(0.0, 0.0, 0.01, false), RATE_2X2);
        assert_eq!(classify_tile(0.0, 0.0, 0.01, true), RATE_4X4);
        assert_eq!(classify_tile(0.0, SKY_DEPTH * 0.5, 0.01, true), RATE_4X4);
    }

    #[test]
    fn far_flat_terrain_stays_2x2_even_when_4x4_exists() {
        assert_eq!(classify_tile(0.001, 0.002, 0.01, true), RATE_2X2);
        assert_eq!(classify_tile(0.001, 0.002, 0.01, false), RATE_2X2);
    }

    #[test]
    fn near_or_discontinuous_tiles_are_full_rate() {
        assert_eq!(classify_tile(0.5, 0.6, 0.01, true), RATE_1X1);
        // Far but a silhouette crosses the tile (range > half the threshold).
        assert_eq!(classify_tile(0.0, 0.009, 0.01, true), RATE_1X1);
    }

    #[test]
    fn conservative_holds_last_near_and_dilates_full_rate() {
        assert_eq!(conservative_rate(RATE_4X4, RATE_1X1, false), RATE_1X1);
        assert_eq!(conservative_rate(RATE_2X2, RATE_2X2, true), RATE_1X1);
        assert_eq!(conservative_rate(RATE_4X4, RATE_2X2, false), RATE_4X4);
        assert_eq!(conservative_rate(RATE_2X2, RATE_4X4, false), RATE_2X2);
        // Finer is always allowed.
        assert_eq!(conservative_rate(RATE_1X1, RATE_4X4, false), RATE_1X1);
    }

    #[test]
    fn rate_encoding_matches_vk_fragment_size_pack() {
        assert_eq!(RATE_1X1, 0);
        assert_eq!(RATE_2X2, 0b0101);
        assert_eq!(RATE_4X4, 0b1010);
    }

    #[test]
    fn push_layout_is_tight_u32s() {
        assert_eq!(size_of::<VrsPush>(), 32);
    }

    #[test]
    fn flag_bits_do_not_overlap() {
        assert_eq!(FLAG_ALLOW_4X4, 1);
        assert_eq!(FLAG_USE_HISTORY, 2);
        assert_eq!(FLAG_WRITE_MIX, 4);
        assert_eq!(FLAG_ALLOW_4X4 | FLAG_USE_HISTORY | FLAG_WRITE_MIX, 7);
    }
}
