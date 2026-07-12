//! HDR bloom — soft-threshold + downsample pyramid, composited by tonemap.
//!
//! One compute pipeline per stage (`threshold`, `downsample`, both entry points
//! of `bloom.comp.slang`) fills the per-slot [`BloomChain`] mip pyramid owned by
//! `RenderTargets`. The `threshold` dispatch bilinear-downsamples the finalized
//! HDR offscreen into mip 0 keeping only the exposed bright spill; a chain of
//! `downsample` dispatches builds the rest. The tonemap pass then samples the
//! chain with a golden-angle spiral and adds the spill before its sigmoid.
//!
//! No SPD / subgroup ops: a plain per-level dispatch chain is simpler and
//! subgroup-size-portable. The pyramid stays in `GENERAL` layout across the whole chain
//! (storage read+write); one transition to `SHADER_READ_ONLY` hands it to the
//! tonemap sampler. This is the ONE new target and its first (and only)
//! consumer, so nothing here is generalized into a reusable HDR-mip abstraction.
//!
//! Determinism: bloom is a pure function of this frame's HDR — no temporal state,
//! no wall-clock. Recorded on the render command buffer right after the offscreen
//! is finalized, so the render→present semaphore makes the pyramid visible to the
//! tonemap sample exactly as it does for the offscreen itself.

use ash::vk;

use crate::genconst;
use crate::skeleton::FrameSlot;

const BLOOM_THRESHOLD_COMP: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bloom_threshold.comp.spv"));
const BLOOM_DOWNSAMPLE_COMP: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bloom_downsample.comp.spv"));

/// Push constants for both `bloom.comp` entry points (layout matches the Slang
/// `Push` struct exactly). `src_dim`/`exposure`/`thr_*` are used by whichever
/// stage needs them; the other stage ignores the unused fields.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomPush {
    dst_dim: [u32; 2],
    src_dim: [u32; 2],
    exposure: f32,
    thr_lo: f32,
    thr_hi: f32,
    thr_scale: f32,
}

fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> vk::ShaderModule {
    let code =
        ash::util::read_spv(&mut std::io::Cursor::new(bytes)).expect("invalid embedded SPIR-V");
    unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .expect("create bloom shader module")
    }
}

/// The two bloom compute pipelines (shared layout) plus the linear-clamp sampler
/// the `threshold` stage reads the HDR offscreen through. Set 0 is push-descriptor:
/// binding 0 = HDR offscreen (combined image sampler), binding 1 = source mip
/// (storage), binding 2 = destination mip (storage).
pub(crate) struct BloomState {
    threshold: vk::Pipeline,
    downsample: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    /// HDR-offscreen read sampler for the `threshold` stage (linear, no mips).
    sampler: vk::Sampler,
    /// Mip-filtered sampler the tonemap pass composites the pyramid through.
    composite_sampler: vk::Sampler,
}

impl BloomState {
    /// The mip-filtered sampler the tonemap present-copy pairs with the pyramid's
    /// `sample_view` (binding 1) — LINEAR mip mode so `SampleLevel` blurs across
    /// the downsampled levels.
    pub(crate) fn composite_sampler(&self) -> vk::Sampler {
        self.composite_sampler
    }
}

impl BloomState {
    pub(crate) fn new(device: &ash::Device, cache: vk::PipelineCache) -> BloomState {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(&bindings),
                    None,
                )
                .expect("create bloom set layout")
        };
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<BloomPush>() as u32)];
        let set_layouts = [set_layout];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .expect("create bloom pipeline layout")
        };

        let build = |bytes: &[u8]| {
            let module = create_shader_module(device, bytes);
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .module(module)
                .name(c"main")
                .stage(vk::ShaderStageFlags::COMPUTE);
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let pipeline = unsafe {
                device
                    .create_compute_pipelines(cache, &[info], None)
                    .map_err(|(_, err)| err)
                    .expect("create bloom compute pipeline")[0]
            };
            unsafe { device.destroy_shader_module(module, None) };
            pipeline
        };
        let threshold = build(BLOOM_THRESHOLD_COMP);
        let downsample = build(BLOOM_DOWNSAMPLE_COMP);

        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .expect("create bloom HDR sampler")
        };
        let composite_sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .max_lod(vk::LOD_CLAMP_NONE)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .expect("create bloom composite sampler")
        };

        BloomState {
            threshold,
            downsample,
            layout,
            set_layout,
            sampler,
            composite_sampler,
        }
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.threshold, None);
            device.destroy_pipeline(self.downsample, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler(self.composite_sampler, None);
        }
    }
}

impl super::Renderer {
    /// Build this slot's bloom pyramid from its finalized HDR offscreen. Recorded
    /// after the offscreen reaches `SHADER_READ_ONLY_OPTIMAL` (the [`HdrReadable`]
    /// finalize); leaves the pyramid in `SHADER_READ_ONLY_OPTIMAL` for the tonemap
    /// present-copy. `exposure` is the metered multiplier the tonemap will apply,
    /// used only in the threshold knee so the spill tracks the displayed frame.
    ///
    /// [`HdrReadable`]: super::HdrReadable
    pub(crate) fn record_bloom_pass(&self, cmd: vk::CommandBuffer, slot: FrameSlot) {
        let device = &self.device.device;
        let bloom = &self.bloom;
        let chain = &self.targets.bloom[slot.index()];
        let offscreen = &self.targets.offscreen[slot.index()];
        let exposure = self.exposure.current().0;
        let levels = chain.mip_views.len();

        // Subresource spanning every mip (color_range() is a single level).
        let all_mips = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: levels as u32,
            base_array_layer: 0,
            layer_count: 1,
        };

        // Lane off: clear the pyramid to black and hand it to the tonemap as-is —
        // the composite `hdr += spill·…` then adds zero, a no-op with no shader or
        // push-constant branch. Cheap enough to run unconditionally each frame.
        if !self.flags.bloom {
            unsafe {
                let to_dst = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(chain.image)
                    .subresource_range(all_mips)];
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_dst),
                );
                device.cmd_clear_color_image(
                    cmd,
                    chain.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: [0.0; 4] },
                    &[all_mips],
                );
                let to_sampled = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(chain.image)
                    .subresource_range(all_mips)];
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
                );
            }
            return;
        }

        unsafe {
            // (1) Offscreen is SHADER_READ_ONLY (finalized) but was only made
            // visible to the tonemap's FRAGMENT stage; extend visibility to this
            // COMPUTE read. src masks cover every finalize path (render-pass store,
            // TAA transfer copy, exposure compute) so the barrier is path-agnostic.
            let to_compute = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::TRANSFER,
                )
                .src_access_mask(
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE
                        | vk::AccessFlags2::TRANSFER_WRITE,
                )
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(offscreen.image)
                .subresource_range(super::color_range())];
            // Bloom image → GENERAL for storage writes. Fully overwritten every
            // frame, so discard the old contents (old layout UNDEFINED).
            let to_general = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(chain.image)
                .subresource_range(all_mips)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&[to_compute[0], to_general[0]]),
            );

            // (2) Threshold → mip 0 (bilinear tap = 2×2 average + soft knee).
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bloom.threshold);
            let hdr_info = [vk::DescriptorImageInfo::default()
                .sampler(bloom.sampler)
                .image_view(offscreen.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let mip0_info = [vk::DescriptorImageInfo::default()
                .image_view(chain.mip_views[0])
                .image_layout(vk::ImageLayout::GENERAL)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&hdr_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&mip0_info),
            ];
            self.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                bloom.layout,
                0,
                &writes,
            );
            let mip0 = chain.mip_extents[0];
            let push = BloomPush {
                dst_dim: [mip0.width, mip0.height],
                // Threshold samples the offscreen via uv, so src_dim is unused here.
                src_dim: [self.render_extent.width, self.render_extent.height],
                exposure,
                thr_lo: genconst::BLOOM_THRESHOLD_LO,
                thr_hi: genconst::BLOOM_THRESHOLD_HI,
                thr_scale: genconst::BLOOM_THRESHOLD_SCALE,
            };
            device.cmd_push_constants(
                cmd,
                bloom.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(cmd, mip0.width.div_ceil(8), mip0.height.div_ceil(8), 1);

            // (3) Downsample chain: each level reads the previous (RAW → a compute
            // write→read barrier between dispatches).
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bloom.downsample);
            for i in 1..levels {
                let rw = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&rw),
                );

                let src_info = [vk::DescriptorImageInfo::default()
                    .image_view(chain.mip_views[i - 1])
                    .image_layout(vk::ImageLayout::GENERAL)];
                let dst_info = [vk::DescriptorImageInfo::default()
                    .image_view(chain.mip_views[i])
                    .image_layout(vk::ImageLayout::GENERAL)];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&src_info),
                    vk::WriteDescriptorSet::default()
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&dst_info),
                ];
                self.device.push_descriptor.cmd_push_descriptor_set(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    bloom.layout,
                    0,
                    &writes,
                );
                let src = chain.mip_extents[i - 1];
                let dst = chain.mip_extents[i];
                let push = BloomPush {
                    dst_dim: [dst.width, dst.height],
                    src_dim: [src.width, src.height],
                    exposure,
                    thr_lo: genconst::BLOOM_THRESHOLD_LO,
                    thr_hi: genconst::BLOOM_THRESHOLD_HI,
                    thr_scale: genconst::BLOOM_THRESHOLD_SCALE,
                };
                device.cmd_push_constants(
                    cmd,
                    bloom.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_dispatch(cmd, dst.width.div_ceil(8), dst.height.div_ceil(8), 1);
            }

            // (4) Whole pyramid → SHADER_READ_ONLY for the tonemap composite.
            let to_sampled = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(chain.image)
                .subresource_range(all_mips)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        }
    }
}
