//! HDR bloom pyramid + quarter-resolution spill composite.
//!
//! One compute pipeline per bloom stage (`threshold`, `downsample`, both entry
//! points of `bloom.comp.slang`) fills the per-slot [`BloomChain`] mip pyramid
//! owned by `RenderTargets`. The `threshold` dispatch bilinear-downsamples the
//! finalized HDR offscreen into mip 0 keeping only the exposed bright spill; a
//! chain of `downsample` dispatches builds the rest.
//!
//! A third compute dispatch (`spill.comp.slang`) then writes a quarter-res
//! RGBA16F image: the same 6-tap golden-angle gather over the bloom mip at
//! [`crate::genconst::BLOOM_SPIRAL_LOD`], plus the 4-tap godray depth march.
//! The tonemap fragment takes one bilinear tap of that image and adds it.
//!
//! No SPD / subgroup ops: a plain per-level dispatch chain is simpler and
//! subgroup-size-portable. The pyramid stays in `GENERAL` layout across the
//! whole chain (storage read+write); one transition to `SHADER_READ_ONLY`
//! hands it to the spill sampler. The spill image is written in `GENERAL` and
//! rests in `SHADER_READ_ONLY` for the tonemap fragment — exactly one image
//! barrier each way. The write-side barrier is batched with the pyramid's
//! rest transition when the pyramid ran this frame.
//!
//! Layout across frames in flight: one spill image per slot, like the bloom
//! pyramid. A slot is only re-recorded after its previous present copy has
//! retired (`acquire_slot` waits `copy_value` when `copy_slot` matches), so
//! the previous fragment sample of this image is done and discarding from
//! `UNDEFINED` on the write is valid. After the compute write it rests in
//! `SHADER_READ_ONLY_OPTIMAL` until this slot is recorded again. Recreate
//! tears the per-slot images down with `RenderTargets` (new ones begin
//! `UNDEFINED`); the 1×1 black fallback lives on [`BloomState`] (extent-
//! independent) and is not rebuilt on resize. Screenshot readback still
//! copies the swapchain image after the tonemap draw — spill is an
//! intermediate the present pass samples, never a capture source.
//!
//! Determinism: bloom and spill are a pure function of this frame's HDR +
//! depth — no temporal state, no wall-clock. Recorded on the render command
//! buffer of presented frames only (tonemap never samples an unpresented
//! pyramid/spill), right after the offscreen is finalized, so the
//! render→present semaphore makes them visible to the tonemap sample exactly
//! as it does for the offscreen itself. The mip chain is capped at
//! [`crate::genconst::BLOOM_MAX_MIPS`] (the LOD the spill pass actually
//! samples); bloom-off clears each slot's pyramid once, not every frame.
//! When bloom AND godrays are both off the spill dispatch is skipped and
//! tonemap binds the 1×1 black image.

use ash::vk;

use super::image::{ImageDesc, ImageResource, LayoutUse};
use super::pass;
use super::targets::HDR_COLOR_FORMAT;
use super::{SAMPLEABLE_DEPTH_REST_LAYOUT, color_range};
use crate::camera::{Godray, WarpMap};
use crate::genconst;
use crate::rev::FrameSlot;

const BLOOM_THRESHOLD_COMP: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bloom_threshold.comp.spv"));
const BLOOM_DOWNSAMPLE_COMP: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/bloom_downsample.comp.spv"));
const SPILL_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spill.comp.spv"));

// Shared push constants for threshold and downsample pipelines.
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpillPush {
    dst_dim: [u32; 2],
    s: f32,
    atan_s: f32,
    godray0: [f32; 4],
    godray1: [f32; 4],
    bloom_strength: f32,
    _pad: [f32; 3],
}

pub(crate) struct BloomState {
    threshold: vk::Pipeline,
    downsample: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    composite_sampler: vk::Sampler,
    spill: vk::Pipeline,
    spill_layout: vk::PipelineLayout,
    spill_set_layout: vk::DescriptorSetLayout,
    depth_sampler: vk::Sampler,
    /// 1×1 black RGBA16F, bound by tonemap when the spill dispatch is skipped.
    /// Primed once (clear → `SHADER_READ_ONLY`) on the render command buffer;
    /// extent-independent, so recreate does not rebuild it.
    black: ImageResource,
    black_primed: bool,
}

impl BloomState {
    pub(crate) fn black_view(&self) -> vk::ImageView {
        self.black.view()
    }

    fn ensure_black(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        if self.black_primed {
            return;
        }
        self.black.transition(device, cmd, LayoutUse::TransferClear);
        unsafe {
            device.cmd_clear_color_image(
                cmd,
                self.black.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: [0.0; 4] },
                &[self.black.subresource_range()],
            );
        }
        self.black
            .transition(device, cmd, LayoutUse::FragmentSampledAfterClear);
        self.black_primed = true;
    }
}

impl BloomState {
    pub(crate) fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        cache: vk::PipelineCache,
    ) -> BloomState {
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
        let (set_layout, layout) = pass::push_descriptor_layouts(
            device,
            &bindings,
            size_of::<BloomPush>() as u32,
            "bloom",
        );

        let threshold = pass::compute_pipeline(
            device,
            cache,
            layout,
            BLOOM_THRESHOLD_COMP,
            "bloom threshold",
        );
        let downsample = pass::compute_pipeline(
            device,
            cache,
            layout,
            BLOOM_DOWNSAMPLE_COMP,
            "bloom downsample",
        );

        let sampler = pass::linear_clamp_sampler(device, "bloom HDR");
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

        let spill_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let (spill_set_layout, spill_layout) = pass::push_descriptor_layouts(
            device,
            &spill_bindings,
            size_of::<SpillPush>() as u32,
            "spill",
        );
        let spill = pass::compute_pipeline(device, cache, spill_layout, SPILL_COMP, "spill");
        let depth_sampler = pass::nearest_clamp_sampler(device, "spill depth");

        let black = ImageResource::create(
            device,
            memory_props,
            &ImageDesc {
                extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                format: HDR_COLOR_FORMAT,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                layers: 1,
                aspect: vk::ImageAspectFlags::COLOR,
                samples: vk::SampleCountFlags::TYPE_1,
            },
        );

        BloomState {
            threshold,
            downsample,
            layout,
            set_layout,
            sampler,
            composite_sampler,
            spill,
            spill_layout,
            spill_set_layout,
            depth_sampler,
            black,
            black_primed: false,
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
            device.destroy_pipeline(self.spill, None);
            device.destroy_pipeline_layout(self.spill_layout, None);
            device.destroy_descriptor_set_layout(self.spill_set_layout, None);
            device.destroy_sampler(self.depth_sampler, None);
            self.black.destroy(device);
        }
    }
}

fn spill_to_general(image: vk::Image) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .image(image)
        .subresource_range(color_range())
}

fn spill_to_sampled(image: vk::Image) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image(image)
        .subresource_range(color_range())
}

impl super::Renderer {
    pub(crate) fn record_bloom_pass(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: FrameSlot,
        warp_map: WarpMap,
        godray: Godray,
    ) {
        let spill_live = self.flags.bloom || godray.strength > 0.0;
        let mut spill_write_issued = false;

        // All mip levels.
        let levels = self.targets.bloom[slot.index()].mip_views.len();
        let all_mips = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: levels as u32,
            base_array_layer: 0,
            layer_count: 1,
        };

        if self.flags.bloom {
            self.record_bloom_pyramid(cmd, slot, all_mips);
            // Pyramid rest + spill write-side, one barrier batch.
            let chain_image = self.targets.bloom[slot.index()].image;
            let spill_image = self.targets.spill[slot.index()].image();
            let bloom_to_sampled = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(chain_image)
                .subresource_range(all_mips);
            let barriers = [bloom_to_sampled, spill_to_general(spill_image)];
            unsafe {
                self.device.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&barriers),
                );
            }
            spill_write_issued = true;
        } else {
            // Lane off: clear the pyramid to black once. The spill pass then
            // skips the gather (`bloom_strength = 0`); subsequent frames reuse
            // the black pyramid. Toggling the lane back on invalidates `cleared`.
            if !self.targets.bloom[slot.index()].cleared {
                let chain_image = self.targets.bloom[slot.index()].image;
                let spill_image = self.targets.spill[slot.index()].image();
                let device = &self.device.device;
                unsafe {
                    let to_dst = [vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                        .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .image(chain_image)
                        .subresource_range(all_mips)];
                    device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&to_dst),
                    );
                    device.cmd_clear_color_image(
                        cmd,
                        chain_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue { float32: [0.0; 4] },
                        &[all_mips],
                    );
                    let bloom_to_sampled = vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::COMPUTE_SHADER
                                | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        )
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(chain_image)
                        .subresource_range(all_mips);
                    if spill_live {
                        let barriers = [bloom_to_sampled, spill_to_general(spill_image)];
                        device.cmd_pipeline_barrier2(
                            cmd,
                            &vk::DependencyInfo::default().image_memory_barriers(&barriers),
                        );
                        spill_write_issued = true;
                    } else {
                        let barriers = [bloom_to_sampled];
                        device.cmd_pipeline_barrier2(
                            cmd,
                            &vk::DependencyInfo::default().image_memory_barriers(&barriers),
                        );
                    }
                }
                self.targets.bloom[slot.index()].cleared = true;
            }
        }

        if spill_live {
            if !spill_write_issued {
                let spill_image = self.targets.spill[slot.index()].image();
                let barriers = [spill_to_general(spill_image)];
                unsafe {
                    self.device.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&barriers),
                    );
                }
            }
            self.record_spill_dispatch(cmd, slot, warp_map, godray);
            let spill_image = self.targets.spill[slot.index()].image();
            let barriers = [spill_to_sampled(spill_image)];
            unsafe {
                self.device.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&barriers),
                );
            }
        } else {
            self.bloom.ensure_black(&self.device.device, cmd);
        }
    }

    fn record_bloom_pyramid(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: FrameSlot,
        all_mips: vk::ImageSubresourceRange,
    ) {
        let device = &self.device.device;
        let bloom = &self.bloom;
        let (hdr_image, hdr_view) = self.hdr_of(slot.index());
        let exposure = self.exposure.current().0;
        self.targets.bloom[slot.index()].cleared = false;
        let chain = &self.targets.bloom[slot.index()];

        unsafe {
            // Transition HDR image for compute sampling.
            let to_compute = vk::ImageMemoryBarrier2::default()
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
                .image(hdr_image)
                .subresource_range(color_range());
            // Transition pyramid to storage layout.
            let to_general = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(chain.image)
                .subresource_range(all_mips);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_compute, to_general]),
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bloom.threshold);
            let hdr_info = [vk::DescriptorImageInfo::default()
                .sampler(bloom.sampler)
                .image_view(hdr_view)
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

            // Downsample remaining mip levels.
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bloom.downsample);
            for i in 1..chain.mip_views.len() {
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
        }
    }

    fn record_spill_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        slot: FrameSlot,
        warp_map: WarpMap,
        godray: Godray,
    ) {
        let device = &self.device.device;
        let bloom = &self.bloom;
        let extent = super::targets::spill_extent(self.render_extent);
        let (s, atan_s) = match warp_map {
            WarpMap::Identity => (0.0, 0.0),
            WarpMap::Active { s, atan_s } => (s, atan_s),
        };
        let bloom_strength = if self.flags.bloom {
            genconst::BLOOM_STRENGTH
        } else {
            0.0
        };

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, bloom.spill);
            let bloom_info = [vk::DescriptorImageInfo::default()
                .sampler(bloom.composite_sampler)
                .image_view(self.targets.bloom[slot.index()].sample_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(bloom.depth_sampler)
                .image_view(self.targets.sampleable_depth(slot.index()).view())
                .image_layout(SAMPLEABLE_DEPTH_REST_LAYOUT)];
            let spill_info = [vk::DescriptorImageInfo::default()
                .image_view(self.targets.spill[slot.index()].view())
                .image_layout(vk::ImageLayout::GENERAL)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&bloom_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&spill_info),
            ];
            self.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                bloom.spill_layout,
                0,
                &writes,
            );
            let push = SpillPush {
                dst_dim: [extent.width, extent.height],
                s,
                atan_s,
                godray0: [
                    godray.sun_uv[0],
                    godray.sun_uv[1],
                    godray.strength,
                    godray.jitter_uv[0],
                ],
                godray1: [
                    godray.tint[0],
                    godray.tint[1],
                    godray.tint[2],
                    godray.jitter_uv[1],
                ],
                bloom_strength,
                _pad: [0.0; 3],
            };
            device.cmd_push_constants(
                cmd,
                bloom.spill_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            let wg = genconst::SPILL_WG;
            device.cmd_dispatch(
                cmd,
                extent.width.div_ceil(wg),
                extent.height.div_ceil(wg),
                1,
            );
        }
    }
}
