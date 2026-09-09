//! Hi-Z depth pyramid: per-slot R32F mip chain and the compute pass that
//! builds it after the scene pass.
//!
//! Level 0 is half render resolution; each later mip is MIN (farthest
//! reversed-Z) of a 2×2, or 3×3 when a source axis is odd, down to 1×1.
//! The pyramid is sampled by next frame's cull compute, so it rests in
//! `SHADER_READ_ONLY_OPTIMAL` after the last mip write.

use ash::vk;

use super::pass;
use super::targets::HizChain;
use super::{SAMPLEABLE_DEPTH_REST_LAYOUT, color_range};

const HIZ_DEPTH_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hiz_depth.comp.spv"));
const HIZ_MIP_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hiz_mip.comp.spv"));

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HizPush {
    dst_dim: [u32; 2],
    src_dim: [u32; 2],
}

pub(crate) struct HizState {
    depth: vk::Pipeline,
    mip: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    depth_sampler: vk::Sampler,
}

impl HizState {
    pub(crate) fn new(device: &ash::Device, cache: vk::PipelineCache) -> Self {
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
        let (set_layout, layout) =
            pass::push_descriptor_layouts(device, &bindings, size_of::<HizPush>() as u32, "hiz");
        let depth = pass::compute_pipeline(device, cache, layout, HIZ_DEPTH_COMP, "hiz depth");
        let mip = pass::compute_pipeline(device, cache, layout, HIZ_MIP_COMP, "hiz mip");
        let depth_sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .expect("create Hi-Z depth sampler")
        };
        Self {
            depth,
            mip,
            layout,
            set_layout,
            depth_sampler,
        }
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.depth, None);
            device.destroy_pipeline(self.mip, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.depth_sampler, None);
        }
    }
}

impl super::Renderer {
    /// Pyramid → GENERAL for the end-of-frame reduce.
    ///
    /// Fully overwritten, so UNDEFINED is a valid old layout. When this slot
    /// was sampled by a later frame's cull (`ready`), src is that sampled read
    /// (`COMPUTE_SHADER` / `SHADER_SAMPLED_READ`) and old layout is
    /// `SHADER_READ_ONLY` so the execution dependency on the consumer is
    /// explicit — same pattern as [`Self::vrs_rate_to_general_barrier`].
    pub(super) fn hiz_to_general_barrier(&self, slot: usize) -> vk::ImageMemoryBarrier2<'_> {
        let chain = &self.targets.hiz[slot];
        let ready = chain.ready;
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(if ready {
                vk::PipelineStageFlags2::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags2::NONE
            })
            .src_access_mask(if ready {
                vk::AccessFlags2::SHADER_SAMPLED_READ
            } else {
                vk::AccessFlags2::NONE
            })
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .old_layout(if ready {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::UNDEFINED
            })
            .new_layout(vk::ImageLayout::GENERAL)
            .image(chain.image)
            .subresource_range(chain.all_mips())
    }

    /// Fill one slot's pyramid with 0 (reversed-Z far / empty). Used so the
    /// first cull after create, resize, or a flag toggle samples a pyramid
    /// that never occludes. Leaves the image in `SHADER_READ_ONLY`.
    #[allow(dead_code)] // sampled by the next-frame cull (occlusion test).
    pub(super) unsafe fn record_hiz_clear(&mut self, cmd: vk::CommandBuffer, slot: usize) {
        let device = &self.device.device;
        let image = self.targets.hiz[slot].image;
        let range = self.targets.hiz[slot].all_mips();
        unsafe {
            let to_dst = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .subresource_range(range)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_dst),
            );
            device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
                &[range],
            );
            let to_sampled = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .subresource_range(range)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        }
        self.targets.hiz[slot].ready = true;
    }

    /// Build this slot's pyramid from the just-written sampleable depth.
    /// Depth already rests in [`SAMPLEABLE_DEPTH_REST_LAYOUT`]; the pyramid
    /// is already GENERAL (joined into the post-scene barrier). Rests the
    /// pyramid in `SHADER_READ_ONLY` for the next frame's cull compute.
    ///
    /// `cmd` must be recording, outside a render pass.
    pub(super) unsafe fn record_hiz_generate(&mut self, cmd: vk::CommandBuffer, slot: usize) {
        let device = &self.device.device;
        let hiz = &self.hiz;
        let depth = self.targets.sampleable_depth(slot);
        let levels = self.targets.hiz[slot].mip_views.len();
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, hiz.depth);
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(hiz.depth_sampler)
                .image_view(depth.view())
                .image_layout(SAMPLEABLE_DEPTH_REST_LAYOUT)];
            let mip0_info = [vk::DescriptorImageInfo::default()
                .image_view(self.targets.hiz[slot].mip_views[0])
                .image_layout(vk::ImageLayout::GENERAL)];
            // Binding 1 is unused by reduce_depth; a valid storage view is
            // still required if the module declares it. Mip 0 is write-only
            // in that entry, so reusing it is a no-op read.
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&mip0_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&mip0_info),
            ];
            self.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                hiz.layout,
                0,
                &writes,
            );
            let mip0 = self.targets.hiz[slot].mip_extents[0];
            let push = HizPush {
                dst_dim: [mip0.width, mip0.height],
                src_dim: [self.render_extent.width, self.render_extent.height],
            };
            device.cmd_push_constants(
                cmd,
                hiz.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(cmd, mip0.width.div_ceil(8), mip0.height.div_ceil(8), 1);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, hiz.mip);
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
                    .image_view(self.targets.hiz[slot].mip_views[i - 1])
                    .image_layout(vk::ImageLayout::GENERAL)];
                let dst_info = [vk::DescriptorImageInfo::default()
                    .image_view(self.targets.hiz[slot].mip_views[i])
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
                    hiz.layout,
                    0,
                    &writes,
                );
                let src = self.targets.hiz[slot].mip_extents[i - 1];
                let dst = self.targets.hiz[slot].mip_extents[i];
                let push = HizPush {
                    dst_dim: [dst.width, dst.height],
                    src_dim: [src.width, src.height],
                };
                device.cmd_push_constants(
                    cmd,
                    hiz.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_dispatch(cmd, dst.width.div_ceil(8), dst.height.div_ceil(8), 1);
            }

            let chain = &self.targets.hiz[slot];
            let to_sampled = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(chain.image)
                .subresource_range(chain.all_mips())];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        }
        self.targets.hiz[slot].ready = true;
    }
}

impl HizChain {
    pub(crate) fn all_mips(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: self.mip_views.len() as u32,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_layout_is_two_uint2s() {
        assert_eq!(size_of::<HizPush>(), 16);
    }

    #[test]
    fn unused_color_range_helper_stays_single_mip() {
        // The pyramid uses `HizChain::all_mips`; the shared colour range is
        // still the single-mip helper used by every other colour image.
        let r = color_range();
        assert_eq!(r.level_count, 1);
    }
}
