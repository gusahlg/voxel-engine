//! Per-frame octahedral cloud LUT: compute march → bilinear tap in sky.frag.

use ash::vk;

use super::image::LayoutUse;
use super::pass;
use crate::genconst;
use crate::rev::FrameSlot;

const SKY_CLOUD_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sky_cloud.comp.spv"));

pub(crate) struct SkyCloudState {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

impl SkyCloudState {
    pub(crate) fn new(device: &ash::Device, cache: vk::PipelineCache) -> SkyCloudState {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let (set_layout, layout) =
            pass::push_descriptor_layouts(device, &bindings, 0, "sky cloud LUT");
        let pipeline = pass::compute_pipeline(device, cache, layout, SKY_CLOUD_COMP, "sky cloud");
        SkyCloudState {
            pipeline,
            layout,
            set_layout,
        }
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// True when the slab is entirely behind the camera (including the game's
/// `camera_y = f32::MAX` clouds-off sentinel).
pub(crate) fn clouds_hidden(camera_y: f32) -> bool {
    camera_y >= genconst::CLOUD_BOTTOM + genconst::CLOUD_THICKNESS
}

impl super::Renderer {
    /// March (or zero) this slot's cloud LUT before the scene pass. Zeros the LUT
    /// when the slab is hidden so the fragment's bilinear tap composites as a no-op.
    pub(crate) fn record_sky_cloud_lut(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: usize,
        camera_y: f32,
    ) {
        let device = &self.device.device;

        if clouds_hidden(camera_y) {
            let lut = &mut self.targets.sky_cloud[slot];
            lut.transition_discard(device, cmd, LayoutUse::TransferClear);
            unsafe {
                device.cmd_clear_color_image(
                    cmd,
                    lut.image(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: [0.0; 4] },
                    &[lut.subresource_range()],
                );
            }
            lut.transition(device, cmd, LayoutUse::FragmentSampledAfterClear);
            return;
        }

        self.targets.sky_cloud[slot].transition_discard(
            device,
            cmd,
            LayoutUse::ComputeStorageWrite,
        );
        let lut_view = self.targets.sky_cloud[slot].view();

        let n = genconst::SKY_CLOUD_LUT_SIZE;
        let wg = genconst::SKY_CLOUD_LUT_WG;
        let ubo = self.ubo_ring.buffer(FrameSlot::new(slot));
        let pipeline = self.sky_cloud.pipeline;
        let layout = self.sky_cloud.layout;
        let ubo_infos = [vk::DescriptorBufferInfo::default()
            .buffer(ubo)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let image_infos = [vk::DescriptorImageInfo::default()
            .image_view(lut_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&ubo_infos),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&image_infos),
        ];
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &writes,
            );
            device.cmd_dispatch(cmd, n.div_ceil(wg), n.div_ceil(wg), 1);
        }
        self.targets.sky_cloud[slot].transition(device, cmd, LayoutUse::SampledAfterComputeWrite);
    }
}

#[cfg(test)]
mod tests {
    use super::clouds_hidden;
    use crate::genconst;

    #[test]
    fn clouds_hidden_at_or_above_slab_top() {
        let top = genconst::CLOUD_BOTTOM + genconst::CLOUD_THICKNESS;
        assert!(!clouds_hidden(0.0));
        assert!(!clouds_hidden(genconst::CLOUD_BOTTOM));
        assert!(clouds_hidden(top));
        assert!(clouds_hidden(f32::MAX));
    }

    fn oct_encode(d: [f32; 3]) -> [f32; 2] {
        let inv = d[0].abs() + d[1].abs() + d[2].abs();
        [d[0] / inv * 0.5 + 0.5, d[2] / inv * 0.5 + 0.5]
    }

    fn oct_decode(uv: [f32; 2]) -> [f32; 3] {
        let fx = uv[0] * 2.0 - 1.0;
        let fy = uv[1] * 2.0 - 1.0;
        let n = [fx, 1.0 - fx.abs() - fy.abs(), fy];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        [n[0] / len, n[1] / len, n[2] / len]
    }

    #[test]
    fn hemisphere_oct_roundtrips_zenith_and_horizon() {
        for d in [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.70710677, 0.70710677, 0.0],
        ] {
            let back = oct_decode(oct_encode(d));
            for (b, want) in back.iter().zip(d) {
                assert!((b - want).abs() < 1e-5, "roundtrip {d:?} -> {back:?}");
            }
        }
    }
}
