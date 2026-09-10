//! Shared boilerplate for compute and graphics passes: shader module creation,
//! push-descriptor layouts, pipeline construction.

use ash::vk;

/// Load embedded SPIR-V and create shader module (label for panic messages).
pub(crate) fn shader_module(device: &ash::Device, bytes: &[u8], label: &str) -> vk::ShaderModule {
    let code =
        ash::util::read_spv(&mut std::io::Cursor::new(bytes)).expect("invalid embedded SPIR-V");
    unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .unwrap_or_else(|e| panic!("create {label} shader module: {e:?}"))
    }
}

/// Push-constant range for a pipeline layout, or `None` when `size` is 0
/// (Vulkan forbids a zero-sized range; sky LUT and similar passes omit it).
fn optional_push_range(stages: vk::ShaderStageFlags, size: u32) -> Option<vk::PushConstantRange> {
    (size > 0).then_some(
        vk::PushConstantRange::default()
            .stage_flags(stages)
            .offset(0)
            .size(size),
    )
}

/// Push-descriptor set layout plus a matching pipeline layout.
///
/// `push_stages` is the shader stage mask on the push-constant range (compute
/// or graphics). A `push_constant_size` of 0 omits the range entirely.
pub(crate) fn push_descriptor_layouts(
    device: &ash::Device,
    bindings: &[vk::DescriptorSetLayoutBinding],
    push_stages: vk::ShaderStageFlags,
    push_constant_size: u32,
    label: &str,
) -> (vk::DescriptorSetLayout, vk::PipelineLayout) {
    let set_layout = unsafe {
        device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(bindings),
                None,
            )
            .unwrap_or_else(|e| panic!("create {label} set layout: {e:?}"))
    };
    let push = optional_push_range(push_stages, push_constant_size).map(|range| [range]);
    let set_layouts = [set_layout];
    let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    if let Some(ref ranges) = push {
        info = info.push_constant_ranges(ranges);
    }
    let layout = unsafe {
        device
            .create_pipeline_layout(&info, None)
            .unwrap_or_else(|e| panic!("create {label} pipeline layout: {e:?}"))
    };
    (set_layout, layout)
}

/// One compute pipeline linked against `layout` from embedded SPIR-V; the
/// transient shader module is destroyed before returning, as every call site
/// already did by hand.
pub(crate) fn compute_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    bytes: &[u8],
    label: &str,
) -> vk::Pipeline {
    let module = shader_module(device, bytes, label);
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .module(module)
        .name(c"main")
        .stage(vk::ShaderStageFlags::COMPUTE);
    let pipeline = unsafe {
        device
            .create_compute_pipelines(
                cache,
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            )
            .map_err(|(_, err)| err)
            .unwrap_or_else(|e| panic!("create {label} compute pipeline: {e:?}"))[0]
    };
    unsafe { device.destroy_shader_module(module, None) };
    pipeline
}

/// Linear-filter, clamp-to-edge sampler — the primary read sampler exposure,
/// bloom's threshold stage, and the tonemap/sky LUT paths each build
/// identically (bloom's second, mip-filtered composite sampler is
/// pass-specific and stays put).
pub(crate) fn linear_clamp_sampler(device: &ash::Device, label: &str) -> vk::Sampler {
    unsafe {
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
            .unwrap_or_else(|e| panic!("create {label} sampler: {e:?}"))
    }
}

/// Nearest-filter, clamp-to-edge sampler — point depth fetches (fused TAA
/// reprojection, VRS) must not interpolate reversed-Z.
pub(crate) fn nearest_clamp_sampler(device: &ash::Device, label: &str) -> vk::Sampler {
    unsafe {
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
            .unwrap_or_else(|e| panic!("create {label} sampler: {e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_range_forwards_stage_and_size() {
        let compute = optional_push_range(vk::ShaderStageFlags::COMPUTE, 4).unwrap();
        assert_eq!(compute.stage_flags, vk::ShaderStageFlags::COMPUTE);
        assert_eq!(compute.offset, 0);
        assert_eq!(compute.size, 4);

        let frag = optional_push_range(vk::ShaderStageFlags::FRAGMENT, 48).unwrap();
        assert_eq!(frag.stage_flags, vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(frag.offset, 0);
        assert_eq!(frag.size, 48);

        let vert = optional_push_range(vk::ShaderStageFlags::VERTEX, 8).unwrap();
        assert_eq!(vert.stage_flags, vk::ShaderStageFlags::VERTEX);
        assert_eq!(vert.size, 8);
    }

    #[test]
    fn zero_size_push_is_omitted() {
        assert!(optional_push_range(vk::ShaderStageFlags::COMPUTE, 0).is_none());
        assert!(optional_push_range(vk::ShaderStageFlags::FRAGMENT, 0).is_none());
        assert!(optional_push_range(vk::ShaderStageFlags::VERTEX, 16).is_some());
    }
}
