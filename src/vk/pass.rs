//! Shared boilerplate for compute/post passes: shader module creation,
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

/// Push-descriptor layout and pipeline layout with compute push constants.
pub(crate) fn push_descriptor_layouts(
    device: &ash::Device,
    bindings: &[vk::DescriptorSetLayoutBinding],
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
    let push = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size)];
    let set_layouts = [set_layout];
    let layout = unsafe {
        device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )
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

/// Linear-filter, clamp-to-edge sampler — the primary read sampler taa,
/// exposure, and bloom's threshold stage each build identically (bloom's
/// second, mip-filtered composite sampler is pass-specific and stays put).
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
