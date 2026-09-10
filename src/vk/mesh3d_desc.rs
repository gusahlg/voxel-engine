use ash::{khr, vk};

/// Create mesh3d push-descriptor set layout.
pub fn create_mesh3d_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
        .bindings(&bindings);
    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .expect("Failed to create mesh3d set layout")
    }
}

/// Push mesh3d descriptors.
#[allow(clippy::too_many_arguments)]
pub fn push_mesh3d_descriptors(
    push: &khr::push_descriptor::Device,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    records: vk::Buffer,
    dyns: vk::Buffer,
    tex_sampler: vk::Sampler,
    tex_view: vk::ImageView,
    ubo: vk::Buffer,
    cascade_ubo: vk::Buffer,
    shadow_sampler: vk::Sampler,
    shadow_view: vk::ImageView,
) {
    let buffer_infos = [vk::DescriptorBufferInfo::default()
        .buffer(records)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let dyn_infos = [vk::DescriptorBufferInfo::default()
        .buffer(dyns)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let image_infos = [vk::DescriptorImageInfo::default()
        .sampler(tex_sampler)
        .image_view(tex_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let ubo_infos = [vk::DescriptorBufferInfo::default()
        .buffer(ubo)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let cascade_infos = [vk::DescriptorBufferInfo::default()
        .buffer(cascade_ubo)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let shadow_infos = [vk::DescriptorImageInfo::default()
        .sampler(shadow_sampler)
        .image_view(shadow_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&ubo_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&cascade_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&shadow_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&dyn_infos),
    ];
    unsafe {
        push.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &writes);
    }
}

/// Pushes only binding 5 (previous slot's sampleable depth + nearest/clamp
/// sampler) for the water-absorption blend variant. Layered on top of an
/// already-pushed 0-4 set (same compatible layout, so the earlier writes stay
/// live). The image rests in `SHADER_READ_ONLY_OPTIMAL` from the previous
/// frame's rest barrier (or the 1×1 dummy, primed to the same layout).
pub fn push_prev_depth(
    push: &khr::push_descriptor::Device,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    sampler: vk::Sampler,
    depth_view: vk::ImageView,
) {
    let image_infos = [vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(depth_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_binding(5)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_infos)];
    unsafe {
        push.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &writes);
    }
}
