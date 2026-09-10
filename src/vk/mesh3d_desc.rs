use ash::{khr, vk};

/// mesh3d push-descriptor bindings (set 0). Binding 5 is the previous frame's
/// sampleable depth (water absorb); 7 is the material SSBO after dyns at 6.
pub const MESH3D_BINDING_RECORDS: u32 = 0;
pub const MESH3D_BINDING_BLOCK_TEX: u32 = 1;
pub const MESH3D_BINDING_FRAME_UBO: u32 = 2;
pub const MESH3D_BINDING_CASCADE_UBO: u32 = 3;
pub const MESH3D_BINDING_SHADOW_MAP: u32 = 4;
pub const MESH3D_BINDING_DEPTH_INPUT: u32 = 5;
pub const MESH3D_BINDING_DYNS: u32 = 6;
pub const MESH3D_BINDING_MATERIALS: u32 = 7;

/// Create mesh3d push-descriptor set layout.
pub fn create_mesh3d_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(MESH3D_BINDING_RECORDS)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        vk::DescriptorSetLayoutBinding::default()
            .binding(MESH3D_BINDING_BLOCK_TEX)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(MESH3D_BINDING_FRAME_UBO)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(MESH3D_BINDING_CASCADE_UBO)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(MESH3D_BINDING_SHADOW_MAP)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(MESH3D_BINDING_DEPTH_INPUT)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(MESH3D_BINDING_DYNS)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        vk::DescriptorSetLayoutBinding::default()
            .binding(MESH3D_BINDING_MATERIALS)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
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
    materials: vk::Buffer,
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
    let mat_infos = [vk::DescriptorBufferInfo::default()
        .buffer(materials)
        .offset(0)
        .range(vk::WHOLE_SIZE)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_binding(MESH3D_BINDING_RECORDS)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(MESH3D_BINDING_BLOCK_TEX)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(MESH3D_BINDING_FRAME_UBO)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&ubo_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(MESH3D_BINDING_CASCADE_UBO)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&cascade_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(MESH3D_BINDING_SHADOW_MAP)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&shadow_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(MESH3D_BINDING_DYNS)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&dyn_infos),
        vk::WriteDescriptorSet::default()
            .dst_binding(MESH3D_BINDING_MATERIALS)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&mat_infos),
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
        .dst_binding(MESH3D_BINDING_DEPTH_INPUT)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_infos)];
    unsafe {
        push.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &writes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_ssbo_is_the_next_free_binding() {
        // 0 records, 1 block tex, 2 frame UBO, 3 cascade UBO, 4 shadow map,
        // 5 previous-frame sampleable depth, 6 dyns, 7 materials.
        assert_eq!(MESH3D_BINDING_RECORDS, 0);
        assert_eq!(MESH3D_BINDING_BLOCK_TEX, 1);
        assert_eq!(MESH3D_BINDING_FRAME_UBO, 2);
        assert_eq!(MESH3D_BINDING_CASCADE_UBO, 3);
        assert_eq!(MESH3D_BINDING_SHADOW_MAP, 4);
        assert_eq!(MESH3D_BINDING_DEPTH_INPUT, 5);
        assert_eq!(MESH3D_BINDING_DYNS, 6);
        assert_eq!(MESH3D_BINDING_MATERIALS, 7);
        let mut used = vec![
            MESH3D_BINDING_RECORDS,
            MESH3D_BINDING_BLOCK_TEX,
            MESH3D_BINDING_FRAME_UBO,
            MESH3D_BINDING_CASCADE_UBO,
            MESH3D_BINDING_SHADOW_MAP,
            MESH3D_BINDING_DEPTH_INPUT,
            MESH3D_BINDING_DYNS,
            MESH3D_BINDING_MATERIALS,
        ];
        used.sort_unstable();
        let n = used.len();
        used.dedup();
        assert_eq!(used.len(), n, "mesh3d bindings must be unique");
        assert_eq!(
            MESH3D_BINDING_MATERIALS,
            MESH3D_BINDING_DYNS + 1,
            "materials take the next free binding after dyns"
        );
    }
}
