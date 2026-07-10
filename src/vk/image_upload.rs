/// Shared image upload: create device-local image, stage via temp buffer,
/// and copy with one blocking submit. Font atlas and block textures both use
/// this; callers pack their own bytes and copy regions.
use ash::{khr, vk};

use super::alloc::find_memory_type;

/// Push-descriptor set layout shared by the font atlas and block texture
/// array: binding 0 = combined image sampler, fragment stage.
pub fn create_sampler_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
        .bindings(&bindings);
    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .expect("Failed to create descriptor set layout")
    }
}

/// Pushes `sampler`/`view` as binding 0 (combined image sampler) of `set` in
/// the bound `layout`. Shared by the font atlas and block texture array.
pub fn push_combined_image_sampler(
    push: &khr::push_descriptor::Device,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    set: u32,
    sampler: vk::Sampler,
    view: vk::ImageView,
) {
    let image_infos = [vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_infos)];
    unsafe {
        push.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::GRAPHICS, layout, set, &writes);
    }
}

/// Describes one image to upload. `bytes` is the fully-packed staging blob;
/// `regions` addresses ranges of it into the image's mip levels / array
/// layers (built by the caller, since packing differs per texture kind).
pub struct ImageUpload<'a> {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub view_type: vk::ImageViewType,
    pub bytes: &'a [u8],
    pub regions: &'a [vk::BufferImageCopy],
}

/// Creates image + memory + view, uploads bytes, destroys staging buffer.
/// Returns (image, memory, view); blocks until copy completes.
pub fn upload_image(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    params: &ImageUpload,
) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
    // Query memory properties once and reuse for both the image and the
    // staging buffer.
    let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let size = params.bytes.len() as vk::DeviceSize;

    // Image
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(params.format)
        .extent(vk::Extent3D {
            width: params.extent.width,
            height: params.extent.height,
            depth: 1,
        })
        .mip_levels(params.mip_levels)
        .array_layers(params.array_layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe {
        device
            .create_image(&image_info, None)
            .expect("Failed to create image")
    };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(find_memory_type(
            &memory_props,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ));
    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .expect("Failed to allocate image memory")
    };
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .expect("Failed to bind image memory");
    }

    // Staging buffer (temporary, destroyed after the blocking upload).
    let staging_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let staging = unsafe {
        device
            .create_buffer(&staging_info, None)
            .expect("Failed to create staging buffer")
    };
    let staging_req = unsafe { device.get_buffer_memory_requirements(staging) };
    let staging_alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(staging_req.size)
        .memory_type_index(find_memory_type(
            &memory_props,
            staging_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ));
    let staging_memory = unsafe {
        device
            .allocate_memory(&staging_alloc, None)
            .expect("Failed to allocate staging memory")
    };
    unsafe {
        device
            .bind_buffer_memory(staging, staging_memory, 0)
            .expect("Failed to bind staging memory");
        let ptr = device
            .map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())
            .expect("Failed to map staging memory");
        std::ptr::copy_nonoverlapping(params.bytes.as_ptr(), ptr as *mut u8, params.bytes.len());
        device.unmap_memory(staging_memory);
    }

    // One-time upload with layout transitions.
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe {
        device
            .allocate_command_buffers(&alloc)
            .expect("Failed to allocate upload command buffer")[0]
    };
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    let subresource = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: params.mip_levels,
        base_array_layer: 0,
        layer_count: params.array_layers,
    };
    unsafe {
        device
            .begin_command_buffer(cmd, &begin)
            .expect("Failed to begin upload command buffer");

        let to_transfer = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(subresource)];
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&to_transfer),
        );

        device.cmd_copy_buffer_to_image(
            cmd,
            staging,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            params.regions,
        );

        let to_sampled = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image)
            .subresource_range(subresource)];
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
        );

        device
            .end_command_buffer(cmd)
            .expect("Failed to end upload command buffer");

        let buffers = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);
        device
            .queue_submit(graphics_queue, &[submit], vk::Fence::null())
            .expect("Failed to submit image upload");
        device
            .queue_wait_idle(graphics_queue)
            .expect("Image upload wait failed");

        device.free_command_buffers(command_pool, &buffers);
        device.destroy_buffer(staging, None);
        device.free_memory(staging_memory, None);
    }

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(params.view_type)
        .format(params.format)
        .subresource_range(subresource);
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .expect("Failed to create image view")
    };

    (image, memory, view)
}
