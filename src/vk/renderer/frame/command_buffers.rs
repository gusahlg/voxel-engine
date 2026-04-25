/// Provides helpers for command buffers
use ash::vk;

use crate::vk::renderer::swapchain::SwapchainInfo;
use crate::vk::renderer::device::Device;

pub fn record_command_buffer(device: &Device,
                            swapchain_info: &SwapchainInfo,
                            graphics_pipeline: vk::Pipeline,
                            command_buffer: vk::CommandBuffer, 
                            image_index: usize,
                            ) -> Result<(), vk::Result> {
    let begin_info = vk::CommandBufferBeginInfo::default();

    unsafe {
        device.logical_device.begin_command_buffer(command_buffer, &begin_info)?;

        let pre_render_barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(swapchain_info.images[image_index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        let pre_render_dependency = vk::DependencyInfo::default()
            .image_memory_barriers(&pre_render_barrier);
        device.logical_device.cmd_pipeline_barrier2(
            command_buffer,
            &pre_render_dependency,
        );

        let color_attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(swapchain_info.image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            })];

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain_info.extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachment);

        device.logical_device.cmd_begin_rendering(command_buffer, &rendering_info);

        device.logical_device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            graphics_pipeline,
        );

        device.logical_device.cmd_draw(command_buffer, 3, 1, 0, 0);

        device.logical_device.cmd_end_rendering(command_buffer);

        let post_render_barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(swapchain_info.images[image_index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        let post_render_dependency = vk::DependencyInfo::default()
            .image_memory_barriers(&post_render_barrier);
        device.logical_device.cmd_pipeline_barrier2(
            command_buffer,
            &post_render_dependency,
        );

        device.logical_device.end_command_buffer(command_buffer)?;
    }

    Ok(())
}
