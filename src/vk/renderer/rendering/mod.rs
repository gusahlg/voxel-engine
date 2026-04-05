/// This file defines everything that is related to the graphics pipeline and how commands are
/// processed to be data that can be rendered on screen.
use ash::{vk, Device};

use std::ffi::CString;

mod shader_helpers;
use shader_helpers::*;

pub struct RenderingBundle {
    pub pipeline_layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
    pub framebuffers: Vec<vk::Framebuffer>,
}

impl RenderingBundle {
    pub fn new(
        device: &Device,
        render_pass: vk::RenderPass,
        swapchain_extent: vk::Extent2D,
        swapchain_image_views: &[vk::ImageView],
    ) -> Self {
        let vert_code = read_spv("shaders_spv/tri.vert.spv");
        let frag_code = read_spv("shaders_spv/tri.frag.spv");

        let vert_module = unsafe { create_shader_module(device, &vert_code) };
        let frag_module = unsafe { create_shader_module(device, &frag_code) };

        let entry_point = CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .module(vert_module)
                .name(&entry_point)
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo::default()
                .module(frag_module)
                .name(&entry_point)
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        };

        let viewports = [viewport];
        let scissors = [scissor];

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);

        let color_blend_attachments = [color_blend_attachment];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| err)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        let mut framebuffers = Vec::with_capacity(swapchain_image_views.len());

        for &image_view in swapchain_image_views {
            let attachments = [image_view];

            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain_extent.width)
                .height(swapchain_extent.height)
                .layers(1);

            let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() };
            framebuffers.push(framebuffer);
        }

        Self {
            pipeline_layout,
            graphics_pipeline,
            framebuffers,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
        // Destroy framebuffers first
        for &fb in &self.framebuffers {
            device.destroy_framebuffer(fb, None);
        }

        // Then pipeline
        device.destroy_pipeline(self.graphics_pipeline, None);

        // Then pipeline layout
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
