/// Graphics pipelines. All use dynamic rendering, dynamic viewport/scissor
/// (never rebuilt on resize — only on MSAA changes), reversed-Z depth, and
/// SPIR-V embedded at compile time.
///
/// - `mesh3d`:  triangle list, Vertex{pos f32x3, uv f32x2, color u8x4}, depth
///   RW, cull back; samples the block texture array (set 0)
/// - `lines3d`: line list, same vertex/set, depth read only, no cull
/// - `tris2d`:  triangle list, Vertex2D{pos px, uv, color}, no depth, alpha blend
use ash::vk;
use std::io::Cursor;

use crate::mesh::Vertex;

pub const PUSH_BYTES_3D: u32 = 64; // Mat4
pub const PUSH_BYTES_2D: u32 = 8; // vec2 pixels_to_ndc

/// 2D overlay vertex: pixel position, atlas UV, RGBA8 color.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2D {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
}

const MESH3D_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d.vert.spv"));
const MESH3D_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d.frag.spv"));
const TRIS2D_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tris2d.vert.spv"));
const TRIS2D_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tris2d.frag.spv"));

pub struct Pipelines {
    pub layout_3d: vk::PipelineLayout,
    pub layout_2d: vk::PipelineLayout,
    pub mesh3d: vk::Pipeline,
    pub lines3d: vk::Pipeline,
    pub tris2d: vk::Pipeline,
}

impl Pipelines {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ash::Device,
        cache: vk::PipelineCache,
        color_format: vk::Format,
        depth_format: vk::Format,
        samples: vk::SampleCountFlags,
        atlas_set_layout: vk::DescriptorSetLayout,
        block_set_layout: vk::DescriptorSetLayout,
    ) -> Self {
        // Layouts
        let push_3d = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(PUSH_BYTES_3D)];
        let set_layouts_3d = [block_set_layout];
        let layout_3d_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts_3d)
            .push_constant_ranges(&push_3d);
        let layout_3d = unsafe {
            device
                .create_pipeline_layout(&layout_3d_info, None)
                .expect("Failed to create 3D pipeline layout")
        };

        let push_2d = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(PUSH_BYTES_2D)];
        let set_layouts = [atlas_set_layout];
        let layout_2d_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_2d);
        let layout_2d = unsafe {
            device
                .create_pipeline_layout(&layout_2d_info, None)
                .expect("Failed to create 2D pipeline layout")
        };

        // Vertex layouts
        let bindings_3d = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let attributes_3d = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32_SFLOAT,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 20,
            },
        ];

        let bindings_2d = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex2D>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let attributes_2d = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 16,
            },
        ];

        let mesh_vert = create_shader_module(device, MESH3D_VERT);
        let mesh_frag = create_shader_module(device, MESH3D_FRAG);
        let tri2d_vert = create_shader_module(device, TRIS2D_VERT);
        let tri2d_frag = create_shader_module(device, TRIS2D_FRAG);

        let builder = PipelineBuilder {
            device,
            cache,
            color_format,
            depth_format,
            samples,
        };

        // Depth: reversed-Z, so GREATER_OR_EQUAL and clear to 0.0.
        let mesh3d = builder.build(
            mesh_vert,
            mesh_frag,
            &bindings_3d,
            &attributes_3d,
            layout_3d,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            DepthMode::ReadWrite,
            vk::CullModeFlags::BACK,
            false,
        );
        let lines3d = builder.build(
            mesh_vert,
            mesh_frag,
            &bindings_3d,
            &attributes_3d,
            layout_3d,
            vk::PrimitiveTopology::LINE_LIST,
            DepthMode::ReadOnly,
            vk::CullModeFlags::NONE,
            false,
        );
        let tris2d = builder.build(
            tri2d_vert,
            tri2d_frag,
            &bindings_2d,
            &attributes_2d,
            layout_2d,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            DepthMode::Disabled,
            vk::CullModeFlags::NONE,
            true,
        );

        unsafe {
            device.destroy_shader_module(mesh_vert, None);
            device.destroy_shader_module(mesh_frag, None);
            device.destroy_shader_module(tri2d_vert, None);
            device.destroy_shader_module(tri2d_frag, None);
        }

        Self {
            layout_3d,
            layout_2d,
            mesh3d,
            lines3d,
            tris2d,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.mesh3d, None);
            device.destroy_pipeline(self.lines3d, None);
            device.destroy_pipeline(self.tris2d, None);
            device.destroy_pipeline_layout(self.layout_3d, None);
            device.destroy_pipeline_layout(self.layout_2d, None);
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum DepthMode {
    ReadWrite,
    ReadOnly,
    Disabled,
}

struct PipelineBuilder<'a> {
    device: &'a ash::Device,
    /// Renderer-owned, disk-backed cache; null is valid (no caching).
    cache: vk::PipelineCache,
    color_format: vk::Format,
    depth_format: vk::Format,
    samples: vk::SampleCountFlags,
}

impl PipelineBuilder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        vert: vk::ShaderModule,
        frag: vk::ShaderModule,
        bindings: &[vk::VertexInputBindingDescription],
        attributes: &[vk::VertexInputAttributeDescription],
        layout: vk::PipelineLayout,
        topology: vk::PrimitiveTopology,
        depth: DepthMode,
        cull: vk::CullModeFlags,
        blend: bool,
    ) -> vk::Pipeline {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .module(vert)
                .name(c"main")
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo::default()
                .module(frag)
                .name(c"main")
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(bindings)
            .vertex_attribute_descriptions(attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(topology)
            .primitive_restart_enable(false);

        // Dynamic — counts only.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Negative-viewport y flip keeps GL winding: visually-CCW = front.
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(cull)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let multisampling =
            vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(self.samples);

        let color_attachment = if blend {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
        } else {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)
        };
        let color_attachments = [color_attachment];
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachments);

        let depth_stencil = match depth {
            DepthMode::ReadWrite => vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL),
            DepthMode::ReadOnly => vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL),
            DepthMode::Disabled => vk::PipelineDepthStencilStateCreateInfo::default(),
        };

        let color_formats = [self.color_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(self.depth_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .dynamic_state(&dynamic_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&depth_stencil)
            .layout(layout)
            .push_next(&mut rendering_info);

        unsafe {
            self.device
                .create_graphics_pipelines(self.cache, &[pipeline_info], None)
                .map_err(|(_, err)| err)
                .expect("Failed to create graphics pipeline")[0]
        }
    }
}

fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> vk::ShaderModule {
    let code = ash::util::read_spv(&mut Cursor::new(bytes)).expect("Invalid embedded SPIR-V");
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe {
        device
            .create_shader_module(&info, None)
            .expect("Failed to create shader module")
    }
}
