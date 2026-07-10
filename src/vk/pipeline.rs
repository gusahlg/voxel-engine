/// Graphics pipelines. All use dynamic rendering, dynamic viewport/scissor
/// (never rebuilt on resize — only on MSAA changes), reversed-Z depth, and
/// SPIR-V embedded at compile time.
///
/// - `mesh3d`:      triangle list, MeshVertex{packed u32x2}, depth RW, cull
///   back; samples the block texture array (set 0, `layout_3d`)
/// - `debug_tris`:  triangle list, DebugVertex{pos f32x3, color u8x4}, depth RW,
///   cull back; view_proj push constant only (`layout_debug`, no descriptor set)
/// - `debug_lines`: line list, DebugVertex, depth read only, no cull, `layout_debug`
/// - `tris2d`:      triangle list, Vertex2D{pos px, uv, color}, no depth, alpha blend
use ash::vk;
use glam::Mat4;
use std::io::Cursor;

use crate::frame::SkyDesc;
use crate::mesh::{DebugVertex, MeshVertex, Pass};
use crate::vk::device::FragmentShadingRate;
use crate::vk::vertex_input::{VertexInput, vertex_struct};

pub const PUSH_BYTES_3D: u32 = size_of::<Mesh3dPush>() as u32; // view_proj + SkyLight
pub const PUSH_BYTES_DEBUG: u32 = size_of::<Mat4>() as u32; // view_proj only (unlit)
pub const PUSH_BYTES_2D: u32 = size_of::<[f32; 2]>() as u32; // pixels_to_ndc
pub const PUSH_BYTES_SKY: u32 = size_of::<SkyParams>() as u32; // full 128-byte sky block
pub const PUSH_BYTES_TONEMAP: u32 = size_of::<f32>() as u32; // exposure

/// Sky lighting/fog for the mesh pipeline — the tail of [`Mesh3dPush`], read by
/// `mesh3d`'s fragment stage. `sun_light`/`ambient` are rgb in `.xyz` (`.w`
/// unused); `fog` is rgb with density in `.w`. [`SkyLight::IDENTITY`] (white
/// sun, black ambient, zero density) reproduces the unlit look.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyLight {
    pub sun_light: [f32; 4],
    pub ambient: [f32; 4],
    pub fog: [f32; 4],
}

impl SkyLight {
    pub const IDENTITY: Self = Self {
        sun_light: [1.0, 1.0, 1.0, 1.0],
        ambient: [0.0, 0.0, 0.0, 1.0],
        fog: [0.0, 0.0, 0.0, 0.0],
    };
}

/// The mesh pipeline's push constant: camera matrix plus this frame's sky
/// lighting. Composed at record time from the frame's `view_proj` and
/// [`SkyLight`]; a single push replaces the former split at offset 64.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Mesh3dPush {
    pub view_proj: Mat4,
    pub sky: SkyLight,
}

/// The sky background pass's push constant, exactly 128 bytes (the guaranteed
/// `maxPushConstantsSize` minimum). Scalars ride in the `.w` lanes rather than
/// adding fields. Composed by the engine at record time from a [`SkyDesc`];
/// the app never builds it directly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyParams {
    inv_view_proj: Mat4, // 64 — clip→world; ray reconstructed camera-independently
    sun: [f32; 4],       // xyz sun dir, w = exposure
    zenith: [f32; 4],    // rgb, w = daylight (reserved for future use)
    horizon: [f32; 4],   // rgb, w = cos_inner (sun-disc inner edge)
    sun_tint: [f32; 4],  // rgb, w = cos_outer (sun-disc outer edge)
}

impl SkyParams {
    pub fn compose(inv_view_proj: Mat4, desc: &SkyDesc) -> Self {
        let n = |c: crate::color::Color| [c.r, c.g, c.b].map(|v| v as f32 / 255.0);
        let s = desc.sun_dir.normalize_or_zero();
        let [zr, zg, zb] = n(desc.zenith);
        let [hr, hg, hb] = n(desc.horizon);
        let [tr, tg, tb] = n(desc.sun_tint);
        let r = desc.sun_angular_radius;
        Self {
            inv_view_proj,
            sun: [s.x, s.y, s.z, desc.exposure],
            zenith: [zr, zg, zb, 0.0],
            horizon: [hr, hg, hb, r.cos()],
            sun_tint: [tr, tg, tb, (r * 2.0).cos()],
        }
    }
}

vertex_struct! {
    /// 2D overlay vertex: pixel position, atlas UV, RGBA8 color.
    pub struct Vertex2D {
        pub pos: [f32; 2],
        pub uv: [f32; 2],
        pub color: [u8; 4],
    }
}

const MESH3D_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d.vert.spv"));
const MESH3D_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d.frag.spv"));
const DEBUG_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/debug.vert.spv"));
const DEBUG_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/debug.frag.spv"));
const TRIS2D_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tris2d.vert.spv"));
const TRIS2D_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tris2d.frag.spv"));
const TRIS2D_TEX_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tris2d_tex.frag.spv"));
const SKY_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sky.vert.spv"));
const SKY_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sky.frag.spv"));
const TONEMAP_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.vert.spv"));
const TONEMAP_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.frag.spv"));
const VRS_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vrs.comp.spv"));

/// The VRS classifier compute pipeline plus the depth sampler it reads through.
/// Present exactly when attachment VRS is enabled. Set 0 is push-descriptor:
/// binding 0 = depth (combined image sampler), binding 1 = rate storage image.
pub struct VrsCompute {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
    pub depth_sampler: vk::Sampler,
}

pub struct Pipelines {
    pub layout_3d: vk::PipelineLayout,
    /// Push-constant-only (view_proj) layout for immediate debug geometry.
    pub layout_debug: vk::PipelineLayout,
    pub layout_2d: vk::PipelineLayout,
    pub mesh3d: vk::Pipeline,
    /// Same config as `mesh3d`, but with depth bias enabled so tiles render
    /// slightly toward the reversed-Z far plane — full-res chunks win at
    /// coincident depth. Selected per-draw via `DrawEntry::biased`.
    pub mesh3d_biased: vk::Pipeline,
    /// Same vertex/fragment modules and `layout_3d` as `mesh3d`, but alpha
    /// blends and reads (never writes) depth. Selected for [`Pass::Transparent`].
    pub mesh3d_transparent: vk::Pipeline,
    pub debug_tris: vk::Pipeline,
    /// Same debug modules/layout as `debug_tris`, but alpha blends and reads
    /// (never writes) depth — for translucent ground decals (contact shadows).
    pub debug_tris_blend: vk::Pipeline,
    pub debug_lines: vk::Pipeline,
    pub tris2d: vk::Pipeline,
    /// Variant of `tris2d` that samples RGBA texture instead of R8 atlas.
    pub tris2d_tex: vk::Pipeline,
    /// Vertex-less fullscreen background pass; push-constant only, no descriptor
    /// set. Depth-tests (read-only) at the reversed-Z far plane so it shades
    /// only pixels the terrain left uncovered.
    pub sky: vk::Pipeline,
    pub layout_sky: vk::PipelineLayout,
    /// Fullscreen AgX tonemap: samples the HDR offscreen (set 0 push descriptor,
    /// `tonemap_set_layout`) and writes the LDR swapchain image.
    pub tonemap: vk::Pipeline,
    pub layout_tonemap: vk::PipelineLayout,
    pub tonemap_set_layout: vk::DescriptorSetLayout,
    /// Linear-clamp sampler pushed with the HDR image for the tonemap draw.
    pub tonemap_sampler: vk::Sampler,
    /// `Some` exactly when attachment VRS is enabled (`fsr.is_some()`).
    pub vrs_compute: Option<VrsCompute>,
}

impl Pipelines {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ash::Device,
        cache: vk::PipelineCache,
        color_format: vk::Format,
        present_format: vk::Format,
        depth_format: vk::Format,
        samples: vk::SampleCountFlags,
        atlas_set_layout: vk::DescriptorSetLayout,
        mesh3d_set_layout: vk::DescriptorSetLayout,
        fsr: Option<&FragmentShadingRate>,
    ) -> Self {
        // 3D set 0: binding 0 = offsets SSBO (vertex), binding 1 = texture
        // array (fragment) — one push set (Vulkan allows at most one per
        // layout). 2D layout uses its own set 0 for the atlas.
        let push_3d = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(PUSH_BYTES_3D)];
        let set_layouts_3d = [mesh3d_set_layout];
        let layout_3d_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts_3d)
            .push_constant_ranges(&push_3d);
        let layout_3d = unsafe {
            device
                .create_pipeline_layout(&layout_3d_info, None)
                .expect("Failed to create 3D pipeline layout")
        };

        // Debug layout: view_proj push constant only, no descriptor set.
        let push_debug = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(PUSH_BYTES_DEBUG)];
        let layout_debug_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_debug);
        let layout_debug = unsafe {
            device
                .create_pipeline_layout(&layout_debug_info, None)
                .expect("Failed to create debug pipeline layout")
        };

        // Sky layout: one 128-byte push constant across both stages, no set.
        let push_sky = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(PUSH_BYTES_SKY)];
        let layout_sky_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_sky);
        let layout_sky = unsafe {
            device
                .create_pipeline_layout(&layout_sky_info, None)
                .expect("Failed to create sky pipeline layout")
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

        // Tonemap: set 0 binding 0 = combined image sampler (the HDR offscreen),
        // pushed at record time. Plus a 4-byte exposure push constant.
        let tonemap_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let tonemap_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(&tonemap_binding),
                    None,
                )
                .expect("Failed to create tonemap set layout")
        };
        let tonemap_sampler = unsafe {
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
                .expect("Failed to create tonemap sampler")
        };
        let push_tonemap = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(PUSH_BYTES_TONEMAP)];
        let set_layouts_tonemap = [tonemap_set_layout];
        let layout_tonemap_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts_tonemap)
            .push_constant_ranges(&push_tonemap);
        let layout_tonemap = unsafe {
            device
                .create_pipeline_layout(&layout_tonemap_info, None)
                .expect("Failed to create tonemap pipeline layout")
        };

        // Vertex layouts derived from struct fields (see vertex_input).
        // Locations, offsets, and formats are kept in sync automatically.
        let bindings_3d = [MeshVertex::binding()];
        let attributes_3d = MeshVertex::ATTRIBUTES;

        let bindings_debug = [DebugVertex::binding()];
        let attributes_debug = DebugVertex::ATTRIBUTES;

        let bindings_2d = [Vertex2D::binding()];
        let attributes_2d = Vertex2D::ATTRIBUTES;

        let mesh_vert = create_shader_module(device, MESH3D_VERT);
        let mesh_frag = create_shader_module(device, MESH3D_FRAG);
        let debug_vert = create_shader_module(device, DEBUG_VERT);
        let debug_frag = create_shader_module(device, DEBUG_FRAG);
        let tri2d_vert = create_shader_module(device, TRIS2D_VERT);
        let tri2d_frag = create_shader_module(device, TRIS2D_FRAG);
        let tri2d_tex_frag = create_shader_module(device, TRIS2D_TEX_FRAG);
        let sky_vert = create_shader_module(device, SKY_VERT);
        let sky_frag = create_shader_module(device, SKY_FRAG);

        let builder = PipelineBuilder {
            device,
            cache,
            color_format,
            depth_format,
            samples,
            fsr_enabled: fsr.is_some(),
        };

        // Depth: reversed-Z, so GREATER_OR_EQUAL and clear to 0.0.
        let mesh3d = builder.build(
            mesh_vert,
            mesh_frag,
            &bindings_3d,
            attributes_3d,
            layout_3d,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::ReadWrite,
                cull: vk::CullModeFlags::BACK,
                blend: false,
                vrs: true,
                depth_bias: None,
            },
        );
        let mesh3d_biased = builder.build(
            mesh_vert,
            mesh_frag,
            &bindings_3d,
            attributes_3d,
            layout_3d,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::ReadWrite,
                cull: vk::CullModeFlags::BACK,
                blend: false,
                vrs: true,
                depth_bias: Some((-2.0, -1.0)),
            },
        );
        // Transparent world geometry: same modules/layout, alpha blend, depth
        // read-only (all opaque wrote depth first; water tests but never writes).
        let mesh3d_transparent = builder.build(
            mesh_vert,
            mesh_frag,
            &bindings_3d,
            attributes_3d,
            layout_3d,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::ReadOnly,
                cull: vk::CullModeFlags::BACK,
                blend: true,
                vrs: true,
                depth_bias: None,
            },
        );
        let debug_tris = builder.build(
            debug_vert,
            debug_frag,
            &bindings_debug,
            attributes_debug,
            layout_debug,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::ReadWrite,
                cull: vk::CullModeFlags::BACK,
                blend: false,
                vrs: false,
                depth_bias: None,
            },
        );
        // Translucent debug geometry (contact shadows): alpha blend, depth
        // read-only so decals blend over terrain without occluding it.
        let debug_tris_blend = builder.build(
            debug_vert,
            debug_frag,
            &bindings_debug,
            attributes_debug,
            layout_debug,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::ReadOnly,
                cull: vk::CullModeFlags::NONE,
                blend: true,
                vrs: false,
                depth_bias: None,
            },
        );
        let debug_lines = builder.build(
            debug_vert,
            debug_frag,
            &bindings_debug,
            attributes_debug,
            layout_debug,
            PipelineConfig {
                topology: vk::PrimitiveTopology::LINE_LIST,
                depth: DepthMode::ReadOnly,
                cull: vk::CullModeFlags::NONE,
                blend: false,
                vrs: false,
                depth_bias: None,
            },
        );
        let tris2d = builder.build(
            tri2d_vert,
            tri2d_frag,
            &bindings_2d,
            attributes_2d,
            layout_2d,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::Disabled,
                cull: vk::CullModeFlags::NONE,
                blend: true,
                vrs: false,
                depth_bias: None,
            },
        );

        // Minimap pipeline: same vertex/layout as tris2d, only fragment sampler changed.
        let tris2d_tex = builder.build(
            tri2d_vert,
            tri2d_tex_frag,
            &bindings_2d,
            attributes_2d,
            layout_2d,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::Disabled,
                cull: vk::CullModeFlags::NONE,
                blend: true,
                vrs: false,
                depth_bias: None,
            },
        );

        // Sky: no vertex input (verts synthesised from SV_VertexID), depth
        // read-only at the far plane, opaque, no cull. Same GREATER_OR_EQUAL
        // compare as the scene, so it passes only where depth is still cleared.
        let sky = builder.build(
            sky_vert,
            sky_frag,
            &[],
            &[],
            layout_sky,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::ReadOnly,
                cull: vk::CullModeFlags::NONE,
                blend: false,
                vrs: true,
                depth_bias: None,
            },
        );

        // Tonemap: its own builder — writes the present format at single-sample
        // with no depth attachment; never VRS.
        let tonemap_vert = create_shader_module(device, TONEMAP_VERT);
        let tonemap_frag = create_shader_module(device, TONEMAP_FRAG);
        let tonemap_builder = PipelineBuilder {
            device,
            cache,
            color_format: present_format,
            depth_format: vk::Format::UNDEFINED,
            samples: vk::SampleCountFlags::TYPE_1,
            fsr_enabled: false,
        };
        let tonemap = tonemap_builder.build(
            tonemap_vert,
            tonemap_frag,
            &[],
            &[],
            layout_tonemap,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::Disabled,
                cull: vk::CullModeFlags::NONE,
                blend: false,
                vrs: false,
                depth_bias: None,
            },
        );

        unsafe {
            device.destroy_shader_module(tonemap_vert, None);
            device.destroy_shader_module(mesh_vert, None);
            device.destroy_shader_module(mesh_frag, None);
            device.destroy_shader_module(debug_vert, None);
            device.destroy_shader_module(debug_frag, None);
            device.destroy_shader_module(tri2d_vert, None);
            device.destroy_shader_module(tri2d_frag, None);
            device.destroy_shader_module(tri2d_tex_frag, None);
            device.destroy_shader_module(sky_vert, None);
            device.destroy_shader_module(sky_frag, None);
        }

        let vrs_compute = fsr.map(|_| create_vrs_compute(device, cache));

        Self {
            vrs_compute,
            layout_3d,
            layout_debug,
            layout_2d,
            mesh3d,
            mesh3d_biased,
            mesh3d_transparent,
            debug_tris,
            debug_tris_blend,
            debug_lines,
            tris2d,
            tris2d_tex,
            sky,
            layout_sky,
            tonemap,
            layout_tonemap,
            tonemap_set_layout,
            tonemap_sampler,
        }
    }

    /// The 3D pipeline for a mesh's draw pass. Exhaustive so a new [`Pass`]
    /// variant forces a matching pipeline here.
    pub fn pipeline_for(&self, pass: Pass) -> vk::Pipeline {
        match pass {
            Pass::Opaque => self.mesh3d,
            Pass::Transparent => self.mesh3d_transparent,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if let Some(v) = &self.vrs_compute {
                device.destroy_pipeline(v.pipeline, None);
                device.destroy_pipeline_layout(v.layout, None);
                device.destroy_descriptor_set_layout(v.set_layout, None);
                device.destroy_sampler(v.depth_sampler, None);
            }
            device.destroy_pipeline(self.mesh3d, None);
            device.destroy_pipeline(self.mesh3d_biased, None);
            device.destroy_pipeline(self.mesh3d_transparent, None);
            device.destroy_pipeline(self.debug_tris, None);
            device.destroy_pipeline(self.debug_tris_blend, None);
            device.destroy_pipeline(self.debug_lines, None);
            device.destroy_pipeline(self.tris2d, None);
            device.destroy_pipeline(self.tris2d_tex, None);
            device.destroy_pipeline(self.sky, None);
            device.destroy_pipeline(self.tonemap, None);
            device.destroy_pipeline_layout(self.layout_3d, None);
            device.destroy_pipeline_layout(self.layout_debug, None);
            device.destroy_pipeline_layout(self.layout_2d, None);
            device.destroy_pipeline_layout(self.layout_sky, None);
            device.destroy_pipeline_layout(self.layout_tonemap, None);
            device.destroy_descriptor_set_layout(self.tonemap_set_layout, None);
            device.destroy_sampler(self.tonemap_sampler, None);
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
    /// Whether attachment VRS is enabled; when true, `vrs` configs chain the
    /// shading-rate state so the rate attachment drives coarse shading.
    fsr_enabled: bool,
}

/// Per-pipeline knobs for `PipelineBuilder::build`, named at each call site
/// to avoid a positional-bool footgun.
struct PipelineConfig {
    topology: vk::PrimitiveTopology,
    depth: DepthMode,
    cull: vk::CullModeFlags,
    blend: bool,
    /// Opt this pipeline into attachment VRS (geometry passes only).
    vrs: bool,
    depth_bias: Option<(f32, f32)>,
}

impl PipelineBuilder<'_> {
    fn build(
        &self,
        vert: vk::ShaderModule,
        frag: vk::ShaderModule,
        bindings: &[vk::VertexInputBindingDescription],
        attributes: &[vk::VertexInputAttributeDescription],
        layout: vk::PipelineLayout,
        cfg: PipelineConfig,
    ) -> vk::Pipeline {
        let PipelineConfig {
            topology,
            depth,
            cull,
            blend,
            vrs,
            depth_bias,
        } = cfg;
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

        // Viewport and scissor are dynamic (set at render time).
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
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(depth_bias.is_some())
            .depth_bias_constant_factor(depth_bias.map_or(0.0, |b| b.0))
            .depth_bias_slope_factor(depth_bias.map_or(0.0, |b| b.1));

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

        let mut fsr_state = vk::PipelineFragmentShadingRateStateCreateInfoKHR::default()
            .fragment_size(vk::Extent2D {
                width: 1,
                height: 1,
            })
            .combiner_ops([
                vk::FragmentShadingRateCombinerOpKHR::KEEP,
                vk::FragmentShadingRateCombinerOpKHR::REPLACE,
            ]);

        // Every pipeline drawn in a pass that binds a rate attachment must
        // carry this flag — even the non-VRS ones (debug/2D shade at 1×1). So
        // it's keyed on the builder's `fsr_enabled`, not the per-pipeline `vrs`.
        let create_flags = if self.fsr_enabled {
            vk::PipelineCreateFlags::RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_KHR
        } else {
            vk::PipelineCreateFlags::empty()
        };

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .flags(create_flags)
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
        if vrs && self.fsr_enabled {
            pipeline_info = pipeline_info.push_next(&mut fsr_state);
        }

        unsafe {
            self.device
                .create_graphics_pipelines(self.cache, &[pipeline_info], None)
                .map_err(|(_, err)| err)
                .expect("Failed to create graphics pipeline")[0]
        }
    }
}

fn create_vrs_compute(device: &ash::Device, cache: vk::PipelineCache) -> VrsCompute {
    let bindings = [
        // Depth, sampled by the classifier.
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // Rate image, written by the classifier.
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let set_layout = unsafe {
        device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&bindings),
                None,
            )
            .expect("Failed to create VRS set layout")
    };

    let push = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(size_of::<super::vrs::VrsPush>() as u32)];
    let set_layouts = [set_layout];
    let layout = unsafe {
        device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )
            .expect("Failed to create VRS pipeline layout")
    };

    let module = create_shader_module(device, VRS_COMP);
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .module(module)
        .name(c"main")
        .stage(vk::ShaderStageFlags::COMPUTE);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipeline = unsafe {
        device
            .create_compute_pipelines(cache, &[info], None)
            .map_err(|(_, err)| err)
            .expect("Failed to create VRS compute pipeline")[0]
    };
    unsafe { device.destroy_shader_module(module, None) };

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
            .expect("Failed to create VRS depth sampler")
    };

    VrsCompute {
        pipeline,
        layout,
        set_layout,
        depth_sampler,
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
