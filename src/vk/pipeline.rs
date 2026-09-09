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

use crate::frame::SkyDesc;
use crate::mesh::{DebugVertex, MeshVertex, Pass};
use crate::vk::device::FragmentShadingRate;
use crate::vk::pass;
use crate::vk::vertex_input::{VertexInput, vertex_struct};

pub const PUSH_BYTES_3D: u32 = size_of::<Mesh3dPush>() as u32;
pub const PUSH_BYTES_DEBUG: u32 = size_of::<DebugPush>() as u32;
pub const PUSH_BYTES_2D: u32 = size_of::<[f32; 2]>() as u32; // pixels_to_ndc
pub const PUSH_BYTES_SKY: u32 = size_of::<SkyParams>() as u32; // inv_view_proj + disc cosines
const _: () = assert!(size_of::<SkyParams>() <= 128);
// exposure + wide-FOV remap coefficients (s, atan_s) + vignette; see camera::WarpPush.
pub const PUSH_BYTES_TONEMAP: u32 = size_of::<crate::camera::WarpPush>() as u32;
pub const PUSH_BYTES_TONEMAP_TAA: u32 = size_of::<super::taa::TonemapTaaPush>() as u32;

/// Frame camera eye split: exact integer block + fractional part.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EyeSplit {
    pub block: [i32; 3],
    pub _pad0: i32,
    pub frac: [f32; 3],
    pub _pad1: f32,
}

impl EyeSplit {
    pub fn of(eye: glam::DVec3) -> Self {
        let block = eye.floor();
        Self {
            block: block.as_ivec3().to_array(),
            _pad0: 0,
            frac: (eye - block).as_vec3().to_array(),
            _pad1: 0.0,
        }
    }
}

/// 3D push constant data.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Mesh3dPush {
    pub view_proj: Mat4,
    pub clip: f32,
    /// Vertical half-height of the full-res slab.
    pub clip_v: f32,
    /// 1/render_extent for previous-depth UV. Zero when that depth is invalid.
    pub inv_render_extent: [f32; 2],
    pub eye: EyeSplit,
}

// Struct must fit within 128-byte push budget.
const _: () = assert!(size_of::<Mesh3dPush>() <= 128);

/// Debug push constant: view_proj only.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugPush {
    pub view_proj: Mat4,
}

/// Sky push constant: inverse view-proj, unit sun dir, disc tint, precomputed
/// sun/moon cone cosines (the per-pixel `cos(radius·SUN_DISC_*)` hoist).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyParams {
    inv_view_proj: Mat4,
    sun: [f32; 4],
    sun_tint: [f32; 4],
    moon: [f32; 4],
}

impl SkyParams {
    pub fn compose(inv_view_proj: Mat4, desc: &SkyDesc) -> Self {
        let s = desc.sun_dir.normalize_or_zero();
        // sun_tint is ALREADY linear (LinearRgb, non-quantising boundary) — no
        // /255 decode: the disc composites in linear light shader-side.
        let [tr, tg, tb] = desc.sun_tint.0;
        let r = desc.sun_angular_radius;
        let moon_r = r * crate::genconst::MOON_RADIUS_SCALE;
        Self {
            inv_view_proj,
            sun: [s.x, s.y, s.z, (r * crate::genconst::SUN_DISC_CORE).cos()],
            sun_tint: [tr, tg, tb, (r * crate::genconst::SUN_DISC_RIM).cos()],
            moon: [
                (moon_r * crate::genconst::SUN_DISC_CORE).cos(),
                (moon_r * crate::genconst::SUN_DISC_RIM).cos(),
                0.0,
                0.0,
            ],
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
/// Full mesh3d.frag module (water/absorb branch + LOD-slab `discard`): the
/// Blend pipelines only.
const MESH3D_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d.frag.spv"));
/// Full-res opaque variant: no water code and no `discard`, so the driver
/// keeps early depth writes on for every opaque terrain fragment.
const MESH3D_OPAQUE_FRAG: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d_opaque.frag.spv"));
/// Coarse-LOD opaque variant: the slab-clip `discard`, no cascade sampling.
const MESH3D_LOD_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d_lod.frag.spv"));
/// Full-res opaque with every optional lane compiled out (`MESH3D_LEAN`).
const MESH3D_OPAQUE_LEAN_FRAG: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d_opaque_lean.frag.spv"));
/// Coarse-LOD opaque + the same lane-off diet.
const MESH3D_LOD_LEAN_FRAG: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d_lod_lean.frag.spv"));
/// Shader variant that samples the previous frame's depth for water absorption;
/// built when MSAA is off (single-sample depth is directly sampleable).
const MESH3D_WATER_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh3d_water.frag.spv"));

pub(crate) const DEBUG_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/debug.vert.spv"));
const DEBUG_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/debug.frag.spv"));
const TRIS2D_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tris2d.vert.spv"));
const TRIS2D_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tris2d.frag.spv"));
const TRIS2D_TEX_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tris2d_tex.frag.spv"));
const SKY_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sky.vert.spv"));
const SKY_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sky.frag.spv"));
const TONEMAP_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.vert.spv"));
const TONEMAP_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.frag.spv"));
const TONEMAP_TAA_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tonemap_taa.frag.spv"));
const VRS_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vrs.comp.spv"));

/// The VRS classifier compute pipeline plus the depth sampler it reads through.
/// Present exactly when attachment VRS is enabled. Set 0 is push-descriptor:
/// binding 0 = depth (combined image sampler), binding 1 = rate storage image,
/// binding 2 = history storage image, binding 3 = mix histogram SSBO.
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
    /// Full-res opaque terrain (`cull::Group::Opaque`): the no-`discard`
    /// fragment variant, depth read/write, cull back.
    pub mesh3d: vk::Pipeline,
    /// Coarse-LOD opaque terrain (`cull::Group::OpaqueLod`): same state as
    /// `mesh3d` with the slab-clip `discard` fragment variant.
    pub mesh3d_lod: vk::Pipeline,
    /// `mesh3d` fragment with `MESH3D_LEAN` (no cascade/candle/ambient/fog).
    pub mesh3d_lean: vk::Pipeline,
    /// `mesh3d_lod` fragment with `MESH3D_LEAN`. Same layout as `mesh3d`.
    pub mesh3d_lod_lean: vk::Pipeline,
    /// The full fragment module with `layout_3d`, alpha blended, reads (never
    /// writes) depth. Selected for [`Pass::Blend`].
    pub mesh3d_transparent: vk::Pipeline,
    /// Water-absorption variant when MSAA is off (samples previous-frame depth);
    /// fallback to mesh3d_transparent otherwise.
    pub mesh3d_transparent_absorb: Option<vk::Pipeline>,
    pub debug_tris: vk::Pipeline,
    /// Same debug modules/layout as `debug_tris`, but alpha blends and reads
    /// (never writes) depth — for translucent ground decals (contact shadows).
    pub debug_tris_blend: vk::Pipeline,
    pub debug_lines: vk::Pipeline,
    pub tris2d: vk::Pipeline,
    /// Variant of `tris2d` that samples RGBA texture instead of R8 atlas.
    pub tris2d_tex: vk::Pipeline,
    /// Present-format, single-sample variants of the two overlay pipelines, drawn
    /// AFTER the tonemap resample so the wide-FOV warp never bends the HUD/minimap.
    /// Unused in rectilinear mode (the overlay stays in the offscreen scene pass).
    pub tris2d_present: vk::Pipeline,
    pub tris2d_tex_present: vk::Pipeline,
    /// Vertex-less fullscreen background pass: geometry push constant + set 0
    /// binding 0 (cloud LUT) and binding 1 (the shared per-frame `FrameUniforms`).
    /// Depth-tests (read-only) at the reversed-Z far plane so it shades only
    /// pixels the terrain left uncovered.
    pub sky: vk::Pipeline,
    pub layout_sky: vk::PipelineLayout,
    pub sky_set_layout: vk::DescriptorSetLayout,
    /// Linear-clamp sampler pushed with the octahedral cloud LUT.
    pub sky_lut_sampler: vk::Sampler,
    /// Fullscreen tonemap: samples the HDR offscreen and the quarter-res spill
    /// (set 0 push descriptor, `tonemap_set_layout`) and writes the LDR swapchain.
    /// TAA-off path: one color attachment, no TAA ALU (`tonemap.frag` without
    /// `-DTAA_FUSED`).
    pub tonemap: vk::Pipeline,
    pub layout_tonemap: vk::PipelineLayout,
    pub tonemap_set_layout: vk::DescriptorSetLayout,
    /// Linear-clamp sampler pushed with the HDR image, the spill image, and
    /// (when TAA is on) the read-history image.
    pub tonemap_sampler: vk::Sampler,
    /// Fused TAA tonemap (`-DTAA_FUSED`): two color attachments (swapchain +
    /// RGBA16F history). Own layout: extra history/depth bindings and the
    /// larger push block. Off path keeps `tonemap` so TAA-off costs nothing.
    pub tonemap_taa: vk::Pipeline,
    pub layout_tonemap_taa: vk::PipelineLayout,
    pub tonemap_taa_set_layout: vk::DescriptorSetLayout,
    /// Nearest-clamp sampler for the render-res depth tap (reversed-Z).
    pub tonemap_depth_sampler: vk::Sampler,
    /// Overlay variants with a second (history) color attachment, write-mask
    /// empty on attachment 1, so they can draw in the same rendering as the
    /// fused tonemap. Attachment 1 is not written: HUD stays out of history.
    /// `Some` only when `independentBlend` is enabled (distinct per-attachment
    /// blend states). Without it the present path draws overlay in a second
    /// one-attachment rendering using `tris2d_present` / `tris2d_tex_present`.
    pub tris2d_present_taa: Option<vk::Pipeline>,
    pub tris2d_tex_present_taa: Option<vk::Pipeline>,
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
        independent_blend: bool,
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

        // Sky layout: fragment push constant (inv VP + disc cosines) plus set 0
        // binding 0 = cloud LUT, binding 1 = FrameUniforms. Dedicated rather than
        // sharing mesh3d_set_layout: the LUT is a sampled image the mesh pass
        // never touches.
        let push_sky = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(PUSH_BYTES_SKY)];
        let sky_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let sky_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(&sky_bindings),
                    None,
                )
                .expect("Failed to create sky set layout")
        };
        let set_layouts_sky = [sky_set_layout];
        let layout_sky_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts_sky)
            .push_constant_ranges(&push_sky);
        let layout_sky = unsafe {
            device
                .create_pipeline_layout(&layout_sky_info, None)
                .expect("Failed to create sky pipeline layout")
        };
        let sky_lut_sampler = pass::linear_clamp_sampler(device, "sky cloud LUT");

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

        // Tonemap: set 0 binding 0 = the HDR offscreen, binding 1 = the
        // quarter-res spill (bloom composite + godrays). Both combined image
        // samplers pushed at record time. Plus the tonemap push constant.
        let tonemap_binding = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
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

        // Fused TAA tonemap: HDR + spill + history + depth, larger push.
        let tonemap_taa_binding = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(super::taa::TONEMAP_TAA_HDR_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(super::taa::TONEMAP_TAA_SPILL_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(super::taa::TONEMAP_TAA_HISTORY_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(super::taa::TONEMAP_TAA_DEPTH_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let tonemap_taa_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(&tonemap_taa_binding),
                    None,
                )
                .expect("Failed to create tonemap TAA set layout")
        };
        let tonemap_depth_sampler = pass::nearest_clamp_sampler(device, "tonemap depth");
        let push_tonemap_taa = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(PUSH_BYTES_TONEMAP_TAA)];
        let set_layouts_tonemap_taa = [tonemap_taa_set_layout];
        let layout_tonemap_taa_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts_tonemap_taa)
            .push_constant_ranges(&push_tonemap_taa);
        let layout_tonemap_taa = unsafe {
            device
                .create_pipeline_layout(&layout_tonemap_taa_info, None)
                .expect("Failed to create tonemap TAA pipeline layout")
        };

        // Vertex layouts derived from struct fields (see vertex_input).
        // Locations, offsets, and formats are kept in sync automatically.
        let bindings_3d = [MeshVertex::binding()];
        let attributes_3d = MeshVertex::ATTRIBUTES;

        let bindings_debug = [DebugVertex::binding()];
        let attributes_debug = DebugVertex::ATTRIBUTES;

        let bindings_2d = [Vertex2D::binding()];
        let attributes_2d = Vertex2D::ATTRIBUTES;

        let mesh_vert = pass::shader_module(device, MESH3D_VERT, "mesh3d vertex");
        let mesh_frag = pass::shader_module(device, MESH3D_FRAG, "mesh3d fragment");
        let mesh_opaque_frag =
            pass::shader_module(device, MESH3D_OPAQUE_FRAG, "mesh3d opaque fragment");
        let mesh_lod_frag = pass::shader_module(device, MESH3D_LOD_FRAG, "mesh3d lod fragment");
        let mesh_opaque_lean_frag = pass::shader_module(
            device,
            MESH3D_OPAQUE_LEAN_FRAG,
            "mesh3d opaque lean fragment",
        );
        let mesh_lod_lean_frag =
            pass::shader_module(device, MESH3D_LOD_LEAN_FRAG, "mesh3d lod lean fragment");
        let debug_vert = pass::shader_module(device, DEBUG_VERT, "debug vertex");
        let debug_frag = pass::shader_module(device, DEBUG_FRAG, "debug fragment");
        let tri2d_vert = pass::shader_module(device, TRIS2D_VERT, "2d vertex");
        let tri2d_frag = pass::shader_module(device, TRIS2D_FRAG, "2d fragment");
        let tri2d_tex_frag = pass::shader_module(device, TRIS2D_TEX_FRAG, "textured 2d fragment");
        let sky_vert = pass::shader_module(device, SKY_VERT, "sky vertex");
        let sky_frag = pass::shader_module(device, SKY_FRAG, "sky fragment");

        let builder = PipelineBuilder {
            device,
            cache,
            color_format,
            depth_format,
            samples,
            fsr_enabled: fsr.is_some(),
            second_color: None,
        };

        // Depth: reversed-Z, so GREATER_OR_EQUAL and clear to 0.0.
        let opaque_config = || PipelineConfig {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            depth: DepthMode::ReadWrite,
            cull: vk::CullModeFlags::BACK,
            blend: false,
            vrs: true,
            depth_bias: None,
        };
        let mesh3d = builder.build(
            mesh_vert,
            mesh_opaque_frag,
            &bindings_3d,
            attributes_3d,
            layout_3d,
            opaque_config(),
        );
        let mesh3d_lod = builder.build(
            mesh_vert,
            mesh_lod_frag,
            &bindings_3d,
            attributes_3d,
            layout_3d,
            opaque_config(),
        );
        let mesh3d_lean = builder.build(
            mesh_vert,
            mesh_opaque_lean_frag,
            &bindings_3d,
            attributes_3d,
            layout_3d,
            opaque_config(),
        );
        let mesh3d_lod_lean = builder.build(
            mesh_vert,
            mesh_lod_lean_frag,
            &bindings_3d,
            attributes_3d,
            layout_3d,
            opaque_config(),
        );
        // Blend world geometry: same modules/layout, alpha blend, depth read-only
        // (all opaque wrote depth first; blend tests but never writes). Double-sided
        // (cull NONE): the mesher emits translucent bodies as open shells (interior
        // and against-opaque faces culled), so a single kept face — e.g. a water
        // surface — must show from both sides, notably from underwater.
        let mesh3d_transparent = builder.build(
            mesh_vert,
            mesh_frag,
            &bindings_3d,
            attributes_3d,
            layout_3d,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::ReadOnly,
                cull: vk::CullModeFlags::NONE,
                blend: true,
                vrs: true,
                depth_bias: None,
            },
        );
        // Water absorption: previous-frame depth sample. Single-sample only
        // (MSAA stores the sampleable image on a separate resolve target the
        // absorb path does not read).
        let absorb_ok = samples == vk::SampleCountFlags::TYPE_1;
        let mesh3d_water_frag =
            absorb_ok.then(|| pass::shader_module(device, MESH3D_WATER_FRAG, "water fragment"));
        let mesh3d_transparent_absorb = mesh3d_water_frag.map(|water_frag| {
            builder.build(
                mesh_vert,
                water_frag,
                &bindings_3d,
                attributes_3d,
                layout_3d,
                PipelineConfig {
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    depth: DepthMode::ReadOnly,
                    cull: vk::CullModeFlags::NONE,
                    blend: true,
                    vrs: true,
                    depth_bias: None,
                },
            )
        });

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
        let tonemap_vert = pass::shader_module(device, TONEMAP_VERT, "tonemap vertex");
        let tonemap_frag = pass::shader_module(device, TONEMAP_FRAG, "tonemap fragment");
        let tonemap_builder = PipelineBuilder {
            device,
            cache,
            color_format: present_format,
            depth_format: vk::Format::UNDEFINED,
            samples: vk::SampleCountFlags::TYPE_1,
            fsr_enabled: false,
            second_color: None,
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

        let tonemap_taa_frag =
            pass::shader_module(device, TONEMAP_TAA_FRAG, "tonemap TAA fragment");
        let tonemap_taa_builder = PipelineBuilder {
            second_color: Some((
                super::taa::TAA_HISTORY_FORMAT,
                vk::ColorComponentFlags::RGBA,
            )),
            ..tonemap_builder
        };
        let tonemap_taa = tonemap_taa_builder.build(
            tonemap_vert,
            tonemap_taa_frag,
            &[],
            &[],
            layout_tonemap_taa,
            PipelineConfig {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                depth: DepthMode::Disabled,
                cull: vk::CullModeFlags::NONE,
                blend: false,
                vrs: false,
                depth_bias: None,
            },
        );
        // Overlay variants at present format / single-sample (same modules, layout,
        // and blend as tris2d/tris2d_tex) for the post-tonemap swapchain draw.
        let overlay_2d_config = || PipelineConfig {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            depth: DepthMode::Disabled,
            cull: vk::CullModeFlags::NONE,
            blend: true,
            vrs: false,
            depth_bias: None,
        };
        let tris2d_present = tonemap_builder.build(
            tri2d_vert,
            tri2d_frag,
            &bindings_2d,
            attributes_2d,
            layout_2d,
            overlay_2d_config(),
        );
        let tris2d_tex_present = tonemap_builder.build(
            tri2d_vert,
            tri2d_tex_frag,
            &bindings_2d,
            attributes_2d,
            layout_2d,
            overlay_2d_config(),
        );
        // Distinct blend states (att0 alpha-blend, att1 empty write mask) are
        // legal only with independentBlend. Without it these pipelines are
        // omitted and present.rs draws overlay in a second one-attachment scope.
        let (tris2d_present_taa, tris2d_tex_present_taa) = if independent_blend {
            let overlay_taa_builder = PipelineBuilder {
                second_color: Some((
                    super::taa::TAA_HISTORY_FORMAT,
                    vk::ColorComponentFlags::empty(),
                )),
                ..tonemap_builder
            };
            (
                Some(overlay_taa_builder.build(
                    tri2d_vert,
                    tri2d_frag,
                    &bindings_2d,
                    attributes_2d,
                    layout_2d,
                    overlay_2d_config(),
                )),
                Some(overlay_taa_builder.build(
                    tri2d_vert,
                    tri2d_tex_frag,
                    &bindings_2d,
                    attributes_2d,
                    layout_2d,
                    overlay_2d_config(),
                )),
            )
        } else {
            (None, None)
        };

        unsafe {
            device.destroy_shader_module(tonemap_vert, None);
            device.destroy_shader_module(tonemap_frag, None);
            device.destroy_shader_module(tonemap_taa_frag, None);
            device.destroy_shader_module(mesh_vert, None);
            device.destroy_shader_module(mesh_frag, None);
            device.destroy_shader_module(mesh_opaque_frag, None);
            device.destroy_shader_module(mesh_lod_frag, None);
            device.destroy_shader_module(mesh_opaque_lean_frag, None);
            device.destroy_shader_module(mesh_lod_lean_frag, None);
            if let Some(m) = mesh3d_water_frag {
                device.destroy_shader_module(m, None);
            }
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
            mesh3d_lod,
            mesh3d_lean,
            mesh3d_lod_lean,
            mesh3d_transparent,
            mesh3d_transparent_absorb,
            debug_tris,
            debug_tris_blend,
            debug_lines,
            tris2d,
            tris2d_tex,
            tris2d_present,
            tris2d_tex_present,
            sky,
            layout_sky,
            sky_set_layout,
            sky_lut_sampler,
            tonemap,
            layout_tonemap,
            tonemap_set_layout,
            tonemap_sampler,
            tonemap_taa,
            layout_tonemap_taa,
            tonemap_taa_set_layout,
            tonemap_depth_sampler,
            tris2d_present_taa,
            tris2d_tex_present_taa,
        }
    }

    /// Full-res and coarse-LOD opaque pipelines. `lean` selects the compile-time
    /// lane-off fragment variants (no cascade taps, no candle, no ambient, no fog).
    pub fn opaque_pipelines(&self, lean: bool) -> (vk::Pipeline, vk::Pipeline) {
        if lean {
            (self.mesh3d_lean, self.mesh3d_lod_lean)
        } else {
            (self.mesh3d, self.mesh3d_lod)
        }
    }

    /// The 3D pipeline for a mesh's draw pass. Exhaustive so a new [`Pass`]
    /// variant forces a matching pipeline here. Opaque means the FULL-RES
    /// partition; the coarse-LOD partition binds [`Self::mesh3d_lod`].
    pub fn pipeline_for(&self, pass: Pass) -> vk::Pipeline {
        match pass {
            Pass::Opaque => self.mesh3d,
            // Reserved: shares the opaque pipeline (depth write, cull back) until an
            // alpha-test `discard` frag variant lands (it must NOT reuse the
            // no-discard opaque module then). Sound because nothing emits Cutout yet.
            Pass::Cutout => self.mesh3d,
            Pass::Blend => self.blend_pipeline(),
        }
    }

    /// The pipeline for the transparent [`Pass::Blend`] draw: the water
    /// previous-depth absorption variant when available, else the interim-tint fallback.
    pub fn blend_pipeline(&self) -> vk::Pipeline {
        self.mesh3d_transparent_absorb
            .unwrap_or(self.mesh3d_transparent)
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
            device.destroy_pipeline(self.mesh3d_lod, None);
            device.destroy_pipeline(self.mesh3d_lean, None);
            device.destroy_pipeline(self.mesh3d_lod_lean, None);
            device.destroy_pipeline(self.mesh3d_transparent, None);
            if let Some(p) = self.mesh3d_transparent_absorb {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline(self.debug_tris, None);
            device.destroy_pipeline(self.debug_tris_blend, None);
            device.destroy_pipeline(self.debug_lines, None);
            device.destroy_pipeline(self.tris2d, None);
            device.destroy_pipeline(self.tris2d_tex, None);
            device.destroy_pipeline(self.tris2d_present, None);
            device.destroy_pipeline(self.tris2d_tex_present, None);
            if let Some(p) = self.tris2d_present_taa {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.tris2d_tex_present_taa {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline(self.sky, None);
            device.destroy_pipeline(self.tonemap, None);
            device.destroy_pipeline(self.tonemap_taa, None);
            device.destroy_pipeline_layout(self.layout_3d, None);
            device.destroy_pipeline_layout(self.layout_debug, None);
            device.destroy_pipeline_layout(self.layout_2d, None);
            device.destroy_pipeline_layout(self.layout_sky, None);
            device.destroy_descriptor_set_layout(self.sky_set_layout, None);
            device.destroy_sampler(self.sky_lut_sampler, None);
            device.destroy_pipeline_layout(self.layout_tonemap, None);
            device.destroy_descriptor_set_layout(self.tonemap_set_layout, None);
            device.destroy_sampler(self.tonemap_sampler, None);
            device.destroy_pipeline_layout(self.layout_tonemap_taa, None);
            device.destroy_descriptor_set_layout(self.tonemap_taa_set_layout, None);
            device.destroy_sampler(self.tonemap_depth_sampler, None);
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum DepthMode {
    ReadWrite,
    ReadOnly,
    Disabled,
}

#[derive(Clone, Copy)]
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
    /// Optional second color attachment (fused TAA history). The write mask
    /// is RGBA for the tonemap write, empty for overlay variants that must
    /// match the 2-attachment rendering without touching history. Overlay
    /// variants with a distinct empty mask require `independentBlend`.
    second_color: Option<(vk::Format, vk::ColorComponentFlags)>,
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
    #[allow(clippy::too_many_arguments)]
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

        let write_mask = color_write_mask(self.color_format);
        let color_attachment = if blend {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(write_mask)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
        } else {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(write_mask)
                .blend_enable(false)
        };
        let second_blend = self.second_color.map(|(_, mask)| {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(mask)
                .blend_enable(false)
        });
        let color_attachments_1 = [color_attachment];
        let color_attachments_2 = [color_attachment, second_blend.unwrap_or_default()];
        let color_attachments: &[vk::PipelineColorBlendAttachmentState] =
            if self.second_color.is_some() {
                &color_attachments_2
            } else {
                &color_attachments_1
            };
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(color_attachments);

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

        let color_formats_1 = [self.color_format];
        let color_formats_2 = [
            self.color_format,
            self.second_color
                .map(|(f, _)| f)
                .unwrap_or(vk::Format::UNDEFINED),
        ];
        let color_formats: &[vk::Format] = if self.second_color.is_some() {
            &color_formats_2
        } else {
            &color_formats_1
        };
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(color_formats)
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
        // Previous raw classification (same-texel read, then write).
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // Tile-mix histogram [1x1, 2x2, 4x4].
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let (set_layout, layout) = pass::push_descriptor_layouts(
        device,
        &bindings,
        size_of::<super::vrs::VrsPush>() as u32,
        "vrs",
    );
    let pipeline = pass::compute_pipeline(device, cache, layout, VRS_COMP, "vrs");

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

/// Color write mask for a pipeline's first attachment. Packed 11-bit HDR has
/// no alpha; `A` in the mask is ignored by the spec but we omit it so blend
/// state does not assume a channel the format does not have.
fn color_write_mask(format: vk::Format) -> vk::ColorComponentFlags {
    if super::targets::color_format_has_alpha(format) {
        vk::ColorComponentFlags::RGBA
    } else {
        vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B
    }
}

#[cfg(test)]
mod tests {
    use ash::vk;

    /// Scan a SPIR-V module for an opcode (low 16 bits of each instruction word).
    fn spirv_has_opcode(bytes: &[u8], opcode: u32) -> bool {
        assert!(bytes.len() >= 20 && bytes.len().is_multiple_of(4));
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(words[0], 0x0723_0203, "missing SPIR-V magic");
        let mut i = 5usize;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            let op = words[i] & 0xffff;
            if wc == 0 || i + wc > words.len() {
                break;
            }
            if op == opcode {
                return true;
            }
            i += wc;
        }
        false
    }

    const OP_KILL: u32 = 252;
    const OP_DEMOTE: u32 = 5380;

    #[test]
    fn packed_11bit_hdr_write_mask_has_no_alpha() {
        assert_eq!(
            super::color_write_mask(vk::Format::B10G11R11_UFLOAT_PACK32),
            vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B
        );
        assert_eq!(
            super::color_write_mask(vk::Format::R16G16B16A16_SFLOAT),
            vk::ColorComponentFlags::RGBA
        );
    }

    #[test]
    fn opaque_frag_has_no_discard() {
        for (name, bytes) in [
            ("MESH3D_OPAQUE", super::MESH3D_OPAQUE_FRAG),
            ("MESH3D_OPAQUE_LEAN", super::MESH3D_OPAQUE_LEAN_FRAG),
        ] {
            assert!(
                !spirv_has_opcode(bytes, OP_KILL) && !spirv_has_opcode(bytes, OP_DEMOTE),
                "{name} must not OpKill/OpDemote (early depth write)"
            );
        }
    }

    #[test]
    fn lod_and_blend_frags_keep_slab_discard() {
        assert!(
            spirv_has_opcode(super::MESH3D_LOD_FRAG, OP_KILL)
                || spirv_has_opcode(super::MESH3D_LOD_FRAG, OP_DEMOTE),
            "MESH3D_LOD must keep the slab-clip discard"
        );
        assert!(
            spirv_has_opcode(super::MESH3D_LOD_LEAN_FRAG, OP_KILL)
                || spirv_has_opcode(super::MESH3D_LOD_LEAN_FRAG, OP_DEMOTE),
            "MESH3D_LOD_LEAN must keep the slab-clip discard"
        );
        assert!(
            spirv_has_opcode(super::MESH3D_FRAG, OP_KILL)
                || spirv_has_opcode(super::MESH3D_FRAG, OP_DEMOTE),
            "Blend mesh3d.frag must keep the slab-clip discard"
        );
    }
}
