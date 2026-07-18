//! Cascaded shadow map, producer half (occluder rendering and cascade fitting).
//!
//! This module owns everything the shadow *producer* needs and nothing the
//! receiver does: the depth-only occluder pipeline + `shadow_depth.vert`, the
//! per-frame cascade `fit()`, `Renderer::record_shadow_pass` (renders occluders
//! into each cascade layer), and the binding-3 `CascadeUniformsGpu` UBO the
//! receiver's PCF samples (populated here, sampled in mesh3d.frag by the
//! Frame-lighting agent). The receiver-side PCF / SHADOW_LIMIT fade lives THERE,
//! not here.

use ash::vk;
use glam::{DVec3, Mat4};

use super::cull;

use super::pass::shader_module;
use super::taa::CleanViewProj;
use crate::rev::FrameSlot;
use crate::vk::Renderer;
use crate::vk::buffers::HostBuffer;
use crate::vk::targets::{SHADOW_CASCADES, SHADOW_FORMAT, SHADOW_RESOLUTION};
use crate::vk::vertex_input::VertexInput;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cascade {
    Near,
    Far,
}

pub struct PerCascade<T>([T; 2]);

impl<T> PerCascade<T> {
    pub fn new(pair: [T; 2]) -> Self {
        PerCascade(pair)
    }
}

impl<T> std::ops::Index<Cascade> for PerCascade<T> {
    type Output = T;
    fn index(&self, c: Cascade) -> &T {
        &self.0[c as usize]
    }
}

/// One cascade's light-space view-proj, split distance, and texel scale.
#[derive(Clone, Copy, Debug)]
pub struct CascadeFit {
    pub view_proj: CleanViewProj,
    pub split: f32,
    pub texel_world: f32,
}

pub struct ShadowCfg {
    pub resolution: u32,
    pub blur_texels: f32,
    pub slope_bias: f32,
    pub dist_bias: f32,
    pub fade_band: f32,
    pub splits: [f32; 2],
}

/// Sampling-pass uniforms (separate from frame uniforms; depth pass uses push constants).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CascadeUniformsGpu {
    pub view_proj: [[[f32; 4]; 4]; 2],
    /// x,y = split distances; z = fade_band; w = SHADOW_LIMIT.
    pub splits_fade: [f32; 4],
    /// x = blur_texels, y = slope_bias, z = dist_bias, w = texel_world(near).
    pub bias: [f32; 4],
}

pub const CASCADE_UNIFORMS_BINDING: u32 = 3;

const _: () = assert!(size_of::<CascadeUniformsGpu>() == 160);
const _: () = assert!(std::mem::offset_of!(CascadeUniformsGpu, splits_fade) == 128);

const SHADOW_DEPTH_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow_depth.vert.spv"));

/// The two cascades in render/index order, so callers never spell `as usize`.
const CASCADES: [Cascade; SHADOW_CASCADES as usize] = [Cascade::Near, Cascade::Far];

impl ShadowCfg {
    pub const PROVISIONAL: Self = Self {
        resolution: SHADOW_RESOLUTION,
        blur_texels: 2.0,
        slope_bias: 1.5,
        dist_bias: 2.0,
        fade_band: 16.0,
        splits: [64.0, 256.0],
    };

    fn texel_world_at(&self, radius: f32) -> f32 {
        2.0 * radius / self.resolution as f32
    }
}

/// Inputs that determine shadow map content (world-anchored, whole-texel snapped).
#[derive(Clone, Copy)]
pub(crate) struct ShadowKey {
    /// Toward-sun direction, normalized (compared by angular chord).
    sun: DVec3,
    /// Per-cascade, per-lateral-axis whole-texel snap of the eye.
    snap: [[i64; 2]; SHADOW_CASCADES as usize],
    occluders: u64,
    /// Hash of the immediate caster geometry (avatar boxes); any motion re-renders.
    casters: u64,
}

impl ShadowKey {
    pub(crate) fn of(
        eye: DVec3,
        sun: DVec3,
        occluders: u64,
        casters: u64,
        cfg: &ShadowCfg,
    ) -> Self {
        let sun = sun.normalize_or_zero();
        let light_dir = (-sun).normalize_or_zero();
        let up_hint = if light_dir.y.abs() > 0.99 {
            DVec3::Z
        } else {
            DVec3::Y
        };
        let l_right = light_dir.cross(up_hint).normalize_or_zero();
        let l_up = l_right.cross(light_dir).normalize_or_zero();
        let snap = CASCADES.map(|c| {
            let radius = (cfg.splits[c as usize] + BIAS_MARGIN) as f64;
            let t = cfg.texel_world_at(radius as f32) as f64;
            let cell = |axis: DVec3| (eye.dot(axis) / t).floor() as i64;
            [cell(l_right), cell(l_up)]
        });
        Self {
            sun,
            snap,
            occluders,
            casters,
        }
    }

    /// True when the cached depth would no longer match a fresh fit. Sun
    /// threshold = one far-cascade texel of angular movement (texel_world /
    /// radius): the whole-terrain shadow shift from swinging the sun by θ is
    /// ~θ·radius, so θ below one texel is invisible.
    fn differs(&self, other: &Self, cfg: &ShadowCfg) -> bool {
        if self.occluders != other.occluders
            || self.snap != other.snap
            || self.casters != other.casters
        {
            return true;
        }
        let radius = cfg.splits[1] + BIAS_MARGIN;
        let threshold = (cfg.texel_world_at(radius) / radius) as f64;
        self.sun.distance(other.sun) > threshold
    }
}

/// Dirty cache for shadow depth image (per-slot; entire cache invalidates on change).
pub(crate) struct ShadowCache {
    /// Inputs of the currently-cached generation; `None` forces a render.
    key: Option<ShadowKey>,
    dirty: [bool; SHADOW_CACHE_SLOTS],
}

const SHADOW_CACHE_SLOTS: usize = crate::vk::buffers::FRAMES_IN_FLIGHT as usize;

impl ShadowCache {
    pub(crate) fn new() -> Self {
        Self {
            key: None,
            dirty: [true; SHADOW_CACHE_SLOTS],
        }
    }

    /// Mark all slots dirty (layout reset or shadows disabled).
    pub(crate) fn invalidate(&mut self) {
        self.key = None;
        self.dirty = [true; SHADOW_CACHE_SLOTS];
    }

    /// Check if this slot must re-render (key change re-arms all slots).
    pub(crate) fn take_render(&mut self, slot: usize, cur: ShadowKey, cfg: &ShadowCfg) -> bool {
        if self.key.is_none_or(|k| k.differs(&cur, cfg)) {
            self.key = Some(cur);
            self.dirty = [true; SHADOW_CACHE_SLOTS];
        }
        std::mem::take(&mut self.dirty[slot])
    }
}

pub(crate) struct ShadowPass {
    pipeline: vk::Pipeline,
    /// Depth-only caster for immediate `DebugVertex` boxes (player avatars).
    debug_pipeline: vk::Pipeline,
    ubo: [HostBuffer; 2],
}

impl ShadowPass {
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        cache: vk::PipelineCache,
        layout_3d: vk::PipelineLayout,
        layout_debug: vk::PipelineLayout,
    ) -> Self {
        let pipeline = build_depth_only_pipeline(
            device,
            cache,
            layout_3d,
            SHADOW_DEPTH_VERT,
            &[crate::mesh::MeshVertex::binding()],
            crate::mesh::MeshVertex::ATTRIBUTES,
        );
        let debug_pipeline = build_depth_only_pipeline(
            device,
            cache,
            layout_debug,
            crate::vk::pipeline::DEBUG_VERT,
            &[crate::mesh::DebugVertex::binding()],
            crate::mesh::DebugVertex::ATTRIBUTES,
        );

        let make_ubo = || {
            let mut b = HostBuffer::new(vk::BufferUsageFlags::UNIFORM_BUFFER);
            unsafe {
                b.maintain(
                    instance,
                    device,
                    physical,
                    size_of::<CascadeUniformsGpu>() as u64,
                )
            };
            b
        };
        Self {
            pipeline,
            debug_pipeline,
            ubo: [make_ubo(), make_ubo()],
        }
    }

    pub(crate) fn ubo(&self, slot: usize) -> vk::Buffer {
        self.ubo[slot]
            .bound()
            .expect("the cascade UBO is written before any receiver binds it")
    }

    pub(crate) fn write_uniforms(&mut self, slot: usize, u: &CascadeUniformsGpu) {
        unsafe { self.ubo[slot].write(0, bytemuck::bytes_of(u)) };
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline(self.debug_pipeline, None);
            self.ubo[0].destroy(device);
            self.ubo[1].destroy(device);
        }
    }
}

/// Build depth-only pipeline (no cull) for a given caster vertex layout.
fn build_depth_only_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    vert: &[u8],
    bindings: &[vk::VertexInputBindingDescription],
    attributes: &[vk::VertexInputAttributeDescription],
) -> vk::Pipeline {
    let module = shader_module(device, vert, "shadow-depth");
    let stages = [vk::PipelineShaderStageCreateInfo::default()
        .module(module)
        .name(c"main")
        .stage(vk::ShaderStageFlags::VERTEX)];

    // Vert reads only position; layout must match the caster meshes.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(bindings)
        .vertex_attribute_descriptions(attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_blending = vk::PipelineColorBlendStateCreateInfo::default();

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);

    let mut rendering_info =
        vk::PipelineRenderingCreateInfo::default().depth_attachment_format(SHADOW_FORMAT);

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

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(cache, &[pipeline_info], None)
            .map_err(|(_, err)| err)
            .expect("Failed to create shadow depth pipeline")[0]
    };
    unsafe { device.destroy_shader_module(module, None) };
    pipeline
}

/// Pullback margin for tall occluders.
const PULLBACK: f32 = 100.0;

/// Coverage margin for bias and blur taps.
const BIAS_MARGIN: f32 = 4.0;

/// Fit cascade as eye-centered sphere (stable across camera rotation).
/// Texel grid anchored in world f64 space, whole-texel-snapped. Reversed-Z ortho.
pub(crate) fn fit(eye: DVec3, sun: DVec3, c: Cascade, cfg: &ShadowCfg) -> CascadeFit {
    let split = cfg.splits[c as usize];
    let radius = split + BIAS_MARGIN;
    let texel_world = cfg.texel_world_at(radius);

    let light_dir = (-sun).normalize_or_zero();
    let up_hint = if light_dir.y.abs() > 0.99 {
        DVec3::Z
    } else {
        DVec3::Y
    };
    let l_right = light_dir.cross(up_hint).normalize_or_zero();
    let l_up = l_right.cross(light_dir).normalize_or_zero();

    let t = texel_world as f64;
    let phase = |axis: DVec3| (eye.dot(axis).rem_euclid(t)) as f32;
    let centre = -(l_right.as_vec3() * phase(l_right) + l_up.as_vec3() * phase(l_up));

    let light_dir = light_dir.as_vec3();
    let ls_eye = centre - light_dir * (radius + PULLBACK);
    let view = Mat4::look_at_rh(ls_eye, centre, up_hint.as_vec3());
    let proj = Mat4::orthographic_rh(
        -radius,
        radius,
        -radius,
        radius,
        2.0 * radius + PULLBACK, // far arg (maps to 0)
        0.0,                     // near arg (maps to 1)
    );

    CascadeFit {
        view_proj: CleanViewProj(proj * view),
        split,
        texel_world,
    }
}

impl Renderer {
    pub(crate) fn shadow_uniforms(
        &self,
        eye: DVec3,
        sun: DVec3,
        cfg: &ShadowCfg,
    ) -> CascadeUniformsGpu {
        let fits = PerCascade::new(CASCADES.map(|c| fit(eye, sun, c, cfg)));
        let near = &fits[Cascade::Near];
        let far = &fits[Cascade::Far];
        let shadow_limit = if self.flags.shadows {
            cfg.splits[1]
        } else {
            f32::MAX
        };
        CascadeUniformsGpu {
            view_proj: [
                near.view_proj.0.to_cols_array_2d(),
                far.view_proj.0.to_cols_array_2d(),
            ],
            splits_fade: [near.split, far.split, cfg.fade_band, shadow_limit],
            bias: [
                cfg.blur_texels,
                cfg.slope_bias,
                cfg.dist_bias,
                near.texel_world,
            ],
        }
    }

    pub(crate) fn record_shadow_pass(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        eye: DVec3,
        sun: DVec3,
        cfg: &ShadowCfg,
        caster_verts: u32,
    ) {
        let device = &self.device.device;
        let shadow = &self.targets.shadow[FrameSlot::new(slot)];
        let layout = self.pipelines.layout_3d;

        let full_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: SHADOW_CASCADES,
        };

        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .image(shadow.image)
                        .subresource_range(full_range),
                ]),
            );

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: SHADOW_RESOLUTION as f32,
                height: SHADOW_RESOLUTION as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: SHADOW_RESOLUTION,
                    height: SHADOW_RESOLUTION,
                },
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            let occluders = self
                .flags
                .shadows
                .then(|| self.record_buffers.map(|b| b.records))
                .flatten();
            if let Some(records_buffer) = occluders {
                let records_info = [vk::DescriptorBufferInfo::default()
                    .buffer(records_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)];
                let write = [vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&records_info)];
                self.device.push_descriptor.cmd_push_descriptor_set(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &write,
                );
            }

            for c in CASCADES {
                let f = fit(eye, sun, c, cfg);
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(shadow.layer_views[c as usize])
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0, // reversed-Z far
                            stencil: 0,
                        },
                    });
                let rendering_info = vk::RenderingInfo::default()
                    .render_area(scissor)
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);

                device.cmd_begin_rendering(cmd, &rendering_info);
                if occluders.is_some() {
                    // Terrain occluders: MeshVertex depth pipeline, layout_3d push.
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.shadow.pipeline,
                    );
                    let push = crate::vk::pipeline::Mesh3dPush {
                        view_proj: f.view_proj.0,
                        clip: 0.0,
                        clip_v: 0.0,
                        _pad: [0.0; 2],
                        eye: crate::vk::pipeline::EyeSplit::of(eye),
                    };
                    device.cmd_push_constants(
                        cmd,
                        layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push),
                    );
                    self.record_shadow_occluders(cmd);
                }
                if caster_verts > 0 {
                    // Avatar boxes: same eye-relative space as the cascade fit, so
                    // the debug view_proj is the cascade matrix unchanged. Immediate
                    // cube verts sit at offset 0 of the slot's `imm` buffer.
                    let imm = self.slots[FrameSlot::new(slot)]
                        .imm
                        .bound()
                        .expect("a non-zero caster count implies an allocated imm buffer");
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.shadow.debug_pipeline,
                    );
                    let dpush = crate::vk::pipeline::DebugPush {
                        view_proj: f.view_proj.0,
                    };
                    device.cmd_push_constants(
                        cmd,
                        self.pipelines.layout_debug,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytemuck::bytes_of(&dpush),
                    );
                    device.cmd_bind_vertex_buffers(cmd, 0, &[imm], &[0]);
                    device.cmd_draw(cmd, caster_verts, 1, 0, 0);
                }
                device.cmd_end_rendering(cmd);
            }

            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(shadow.image)
                        .subresource_range(full_range),
                ]),
            );
        }
    }

    unsafe fn record_shadow_occluders(&self, cmd: vk::CommandBuffer) {
        let Some(frame) = &self.cull_frame else {
            return;
        };
        let device = &self.device.device;
        let quad_ibo = self
            .quad_ibo
            .bound()
            .expect("live records imply the quad IBO is allocated");
        unsafe { device.cmd_bind_index_buffer(cmd, quad_ibo, 0, vk::IndexType::UINT32) };
        let base = 2 * frame.arena_count;
        for arena in 0..frame.arena_count {
            let part = frame.partitions[base + arena];
            if part.capacity == 0 {
                continue;
            }
            unsafe {
                device.cmd_bind_vertex_buffers(cmd, 0, &[self.arena_dir.arena_buffer(arena)], &[0]);
                device.cmd_draw_indexed_indirect_count(
                    cmd,
                    frame.commands,
                    u64::from(part.offset) * cull::CMD_STRIDE,
                    frame.counts,
                    ((base + arena) * 4) as u64,
                    part.capacity,
                    cull::CMD_STRIDE as u32,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// An arbitrary non-axis-aligned daytime sun.
    const SUN: DVec3 = DVec3::new(0.3, 0.8, 0.25);

    /// Project world point through cascade view-proj to NDC.
    fn ndc(eye: DVec3, p: DVec3, c: Cascade) -> Vec3 {
        let f = fit(eye, SUN, c, &ShadowCfg::PROVISIONAL);
        let rel = (p - eye).as_vec3();
        let clip = f.view_proj.0 * rel.extend(1.0);
        clip.truncate() / clip.w
    }

    /// All points within selection distance must be covered by the map.
    #[test]
    fn every_selectable_fragment_is_covered() {
        let cfg = ShadowCfg::PROVISIONAL;
        let eye = DVec3::new(1000.0, 80.0, -2000.0);
        let dirs = [
            DVec3::X,
            DVec3::NEG_X,
            DVec3::Y,
            DVec3::NEG_Y,
            DVec3::Z,
            DVec3::NEG_Z,
            DVec3::new(0.6, -0.5, 0.62),
            DVec3::new(-0.7, 0.7, 0.14),
        ];
        for c in [Cascade::Near, Cascade::Far] {
            let split = cfg.splits[c as usize] as f64;
            for d in dirs {
                let p = eye + d.normalize() * split;
                let n = ndc(eye, p, c);
                assert!(
                    n.x.abs() <= 1.0 && n.y.abs() <= 1.0,
                    "{c:?} {d:?}: lateral {n:?} outside footprint"
                );
                assert!(
                    (0.0..=1.0).contains(&n.z),
                    "{c:?} {d:?}: depth {n:?} outside range"
                );
            }
        }
    }

    /// The fit takes no camera orientation input, so the matrices are
    /// bitwise-identical from frame to frame while the eye stands still —
    /// looking around can never move a shadow.
    #[test]
    fn fit_is_deterministic_per_eye() {
        let eye = DVec3::new(-31.7, 12.0, 98765.4);
        for c in [Cascade::Near, Cascade::Far] {
            let a = fit(eye, SUN, c, &ShadowCfg::PROVISIONAL);
            let b = fit(eye, SUN, c, &ShadowCfg::PROVISIONAL);
            assert_eq!(a.view_proj.0, b.view_proj.0);
            assert_eq!(a.texel_world, b.texel_world);
        }
    }

    /// World-anchored texel grid: as the eye translates (including sub-texel
    /// steps), a fixed world point's map coordinate moves by WHOLE texels, so
    /// shadow edges stay locked to world geometry instead of crawling.
    #[test]
    fn texel_grid_is_world_anchored_under_translation() {
        let cfg = ShadowCfg::PROVISIONAL;
        let p = DVec3::new(12.3, 4.5, -67.8);
        let base = DVec3::new(3.0, 20.0, 5.0);
        let eyes = [
            base,
            base + DVec3::new(0.013, 0.0, 0.007), // sub-texel drift
            base + DVec3::new(1.37, -0.5, 2.11),  // walking
            base + DVec3::new(-25.0, 3.0, 17.9),  // sprinting away
        ];
        for c in [Cascade::Near, Cascade::Far] {
            let res = cfg.resolution as f32;
            let texel = |eye: DVec3| {
                let n = ndc(eye, p, c);
                Vec3::new((n.x * 0.5 + 0.5) * res, (n.y * 0.5 + 0.5) * res, 0.0)
            };
            let t0 = texel(eyes[0]);
            for &e in &eyes[1..] {
                let d = texel(e) - t0;
                for frac in [d.x - d.x.round(), d.y - d.y.round()] {
                    assert!(
                        frac.abs() < 1e-2,
                        "{c:?}: eye {e:?} moved grid by fractional texel {frac}"
                    );
                }
            }
        }
    }

    /// Same anchoring must survive extreme coordinates: the phase is computed
    /// in f64 because one f32 ULP out there is already comparable to a texel.
    #[test]
    fn texel_grid_stays_anchored_at_far_coordinates() {
        let cfg = ShadowCfg::PROVISIONAL;
        let base = DVec3::new(1.0e6, 40.0, -2.5e5);
        let p = base + DVec3::new(9.13, -6.0, 21.7);
        let eyes = [base, base + DVec3::new(0.51, 0.25, -1.03)];
        for c in [Cascade::Near, Cascade::Far] {
            let res = cfg.resolution as f32;
            let uv_texels = |eye: DVec3| {
                let n = ndc(eye, p, c);
                ((n.x * 0.5 + 0.5) * res, (n.y * 0.5 + 0.5) * res)
            };
            let (x0, y0) = uv_texels(eyes[0]);
            let (x1, y1) = uv_texels(eyes[1]);
            for d in [x1 - x0, y1 - y0] {
                assert!(
                    (d - d.round()).abs() < 5e-2,
                    "{c:?}: far-coordinate translation broke the grid by {d}"
                );
            }
        }
    }

    /// Occluders standing up to PULLBACK above the covered sphere still land
    /// inside the reversed-Z depth range (they must cast onto it).
    #[test]
    fn tall_occluders_fall_inside_depth_range() {
        let cfg = ShadowCfg::PROVISIONAL;
        let eye = DVec3::new(50.0, 10.0, 50.0);
        let toward_sun = SUN.normalize();
        for c in [Cascade::Near, Cascade::Far] {
            let radius = cfg.splits[c as usize] as f64;
            let p = eye + toward_sun * (radius + PULLBACK as f64 * 0.95);
            let n = ndc(eye, p, c);
            assert!(
                (0.0..=1.0).contains(&n.z),
                "{c:?}: tall occluder at {n:?} clipped"
            );
        }
    }
}
