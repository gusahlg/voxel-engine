//! Cascaded shadow map, producer half (occluder rendering and cascade fitting).
//!
//! This module owns everything the shadow *producer* needs and nothing the
//! receiver does: the depth-only occluder pipeline + `shadow_depth.vert`, the
//! per-frame cascade `fit()`, `Renderer::record_shadow_pass` (renders occluders
//! into each cascade layer), and the binding-3 `CascadeUniformsGpu` UBO the
//! receiver's PCF samples (populated here, sampled in mesh3d.frag by the
//! Frame-lighting agent). The receiver-side PCF / SHADOW_LIMIT fade lives THERE,
//! not here.
//!
//! ── MERGE SEAMS (this is an in-flight, shared-tree slice; it does not compile
//! standalone — the orchestrator wires the rest into vk/mod.rs at merge) ──
//!  * `mod.rs`: declare `pub(crate) mod shadow;`, add a `shadow: ShadowPass`
//!    field to `Renderer`, build it in `Renderer::new` (after `pipelines`), and
//!    `destroy` it on drop.
//!  * `mod.rs`: call `record_shadow_pass` on the frame command buffer BEFORE the
//!    main color pass, and write `shadow_uniforms(..)` into `shadow.ubo(slot)`
//!    each frame.
//!  * `buffers.rs` `create_mesh3d_set_layout` / `push_mesh3d_descriptors`: add
//!    set-0 binding 3 (`CascadeUniformsGpu` UBO) and binding 4 (shadow map
//!    combined image sampler) so the receiver sees them. FROZEN binding numbers:
//!    `CASCADE_UNIFORMS_BINDING == 3`.
//!  * `build.rs`: register `shadow_depth.vert.slang → shadow_depth.vert.spv`.

use ash::vk;
use glam::{DVec3, Mat4, Vec3};

use crate::camera::{Camera3D, Z_NEAR};
use crate::skeleton::{
    Cascade, CascadeFit, CascadeUniformsGpu, CleanViewProj, PerCascade, ShadowCfg,
};
use crate::vk::buffers::{DrawIndexedIndirect, HostBuffer};
use crate::vk::targets::{SHADOW_CASCADES, SHADOW_FORMAT, SHADOW_RESOLUTION};
use crate::vk::vertex_input::VertexInput;
use crate::vk::Renderer;

/// Local mirror of `pipeline::create_shader_module` (that one is private to the
/// `pipeline` module; a sibling module cannot reach it).
fn shader_module(device: &ash::Device, bytes: &[u8]) -> vk::ShaderModule {
    let code =
        ash::util::read_spv(&mut std::io::Cursor::new(bytes)).expect("Invalid embedded SPIR-V");
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe {
        device
            .create_shader_module(&info, None)
            .expect("Failed to create shadow shader module")
    }
}

const SHADOW_DEPTH_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow_depth.vert.spv"));

/// The two cascades in render/index order, so callers never spell `as usize`.
const CASCADES: [Cascade; SHADOW_CASCADES as usize] = [Cascade::Near, Cascade::Far];

// Starting shadow configuration constants. Tentative, pending tuning.
impl ShadowCfg {
    /// Ships-now configuration; the PCF/bias/split constants are tentative and
    /// expected to be re-tuned.
    pub fn provisional() -> Self {
        Self {
            resolution: SHADOW_RESOLUTION,
            blur_texels: 2.0,   // rotated 4-tap radius
            // Normal-offset bias, in shadow texels (depth-range independent).
            slope_bias: 1.5,    // base normal offset (texels)
            dist_bias: 2.0,     // extra offset per unit tan(grazing) (texels)
            fade_band: 16.0,    // map→fallback smoothstep width (m)
            splits: [64.0, 256.0], // near/far far-distances (SHADOW_LIMIT = 256)
        }
    }

    /// Metres per shadow texel for cascade `c` at its bounding radius — the
    /// receiver's bias scale and the CPU stable-snap increment.
    fn texel_world_at(&self, radius: f32) -> f32 {
        2.0 * radius / self.resolution as f32
    }
}

/// The producer's own resources: the depth-only occluder pipeline (reuses the
/// renderer's `layout_3d`, so no new descriptor set layout) and the per-slot
/// binding-3 UBO ring the receiver samples. `layout_3d` already declares set-0
/// binding 0 (the offsets SSBO the vert reads); its unused bindings are fine.
pub(crate) struct ShadowPass {
    pipeline: vk::Pipeline,
    /// One host-visible UBO per frame-in-flight, each exactly a
    /// `CascadeUniformsGpu`; written per frame, bound at set 0 binding 3.
    ubo: [HostBuffer; 2],
}

impl ShadowPass {
    /// Build the depth-only pipeline and allocate the binding-3 UBO ring. Reuses
    /// `layout_3d` (set 0 = mesh3d set layout, 128 B vertex push): the shadow
    /// vert reads only binding 0 + pushes a 64 B `view_proj`.
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        cache: vk::PipelineCache,
        layout_3d: vk::PipelineLayout,
    ) -> Self {
        let pipeline = build_depth_only_pipeline(device, cache, layout_3d);

        let make_ubo = || {
            let mut b = HostBuffer::new(vk::BufferUsageFlags::UNIFORM_BUFFER);
            // GPU idle at init; `maintain` allocates + persistently maps.
            unsafe { b.maintain(instance, device, physical, size_of::<CascadeUniformsGpu>() as u64) };
            b
        };
        Self {
            pipeline,
            ubo: [make_ubo(), make_ubo()],
        }
    }

    /// The binding-3 UBO buffer for `slot` (raw frame-in-flight index). The
    /// cascade UBO is written by the shadow pass (which runs before any mesh
    /// draw that samples binding 3) and persists across the shadows-off skip, so
    /// it is allocated by the time a receiver binds it.
    pub(crate) fn ubo(&self, slot: usize) -> vk::Buffer {
        self.ubo[slot]
            .bound()
            .expect("the cascade UBO is written before any receiver binds it")
    }

    /// Write this frame's cascade uniforms into `slot`'s mapped UBO. Coherent
    /// memory: visible to the GPU with no explicit flush.
    pub(crate) fn write_uniforms(&mut self, slot: usize, u: &CascadeUniformsGpu) {
        unsafe { self.ubo[slot].write(0, bytemuck::bytes_of(u)) };
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            self.ubo[0].destroy(device);
            self.ubo[1].destroy(device);
        }
    }
}

/// Vertex-only, depth-write, no-cull pipeline into the D32 shadow map. No color
/// attachment; reversed-Z (`GREATER_OR_EQUAL`, cleared to 0.0) consistent with
/// the engine's depth policy. Front-face winding is irrelevant with cull off, so
/// occluders cast regardless of orientation (avoids peter-panning on thin faces).
fn build_depth_only_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
) -> vk::Pipeline {
    let module = shader_module(device, SHADOW_DEPTH_VERT);
    let stages = [vk::PipelineShaderStageCreateInfo::default()
        .module(module)
        .name(c"main")
        .stage(vk::ShaderStageFlags::VERTEX)];

    // Same MeshVertex binding/attributes as mesh3d; the vert reads only the
    // position bits, but the vertex buffer layout must match the source meshes.
    let bindings = [crate::mesh::MeshVertex::binding()];
    let attributes = crate::mesh::MeshVertex::ATTRIBUTES;
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
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

    // No color attachments: depth-only.
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

/// Fit cascade `c` around its camera-frustum slice. Camera-relative world
/// (camera at the origin), so the slice corners and the light-space bound are
/// all small — no far-terrain jitter. Stable under camera *rotation* (a
/// corner-average bounding sphere is rotation-invariant) and under *translation*
/// (the light-space origin is snapped to whole `texel_world` increments).
/// Reversed-Z ortho (near→1, far→0), matching the engine depth policy.
pub(crate) fn fit(cam: &Camera3D, sun: DVec3, c: Cascade, cfg: &ShadowCfg) -> CascadeFit {
    // Sun is a normalized direction — f32 is ample; narrow once here (the frozen
    // skeleton signature keeps it DVec3 to match the game's f64 world state).
    let sun = sun.as_vec3();
    // Slice bounds: Near covers [Z_NEAR, split0]; Far covers [split0, split1].
    let split = cfg.splits[c as usize];
    let near = match c {
        Cascade::Near => Z_NEAR,
        Cascade::Far => cfg.splits[0],
    };

    // Camera basis (camera at origin).
    let fwd = (cam.target - cam.position).normalize_or_zero();
    let right = fwd.cross(cam.up).normalize_or_zero();
    let up = right.cross(fwd);
    let tan_y = (cam.fovy.to_radians() * 0.5).tan();
    // A bounding sphere only needs the widest extent, so assume 16:9; a slightly
    // loose sphere only spends a little texel density.
    const ASSUMED_ASPECT: f32 = 16.0 / 9.0;
    let tan_x = tan_y * ASSUMED_ASPECT;

    // Eight slice corners, then the corner-average sphere (rotation-stable).
    let mut corners = [Vec3::ZERO; 8];
    let mut k = 0;
    for &d in &[near, split] {
        let cx = right * (tan_x * d);
        let cy = up * (tan_y * d);
        let centre = fwd * d;
        for &sx in &[-1.0f32, 1.0] {
            for &sy in &[-1.0f32, 1.0] {
                corners[k] = centre + cx * sx + cy * sy;
                k += 1;
            }
        }
    }
    let mut centre = Vec3::ZERO;
    for corner in &corners {
        centre += *corner;
    }
    centre /= 8.0;
    let mut radius = 0.0f32;
    for corner in &corners {
        radius = radius.max((*corner - centre).length());
    }

    let texel_world = cfg.texel_world_at(radius);

    // Light basis: `sun` points TOWARD the sun, so light travels along `-sun`.
    let light_dir = (-sun).normalize_or_zero();
    // Up hint chosen to avoid degeneracy when the sun is near the zenith.
    let up_hint = if light_dir.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let l_right = light_dir.cross(up_hint).normalize_or_zero();
    let l_up = l_right.cross(light_dir).normalize_or_zero();

    // Snap the sphere centre to whole texels along the light's lateral axes so a
    // camera translation slides the shadow texel grid in integer steps (no
    // shimmer). Snapping in WORLD space keeps eye and target in lockstep.
    let snap = |axis: Vec3| {
        let c = centre.dot(axis);
        (c / texel_world).round() * texel_world - c
    };
    let snapped = centre + l_right * snap(l_right) + l_up * snap(l_up);

    // Pull the eye back past the sphere toward the sun so occluders standing
    // above the lit region still fall inside the depth range.
    const PULLBACK: f32 = 100.0; // tall-occluder capture margin (m)
    let eye = snapped - light_dir * (radius + PULLBACK);
    let view = Mat4::look_at_rh(eye, snapped, up_hint);
    // Reversed-Z ortho: near/far arguments swapped so near→1, far→0.
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

// ── impl Renderer — per-frame uniforms + the pass record ─────────────────────

impl Renderer {
    /// Populate the FROZEN binding-3 `CascadeUniformsGpu` for this frame from
    /// both cascade fits. Layout (asserted in skeleton.rs, size 160, splits_fade
    /// at offset 128): `view_proj[2]`,
    /// `splits_fade = [split0, split1, fade_band, SHADOW_LIMIT]`,
    /// `bias = [blur_texels, slope_bias, dist_bias, texel_world_near]`.
    pub(crate) fn shadow_uniforms(
        &self,
        cam: &Camera3D,
        sun: DVec3,
        cfg: &ShadowCfg,
    ) -> CascadeUniformsGpu {
        let fits = PerCascade::new(CASCADES.map(|c| fit(cam, sun, c, cfg)));
        let near = &fits[Cascade::Near];
        let far = &fits[Cascade::Far];
        // SHADOW_LIMIT is the far cascade's far distance (256 m); the
        // receiver smoothsteps map→fallback over `fade_band` up to it.
        // `flags.shadows` false: push the map→fallback blend out of reach so the
        // receiver reads only the (cleared, fully-lit) map — with occluder
        // draws skipped below, every fragment passes the reversed-Z SampleCmp.
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

    /// Render opaque occluders into each cascade; replays the light-space-culled
    /// shadow_runs subset.
    pub(crate) fn record_shadow_pass(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        cam: &Camera3D,
        sun: DVec3,
        cfg: &ShadowCfg,
    ) {
        let device = &self.device.device;
        let shadow = &self.targets.shadow[crate::skeleton::FrameSlot::new(slot)];
        let layout = self.pipelines.layout_3d;

        let full_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: SHADOW_CASCADES,
        };

        unsafe {
            // UNDEFINED → DEPTH_ATTACHMENT for the occluder writes (contents
            // discarded each frame; every texel is cleared then rendered).
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

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.shadow.pipeline);

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

            // Occluders are drawn only when shadows are ON *and* this frame has
            // geometry: `bound()` is `None` on an empty 3D frame (or the
            // shadows-off prime, which only needs the clears), so there is simply
            // no handle to push. Binding 0 is pushed iff it will be consumed by a
            // draw — the push and the draw share this one `Option`, so a null
            // buffer can never reach `cmd_push_descriptor_set`.
            let occluders = self.flags
                .shadows
                .then(|| self.offsets[slot].bound())
                .flatten();
            if let Some(offsets_buffer) = occluders {
                // The offsets SSBO the vert reads (set 0, binding 0). Only binding
                // 0 is pushed — the depth-only vert touches no texture/UBO binding.
                let offsets_info = [vk::DescriptorBufferInfo::default()
                    .buffer(offsets_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)];
                let write = [vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&offsets_info)];
                self.device.push_descriptor.cmd_push_descriptor_set(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &write,
                );
            }

            for c in CASCADES {
                let f = fit(cam, sun, c, cfg);
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
                device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&f.view_proj.0.to_cols_array()),
                );
                // Pass still runs when shadows are off (the clears + layout
                // transitions keep binding 4 valid); only the draws are skipped.
                // Gated on the SAME `Option` that bound the offsets descriptor, so
                // this never draws without its binding nor on an empty frame.
                if occluders.is_some() {
                    self.record_shadow_occluders(cmd, slot);
                }
                device.cmd_end_rendering(cmd);
            }

            // DEPTH_ATTACHMENT → SHADER_READ for the receiver's PCF sample.
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

    /// Draw the light-space-culled occluder runs (`shadow_runs` — the opaque
    /// subset whose AABBs reach a cascade footprint; `mesh3d` and `mesh3d_biased`
    /// share the one depth pipeline) into the currently-bound cascade layer,
    /// indirect from `slot`'s command buffer, mirroring
    /// `record_mesh_indirect`'s feature-level fallback.
    unsafe fn record_shadow_occluders(&self, cmd: vk::CommandBuffer, slot: usize) {
        // Guard against empty shadow_runs; ensures quad IBO is allocated before use.
        if self.shadow_runs.is_empty() {
            return;
        }
        let device = &self.device.device;
        const STRIDE: u64 = size_of::<DrawIndexedIndirect>() as u64;
        // Reached only when the caller found `offsets.bound()` Some — i.e. this
        // frame has draws — so the indirect buffer is likewise allocated.
        let indirect = self.indirect[slot]
            .bound()
            .expect("occluder draws imply an allocated indirect buffer");
        // Shared quad IBO for all occluders (run.buffer is the VERTEX buffer).
        let quad_ibo = self
            .quad_ibo
            .bound()
            .expect("occluder draws imply the quad IBO is allocated");
        unsafe { device.cmd_bind_index_buffer(cmd, quad_ibo, 0, vk::IndexType::UINT32) };
        for run in self.shadow_runs.iter() {
            unsafe {
                device.cmd_bind_vertex_buffers(cmd, 0, &[run.buffer], &[0]);
                if self.device.multi_draw_indirect && self.device.draw_indirect_first_instance {
                    device.cmd_draw_indexed_indirect(
                        cmd,
                        indirect,
                        run.first as u64 * STRIDE,
                        run.count,
                        STRIDE as u32,
                    );
                } else if self.device.draw_indirect_first_instance {
                    for i in run.first..run.first + run.count {
                        device.cmd_draw_indexed_indirect(cmd, indirect, i as u64 * STRIDE, 1, STRIDE as u32);
                    }
                }
            }
        }
    }
}
