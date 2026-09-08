//! Scene-pass recorder: `RenderPass` and the mesh/sky/debug draws it issues.
//! Split out of `mod.rs` so later work can touch recording without opening
//! the frame loop or present path.

use ash::vk;

use crate::frame::DrawLists;
use crate::mesh::Pass;
use crate::skeleton::FrameSlot;

use super::buffers::{self, DrawIndexedIndirect};
use super::cull;
use super::frame_loop::{ImmOffsets, jittered_clip};
use super::gpu_timer::GpuPass;
use super::pipeline;
use super::{HdrReadable, Renderer, color_range, depth_range};

/// Manages dynamic rendering for one frame. Must call `end()` explicitly.
pub(super) struct RenderPass<'a> {
    r: &'a Renderer,
    cmd: vk::CommandBuffer,
    slot: usize,
    lists: &'a DrawLists,
    offsets: ImmOffsets,
    offscreen_image: vk::Image,
    ended: bool,
    /// Whether `layout_3d` descriptors are currently live. Incompatible layouts
    /// (sky, debug, 2D) disturb them; tracking lets mesh passes skip re-pushing.
    mesh_desc_bound: std::cell::Cell<bool>,
}

impl<'a> RenderPass<'a> {
    /// Records attachment layout transitions and begins dynamic rendering.
    pub(super) unsafe fn begin(
        r: &'a Renderer,
        cmd: vk::CommandBuffer,
        slot: usize,
        lists: &'a DrawLists,
        offsets: ImmOffsets,
        do_vrs: bool,
    ) -> RenderPass<'a> {
        let device = &r.device.device;
        let extent = r.render_extent;
        let offscreen_image = r.targets.offscreen[slot].image();
        let profiling = crate::profile::is_enabled();
        unsafe {
            // Generate the rate map first: it samples this slot's depth (leaving
            // it in DEPTH_ATTACHMENT_OPTIMAL, ready for the pass below) and
            // returns the only valid `RateAttachment`. Done before the color
            // barriers so the compute dispatch overlaps nothing it depends on.
            let rate = do_vrs.then(|| {
                let scene = lists.scene.as_ref().expect("do_vrs implies a 3D scene");
                let focal_px = 0.5 * extent.height as f32 / scene.fovy_tan_half.max(1e-4);
                let d_threshold = crate::camera::Z_NEAR / focal_px;
                let rate = r.record_vrs_generate(cmd, slot, d_threshold);
                if profiling {
                    r.gpu_timer.mark(device, cmd, slot, GpuPass::Vrs);
                }
                rate
            });

            // Transition attachments to render targets; old contents discarded.
            // The VRS pass above transitions the sampled depth when `do_vrs` —
            // under MSAA that is `resolved_depth`, so the MS `depth` attachment
            // still needs its own transition here.
            let mut image_barriers = [vk::ImageMemoryBarrier2::default(); 4];
            let mut barrier_count = 0;
            image_barriers[barrier_count] = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(offscreen_image)
                .subresource_range(color_range());
            barrier_count += 1;
            // MS depth needs a fresh-target transition unless the VRS pass
            // already put THIS image there — which it only does single-sampled
            // (under MSAA the VRS pass transitions `resolved_depth` instead).
            if !do_vrs || r.targets.msaa.is_some() {
                image_barriers[barrier_count] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .dst_access_mask(
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(r.depth_pass_layout())
                    .image(r.targets.depth[slot].image())
                    .subresource_range(depth_range());
                barrier_count += 1;
            }
            // The single-sample resolve target: bring it to the attachment layout
            // for the SAMPLE_ZERO resolve. When `do_vrs`, the VRS pass already
            // restored it to DEPTH_ATTACHMENT_OPTIMAL after sampling.
            if !do_vrs && let Some(resolved) = &r.targets.resolved_depth[slot] {
                image_barriers[barrier_count] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .image(resolved.image())
                    .subresource_range(depth_range());
                barrier_count += 1;
            }
            if let Some(msaa) = &r.targets.msaa {
                image_barriers[barrier_count] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(msaa.image())
                    .subresource_range(color_range());
                barrier_count += 1;
            }
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&image_barriers[..barrier_count]),
            );

            let clear_color = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [
                        // Linear-light clear straight into the HDR offscreen — the
                        // tonemap owns the OETF, so no encode here.
                        lists.clear.0[0],
                        lists.clear.0[1],
                        lists.clear.0[2],
                        1.0,
                    ],
                },
            };
            let offscreen_view = r.targets.offscreen[slot].view();
            let mut color_attachment = if let Some(msaa) = &r.targets.msaa {
                vk::RenderingAttachmentInfo::default()
                    .image_view(msaa.view())
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                    .resolve_image_view(offscreen_view)
                    .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
            } else {
                // Offscreen is color target; store contents for present copy.
                vk::RenderingAttachmentInfo::default()
                    .image_view(offscreen_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
            };
            color_attachment = color_attachment.clear_value(clear_color);
            let color_attachments = [color_attachment];

            // Reversed-Z: clear depth to 0.0, GREATER_OR_EQUAL test. Single-
            // sampled: store the depth so a later cycle can classify it for VRS.
            // MSAA: DONT_CARE the MS store — its single-sample SAMPLE_ZERO
            // resolve into `resolved_depth` is what feeds VRS/TAA/godrays.
            let depth_store = if r.targets.msaa.is_some() {
                vk::AttachmentStoreOp::DONT_CARE
            } else {
                vk::AttachmentStoreOp::STORE
            };
            let mut depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(r.targets.depth[slot].view())
                .image_layout(r.depth_pass_layout())
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(depth_store)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });
            if let Some(resolved) = &r.targets.resolved_depth[slot] {
                depth_attachment = depth_attachment
                    .resolve_mode(vk::ResolveModeFlags::SAMPLE_ZERO)
                    .resolve_image_view(resolved.view())
                    .resolve_image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
            }

            // `rate` (generated above) is the only source of a `RateAttachment`,
            // so its image is guaranteed classified and in the shading-rate
            // layout before it is bound here.
            let mut rate_attachment = rate.as_ref().map(|rate| {
                vk::RenderingFragmentShadingRateAttachmentInfoKHR::default()
                    .image_view(rate.view)
                    .image_layout(vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR)
                    .shading_rate_attachment_texel_size(rate.texel_size)
            });

            let mut rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment);
            if let Some(rate_attachment) = &mut rate_attachment {
                rendering_info = rendering_info.push_next(rate_attachment);
            }

            device.cmd_begin_rendering(cmd, &rendering_info);
            // Close the begin span (transitions + load-op clears) so the first
            // draw pass reports only its draws.
            if profiling {
                r.gpu_timer.mark(device, cmd, slot, GpuPass::Clear);
            }

            // Negative height for GL-style y-up NDC.
            let viewport = vk::Viewport {
                x: 0.0,
                y: extent.height as f32,
                width: extent.width as f32,
                height: -(extent.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );

            // Push-constant / push-descriptor state is bound per pass at record
            // time (each pass re-establishes it after interleaved passes bind
            // incompatible layouts), not once here.
        }

        RenderPass {
            r,
            cmd,
            slot,
            lists,
            offsets,
            offscreen_image,
            ended: false,
            mesh_desc_bound: std::cell::Cell::new(false),
        }
    }

    /// Pushes the `layout_3d` constants (view_proj + sky lighting/fog) and the
    /// push descriptors (per-draw offsets SSBO at binding 0, block-texture array
    /// at binding 1) shared by both mesh passes. Called at the head of each mesh
    /// pass rather than once up front, because interleaved passes bind
    /// incompatible layouts that disturb this state. Only sound when at least
    /// one mesh run exists (else the offsets SSBO can be a null buffer).
    unsafe fn bind_mesh3d_state(&self) {
        unsafe {
            self.push_mesh3d_descriptors();
            // LOD slab extents: LOD tiles hard-discard inside the full-res volume.
            self.push_mesh3d_constants(self.lists.lod_clip, self.lists.lod_clip_v);
        }
    }

    /// Pushes `layout_3d` descriptors only if a foreign pass disturbed them.
    /// Skips redundant pushes when adjacent mesh passes share state.
    unsafe fn push_mesh3d_descriptors(&self) {
        if self.mesh_desc_bound.get() {
            return;
        }
        let r = self.r;
        // Mesh passes ensure at least one run exists before pushing.
        let bufs = r
            .record_buffers
            .expect("a mesh run implies the record SSBOs are allocated");
        let layout = r.pipelines.layout_3d;
        buffers::push_mesh3d_descriptors(
            &r.device.push_descriptor,
            self.cmd,
            layout,
            bufs.records,
            bufs.dyns,
            r.block_textures.sampler,
            r.block_textures.view,
            r.ubo_ring.buffer(FrameSlot::new(self.slot)),
            r.shadow.ubo(self.slot),
            r.targets.shadow.sampler,
            r.targets.shadow.sample_view,
        );
        self.mesh_desc_bound.set(true);
    }

    /// Pushes view-proj + LOD slab extents. Pass-specific, so unconditionally
    /// pushed per pass (unlike descriptors). Jitter packaged here as a local.
    unsafe fn push_mesh3d_constants(&self, clip: f32, clip_v: f32) {
        let r = self.r;
        let scene = self
            .lists
            .scene
            .as_ref()
            .expect("a mesh pass implies a 3D scene");
        let push = pipeline::Mesh3dPush {
            view_proj: jittered_clip(scene.view_proj, scene.jitter.0, r.render_extent),
            clip,
            clip_v,
            _pad: [0.0; 2],
            eye: pipeline::EyeSplit::of(scene.eye),
        };
        let layout = r.pipelines.layout_3d;
        unsafe {
            r.device.device.cmd_push_constants(
                self.cmd,
                layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
        }
    }

    /// Marks the `layout_3d` push descriptors stale: call after binding any
    /// pipeline whose layout is not push-compatible with `layout_3d` (sky, debug
    /// cubes/lines/shadows, 2D), so the next `layout_3d` pass re-pushes them.
    fn invalidate_mesh_desc(&self) {
        self.mesh_desc_bound.set(false);
    }

    /// Issues indirect mesh draws for one pass, using the best available
    /// feature level and falling back from multi-draw to single-draw indirect
    /// as needed. Runs are sorted so a pass's runs are contiguous; the pass
    /// pipeline binds once, before the first matching run. Only called when
    /// `lists.scene.is_some()`.
    pub(super) unsafe fn record_mesh_indirect(&self, pass: Pass) {
        // Opaque/Cutout always come from the GPU cull's partitions; only Blend
        // takes the CPU-sorted run path below.
        if pass != Pass::Blend {
            unsafe { self.record_mesh_indirect_count(pass) };
            return;
        }
        if !self.r.draw_runs.iter().any(|run| run.pass == pass) {
            return;
        }
        // Interleaved debug/sky/2D passes bind pipelines with layouts that are
        // not push-compatible with `layout_3d`, which per Vulkan's layout-
        // compatibility rules disturbs this layout's push constants and push
        // descriptors. Re-establish them at the head of every mesh pass so the
        // transparent pass (recorded after sky) draws with valid state.
        unsafe { self.bind_mesh3d_state() };
        // The water-absorption blend variant reads the scene depth as an input
        // attachment (set 0 binding 5). Layered on top of the 0-4 push above
        // (same layout ⇒ those writes stay live); pushed only for Blend when the
        // absorb pipeline is active, and consumed only inside the water branch.
        let absorb_active = self.r.pipelines.mesh3d_transparent_absorb.is_some();
        if pass == Pass::Blend && absorb_active {
            let layout = self.r.pipelines.layout_3d;
            buffers::push_depth_input_attachment(
                &self.r.device.push_descriptor,
                self.cmd,
                layout,
                self.r.targets.depth[self.slot].view(),
            );
            // Framebuffer-local (BY_REGION) dependency INSIDE the render pass
            // — legal exactly because dynamic_rendering_local_read is enabled
            // whenever this pipeline exists: the opaque passes' depth writes
            // must be visible to the water branch's input-attachment reads.
            // No layout change (illegal inside a pass; the frame-wide
            // RENDERING_LOCAL_READ layout is what makes that unnecessary).
            let dep = [vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::INPUT_ATTACHMENT_READ)];
            unsafe {
                self.r.device.device.cmd_pipeline_barrier2(
                    self.cmd,
                    &vk::DependencyInfo::default()
                        .dependency_flags(vk::DependencyFlags::BY_REGION)
                        .memory_barriers(&dep),
                );
            }
            // The instance's input-attachment mapping must MATCH the bound
            // pipeline at every draw. Set the absorb pipeline's mapping (depth
            // → input 0, color not-an-input) for exactly this pass's draws;
            // restored to the implicit identity below so the later sky/debug
            // pipelines (created without a mapping) stay valid.
            unsafe { self.set_input_attachment_mapping(true) };
        }
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            let indirect_buffer = self.r.slots[FrameSlot::new(self.slot)]
                .indirect
                .bound()
                .expect("a draw run implies the indirect buffer is allocated");
            // One shared quad IBO for every run: bucket-permuted vertices make each
            // run's `first_index`/`vertex_offset` address it directly. Bound once —
            // index-buffer binding survives the per-run pipeline rebinds below.
            let quad_ibo = self
                .r
                .quad_ibo
                .bound()
                .expect("a draw run implies the quad IBO is allocated");
            device.cmd_bind_index_buffer(cmd, quad_ibo, 0, vk::IndexType::UINT32);
            const STRIDE: u64 = std::mem::size_of::<DrawIndexedIndirect>() as u64;
            // Rebind only when the pass's pipeline changes.
            let mut bound: Option<vk::Pipeline> = None;
            for run in self.r.draw_runs.iter().filter(|run| run.pass == pass) {
                let pipeline = self.r.mesh_pipeline_for(pass);
                if bound != Some(pipeline) {
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    bound = Some(pipeline);
                }
                device.cmd_bind_vertex_buffers(cmd, 0, &[run.buffer], &[0]);
                if self.r.device.multi_draw_indirect && self.r.device.draw_indirect_first_instance {
                    device.cmd_draw_indexed_indirect(
                        cmd,
                        indirect_buffer,
                        run.first as u64 * STRIDE,
                        run.count,
                        STRIDE as u32,
                    );
                } else if self.r.device.draw_indirect_first_instance {
                    // Fall back to single-draw indirect calls.
                    for i in run.first..run.first + run.count {
                        device.cmd_draw_indexed_indirect(
                            cmd,
                            indirect_buffer,
                            i as u64 * STRIDE,
                            1,
                            STRIDE as u32,
                        );
                    }
                } else {
                    // Fall back to direct draws; replay commands CPU-side.
                    let range = run.first as usize..(run.first + run.count) as usize;
                    for c in &self.r.draw_commands[range] {
                        device.cmd_draw_indexed(
                            cmd,
                            c.index_count,
                            c.instance_count,
                            c.first_index,
                            c.vertex_offset,
                            c.first_instance,
                        );
                    }
                }
            }
        }
        if pass == Pass::Blend && absorb_active {
            unsafe { self.set_input_attachment_mapping(false) };
        }
    }

    /// GPU-culled variant of [`record_mesh_indirect`](Self::record_mesh_indirect):
    /// one `vkCmdDrawIndexedIndirectCount` per non-empty (group, arena, bucket)
    /// partition, near-to-far, consuming the commands the cull dispatch emitted
    /// earlier in this command buffer. Never called for Blend (CPU-sorted path).
    /// Opaque draws its full-res partition (no-`discard` pipeline) first, then
    /// the coarse-LOD partition (slab-clip pipeline) — near before far, so the
    /// LOD skirt behind full-res terrain is mostly depth-rejected.
    unsafe fn record_mesh_indirect_count(&self, pass: Pass) {
        let groups: &[(cull::Group, vk::Pipeline)] = match pass {
            Pass::Opaque => &[
                (cull::Group::Opaque, self.r.pipelines.mesh3d),
                (cull::Group::OpaqueLod, self.r.pipelines.mesh3d_lod),
            ],
            Pass::Cutout => &[(cull::Group::Cutout, self.r.mesh_pipeline_for(pass))],
            Pass::Blend => unreachable!("Blend stays on the CPU path"),
        };
        for &(group, pipeline) in groups {
            unsafe { self.record_group_indirect_count(group, pipeline) };
        }
    }

    /// Draws every non-empty arena partition of one cull group with `pipeline`.
    unsafe fn record_group_indirect_count(&self, group: cull::Group, pipeline: vk::Pipeline) {
        let Some(frame) = &self.r.cull_frame else {
            return; // nothing live to draw
        };
        let span = frame.arena_count * cull::BUCKETS;
        let base = group as usize * span;
        if frame.partitions[base..base + span]
            .iter()
            .all(|p| p.capacity == 0)
        {
            return;
        }
        unsafe { self.bind_mesh3d_state() };
        let device = &self.r.device.device;
        unsafe {
            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            let quad_ibo = self
                .r
                .quad_ibo
                .bound()
                .expect("live records imply the quad IBO is allocated");
            device.cmd_bind_index_buffer(self.cmd, quad_ibo, 0, vk::IndexType::UINT32);
            for arena in 0..frame.arena_count {
                let first = cull::camera_part(group as usize, arena, 0, frame.arena_count);
                if frame.partitions[first..first + cull::BUCKETS]
                    .iter()
                    .all(|p| p.capacity == 0)
                {
                    continue;
                }
                device.cmd_bind_vertex_buffers(
                    self.cmd,
                    0,
                    &[self.r.arena_dir.arena_buffer(arena)],
                    &[0],
                );
                for bucket in 0..cull::BUCKETS {
                    let idx = first + bucket;
                    let part = frame.partitions[idx];
                    if part.capacity == 0 {
                        continue;
                    }
                    device.cmd_draw_indexed_indirect_count(
                        self.cmd,
                        frame.commands,
                        u64::from(part.offset) * cull::CMD_STRIDE,
                        frame.counts,
                        (idx * 4) as u64,
                        part.capacity,
                        cull::CMD_STRIDE as u32,
                    );
                }
            }
        }
    }

    /// Sets the render-pass instance's input-attachment mapping: the absorb
    /// pipeline's custom one (`true`: depth → fragment input 0, the color
    /// attachment not an input), or back to the IMPLICIT identity every
    /// mapping-less pipeline was created with (`false`: color 0 → input 0,
    /// no depth) — the state a pipeline and the instance must agree on at
    /// every draw (VUID-vkCmdDraw*-None-09549/10927).
    unsafe fn set_input_attachment_mapping(&self, absorb: bool) {
        let lr = self
            .r
            .device
            .local_read
            .as_ref()
            .expect("absorb pipeline exists only with local_read");
        let depth_input_index = 0u32;
        let custom_colors = [vk::ATTACHMENT_UNUSED];
        let identity_colors = [0u32];
        let mapping = if absorb {
            vk::RenderingInputAttachmentIndexInfoKHR::default()
                .color_attachment_input_indices(&custom_colors)
                .depth_input_attachment_index(&depth_input_index)
        } else {
            vk::RenderingInputAttachmentIndexInfoKHR::default()
                .color_attachment_input_indices(&identity_colors)
        };
        unsafe { lr.cmd_set_rendering_input_attachment_indices(self.cmd, &mapping) };
    }

    /// Pushes `view_proj` to `layout_debug` for the immediate debug geometry.
    /// Done per debug pass because the mesh passes bind `layout_3d`, whose
    /// incompatible push-constant range disturbs this value.
    unsafe fn push_debug_view_proj(&self) {
        let scene = self
            .lists
            .scene
            .as_ref()
            .expect("a debug pass implies a 3D scene");
        let push = pipeline::DebugPush {
            // Match the mesh pass jitter so debug geometry doesn't shimmer against
            // jittered terrain under TAA.
            view_proj: jittered_clip(scene.view_proj, scene.jitter.0, self.r.render_extent),
        };
        unsafe {
            self.r.device.device.cmd_push_constants(
                self.cmd,
                self.r.pipelines.layout_debug,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push),
            );
        }
    }

    /// The immediate-mode debug cubes (debug_tris pipeline, immediate buffer at
    /// offset 0). Only issued when `lists.scene.is_some()`.
    pub(super) unsafe fn record_immediate_cubes(&self) {
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            if !self.lists.cube_verts.is_empty() {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.r.pipelines.debug_tris,
                );
                self.invalidate_mesh_desc();
                self.push_debug_view_proj();
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.r.slots[FrameSlot::new(self.slot)]
                        .imm
                        .bound()
                        .expect("a non-empty immediate list implies an allocated buffer")],
                    &[0],
                );
                device.cmd_draw(cmd, self.lists.cube_verts.len() as u32, 1, 0, 0);
            }
        }
    }

    /// Translucent ground decals / contact shadows (debug_tris_blend pipeline).
    /// Alpha-blended and depth-read-only, so they draw after the opaque cubes and
    /// blend over terrain without occluding geometry behind them.
    pub(super) unsafe fn record_shadows(&self) {
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            if !self.lists.shadow_verts.is_empty() {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.r.pipelines.debug_tris_blend,
                );
                self.invalidate_mesh_desc();
                self.push_debug_view_proj();
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.r.slots[FrameSlot::new(self.slot)]
                        .imm
                        .bound()
                        .expect("a non-empty immediate list implies an allocated buffer")],
                    &[self.offsets.shadow],
                );
                device.cmd_draw(cmd, self.lists.shadow_verts.len() as u32, 1, 0, 0);
            }
        }
    }

    /// The immediate-mode debug lines (debug_lines pipeline). Only issued when
    /// `lists.scene.is_some()`.
    pub(super) unsafe fn record_lines(&self) {
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            if !self.lists.line_verts.is_empty() {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.r.pipelines.debug_lines,
                );
                self.invalidate_mesh_desc();
                self.push_debug_view_proj();
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.r.slots[FrameSlot::new(self.slot)]
                        .imm
                        .bound()
                        .expect("a non-empty immediate list implies an allocated buffer")],
                    &[self.offsets.line],
                );
                device.cmd_draw(cmd, self.lists.line_verts.len() as u32, 1, 0, 0);
            }
        }
    }

    /// The procedural sky background pass (sky pipeline: fragment push constant,
    /// FrameUniforms at set 0 binding 1, cloud LUT at binding 0, no vertex
    /// buffer). A single fullscreen triangle at the reversed-Z far plane; the
    /// read-only depth test rejects it wherever terrain wrote closer depth, so
    /// it shades only background pixels. Skipped unless the frame set a sky
    /// palette.
    pub(super) unsafe fn record_sky(&self) {
        let Some(desc) = self.lists.sky else {
            return;
        };
        let scene = self
            .lists
            .scene
            .as_ref()
            .expect("a sky pass implies a 3D scene");
        let device = &self.r.device.device;
        let cmd = self.cmd;
        // Same jitter the mesh pass applies, so TAA sees a coherently jittered
        // frame (sky vs terrain silhouettes) and history reprojection is stable.
        let jittered = jittered_clip(scene.view_proj, scene.jitter.0, self.r.render_extent);
        let params = pipeline::SkyParams::compose(jittered.inverse(), &desc);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.r.pipelines.sky);
            self.invalidate_mesh_desc();
            device.cmd_push_constants(
                cmd,
                self.r.pipelines.layout_sky,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&params),
            );
            let lut = &self.r.targets.sky_cloud[self.slot];
            let lut_infos = [vk::DescriptorImageInfo::default()
                .sampler(self.r.pipelines.sky_lut_sampler)
                .image_view(lut.view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let ubo = self.r.ubo_ring.buffer(FrameSlot::new(self.slot));
            let ubo_infos = [vk::DescriptorBufferInfo::default()
                .buffer(ubo)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&lut_infos),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&ubo_infos),
            ];
            self.r.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.r.pipelines.layout_sky,
                0,
                &writes,
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    /// Ends dynamic rendering, transitions the offscreen image to
    /// `SHADER_READ_ONLY_OPTIMAL`, and returns the [`HdrReadable`] proof. The
    /// timeline orders later submits; this barrier owns layout and visibility.
    /// Use when no later pass writes the offscreen, so this pass owns the final
    /// transition for tonemapping.
    pub(super) unsafe fn end_sampled(self) -> HdrReadable {
        let slot = self.slot;
        unsafe { self.end(true) };
        HdrReadable::new(slot)
    }

    /// Ends dynamic rendering WITHOUT the sampled transition: a later offscreen
    /// writer (TAA resolve / exposure metering) runs after this, and one of them
    /// owns the finalization instead (its barrier would otherwise race their
    /// writes). Yields no proof — the deferred finalizer produces it.
    pub(super) unsafe fn end_deferred(self) {
        unsafe { self.end(false) };
    }

    unsafe fn end(mut self, transition_offscreen: bool) {
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            device.cmd_end_rendering(cmd);

            self.ended = true;
            if !transition_offscreen {
                return;
            }

            let to_sampled = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.offscreen_image)
                .subresource_range(color_range())];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        }
    }
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        debug_assert!(self.ended, "RenderPass dropped without calling end()");
    }
}
