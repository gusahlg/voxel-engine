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
use super::gpu_timer::{GpuPass, PipeStatPass};
use super::pipeline;
use super::{HdrReadable, Renderer, SceneDepthUse, color_range, depth_range, scene_depth_use};

/// What this frame's VRS does with the slot's rate and history images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SceneVrs {
    /// No classify and no rate attachment: VRS off, no 3D scene, or no rate
    /// images on this device.
    Off,
    /// First use of the slot since create/recreate: no rate attachment, and
    /// the classifier writes the rate and history images from UNDEFINED.
    Prime,
    /// The pass binds the rate image this slot's last classify wrote
    /// (`FRAMES_IN_FLIGHT` frames ago, the staleness the old begin-of-frame
    /// classify accepted), then the classifier rewrites it.
    Bound,
}

impl SceneVrs {
    /// The pass binds the rate image as its shading-rate attachment.
    pub(super) fn binds_rate(self) -> bool {
        self == Self::Bound
    }

    /// The classifier runs at the end of the pass, priming the next use of
    /// the slot.
    pub(super) fn classifies(self) -> bool {
        self != Self::Off
    }
}

/// Who moves the HDR offscreen to `SHADER_READ_ONLY_OPTIMAL` for the
/// tonemap present copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HdrFinalize {
    /// Exposure metering writes the offscreen after the pass and owns the
    /// transition (presented 3D frames with exposure on).
    Exposure,
    /// [`RenderPass::end`] does, in its post-scene barrier (presented, no
    /// exposure).
    Pass,
    /// Nobody: the frame is not presented, and the next begin discards the
    /// offscreen from UNDEFINED.
    Skip,
}

/// Every per-frame decision of the scene pass and the work recorded right
/// after it in the same command buffer. Made once per frame by
/// [`scene_plan`] in `draw_frame`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScenePlan {
    /// The frame presents. HDR colour is read only by bloom, exposure, spill
    /// and the tonemap, all present-only, so colour is stored (MSAA: resolved)
    /// only then. Minimap is a separate texture, screenshots copy the
    /// swapchain after tonemap, and VRS classify reads depth, not colour.
    pub(super) present: bool,
    pub(super) hdr: HdrFinalize,
    /// The sky pass draws: the colour attachment loads `DONT_CARE`, and the
    /// far table and cloud LUT are written ([`super::frame_loop::sky_drawn`]).
    pub(super) sky: bool,
    pub(super) vrs: SceneVrs,
    /// What later passes this frame read of the depth.
    pub(super) depth_use: SceneDepthUse,
    /// Source stages of the depth attachment's begin barrier: the reads of
    /// this slot's depth since its last pass. `FRAGMENT_SHADER` when a later
    /// frame's water absorption sampled its stored depth, `COMPUTE_SHADER`
    /// when its last classify read the stored MS depth, else `NONE`.
    pub(super) depth_src: vk::PipelineStageFlags2,
    /// Water-absorption Blend draws: binding 5 is the previous slot's
    /// sampleable depth, or the dummy when that depth is invalid.
    pub(super) absorb: bool,
    /// The previous slot stored sampleable depth at this render extent.
    pub(super) prev_depth_valid: bool,
    /// Lean opaque/LOD fragment pipelines: every optional lighting lane is
    /// off and fog is off.
    pub(super) mesh_lean: bool,
    /// The quarter-res spill has work on presented frames: bloom, or a live
    /// godray march.
    pub(super) spill_live: bool,
}

/// Inputs to [`scene_plan`], plain values so every combination is
/// host-testable.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SceneInputs {
    /// The frame has a 3D scene.
    pub(super) scene: bool,
    /// `decide_present` acquired a swapchain image.
    pub(super) present: bool,
    /// [`super::frame_loop::sky_drawn`].
    pub(super) sky: bool,
    /// `RenderFlags::vrs`, and the targets have rate images.
    pub(super) vrs: bool,
    /// This slot's last classify left its rate image in GENERAL.
    pub(super) classified: bool,
    /// MSAA with the sample-0 `vrs_ms` classifier.
    pub(super) classify_ms: bool,
    pub(super) taa: bool,
    pub(super) exposure: bool,
    /// Bloom on, or the godray march has strength.
    pub(super) spill_live: bool,
    /// The absorb Blend pipeline exists and a Blend run is drawn.
    pub(super) absorb: bool,
    pub(super) mesh_lean: bool,
    pub(super) prev_depth_valid: bool,
    /// A later frame's water absorption sampled this slot's stored depth.
    pub(super) depth_sampled: bool,
    /// This slot's last classify read its stored MS depth.
    pub(super) ms_depth_read: bool,
}

/// This frame's [`ScenePlan`].
pub(super) fn scene_plan(i: SceneInputs) -> ScenePlan {
    // First scene pass of a slot after create/recreate skips VRS and still
    // classifies at the end, so the next use is primed.
    let vrs = if !(i.scene && i.vrs) {
        SceneVrs::Off
    } else if i.classified {
        SceneVrs::Bound
    } else {
        SceneVrs::Prime
    };
    let depth_use = scene_depth_use(
        i.classify_ms,
        i.present,
        i.taa,
        i.spill_live,
        vrs.classifies(),
        i.absorb,
    );
    // Exposure metering (like bloom) feeds only the tonemap present copy, so
    // it runs solely on frames that will present (`decide_present` already
    // ran; a forced capture always presents).
    let hdr = if !i.present {
        HdrFinalize::Skip
    } else if i.scene && i.exposure {
        HdrFinalize::Exposure
    } else {
        HdrFinalize::Pass
    };
    let mut depth_src = vk::PipelineStageFlags2::NONE;
    if i.depth_sampled {
        depth_src |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
    }
    if i.ms_depth_read {
        depth_src |= vk::PipelineStageFlags2::COMPUTE_SHADER;
    }
    ScenePlan {
        present: i.present,
        hdr,
        sky: i.sky,
        vrs,
        depth_use,
        depth_src,
        absorb: i.absorb,
        prev_depth_valid: i.prev_depth_valid,
        mesh_lean: i.mesh_lean,
        spill_live: i.spill_live,
    }
}

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
    /// Jittered clip matrix and eye split, computed once in `begin` when a 3D
    /// scene is present. Shared by mesh push constants, debug view_proj, and sky.
    scene_state: Option<(glam::Mat4, pipeline::EyeSplit)>,
    mesh_push_bound: std::cell::Cell<bool>,
    index_bound: std::cell::Cell<bool>,
    /// This frame's decisions: depth use, water absorption and its previous
    /// depth, lean pipelines, VRS, and who finalizes the HDR.
    plan: ScenePlan,
}

impl<'a> RenderPass<'a> {
    /// Records attachment layout transitions and begins dynamic rendering.
    pub(super) unsafe fn begin(
        r: &'a Renderer,
        cmd: vk::CommandBuffer,
        slot: usize,
        lists: &'a DrawLists,
        offsets: ImmOffsets,
        plan: ScenePlan,
    ) -> RenderPass<'a> {
        let device = &r.device.device;
        let extent = r.render_extent;
        let offscreen_image = r.targets.offscreen[slot].image();
        let profiling = crate::profile::is_enabled();
        unsafe {
            // Bind last-use's rate image if this slot has already classified
            // (`SceneVrs::Bound`). The classifier ran at the END of that
            // previous use and left the image in GENERAL; the begin batch
            // below transitions it to FRAGMENT_SHADING_RATE_ATTACHMENT. First
            // use after create/recreate skips VRS (`SceneVrs::Prime`).
            let rate = plan.vrs.binds_rate().then(|| r.vrs_rate_attachment(slot));

            // One vkCmdPipelineBarrier2 for every image the pass writes, plus
            // the rate image when VRS is bound. Sampleable depth always begins
            // from UNDEFINED (cleared every frame; see SAMPLEABLE_DEPTH_REST_LAYOUT).
            let mut image_barriers = [vk::ImageMemoryBarrier2::default(); 5];
            let mut barrier_count = 0;
            // Offscreen: src NONE / NONE. `UNDEFINED` discards the image; a
            // color-out src would serialize against the previous CB even though
            // the slot timeline wait already proved this image idle.
            // Dst COLOR_ATTACHMENT_OUTPUT / COLOR_ATTACHMENT_WRITE.
            // Old UNDEFINED → COLOR_ATTACHMENT_OPTIMAL.
            // MSAA: this image is the AVERAGE resolve target; skip the barrier
            // when colour is not stored (no resolve this frame).
            if plan.present || r.targets.msaa.is_none() {
                image_barriers[barrier_count] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(offscreen_image)
                    .subresource_range(color_range());
                barrier_count += 1;
            }
            // Depth attachment (MS depth when multisampled, else the
            // single-sample depth): src NONE / NONE (`UNDEFINED` discard)
            // when nothing sampled this image; plus FRAGMENT_SHADER when a
            // later frame sampled this slot's stored depth (WAR: that sample
            // is a different command buffer still in flight); plus
            // COMPUTE_SHADER when the VRS classifier read this slot's stored
            // MS depth at the end of its previous use (`plan.depth_src`). Dst
            // EARLY|LATE_FRAGMENT_TESTS / DEPTH_STENCIL_ATTACHMENT_{READ,WRITE}.
            // Old UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL. Contents are cleared
            // every frame.
            image_barriers[barrier_count] = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(plan.depth_src)
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
                .new_layout(super::Renderer::depth_pass_layout())
                .image(r.targets.depth[slot].image())
                .subresource_range(depth_range());
            barrier_count += 1;
            // MSAA SAMPLE_ZERO resolve target: src NONE / NONE (`UNDEFINED`
            // discard; Vulkan would otherwise run a color-out src on this
            // depth resolve). Dst COLOR_ATTACHMENT_OUTPUT / COLOR_ATTACHMENT_WRITE.
            // Old UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL.
            // This image rests in SAMPLEABLE_DEPTH_REST_LAYOUT after `end` only
            // for `Sampleable`. Skip the resolve-target barrier otherwise: no
            // consumer, or the classifier alone reads the stored MS `depth`.
            if plan.depth_use == SceneDepthUse::Sampleable
                && let Some(resolved) = &r.targets.resolved_depth[slot]
            {
                image_barriers[barrier_count] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
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
                // MSAA color: ONE image shared by every slot, so the slot
                // timeline wait does not prove it idle — the previous frame
                // (another slot, possibly still on the GPU) wrote it and its
                // AVERAGE resolve read it. Src COLOR_ATTACHMENT_OUTPUT /
                // COLOR_ATTACHMENT_WRITE orders this discard after both
                // (attachment writes and resolves run in that stage).
                // Dst COLOR_ATTACHMENT_OUTPUT / COLOR_ATTACHMENT_WRITE.
                // Old UNDEFINED → COLOR_ATTACHMENT_OPTIMAL: contents are
                // cleared or fully covered by the sky every frame.
                image_barriers[barrier_count] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(msaa.image())
                    .subresource_range(color_range());
                barrier_count += 1;
            }
            if plan.vrs.binds_rate() {
                image_barriers[barrier_count] = r.vrs_rate_to_attachment_barrier(slot);
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
            // Sky triangle covers every pixel left at reversed-Z far (depth 0).
            // Debug-flat (TerrainKey) frames carry `lists.sky == None` and still clear.
            let color_load = if plan.sky {
                vk::AttachmentLoadOp::DONT_CARE
            } else {
                vk::AttachmentLoadOp::CLEAR
            };
            let mut color_attachment = if let Some(msaa) = &r.targets.msaa {
                // MS store is always DONT_CARE; the offscreen AVERAGE resolve
                // is the colour that bloom/exposure/spill/tonemap read. Skip
                // that resolve when this frame will not present (those
                // consumers are present-only; minimap is a separate texture,
                // screenshots copy the swapchain after tonemap, VRS reads
                // depth not colour).
                let mut att = vk::RenderingAttachmentInfo::default()
                    .image_view(msaa.view())
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(color_load)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE);
                if plan.present {
                    att = att
                        .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                        .resolve_image_view(offscreen_view)
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                }
                att
            } else {
                // Offscreen is color target. Store for present-copy consumers;
                // DONT_CARE when this frame is not presented.
                vk::RenderingAttachmentInfo::default()
                    .image_view(offscreen_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(color_load)
                    .store_op(if plan.present {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
            };
            color_attachment = color_attachment.clear_value(clear_color);
            let color_attachments = [color_attachment];

            // Reversed-Z: clear depth to 0.0, GREATER_OR_EQUAL test. Single-
            // sampled: store the depth so later consumers (end-of-frame
            // classify, spill-godrays, present TAA) can sample it after it
            // rests in SAMPLEABLE_DEPTH_REST_LAYOUT. MSAA `Sampleable`:
            // DONT_CARE the MS store — its single-sample SAMPLE_ZERO resolve
            // into `resolved_depth` is what those consumers read. MSAA
            // `ClassifyMs`: store the MS depth and skip the resolve; the
            // classifier loads sample 0 directly. When nothing samples depth
            // this frame, DONT_CARE the store and skip the resolve; the next
            // begin discards from UNDEFINED.
            let depth_store = if plan.depth_use.stores_attachment(r.targets.msaa.is_some()) {
                vk::AttachmentStoreOp::STORE
            } else {
                vk::AttachmentStoreOp::DONT_CARE
            };
            let mut depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(r.targets.depth[slot].view())
                .image_layout(super::Renderer::depth_pass_layout())
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(depth_store)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });
            if plan.depth_use == SceneDepthUse::Sampleable
                && let Some(resolved) = &r.targets.resolved_depth[slot]
            {
                depth_attachment = depth_attachment
                    .resolve_mode(vk::ResolveModeFlags::SAMPLE_ZERO)
                    .resolve_image_view(resolved.view())
                    .resolve_image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
            }

            // `rate` is the previous classify of this slot, transitioned to
            // the shading-rate layout in the begin batch above.
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
                r.gpu_timer.recorded(slot);
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

        let scene_state = lists.scene.as_ref().map(|scene| {
            (
                jittered_clip(scene.view_proj, scene.jitter.0, r.render_extent),
                pipeline::EyeSplit::of(scene.eye),
            )
        });

        RenderPass {
            r,
            cmd,
            slot,
            lists,
            offsets,
            offscreen_image,
            ended: false,
            mesh_desc_bound: std::cell::Cell::new(false),
            scene_state,
            mesh_push_bound: std::cell::Cell::new(false),
            index_bound: std::cell::Cell::new(false),
            plan,
        }
    }

    /// Pushes the `layout_3d` constants (view_proj + sky lighting/fog) and the
    /// push descriptors (per-draw offsets SSBO at binding 0, block-texture array
    /// at binding 1, material table at binding 7) shared by both mesh passes. Called at the head of each mesh
    /// pass rather than once up front, because interleaved passes bind
    /// incompatible layouts that disturb this state. Only sound when at least
    /// one mesh run exists (else the offsets SSBO can be a null buffer).
    ///
    /// Push-constant bytes are identical for every mesh pass (the LOD clip box);
    /// the quad IBO binding survives pipeline/layout changes.
    /// Both are bound once and skipped until a foreign pass invalidates them
    /// (push constants) — the IBO is never invalidated.
    unsafe fn bind_mesh3d_state(&self) {
        unsafe {
            self.push_mesh3d_descriptors();
            if !self.mesh_push_bound.get() {
                // LOD box: LOD tiles hard-discard inside the full-res volume.
                self.push_mesh3d_constants(self.lists.lod_half, self.lists.lod_centre);
                self.mesh_push_bound.set(true);
            }
            if !self.index_bound.get() {
                let quad_ibo = self
                    .r
                    .quad_ibo
                    .bound()
                    .expect("a mesh pass implies the quad IBO is allocated");
                self.r.device.device.cmd_bind_index_buffer(
                    self.cmd,
                    quad_ibo,
                    0,
                    vk::IndexType::UINT32,
                );
                self.index_bound.set(true);
            }
        }
    }

    /// Pushes `layout_3d` descriptors only if a foreign pass disturbed them.
    /// Skips redundant pushes when adjacent mesh passes share state.
    ///
    /// `mesh3d` / `mesh3d_lod` and their lean twins share `layout_3d`, so one
    /// `vkCmdPushDescriptorSetKHR` remains valid across those pipeline binds
    /// (push-descriptor state is per compatible layout, not per pipeline).
    /// Shadows-off still pushes the cascade UBO + shadow sampler: the layout
    /// requires every binding; the skip is the host UBO memcpy, not this push.
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
            r.materials.buffer(),
            bufs.cages,
        );
        self.mesh_desc_bound.set(true);
    }

    /// Pushes view-proj + LOD box extents. Identical for every mesh pass, so
    /// skipped while `mesh_push_bound`. Jitter packaged once in `begin`.
    unsafe fn push_mesh3d_constants(&self, half: glam::Vec3, centre: glam::Vec3) {
        let r = self.r;
        let (view_proj, eye) = self.scene_state.expect("a mesh pass implies a 3D scene");
        let extent = r.render_extent;
        let inv_render_extent = if self.plan.prev_depth_valid {
            [1.0 / extent.width as f32, 1.0 / extent.height as f32]
        } else {
            [0.0, 0.0]
        };
        let push = pipeline::Mesh3dPush::pack(view_proj, half, centre, inv_render_extent, eye);
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
        self.mesh_push_bound.set(false);
        // Index-buffer binding survives pipeline/layout changes.
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
            crate::profile::gauge(crate::profile::Gauge::CallsBlend, 0);
            return;
        }
        self.r.gpu_timer.recorded(self.slot);
        // Interleaved debug/sky/2D passes bind pipelines with layouts that are
        // not push-compatible with `layout_3d`, which per Vulkan's layout-
        // compatibility rules disturbs this layout's push constants and push
        // descriptors. Re-establish them at the head of every mesh pass so the
        // transparent pass (recorded after sky) draws with valid state.
        unsafe { self.bind_mesh3d_state() };
        // The water-absorption blend variant samples the previous slot's
        // sampleable depth (set 0 binding 5). Layered on top of the 0-4 push
        // above (same layout ⇒ those writes stay live); pushed only for Blend
        // when the absorb pipeline is active, and consumed only inside the
        // water branch. Dummy 1×1 when that depth is invalid (shader falls
        // back to WATER_BODY_MIX).
        if self.plan.absorb {
            let layout = self.r.pipelines.layout_3d;
            let depth_view = if self.plan.prev_depth_valid {
                self.r
                    .targets
                    .sampleable_depth(super::PrevDepthTrack::prev_slot(self.slot))
                    .view()
            } else {
                self.r.prev_depth_dummy.view()
            };
            buffers::push_prev_depth(
                &self.r.device.push_descriptor,
                self.cmd,
                layout,
                self.r.pipelines.tonemap_depth_sampler,
                depth_view,
            );
        }
        let device = &self.r.device.device;
        let cmd = self.cmd;
        let mut blend_calls = 0u64;
        unsafe {
            let indirect_buffer = self.r.slots[FrameSlot::new(self.slot)]
                .indirect
                .bound()
                .expect("a draw run implies the indirect buffer is allocated");
            const STRIDE: u64 = std::mem::size_of::<DrawIndexedIndirect>() as u64;
            // Rebind only when the pass's pipeline changes.
            let mut bound: Option<vk::Pipeline> = None;
            for run in self.r.draw_runs.iter().filter(|run| run.pass == pass) {
                let pipeline = if run.caged {
                    self.r.pipelines.caged_blend()
                } else {
                    self.r.mesh_pipeline_for(pass)
                };
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
                    blend_calls += 1;
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
                    blend_calls += u64::from(run.count);
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
                    blend_calls += u64::from(run.count);
                }
            }
        }
        crate::profile::gauge(crate::profile::Gauge::CallsBlend, blend_calls);
    }

    /// GPU-culled variant of [`record_mesh_indirect`](Self::record_mesh_indirect):
    /// one `vkCmdDrawIndexedIndirectCount` per non-empty (group, arena, bucket)
    /// partition, near-to-far, consuming the commands the cull dispatch emitted
    /// earlier in this command buffer. Never called for Blend (CPU-sorted path).
    /// Opaque draws its full-res partition (no-`discard` pipeline) first, then
    /// the coarse-LOD partition (slab-clip pipeline) — near before far, so the
    /// LOD skirt behind full-res terrain is mostly depth-rejected. Each group's
    /// draws close a GPU timestamp (`OpaqueFull` / `OpaqueLod` / `Cutout`).
    unsafe fn record_mesh_indirect_count(&self, pass: Pass) {
        match pass {
            Pass::Opaque => {
                let (full, lod) = self.r.pipelines.opaque_pipelines(self.plan.mesh_lean);
                let (caged, caged_lod) = self.r.pipelines.caged_opaque(self.plan.mesh_lean);
                // Flat then caged, full-res before LOD. Each pair shares one
                // timestamp: caged calls add into the flat group's gauge.
                unsafe {
                    self.record_groups(&[(cull::Group::Opaque, full), (cull::Group::Caged, caged)]);
                    self.record_groups(&[
                        (cull::Group::OpaqueLod, lod),
                        (cull::Group::CagedLod, caged_lod),
                    ]);
                }
            }
            Pass::Cutout => unsafe {
                self.record_groups(&[(cull::Group::Cutout, self.r.mesh_pipeline_for(pass))]);
            },
            Pass::Blend => unreachable!("Blend stays on the CPU path"),
        }
    }

    /// Draws every non-empty arena partition of each `(group, pipeline)`.
    /// One pipeline-stat query and one GPU timestamp, both named by the first
    /// group, so a caged group does not open a second query. An empty group
    /// binds nothing. [`GpuTimer::mark`] is a no-op when nothing was recorded.
    unsafe fn record_groups(&self, groups: &[(cull::Group, vk::Pipeline)]) {
        let first = groups[0].0;
        self.pipe_begin_group(first);
        let mut calls = 0u32;
        if let Some(frame) = &self.r.cull_frame {
            let mut drew = false;
            let mut expected = 0u32;
            for &(group, pipeline) in groups {
                expected += cull::group_indirect_calls(&frame.partitions, group, frame.arena_count);
                let span = frame.arena_count * cull::BUCKETS;
                let base = group as usize * span;
                if frame.partitions[base..base + span]
                    .iter()
                    .all(|p| p.capacity == 0)
                {
                    continue;
                }
                if !drew {
                    self.r.gpu_timer.recorded(self.slot);
                    unsafe { self.bind_mesh3d_state() };
                    drew = true;
                }
                let device = &self.r.device.device;
                unsafe {
                    device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    for arena in 0..frame.arena_count {
                        let part0 = cull::camera_part(group as usize, arena, 0, frame.arena_count);
                        if frame.partitions[part0..part0 + cull::BUCKETS]
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
                            let idx = part0 + bucket;
                            let part = frame.partitions[idx];
                            // Bucket trimming already zeros unreachable buckets
                            // (and empty live lanes); skip the empty call.
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
                            calls += 1;
                        }
                    }
                }
            }
            debug_assert_eq!(calls, expected);
        }
        match first {
            cull::Group::Opaque | cull::Group::Caged => {
                crate::profile::gauge(crate::profile::Gauge::CallsFull, u64::from(calls));
            }
            cull::Group::OpaqueLod | cull::Group::CagedLod => {
                crate::profile::gauge(crate::profile::Gauge::CallsLod, u64::from(calls));
            }
            cull::Group::Cutout => {}
        }
        self.stamp_group(first);
        self.pipe_end_group(first);
    }

    /// GPU timestamp closing `group`'s draws. No-op when profiling is off or
    /// the group recorded nothing.
    fn stamp_group(&self, group: cull::Group) {
        if !crate::profile::is_enabled() {
            return;
        }
        let pass = match group {
            cull::Group::Opaque | cull::Group::Caged => GpuPass::OpaqueFull,
            cull::Group::Cutout => GpuPass::Cutout,
            cull::Group::OpaqueLod | cull::Group::CagedLod => GpuPass::OpaqueLod,
        };
        unsafe {
            self.r
                .gpu_timer
                .mark(&self.r.device.device, self.cmd, self.slot, pass);
        }
    }

    fn pipe_stat_pass(group: cull::Group) -> PipeStatPass {
        match group {
            cull::Group::Opaque | cull::Group::Caged => PipeStatPass::OpaqueFull,
            cull::Group::OpaqueLod | cull::Group::CagedLod => PipeStatPass::OpaqueLod,
            cull::Group::Cutout => PipeStatPass::Cutout,
        }
    }

    fn pipe_begin_group(&self, group: cull::Group) {
        if !crate::profile::is_enabled() {
            return;
        }
        unsafe {
            self.r.pipe_stats.begin_pass(
                &self.r.device.device,
                self.cmd,
                self.slot,
                Self::pipe_stat_pass(group),
            );
        }
    }

    fn pipe_end_group(&self, group: cull::Group) {
        if !crate::profile::is_enabled() {
            return;
        }
        unsafe {
            self.r.pipe_stats.end_pass(
                &self.r.device.device,
                self.cmd,
                self.slot,
                Self::pipe_stat_pass(group),
            );
        }
    }

    /// Pushes `view_proj` to `layout_debug` for the immediate debug geometry.
    /// Done per debug pass because the mesh passes bind `layout_3d`, whose
    /// incompatible push-constant range disturbs this value.
    unsafe fn push_debug_view_proj(&self) {
        let view_proj = self.scene_state.expect("a debug pass implies a 3D scene").0;
        let push = pipeline::DebugPush {
            // Match the mesh pass jitter so debug geometry doesn't shimmer against
            // jittered terrain under TAA.
            view_proj,
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
                self.r.gpu_timer.recorded(self.slot);
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
                self.r.gpu_timer.recorded(self.slot);
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
                self.r.gpu_timer.recorded(self.slot);
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
    /// FrameUniforms at set 0 binding 1, cloud LUT at binding 0, far-body table
    /// at binding 2, datum buffer at binding 3, albedo cubes at binding 4, no
    /// vertex buffer). Depth is the reversed-Z far plane; the read-only test
    /// rejects it wherever terrain wrote closer depth, so it shades only
    /// background pixels. Skipped unless the frame set a sky palette.
    ///
    /// With a real per-tile mask, body-free tiles draw as instanced quads on
    /// the `SKY_BASE` pipeline. Tiles neither disc can touch, and with stars
    /// off, are a prefix of that run and use the 2×2 pipeline when it exists.
    /// Sphere-only tiles use the sphere variant. A heavy tile whose mask is
    /// exactly one Mapped body uses the loop-free mapsolo fragment; its
    /// interior is a prefix of that run and uses the 2×2 mapsolo pipeline
    /// when it exists. Every other heavy tile uses the no-mapped variant, or
    /// the full variant when the frame kept a mapped body. With mapsolo off,
    /// mapped-interior tiles are a prefix of the whole heavy run and use the
    /// full fragment at 2×2. Edge tiles stay at full rate. Both rates use the
    /// fixed-point mapped march. A heavy tile whose mask is exactly one
    /// Rounded body uses the loop-free roundsolo fragment at 1×1 (the last
    /// run). No tile classification keeps one fullscreen triangle on that
    /// same choice. No bodies use the base pipeline. The fullscreen triangle
    /// never binds mapsolo or roundsolo: it has no per-tile mask.
    /// The runs, their order and their first tiles come from
    /// [`SkyDraw::runs`](super::far_bodies::SkyDraw::runs).
    pub(super) unsafe fn record_sky(&self) {
        let Some(desc) = self.lists.sky_for_pass() else {
            return;
        };
        self.r.gpu_timer.recorded(self.slot);
        let device = &self.r.device.device;
        let cmd = self.cmd;
        // Same jitter the mesh pass applies, so TAA sees a coherently jittered
        // frame (sky vs terrain silhouettes) and history reprojection is stable.
        let jittered = self.scene_state.expect("a sky pass implies a 3D scene").0;
        let params = pipeline::SkyParams::compose(jittered.inverse(), &desc);
        let draw = self.r.far_ring.sky_draw(FrameSlot::new(self.slot));
        let sky = &self.r.pipelines.sky;
        // The coarse switch is the one the CPU split read this frame. No runs
        // means the fullscreen triangle.
        let runs = draw.runs(sky.has_coarse());
        let first = match runs.iter().flatten().next() {
            Some(run) => sky.tile(run.frag, run.rate),
            None => sky.fullscreen(draw.fullscreen_frag()),
        };
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, first);
            self.invalidate_mesh_desc();
            device.cmd_push_constants(
                cmd,
                sky.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&params),
            );
            let lut = &self.r.targets.sky_cloud[self.slot];
            let lut_infos = [vk::DescriptorImageInfo::default()
                .sampler(sky.lut_sampler)
                .image_view(lut.view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let ubo = self.r.ubo_ring.buffer(FrameSlot::new(self.slot));
            let ubo_infos = [vk::DescriptorBufferInfo::default()
                .buffer(ubo)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let far = self.r.far_ring.buffer(FrameSlot::new(self.slot));
            let far_infos = [vk::DescriptorBufferInfo::default()
                .buffer(far)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let datum = self.r.far_maps.buffer();
            let datum_infos = [vk::DescriptorBufferInfo::default()
                .buffer(datum)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let cube_sampler = self.r.far_maps.sampler();
            let cube_info = |id: usize| {
                vk::DescriptorImageInfo::default()
                    .sampler(cube_sampler)
                    .image_view(self.r.far_maps.view(id))
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            };
            let cube_infos = [
                cube_info(0),
                cube_info(1),
                cube_info(2),
                cube_info(3),
                cube_info(4),
                cube_info(5),
                cube_info(6),
                cube_info(7),
            ];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&lut_infos),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&ubo_infos),
                vk::WriteDescriptorSet::default()
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&far_infos),
                vk::WriteDescriptorSet::default()
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&datum_infos),
                vk::WriteDescriptorSet::default()
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&cube_infos),
            ];
            self.r.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                sky.layout,
                0,
                &writes,
            );
            if runs.iter().any(Option::is_some) {
                // firstInstance is the run start; the vertex shader's
                // InstanceIndex includes it, so each run draws the tiles the
                // CPU partition wrote there. A pipeline is rebound only when
                // the run's pipeline differs from the bound one.
                let mut bound = first;
                for run in runs.iter().flatten() {
                    let pipe = sky.tile(run.frag, run.rate);
                    if pipe != bound {
                        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipe);
                        bound = pipe;
                    }
                    device.cmd_draw(cmd, 6, run.count, 0, run.first);
                }
            } else {
                device.cmd_draw(cmd, 3, 1, 0, 0);
            }
        }
    }

    /// Ends dynamic rendering and records the post-scene barrier. The timeline
    /// orders later submits; this barrier owns layout and visibility.
    ///
    /// [`HdrFinalize::Pass`]: no later pass writes the offscreen, so this
    /// barrier moves it to `SHADER_READ_ONLY_OPTIMAL` for tonemapping and
    /// the [`HdrReadable`] proof is returned. Otherwise there is no offscreen
    /// transition and no proof: exposure metering writes the offscreen after
    /// this and owns the finalization (its barrier would otherwise race the
    /// write), or the frame is not presented. Sampleable depth (or,
    /// classify-only under MSAA, the MS depth) rests here when a later pass
    /// samples it (see [`super::SAMPLEABLE_DEPTH_REST_LAYOUT`]); a classifying
    /// frame joins the rate and history images into the same barrier.
    pub(super) unsafe fn end(mut self) -> Option<HdrReadable> {
        let transition_offscreen = self.plan.hdr == HdrFinalize::Pass;
        let classify_vrs = self.plan.vrs.classifies();
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            device.cmd_end_rendering(cmd);
            self.r.gpu_timer.recorded(self.slot);

            self.ended = true;

            // Mix fill (profiling only) must precede the barrier that makes it
            // visible to the classifier. Joins as a CLEAR→COMPUTE memory barrier.
            let mix_filled = classify_vrs && self.r.record_vrs_mix_fill(cmd, self.slot);

            // One vkCmdPipelineBarrier2: offscreen (optional) + depth rest
            // (sampleable, or the MS depth for a classify-only frame; none
            // when nothing samples) + (if classifying) rate/history → GENERAL.
            // No extra VRS pipeline barrier beyond the two the frame already has.
            let mut images = [vk::ImageMemoryBarrier2::default(); 4];
            let mut n = 0;
            if transition_offscreen {
                // Offscreen: src COLOR_ATTACHMENT_OUTPUT / COLOR_ATTACHMENT_WRITE
                // (scene color, including the MSAA average resolve). Dst
                // FRAGMENT_SHADER / SHADER_SAMPLED_READ (tonemap present-copy;
                // bloom/exposure insert their own barriers when they run).
                // Old COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL.
                images[n] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.offscreen_image)
                    .subresource_range(color_range());
                n += 1;
            }
            // Depth rest: see `sampleable_depth_rest_barrier` and
            // `ms_depth_classify_barrier`. Skipped when nothing samples; the
            // next begin is UNDEFINED.
            match self.plan.depth_use {
                SceneDepthUse::Sampleable => {
                    images[n] = self.r.sampleable_depth_rest_barrier(self.slot);
                    n += 1;
                }
                SceneDepthUse::ClassifyMs => {
                    images[n] = self.r.ms_depth_classify_barrier(self.slot);
                    n += 1;
                }
                SceneDepthUse::Discard => {}
            }
            if classify_vrs {
                images[n] = self.r.vrs_rate_to_general_barrier(self.slot);
                n += 1;
                images[n] = self.r.vrs_history_to_general_barrier(self.slot);
                n += 1;
            }

            let mix_mem = [super::Renderer::vrs_mix_fill_memory_barrier()];
            let mem: &[vk::MemoryBarrier2] = if mix_filled { &mix_mem } else { &[] };
            if n > 0 || mix_filled {
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default()
                        .memory_barriers(mem)
                        .image_memory_barriers(&images[..n]),
                );
            }
        }
        transition_offscreen.then(|| HdrReadable::new(self.slot))
    }
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        debug_assert!(self.ended, "RenderPass dropped without calling end()");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use SceneDepthUse::{ClassifyMs, Discard, Sampleable};

    const FRAGMENT: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::FRAGMENT_SHADER;
    const COMPUTE: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::COMPUTE_SHADER;

    /// Every combination of the 14 inputs.
    fn all_inputs() -> impl Iterator<Item = SceneInputs> {
        (0u32..1 << 14).map(|bits| {
            let b = |i: u32| bits & (1 << i) != 0;
            SceneInputs {
                scene: b(0),
                present: b(1),
                sky: b(2),
                vrs: b(3),
                classified: b(4),
                classify_ms: b(5),
                taa: b(6),
                exposure: b(7),
                spill_live: b(8),
                absorb: b(9),
                mesh_lean: b(10),
                prev_depth_valid: b(11),
                depth_sampled: b(12),
                ms_depth_read: b(13),
            }
        })
    }

    /// What `record_render` passed to `RenderPass::begin` (and branched on
    /// at the end of the pass) before the plan, from the same inputs.
    struct Old {
        do_vrs: bool,
        classify_vrs: bool,
        depth_use: SceneDepthUse,
        store_color: bool,
        absorb_this_frame: bool,
        depth_sampled: bool,
        ms_depth_read: bool,
        mesh_lean: bool,
        sky_drawn: bool,
        run_exposure: bool,
        will_present: bool,
    }

    fn old(i: SceneInputs) -> Old {
        let vrs_on = i.scene && i.vrs;
        Old {
            do_vrs: vrs_on && i.classified,
            classify_vrs: vrs_on,
            depth_use: scene_depth_use(
                i.classify_ms,
                i.present,
                i.taa,
                i.spill_live,
                vrs_on,
                i.absorb,
            ),
            store_color: i.present,
            absorb_this_frame: i.absorb,
            depth_sampled: i.depth_sampled,
            ms_depth_read: i.ms_depth_read,
            mesh_lean: i.mesh_lean,
            sky_drawn: i.sky,
            run_exposure: i.scene && i.exposure && i.present,
            will_present: i.present,
        }
    }

    #[test]
    fn scene_plan_matches_the_booleans_it_replaced() {
        for i in all_inputs() {
            let plan = scene_plan(i);
            let old = old(i);
            assert_eq!(plan.vrs.binds_rate(), old.do_vrs, "{i:?}");
            assert_eq!(plan.vrs.classifies(), old.classify_vrs, "{i:?}");
            assert_eq!(plan.depth_use, old.depth_use, "{i:?}");
            assert_eq!(plan.present, old.store_color, "{i:?}");
            assert_eq!(plan.absorb, old.absorb_this_frame, "{i:?}");
            assert_eq!(plan.mesh_lean, old.mesh_lean, "{i:?}");
            assert_eq!(plan.sky, old.sky_drawn, "{i:?}");
            assert_eq!(plan.prev_depth_valid, i.prev_depth_valid, "{i:?}");
            assert_eq!(plan.spill_live, i.spill_live, "{i:?}");
            // The begin barrier's depth source, as `begin` built it.
            let mut depth_src = vk::PipelineStageFlags2::NONE;
            if old.depth_sampled {
                depth_src |= FRAGMENT;
            }
            if old.ms_depth_read {
                depth_src |= COMPUTE;
            }
            assert_eq!(plan.depth_src, depth_src, "{i:?}");
            // The end branches: exposure, else presenting, else unpresented.
            let hdr = if old.run_exposure {
                HdrFinalize::Exposure
            } else if old.will_present {
                HdrFinalize::Pass
            } else {
                HdrFinalize::Skip
            };
            assert_eq!(plan.hdr, hdr, "{i:?}");
        }
    }

    #[test]
    fn scene_plan_combinations_stay_consistent() {
        for i in all_inputs() {
            let plan = scene_plan(i);
            // VRS needs a 3D scene, and a bound rate image is always
            // returned to GENERAL by a classify at the end of the pass.
            assert!(plan.vrs == SceneVrs::Off || i.scene, "{i:?}");
            assert!(!plan.vrs.binds_rate() || plan.vrs.classifies(), "{i:?}");
            // The MS depth is stored only for the sample-0 classifier.
            if plan.depth_use == ClassifyMs {
                assert!(plan.vrs.classifies() && i.classify_ms, "{i:?}");
                assert!(!i.absorb && !(i.present && (i.taa || i.spill_live)));
            }
            // Water absorption stores the sampleable depth for the next frame.
            if i.absorb {
                assert_eq!(plan.depth_use, Sampleable, "{i:?}");
            }
            // Only presented frames finalize the HDR; exposure only in 3D.
            assert_eq!(plan.hdr == HdrFinalize::Skip, !plan.present, "{i:?}");
            if plan.hdr == HdrFinalize::Exposure {
                assert!(i.scene && i.exposure, "{i:?}");
            }
            // The depth source is NONE or the two read stages, nothing else.
            assert!((FRAGMENT | COMPUTE).contains(plan.depth_src), "{i:?}");
        }
    }

    #[test]
    fn scene_plan_cases() {
        let first = SceneInputs {
            scene: true,
            present: true,
            vrs: true,
            ..SceneInputs::default()
        };
        // First VRS frame of a slot classifies without binding a rate image.
        assert_eq!(scene_plan(first).vrs, SceneVrs::Prime);
        let next = SceneInputs {
            classified: true,
            ..first
        };
        assert_eq!(scene_plan(next).vrs, SceneVrs::Bound);
        // No 3D scene: no VRS, and the pass finalizes even with exposure on.
        let flat = SceneInputs {
            scene: false,
            exposure: true,
            ..next
        };
        let plan = scene_plan(flat);
        assert_eq!(
            (plan.vrs, plan.hdr, plan.depth_use),
            (SceneVrs::Off, HdrFinalize::Pass, Discard)
        );
        // Unpresented MSAA frame whose only depth reader is the classifier,
        // after a frame that ended the same way.
        let ms = SceneInputs {
            present: false,
            classify_ms: true,
            ms_depth_read: true,
            ..next
        };
        let plan = scene_plan(ms);
        assert_eq!(
            (plan.depth_use, plan.hdr, plan.depth_src),
            (ClassifyMs, HdrFinalize::Skip, COMPUTE)
        );
        // Presented with exposure: metering owns the HDR finalize.
        let lit = SceneInputs {
            exposure: true,
            depth_sampled: true,
            ..next
        };
        let plan = scene_plan(lit);
        assert_eq!(
            (plan.hdr, plan.depth_src),
            (HdrFinalize::Exposure, FRAGMENT)
        );
    }
}
