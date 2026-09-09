//! Present path: overlay, swapchain copy, and screenshot readback.
//! Split out of `mod.rs` so later work can touch presentation without opening
//! the frame loop or scene recorder.

use ash::vk;

use crate::skeleton::FrameSlot;

use super::alloc;
use super::image_upload;
use super::render_client::Capture;
use super::taa::{
    TONEMAP_TAA_DEPTH_BINDING, TONEMAP_TAA_HDR_BINDING, TONEMAP_TAA_HISTORY_BINDING,
    TONEMAP_TAA_SPILL_BINDING, TaaPresent,
};
use super::timeline::{RenderCompletion, queue_present};
use super::{Env, Renderer, SAMPLEABLE_DEPTH_REST_LAYOUT, color_range};

/// Witness that HDR image is ready for present.
#[must_use = "the offscreen HDR must be finalized to SHADER_READ before present"]
pub(crate) struct HdrReadable {
    slot: usize,
}

impl HdrReadable {
    /// Mint the witness. Only the HDR finalizers may create this:
    /// `frame_loop` (`transition_offscreen_to_sampled` and the mint sites
    /// next to it), `scene_pass::RenderPass::end_sampled`, and
    /// `exposure::record_exposure_pass`.
    pub(in crate::vk) fn new(slot: usize) -> Self {
        HdrReadable { slot }
    }
}

/// 2D overlay draw parameters for present pass.
#[derive(Clone, Copy)]
pub(super) struct OverlayPresent {
    pub(super) d2_offset: u64,
    pub(super) d2_count: u32,
    pub(super) d2_tex_offset: u64,
    pub(super) d2_tex_count: u32,
}

/// Host-visible buffer for screenshot readback.
struct Readback {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl Renderer {
    /// Copies the finished frame into the acquired swapchain image (when one
    /// was acquired in [`Self::decide_present`]) and queues the present. The
    /// [`HdrReadable`] proof is REQUIRED (not merely passed): the present copy's
    /// tonemap samples the offscreen, so this signature makes it impossible to
    /// present a frame whose offscreen was never finalized to
    /// `SHADER_READ_ONLY_OPTIMAL`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn present(
        &mut self,
        slot: usize,
        present_target: Option<u32>,
        warp_map: crate::camera::WarpMap,
        overlay: OverlayPresent,
        hdr_readable: HdrReadable,
        spill_live: bool,
        taa: Option<TaaPresent>,
    ) {
        // The proof must be for the slot we are about to sample.
        debug_assert_eq!(hdr_readable.slot, slot, "HdrReadable slot mismatch");
        if let Some(image_index) = present_target {
            unsafe {
                self.submit_present_copy(slot, image_index, warp_map, overlay, spill_live, taa)
            };
            self.last_present = std::time::Instant::now();
        }
    }

    /// Draws the 2D overlay (text atlas + minimap) into the currently-bound
    /// swapchain attachment, using the present-format pipeline variants. Mirrors
    /// `RenderPass::record_2d` but for the post-tonemap pass; the caller has set a
    /// negative-height viewport so `tris2d.vert`'s pixel→NDC mapping is correct.
    ///
    /// `fused_overlay` selects the two-attachment overlay pipelines (empty
    /// history write mask). That is only legal with `independentBlend`; the
    /// caller must pass false and use a one-attachment rendering otherwise.
    unsafe fn record_overlay_present(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        overlay: OverlayPresent,
        extent: vk::Extent2D,
        fused_overlay: bool,
    ) {
        let device = &self.device.device;
        let pixels_to_ndc = [2.0 / extent.width as f32, 2.0 / extent.height as f32];
        let tris2d = if fused_overlay {
            self.pipelines
                .tris2d_present_taa
                .expect("two-attachment overlay pipelines exist when independentBlend is enabled")
        } else {
            self.pipelines.tris2d_present
        };
        let tris2d_tex = if fused_overlay {
            self.pipelines
                .tris2d_tex_present_taa
                .expect("two-attachment overlay pipelines exist when independentBlend is enabled")
        } else {
            self.pipelines.tris2d_tex_present
        };
        unsafe {
            if overlay.d2_count > 0 {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, tris2d);
                self.atlas.push_descriptor(
                    &self.device.push_descriptor,
                    cmd,
                    self.pipelines.layout_2d,
                    0,
                );
                device.cmd_push_constants(
                    cmd,
                    self.pipelines.layout_2d,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&pixels_to_ndc),
                );
                let imm = self.slots[FrameSlot::new(slot)]
                    .imm
                    .bound()
                    .expect("d2_count > 0 implies the immediate buffer is allocated");
                device.cmd_bind_vertex_buffers(cmd, 0, &[imm], &[overlay.d2_offset]);
                device.cmd_draw(cmd, overlay.d2_count, 1, 0, 0);
            }

            if self.minimap.ready() && overlay.d2_tex_count > 0 {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, tris2d_tex);
                self.minimap.push_descriptor(
                    &self.device.push_descriptor,
                    cmd,
                    self.pipelines.layout_2d,
                    slot,
                );
                device.cmd_push_constants(
                    cmd,
                    self.pipelines.layout_2d,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&pixels_to_ndc),
                );
                let imm = self.slots[FrameSlot::new(slot)]
                    .imm
                    .bound()
                    .expect("d2_tex_count > 0 implies the immediate buffer is allocated");
                device.cmd_bind_vertex_buffers(cmd, 0, &[imm], &[overlay.d2_tex_offset]);
                device.cmd_draw(cmd, overlay.d2_tex_count, 1, 0, 0);
            }
        }
    }

    /// Records and submits the offscreen[slot] -> swapchain copy, then
    /// queues the present. Caller guarantees the previous copy has retired
    /// (its value reached) and the image was just acquired with
    /// `slots[slot].image_available`.
    unsafe fn submit_present_copy(
        &mut self,
        slot: usize,
        image_index: u32,
        warp_map: crate::camera::WarpMap,
        overlay: OverlayPresent,
        spill_live: bool,
        taa: Option<TaaPresent>,
    ) {
        // A pending capture piggybacks on this copy: after the tonemap draw,
        // the swapchain image is read back into `readback` instead of going
        // straight to PRESENT. The quarter-res spill is an intermediate the
        // tonemap samples — never a capture source. Allocate the host buffer
        // before borrowing `device` so the read-back path adds no &mut-self
        // conflicts below.
        let capture = self.pending_capture.take();
        let extent = self.swapchain.extent;
        let readback = capture.as_ref().map(|_| unsafe {
            self.create_readback((extent.width as u64) * (extent.height as u64) * 4)
        });

        let swap_image = self.swapchain.images[image_index as usize];
        let swap_view = self.swapchain.image_views[image_index as usize];
        // Exposure applied before the tonemap curve. Render scale is handled by the
        // tonemap sampler (it reads the HDR image bilinearly at window size), so
        // there is no separate copy/blit path anymore. The same pass also applies
        // the wide-FOV periphery remap: `warp_map` carries the coefficients, and an
        // identity (rectilinear) map pushes `s = 0` so the frag stays a no-op.
        // Metering off pins exposure at 1.0 structurally (the set_flags reset
        // already published DEFAULT; this makes the pin independent of
        // transition ordering).
        let exposure = if self.flags.exposure {
            self.exposure.current().0
        } else {
            crate::skeleton::Exposure::DEFAULT.0
        };
        let vignette = if self.flags.vignette { 1.0 } else { 0.0 };
        let tonemap_push = warp_map.push(exposure, vignette);
        // TAA-off: one-attachment pipeline, history omitted. TAA-on: two
        // attachments (swapchain + write-history). Overlay joins that
        // rendering only when independentBlend is enabled (attachment 1
        // write-mask empty so the HUD never lands in history). Without it,
        // overlay is a second one-attachment rendering after a
        // COLOR_ATTACHMENT_OUTPUT write→read|write barrier.
        let taa_fused = taa.is_some();
        let independent_blend = self.device.independent_blend;
        let taa_push = taa.as_ref().map(|t| self.tonemap_taa_push(t, tonemap_push));
        let hist_write_view = taa_fused.then(|| self.taa.write_view());
        let hist_read_view = taa_fused.then(|| self.taa.read_view());
        let depth_view = taa_fused.then(|| self.targets.sampleable_depth(slot).view());
        let (hist_pre_write, hist_pre_read) = if taa_fused {
            let (w, r) = self.taa.history_pre_barriers();
            (Some(w), r)
        } else {
            (None, None)
        };
        let hdr_view = self.targets.offscreen[slot].view();

        let device = &self.device.device;
        unsafe {
            device
                .reset_command_buffer(self.copy_cmd, vk::CommandBufferResetFlags::empty())
                .expect("command buffer reset failed");
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.copy_cmd, &begin)
                .expect("begin command buffer failed");
            // Time the copy on its own pair: read the previous copy (retired —
            // `decide_present` only acquires once it has) before resetting.
            // TAA resolve is fused into this span (no separate GpuTaa stamp).
            let profiling = crate::profile::is_enabled();
            if profiling {
                if let Some(ms) = self.gpu_timer.read_copy(device) {
                    crate::profile::add_ms(crate::profile::Meter::GpuTonemap, ms);
                }
                self.gpu_timer.begin_copy(device, self.copy_cmd);
            }

            // One barrier before the pass: swapchain UNDEFINED→COLOR, and when
            // TAA is on write-history → COLOR plus first-present read-history
            // UNDEFINED→SHADER_READ. Depth already rests in SHADER_READ_ONLY
            // (SAMPLEABLE_DEPTH_REST_LAYOUT); the render→present timeline wait
            // makes it visible to this fragment shader.
            let swap_to_color = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(swap_image)
                .subresource_range(color_range());
            let mut pre = [vk::ImageMemoryBarrier2::default(); 3];
            pre[0] = swap_to_color;
            let mut pre_n = 1;
            if let Some(w) = hist_pre_write {
                pre[pre_n] = w;
                pre_n += 1;
            }
            if let Some(r) = hist_pre_read {
                pre[pre_n] = r;
                pre_n += 1;
            }
            device.cmd_pipeline_barrier2(
                self.copy_cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&pre[..pre_n]),
            );

            let swap_att = vk::RenderingAttachmentInfo::default()
                .image_view(swap_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);
            let hist_att = hist_write_view.map(|view| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE)
            });
            let color_1 = [swap_att];
            let color_2 = [swap_att, hist_att.unwrap_or_default()];
            let color_attachments: &[vk::RenderingAttachmentInfo] =
                if taa_fused { &color_2 } else { &color_1 };
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(color_attachments);
            device.cmd_begin_rendering(self.copy_cmd, &rendering_info);

            // Standard (positive-height) viewport: the fullscreen triangle's uv
            // maps top→top, matching the offscreen's stored orientation.
            device.cmd_set_viewport(
                self.copy_cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                self.copy_cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );

            let spill_view = if spill_live {
                self.targets.spill[slot].view()
            } else {
                self.bloom.black_view()
            };
            if let (Some(push), Some(hist_view), Some(depth_view)) =
                (taa_push.as_ref(), hist_read_view, depth_view)
            {
                device.cmd_bind_pipeline(
                    self.copy_cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.tonemap_taa,
                );
                let hdr_info = [vk::DescriptorImageInfo::default()
                    .sampler(self.pipelines.tonemap_sampler)
                    .image_view(hdr_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let spill_info = [vk::DescriptorImageInfo::default()
                    .sampler(self.pipelines.tonemap_sampler)
                    .image_view(spill_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let hist_info = [vk::DescriptorImageInfo::default()
                    .sampler(self.pipelines.tonemap_sampler)
                    .image_view(hist_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let depth_info = [vk::DescriptorImageInfo::default()
                    .sampler(self.pipelines.tonemap_depth_sampler)
                    .image_view(depth_view)
                    .image_layout(SAMPLEABLE_DEPTH_REST_LAYOUT)];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_binding(TONEMAP_TAA_HDR_BINDING)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&hdr_info),
                    vk::WriteDescriptorSet::default()
                        .dst_binding(TONEMAP_TAA_SPILL_BINDING)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&spill_info),
                    vk::WriteDescriptorSet::default()
                        .dst_binding(TONEMAP_TAA_HISTORY_BINDING)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&hist_info),
                    vk::WriteDescriptorSet::default()
                        .dst_binding(TONEMAP_TAA_DEPTH_BINDING)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&depth_info),
                ];
                self.device.push_descriptor.cmd_push_descriptor_set(
                    self.copy_cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.layout_tonemap_taa,
                    0,
                    &writes,
                );
                device.cmd_push_constants(
                    self.copy_cmd,
                    self.pipelines.layout_tonemap_taa,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(push),
                );
            } else {
                device.cmd_bind_pipeline(
                    self.copy_cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.tonemap,
                );
                image_upload::push_combined_image_sampler(
                    &self.device.push_descriptor,
                    self.copy_cmd,
                    self.pipelines.layout_tonemap,
                    0,
                    self.pipelines.tonemap_sampler,
                    hdr_view,
                );
                // Binding 1: quarter-res spill (bloom composite + godrays), built
                // in the render submit and made visible here by the render→present
                // semaphore. When bloom and godrays are both off the spill dispatch
                // is skipped and this is a 1×1 black image (tonemap stays a single
                // HDR fetch plus a cached 1×1 add of zero).
                let spill_info = [vk::DescriptorImageInfo::default()
                    .sampler(self.pipelines.tonemap_sampler)
                    .image_view(spill_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let post_writes = [vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&spill_info)];
                self.device.push_descriptor.cmd_push_descriptor_set(
                    self.copy_cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.layout_tonemap,
                    0,
                    &post_writes,
                );
                device.cmd_push_constants(
                    self.copy_cmd,
                    self.pipelines.layout_tonemap,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&tonemap_push),
                );
            }
            device.cmd_draw(self.copy_cmd, 3, 1, 0, 0);
            // Composite the 2D overlay onto the tonemapped swapchain (never drawn in
            // the offscreen scene pass). Post-tonemap on BOTH paths: wide-FOV so the
            // warp never bends the HUD, rectilinear so the TAA resolve never reprojects
            // it. Uses a GL-style negative-height viewport, matching tris2d.vert.
            //
            // independentBlend: overlay stays in the fused two-attachment scope.
            // Without it: end that scope after the tonemap triangle, then a second
            // one-attachment rendering (LOAD) using the single-attachment overlay
            // pipelines. Consecutive dynamic-rendering instances that write the
            // same colour attachment are NOT ordered by an implicit
            // COLOR_ATTACHMENT_OUTPUT WAW (unlike render-pass subpasses); blend
            // LOAD also reads the attachment, so this is a write→read|write
            // barrier on the swapchain. Same layout, no extra image transition.
            let fused_overlay = taa_fused && independent_blend;
            if taa_fused && !independent_blend {
                device.cmd_end_rendering(self.copy_cmd);
                let overlay_sync = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(
                        vk::AccessFlags2::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    )
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(swap_image)
                    .subresource_range(color_range())];
                device.cmd_pipeline_barrier2(
                    self.copy_cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&overlay_sync),
                );
                let overlay_att = vk::RenderingAttachmentInfo::default()
                    .image_view(swap_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);
                let overlay_color = [overlay_att];
                let overlay_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    })
                    .layer_count(1)
                    .color_attachments(&overlay_color);
                device.cmd_begin_rendering(self.copy_cmd, &overlay_info);
            }
            device.cmd_set_viewport(
                self.copy_cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: extent.height as f32,
                    width: extent.width as f32,
                    height: -(extent.height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.record_overlay_present(self.copy_cmd, slot, overlay, extent, fused_overlay);
            device.cmd_end_rendering(self.copy_cmd);

            // Publish write-history to SHADER_READ (next present's read). Folded
            // into the first after-pass swapchain barrier.
            let hist_post = taa_fused.then(|| self.taa.history_post_barrier());

            // When capturing, detour through TRANSFER_SRC to copy the finished
            // image into the host buffer, then continue to PRESENT.
            if let Some(rb) = &readback {
                let mut to_src = [vk::ImageMemoryBarrier2::default(); 2];
                to_src[0] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(swap_image)
                    .subresource_range(color_range());
                let mut n = 1;
                if let Some(h) = hist_post {
                    to_src[n] = h;
                    n += 1;
                }
                device.cmd_pipeline_barrier2(
                    self.copy_cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_src[..n]),
                );
                let region = [vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })];
                device.cmd_copy_image_to_buffer(
                    self.copy_cmd,
                    swap_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    rb.buffer,
                    &region,
                );
            }

            let (old_layout, src_stage, src_access) = if readback.is_some() {
                (
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                )
            } else {
                (
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                )
            };
            let swap_to_present = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(swap_image)
                .subresource_range(color_range());
            // Capture path already published history in the TRANSFER_SRC barrier.
            // The non-capture path folds history into this PRESENT barrier.
            let mut to_present = [swap_to_present, vk::ImageMemoryBarrier2::default()];
            let present_n = if readback.is_none() {
                if let Some(h) = hist_post {
                    to_present[1] = h;
                    2
                } else {
                    1
                }
            } else {
                1
            };
            device.cmd_pipeline_barrier2(
                self.copy_cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_present[..present_n]),
            );
            if profiling {
                self.gpu_timer.end_copy(device, self.copy_cmd);
            }
            crate::profile::count(crate::profile::Counter::Presented);
            device
                .end_command_buffer(self.copy_cmd)
                .expect("end command buffer failed");

            // Wait for acquire + render, then signal present semaphore.
            let value = self.timeline.begin_copy(self.copy_cmd).submit(
                device,
                self.device.graphics_queue,
                &self.timeline,
                self.slots[FrameSlot::new(slot)].image_available,
                RenderCompletion::from_value(self.slots[FrameSlot::new(slot)].render_value),
                self.present_semaphores[image_index as usize],
            );
            self.slots[FrameSlot::new(slot)].copy_value = value;
            self.last_copy_value = value;
            self.track_copy(slot);
            if let Some(t) = taa {
                self.taa.finish_present(t.view_proj, t.eye);
            }

            match queue_present(
                &self.swapchain.loader,
                self.device.present_queue,
                self.present_semaphores[image_index as usize],
                self.swapchain.swapchain,
                image_index,
            ) {
                Ok(sub) => {
                    if sub {
                        self.recreate_if_stale();
                    }
                }
                // OUT_OF_DATE/SURFACE_LOST: recreate next frame. Other errors: fatal.
                Err(err) => match Env::classify(err) {
                    Some(Env::OutOfDate | Env::SurfaceLost) => self.needs_recreate = true,
                    _ => panic!("queue_present failed: {err:?}"),
                },
            }
        }

        // Copy (part of the just-submitted `copy_cmd`) is covered by
        // `last_copy_value`; wait it out, then read the host buffer.
        if let (Some(capture), Some(rb)) = (capture, readback) {
            unsafe { self.finish_screenshot(rb, extent, capture) };
        }
    }

    /// Allocates a host-visible, host-coherent buffer for one frame's readback.
    unsafe fn create_readback(&self, size: vk::DeviceSize) -> Readback {
        let device = &self.device.device;
        unsafe {
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .expect("Failed to create screenshot readback buffer");
            let req = device.get_buffer_memory_requirements(buffer);
            let mem_props = self
                .instance
                .instance
                .get_physical_device_memory_properties(self.device.physical);
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(alloc::find_memory_type(
                            &mem_props,
                            req.memory_type_bits,
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )),
                    None,
                )
                .expect("Failed to allocate screenshot readback memory");
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("Failed to bind screenshot readback memory");
            Readback {
                buffer,
                memory,
                size,
            }
        }
    }

    /// Waits for the readback copy to complete, copies the pixels off the GPU
    /// buffer, and hands the owned bytes to a background thread for the
    /// swizzle → PNG encode → disk write — keeping zlib and file I/O off the
    /// render thread. Only the GPU wait and one memcpy happen inline.
    unsafe fn finish_screenshot(&self, rb: Readback, extent: vk::Extent2D, capture: Capture) {
        let Capture { path, reply } = capture;
        let device = &self.device.device;
        let pixels = unsafe {
            self.timeline.wait(device, self.last_copy_value);
            let ptr = device
                .map_memory(rb.memory, 0, rb.size, vk::MemoryMapFlags::empty())
                .expect("Failed to map screenshot readback memory")
                as *const u8;
            let pixels = std::slice::from_raw_parts(ptr, rb.size as usize).to_vec();
            device.unmap_memory(rb.memory);
            device.destroy_buffer(rb.buffer, None);
            device.free_memory(rb.memory, None);
            pixels
        };

        // Byte order of the swapchain's 8-bit channels relative to PNG's RGBA.
        // The picker only ever selects a BGRA or RGBA UNORM/SRGB format; a
        // fallback to anything else is written best-effort (no swizzle) rather
        // than silently mangled.
        let swap_bgra = match self.swapchain.format {
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => Some(true),
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => Some(false),
            other => {
                log::warn!("screenshot: unhandled swapchain format {other:?}; colors may be off");
                None
            }
        };
        let (width, height) = (extent.width, extent.height);

        std::thread::spawn(move || {
            let mut pixels = pixels;
            // Force alpha opaque (composite alpha is OPAQUE, so it is meaningless).
            for px in pixels.chunks_exact_mut(4) {
                if swap_bgra == Some(true) {
                    px.swap(0, 2);
                }
                px[3] = 255;
            }
            let result = crate::screenshot::write_png(&path, width, height, &pixels)
                .map_err(|e| std::io::Error::other(e.to_string()));
            match &result {
                Ok(()) => log::info!("screenshot saved: {}", path.display()),
                Err(e) => log::error!("screenshot encode failed ({}): {e}", path.display()),
            }
            // Signal a blocking caller ([`crate::screenshot_to`]) with the real
            // outcome; the interactive path leaves `reply` None and ignores it.
            if let Some(reply) = reply {
                let _ = reply.send(result);
            }
        });
    }
}
