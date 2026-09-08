//! Present path: overlay, swapchain copy, and screenshot readback.
//! Split out of `mod.rs` so later work can touch presentation without opening
//! the frame loop or scene recorder.

use ash::vk;

use crate::skeleton::FrameSlot;

use super::alloc;
use super::image_upload;
use super::render_client::Capture;
use super::timeline::{RenderCompletion, queue_present};
use super::{Env, Renderer, color_range, depth_range};

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
    pub(super) fn present(
        &mut self,
        slot: usize,
        present_target: Option<u32>,
        warp_map: crate::camera::WarpMap,
        overlay: OverlayPresent,
        hdr_readable: HdrReadable,
        godray: crate::camera::Godray,
    ) {
        // The proof must be for the slot we are about to sample.
        debug_assert_eq!(hdr_readable.slot, slot, "HdrReadable slot mismatch");
        if let Some(image_index) = present_target {
            unsafe { self.submit_present_copy(slot, image_index, warp_map, overlay, godray) };
            self.last_present = std::time::Instant::now();
        }
    }

    /// Draws the 2D overlay (text atlas + minimap) into the currently-bound
    /// swapchain attachment, using the present-format pipeline variants. Mirrors
    /// `RenderPass::record_2d` but for the post-tonemap pass; the caller has set a
    /// negative-height viewport so `tris2d.vert`'s pixel→NDC mapping is correct.
    unsafe fn record_overlay_present(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        overlay: OverlayPresent,
        extent: vk::Extent2D,
    ) {
        let device = &self.device.device;
        let pixels_to_ndc = [2.0 / extent.width as f32, 2.0 / extent.height as f32];
        unsafe {
            if overlay.d2_count > 0 {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.tris2d_present,
                );
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
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.tris2d_tex_present,
                );
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
        godray: crate::camera::Godray,
    ) {
        // A pending capture piggybacks on this copy: after the tonemap draw,
        // the swapchain image is read back into `readback` instead of going
        // straight to PRESENT. Allocate the host buffer before borrowing
        // `device` so the read-back path adds no &mut-self conflicts below.
        let capture = self.pending_capture.take();
        let extent = self.swapchain.extent;
        let readback = capture.as_ref().map(|_| unsafe {
            self.create_readback((extent.width as u64) * (extent.height as u64) * 4)
        });

        let device = &self.device.device;
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
        let tonemap_push = warp_map.push(exposure, godray, vignette);
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
            let profiling = crate::profile::is_enabled();
            if profiling {
                if let Some(ms) = self.gpu_timer.read_copy(device) {
                    crate::profile::add_ms(crate::profile::Meter::GpuTonemap, ms);
                }
                self.gpu_timer.begin_copy(device, self.copy_cmd);
            }

            // Swapchain image → color attachment; old contents discarded.
            let to_color = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(swap_image)
                .subresource_range(color_range())];
            device.cmd_pipeline_barrier2(
                self.copy_cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_color),
            );

            // Transition the sampleable depth for godray sampling, restore after
            // draw. Under MSAA this is the single-sample resolve target; the MS
            // `depth` is never touched here. Always bound: the tonemap layout
            // declares the depth sampler even when godrays are off (strength 0).
            let depth_image = self.targets.sampleable_depth(slot).image();
            let (depth_layout, depth_stage, depth_access) =
                self.sampleable_depth_attachment_state();
            {
                let depth_to_read = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(depth_stage)
                    .src_access_mask(depth_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(depth_layout)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(depth_image)
                    .subresource_range(depth_range())];
                device.cmd_pipeline_barrier2(
                    self.copy_cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&depth_to_read),
                );
            }

            let color_attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(swap_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachment);
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
            device.cmd_bind_pipeline(
                self.copy_cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.tonemap,
            );
            // Binding 0: the frame HDR — the offscreen, or the TAA output
            // when the resolve ran (the copy-back is gone).
            let hdr_view = self.hdr_of(slot).1;
            image_upload::push_combined_image_sampler(
                &self.device.push_descriptor,
                self.copy_cmd,
                self.pipelines.layout_tonemap,
                0,
                self.pipelines.tonemap_sampler,
                hdr_view,
            );
            // Binding 1: the bloom pyramid (built in the render submit, made
            // visible here by the render→present semaphore) with its mip-filtered
            // composite sampler, for the golden-spiral spill in tonemap.frag.
            let bloom_info = [vk::DescriptorImageInfo::default()
                .sampler(self.bloom.composite_sampler())
                .image_view(self.targets.bloom[slot].sample_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            // Binding 2: single-sample scene depth for the godray sky mask —
            // the MSAA resolve target when multisampled, the depth buffer else.
            let depth_view = self.targets.sampleable_depth(slot).view();
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(self.pipelines.tonemap_depth_sampler)
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let post_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&bloom_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),
            ];
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
            device.cmd_draw(self.copy_cmd, 3, 1, 0, 0);
            // Composite the 2D overlay onto the tonemapped swapchain (never drawn in
            // the offscreen scene pass). Post-tonemap on BOTH paths: wide-FOV so the
            // warp never bends the HUD, rectilinear so the TAA resolve never reprojects
            // it. Uses a GL-style negative-height viewport, matching tris2d.vert.
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
            self.record_overlay_present(self.copy_cmd, slot, overlay, extent);
            device.cmd_end_rendering(self.copy_cmd);

            // Restore the sampled depth so the next 3D pass / VRS classifier
            // finds the layout it expects (DEPTH_ATTACHMENT_OPTIMAL under MSAA,
            // where this is the resolve target, not the MS depth).
            {
                let depth_to_attach = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(depth_stage)
                    .dst_access_mask(depth_access)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(depth_layout)
                    .image(depth_image)
                    .subresource_range(depth_range())];
                device.cmd_pipeline_barrier2(
                    self.copy_cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&depth_to_attach),
                );
            }

            // When capturing, detour through TRANSFER_SRC to copy the finished
            // image into the host buffer, then continue to PRESENT.
            if let Some(rb) = &readback {
                let to_src = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(swap_image)
                    .subresource_range(color_range())];
                device.cmd_pipeline_barrier2(
                    self.copy_cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_src),
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
            let to_present = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(swap_image)
                .subresource_range(color_range())];
            device.cmd_pipeline_barrier2(
                self.copy_cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_present),
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
