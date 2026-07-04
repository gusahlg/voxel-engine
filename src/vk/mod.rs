/// The Vulkan renderer: owns the instance, device, swapchain, render
/// targets, pipelines, GPU memory, and the frame loop. Vulkan 1.3 with
/// dynamic rendering + synchronization2; 2 frames in flight; per-swapchain-
/// image present semaphores; reversed-Z depth; optional MSAA with resolve.
pub(crate) mod alloc;
pub(crate) mod buffers;
pub(crate) mod device;
pub(crate) mod instance;
pub(crate) mod pipeline;
pub(crate) mod swapchain;
pub(crate) mod targets;
pub(crate) mod texture;

use ash::{khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::event_loop::ActiveEventLoop;

use crate::color::Color;
use crate::frame::DrawLists;
use crate::mesh::{MeshData, MeshHandle};
use alloc::GpuAllocator;
use buffers::{FRAMES_IN_FLIGHT, HostBuffer, MeshRegistry};
use device::Device;
use instance::InstanceBundle;
use pipeline::Pipelines;
use swapchain::Swapchain;
use targets::RenderTargets;
use texture::FontAtlas;

struct FrameSlot {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    image_available: vk::Semaphore,
}

pub(crate) struct Renderer {
    pub window: winit::window::Window,

    instance: InstanceBundle,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    device: Device,

    allocator: GpuAllocator,
    meshes: MeshRegistry,

    swapchain: Swapchain,
    targets: RenderTargets,
    pipelines: Pipelines,
    atlas: FontAtlas,

    frames: Vec<FrameSlot>,
    /// One per swapchain image: signaled by the render submit, waited by present.
    present_semaphores: Vec<vk::Semaphore>,
    imm: Vec<HostBuffer>,

    slot: usize,
    frame_no: u64,

    vsync: bool,
    msaa: u32,
    needs_recreate: bool,
    pending_vsync: Option<bool>,
    pending_msaa: Option<u32>,
}

impl Renderer {
    pub fn new(
        event_loop: &ActiveEventLoop,
        title: &str,
        width: u32,
        height: u32,
        resizable: bool,
        fullscreen: bool,
        vsync: bool,
        msaa: u32,
    ) -> Self {
        let mut attrs = winit::window::WindowAttributes::default()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .with_resizable(resizable);
        if fullscreen {
            attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");

        let instance = InstanceBundle::new(
            event_loop
                .display_handle()
                .expect("no display handle")
                .as_raw(),
        );

        let surface_loader = khr::surface::Instance::new(&instance.entry, &instance.instance);
        let surface = unsafe {
            ash_window::create_surface(
                &instance.entry,
                &instance.instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .expect("Failed to create Vulkan surface")
        };

        let device = Device::new(&instance.instance, &surface_loader, surface);
        let mut allocator = unsafe { GpuAllocator::new(&instance.instance, device.physical) };
        if allocator.unified_memory() {
            log::info!("Unified memory detected: mesh uploads bypass staging");
        }
        let meshes = MeshRegistry::new();

        let size = window.inner_size();
        let swapchain = Swapchain::new(
            &instance.instance,
            &device,
            &surface_loader,
            surface,
            vk::Extent2D {
                width: size.width,
                height: size.height,
            },
            vsync,
            vk::SwapchainKHR::null(),
        );

        let msaa = clamp_msaa(msaa, device.max_msaa());
        let targets = RenderTargets::new(
            &instance.instance,
            &device.device,
            device.physical,
            swapchain.extent,
            swapchain.format,
            msaa,
        );

        let atlas = FontAtlas::new(
            &instance.instance,
            &device.device,
            device.physical,
            device.graphics_queue,
            device.command_pool,
        );

        let pipelines = Pipelines::new(
            &device.device,
            swapchain.format,
            targets.depth_format,
            targets.samples,
            atlas.set_layout,
        );

        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        let cmds = unsafe {
            device
                .device
                .allocate_command_buffers(&cmd_info)
                .expect("Failed to allocate command buffers")
        };
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let frames = cmds
            .into_iter()
            .map(|cmd| unsafe {
                FrameSlot {
                    cmd,
                    fence: device
                        .device
                        .create_fence(&fence_info, None)
                        .expect("Failed to create fence"),
                    image_available: device
                        .device
                        .create_semaphore(&semaphore_info, None)
                        .expect("Failed to create semaphore"),
                }
            })
            .collect();

        let present_semaphores = create_present_semaphores(&device.device, swapchain.images.len());
        let imm = (0..FRAMES_IN_FLIGHT).map(|_| HostBuffer::new()).collect();

        Self {
            window,
            instance,
            surface_loader,
            surface,
            device,
            allocator,
            meshes,
            swapchain,
            targets,
            pipelines,
            atlas,
            frames,
            present_semaphores,
            imm,
            slot: 0,
            frame_no: FRAMES_IN_FLIGHT, // so nothing is "completed" before the first real frame
            vsync,
            msaa,
            needs_recreate: false,
            pending_vsync: None,
            pending_msaa: None,
        }
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    pub fn request_recreate(&mut self) {
        self.needs_recreate = true;
    }

    pub fn set_vsync(&mut self, on: bool) {
        if on != self.vsync && self.pending_vsync != Some(on) {
            self.pending_vsync = Some(on);
            self.needs_recreate = true;
        }
    }

    pub fn vsync(&self) -> bool {
        self.pending_vsync.unwrap_or(self.vsync)
    }

    pub fn set_msaa(&mut self, samples: u32) -> u32 {
        let clamped = clamp_msaa(samples, self.device.max_msaa());
        if clamped != self.msaa && self.pending_msaa != Some(clamped) {
            self.pending_msaa = Some(clamped);
            self.needs_recreate = true;
        }
        clamped
    }

    pub fn msaa(&self) -> u32 {
        self.pending_msaa.unwrap_or(self.msaa)
    }

    pub fn max_msaa(&self) -> u32 {
        self.device.max_msaa()
    }

    pub fn upload_mesh(&mut self, data: &MeshData) -> Option<MeshHandle> {
        unsafe {
            self.meshes
                .upload(&self.device.device, &mut self.allocator, data, self.frame_no)
        }
    }

    pub fn free_mesh(&mut self, handle: MeshHandle) {
        self.meshes.free(handle, self.frame_no);
    }

    pub fn mesh_aabb(&self, handle: MeshHandle) -> Option<(glam::Vec3, glam::Vec3)> {
        self.meshes.get(handle).map(|m| (m.aabb_min, m.aabb_max))
    }

    /// Records, submits, and presents one frame from the recorded draw lists.
    pub fn draw_frame(&mut self, lists: &DrawLists) {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return; // minimized
        }
        if self.needs_recreate {
            unsafe { self.apply_pending() };
            if self.swapchain.extent.width == 0 || self.swapchain.extent.height == 0 {
                return;
            }
        }

        let device = &self.device.device;
        let slot = self.slot;
        let frame = &self.frames[slot];

        unsafe {
            device
                .wait_for_fences(&[frame.fence], true, u64::MAX)
                .expect("fence wait failed");
            self.meshes.collect(&mut self.allocator, self.frame_no);
        }

        let (image_index, suboptimal) = unsafe {
            match self.swapchain.loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.needs_recreate = true;
                    return;
                }
                Err(err) => panic!("acquire_next_image failed: {err:?}"),
            }
        };
        if suboptimal {
            self.needs_recreate = true;
        }

        // Immediate geometry for this frame, packed into one host buffer.
        let cube_bytes: &[u8] = bytemuck::cast_slice(&lists.cube_verts);
        let line_bytes: &[u8] = bytemuck::cast_slice(&lists.line_verts);
        let d2_bytes: &[u8] = bytemuck::cast_slice(&lists.verts_2d);
        let line_off = (cube_bytes.len() as u64).next_multiple_of(16);
        let d2_off = (line_off + line_bytes.len() as u64).next_multiple_of(16);
        let imm_total = d2_off + d2_bytes.len() as u64;
        unsafe {
            let imm = &mut self.imm[slot];
            if imm_total > 0 {
                imm.ensure_capacity(
                    &self.instance.instance,
                    device,
                    self.device.physical,
                    imm_total,
                );
                imm.write(0, cube_bytes);
                imm.write(line_off, line_bytes);
                imm.write(d2_off, d2_bytes);
            }
        }

        let frame = &self.frames[slot];
        let extent = self.swapchain.extent;
        unsafe {
            device
                .reset_fences(&[frame.fence])
                .expect("fence reset failed");
            device
                .reset_command_buffer(frame.cmd, vk::CommandBufferResetFlags::empty())
                .expect("command buffer reset failed");
            device
                .begin_command_buffer(frame.cmd, &vk::CommandBufferBeginInfo::default())
                .expect("begin command buffer failed");

            self.meshes.flush_copies(device, frame.cmd, self.frame_no);

            // Layout transitions for this frame's attachments. The swapchain
            // barrier chains with the acquire semaphore (both at
            // COLOR_ATTACHMENT_OUTPUT); old contents are always discarded.
            let mut image_barriers = vec![
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(self.swapchain.images[image_index as usize])
                    .subresource_range(color_range()),
                vk::ImageMemoryBarrier2::default()
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
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .image(self.targets.depth_image)
                    .subresource_range(depth_range()),
            ];
            if self.targets.multisampled() {
                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .image(self.targets.msaa_image)
                        .subresource_range(color_range()),
                );
            }
            device.cmd_pipeline_barrier2(
                frame.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&image_barriers),
            );

            let clear_color = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [
                        lists.clear.r as f32 / 255.0,
                        lists.clear.g as f32 / 255.0,
                        lists.clear.b as f32 / 255.0,
                        1.0,
                    ],
                },
            };
            let swap_view = self.swapchain.image_views[image_index as usize];
            let mut color_attachment = if self.targets.multisampled() {
                vk::RenderingAttachmentInfo::default()
                    .image_view(self.targets.msaa_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                    .resolve_image_view(swap_view)
                    .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
            } else {
                vk::RenderingAttachmentInfo::default()
                    .image_view(swap_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
            };
            color_attachment = color_attachment.clear_value(clear_color);
            let color_attachments = [color_attachment];

            // Reversed-Z: clear depth to 0.0, GREATER_OR_EQUAL test.
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.targets.depth_view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment);

            device.cmd_begin_rendering(frame.cmd, &rendering_info);

            // Negative viewport height = GL-style y-up NDC; keeps the game's
            // CCW-from-outside winding as front faces.
            let viewport = vk::Viewport {
                x: 0.0,
                y: extent.height as f32,
                width: extent.width as f32,
                height: -(extent.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(frame.cmd, 0, &[viewport]);
            device.cmd_set_scissor(
                frame.cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );

            let imm_buffer = self.imm[slot].buffer;

            if lists.has_3d {
                device.cmd_push_constants(
                    frame.cmd,
                    self.pipelines.layout_3d,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&lists.view_proj),
                );

                if !lists.mesh_draws.is_empty() {
                    device.cmd_bind_pipeline(
                        frame.cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.mesh3d,
                    );
                    // One vertex/index bind per suballocation block; draws use
                    // first_index/vertex_offset into the shared buffer.
                    let mut bound = vk::Buffer::null();
                    for &handle in &lists.mesh_draws {
                        let Some(mesh) = self.meshes.get(handle) else {
                            continue;
                        };
                        let buffer = mesh.buffer();
                        if buffer != bound {
                            device.cmd_bind_vertex_buffers(frame.cmd, 0, &[buffer], &[0]);
                            device.cmd_bind_index_buffer(
                                frame.cmd,
                                buffer,
                                0,
                                vk::IndexType::UINT32,
                            );
                            bound = buffer;
                        }
                        device.cmd_draw_indexed(
                            frame.cmd,
                            mesh.index_count,
                            1,
                            mesh.first_index,
                            mesh.vertex_offset,
                            0,
                        );
                    }
                }

                if !lists.cube_verts.is_empty() {
                    device.cmd_bind_pipeline(
                        frame.cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.mesh3d,
                    );
                    device.cmd_bind_vertex_buffers(frame.cmd, 0, &[imm_buffer], &[0]);
                    device.cmd_draw(frame.cmd, lists.cube_verts.len() as u32, 1, 0, 0);
                }

                if !lists.line_verts.is_empty() {
                    device.cmd_bind_pipeline(
                        frame.cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.lines3d,
                    );
                    device.cmd_bind_vertex_buffers(frame.cmd, 0, &[imm_buffer], &[line_off]);
                    device.cmd_draw(frame.cmd, lists.line_verts.len() as u32, 1, 0, 0);
                }
            }

            if !lists.verts_2d.is_empty() {
                device.cmd_bind_pipeline(
                    frame.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.tris2d,
                );
                device.cmd_bind_descriptor_sets(
                    frame.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.layout_2d,
                    0,
                    &[self.atlas.set],
                    &[],
                );
                let pixels_to_ndc = [2.0 / extent.width as f32, 2.0 / extent.height as f32];
                device.cmd_push_constants(
                    frame.cmd,
                    self.pipelines.layout_2d,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&pixels_to_ndc),
                );
                device.cmd_bind_vertex_buffers(frame.cmd, 0, &[imm_buffer], &[d2_off]);
                device.cmd_draw(frame.cmd, lists.verts_2d.len() as u32, 1, 0, 0);
            }

            device.cmd_end_rendering(frame.cmd);

            let present_barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(self.swapchain.images[image_index as usize])
                .subresource_range(color_range())];
            device.cmd_pipeline_barrier2(
                frame.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&present_barrier),
            );

            device
                .end_command_buffer(frame.cmd)
                .expect("end command buffer failed");

            let wait_info = [vk::SemaphoreSubmitInfo::default()
                .semaphore(frame.image_available)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let signal_info = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.present_semaphores[image_index as usize])
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.cmd)];
            let submit = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_info)
                .command_buffer_infos(&cmd_info)
                .signal_semaphore_infos(&signal_info)];
            device
                .queue_submit2(self.device.graphics_queue, &submit, frame.fence)
                .expect("queue submit failed");

            let wait_semaphores = [self.present_semaphores[image_index as usize]];
            let swapchains = [self.swapchain.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            match self
                .swapchain
                .loader
                .queue_present(self.device.present_queue, &present_info)
            {
                Ok(sub) => {
                    if sub {
                        self.needs_recreate = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_recreate = true,
                Err(err) => panic!("queue_present failed: {err:?}"),
            }
        }

        self.frame_no += 1;
        self.slot = (self.slot + 1) % self.frames.len();
    }

    /// Applies pending vsync/MSAA changes and rebuilds swapchain-sized state.
    unsafe fn apply_pending(&mut self) {
        unsafe {
            self.device
                .device
                .device_wait_idle()
                .expect("device_wait_idle failed");

            if let Some(v) = self.pending_vsync.take() {
                self.vsync = v;
            }
            let new_msaa = self.pending_msaa.take();

            let size = self.window.inner_size();
            if size.width == 0 || size.height == 0 {
                // Still minimized; keep the recreate pending.
                if let Some(m) = new_msaa {
                    self.pending_msaa = Some(m);
                }
                return;
            }

            let new_swapchain = Swapchain::new(
                &self.instance.instance,
                &self.device,
                &self.surface_loader,
                self.surface,
                vk::Extent2D {
                    width: size.width,
                    height: size.height,
                },
                self.vsync,
                self.swapchain.swapchain,
            );
            self.swapchain.destroy(&self.device.device);
            let format_changed = new_swapchain.format != self.swapchain.format;
            self.swapchain = new_swapchain;

            if let Some(m) = new_msaa {
                self.msaa = m;
            }

            self.targets.destroy(&self.device.device);
            self.targets = RenderTargets::new(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                self.swapchain.extent,
                self.swapchain.format,
                self.msaa,
            );

            if new_msaa.is_some() || format_changed {
                self.pipelines.destroy(&self.device.device);
                self.pipelines = Pipelines::new(
                    &self.device.device,
                    self.swapchain.format,
                    self.targets.depth_format,
                    self.targets.samples,
                    self.atlas.set_layout,
                );
            }

            for &sem in &self.present_semaphores {
                self.device.device.destroy_semaphore(sem, None);
            }
            self.present_semaphores =
                create_present_semaphores(&self.device.device, self.swapchain.images.len());

            self.needs_recreate = false;
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let device = &self.device.device;
            let _ = device.device_wait_idle();

            self.pipelines.destroy(device);
            self.atlas.destroy(device);
            self.targets.destroy(device);
            for imm in &mut self.imm {
                imm.destroy(device);
            }
            self.meshes.destroy_all(&mut self.allocator);
            self.allocator.destroy(device);
            for &sem in &self.present_semaphores {
                device.destroy_semaphore(sem, None);
            }
            for frame in &self.frames {
                device.destroy_semaphore(frame.image_available, None);
                device.destroy_fence(frame.fence, None);
            }
            self.swapchain.destroy(device);
            self.device.destroy();
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy();
        }
    }
}

fn clamp_msaa(requested: u32, max: u32) -> u32 {
    let requested = match requested {
        0 | 1 => 1,
        2..=3 => 2,
        4..=7 => 4,
        _ => 8,
    };
    requested.min(max)
}

fn create_present_semaphores(device: &ash::Device, count: usize) -> Vec<vk::Semaphore> {
    let info = vk::SemaphoreCreateInfo::default();
    (0..count)
        .map(|_| unsafe {
            device
                .create_semaphore(&info, None)
                .expect("Failed to create present semaphore")
        })
        .collect()
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn depth_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}
