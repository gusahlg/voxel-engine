/// The Vulkan renderer: owns the instance, device, swapchain, render
/// targets, pipelines, GPU memory, and the frame loop. Vulkan 1.3 with
/// dynamic rendering + synchronization2; 2 frames in flight; per-swapchain-
/// image present semaphores; reversed-Z depth; optional MSAA with resolve.
///
/// Rendering and presentation are decoupled (manual mailbox): every frame
/// renders into a per-slot offscreen image, and a separate copy submit blits
/// the finished frame into a swapchain image only when the presentation
/// engine can take one — otherwise the frame is rendered but never shown.
/// On macOS/MoltenVK the drawable wait is compositor-paced at the display
/// refresh regardless of present mode; keeping the swapchain out of the
/// render path uncaps the frame loop with vsync off, while vsync on paces
/// the loop at refresh via presentation backpressure (the copy fence).
pub(crate) mod alloc;
pub(crate) mod block_textures;
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

use crate::frame::DrawLists;
use crate::mesh::{MeshData, MeshHandle};
use alloc::GpuAllocator;
use block_textures::BlockTextures;
use buffers::{DrawIndexedIndirect, FRAMES_IN_FLIGHT, HostBuffer, MeshRegistry};
use device::Device;
use instance::InstanceBundle;
use pipeline::Pipelines;
use swapchain::Swapchain;
use targets::RenderTargets;
use texture::FontAtlas;

struct FrameSlot {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Signaled by `acquire_next_image`, waited by the same frame's present
    /// copy submit (at COPY). Reuse invariant: an acquire is only attempted
    /// once `copy_fence` is signaled — every earlier copy submit (the sole
    /// consumer, serialized on one queue by that same fence) has then fully
    /// executed and consumed any prior signal — and a successful acquire is
    /// ALWAYS followed by the copy submit in the same branch, so the
    /// semaphore is never left with an orphaned pending signal.
    image_available: vk::Semaphore,
    /// Signaled by the render submit, waited by the present copy submit (at
    /// COPY). Only signaled on frames that actually present: a binary
    /// semaphore signal with no consumer would make the slot's next signal
    /// invalid. (Same-queue submission order + the offscreen barrier already
    /// order render before copy; this makes the dependency explicit.)
    render_done: vk::Semaphore,
}

/// Byte offsets of the line and 2D sections inside a frame's packed
/// immediate buffer (cube verts start at offset 0).
#[derive(Clone, Copy)]
struct ImmOffsets {
    line: u64,
    d2: u64,
}

/// One resolved mesh draw, pre-sort scratch for [`Renderer::prepare_mesh_draws`].
#[derive(Clone, Copy)]
struct DrawEntry {
    buffer: vk::Buffer,
    index_count: u32,
    first_index: u32,
    vertex_offset: i32,
    offset: glam::Vec3,
}

/// A contiguous range of indirect commands that all draw from one arena
/// buffer: one vertex/index bind + one indirect call each.
#[derive(Clone, Copy)]
struct DrawRun {
    buffer: vk::Buffer,
    first: u32,
    count: u32,
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
    /// Disk-backed pipeline cache (see [`pipeline_cache_path`]): loaded at
    /// init, passed to every pipeline build, written back on Drop.
    pipeline_cache: vk::PipelineCache,
    atlas: FontAtlas,
    block_textures: BlockTextures,
    /// Persistent descriptor machinery for the block texture array: created
    /// once at init; `set_block_textures` only rewrites `block_set`, so
    /// pipelines never need rebuilding for a texture swap.
    block_set_layout: vk::DescriptorSetLayout,
    block_pool: vk::DescriptorPool,
    block_set: vk::DescriptorSet,

    frames: Vec<FrameSlot>,
    /// One per swapchain image: signaled by the copy submit, waited by present.
    present_semaphores: Vec<vk::Semaphore>,
    imm: Vec<HostBuffer>,
    /// Per-slot per-draw offsets SSBO (one vec4 per draw; slot 0 is a
    /// reserved zero vec4 so immediates draw with first_instance 0 and the
    /// shader path stays uniform).
    offsets: Vec<HostBuffer>,
    /// Per-slot indirect command buffer (20-byte VkDrawIndexedIndirectCommand
    /// per mesh draw, instance_count 1, first_instance = SSBO slot).
    indirect: Vec<HostBuffer>,
    /// Descriptor machinery for `offsets`: one set per slot, rewritten when
    /// that slot's SSBO buffer handle changes (growth/decay recreation).
    offsets_set_layout: vk::DescriptorSetLayout,
    offsets_pool: vk::DescriptorPool,
    offsets_sets: Vec<vk::DescriptorSet>,

    /// Frame scratch (persistent capacity): resolved draws, sorted by arena.
    draw_scratch: Vec<DrawEntry>,
    /// Frame scratch: the offsets SSBO contents ([0] = zero vec4).
    draw_offsets_data: Vec<[f32; 4]>,
    /// Frame scratch: indirect commands, arena-contiguous.
    draw_commands: Vec<DrawIndexedIndirect>,
    /// Frame scratch: one entry per arena buffer with visible draws.
    draw_runs: Vec<DrawRun>,

    /// Command buffer for the offscreen->swapchain present copy. A single
    /// one suffices: at most one copy is ever in flight, guarded by
    /// `copy_fence`.
    copy_cmd: vk::CommandBuffer,
    /// Signaled when the in-flight present copy has finished. Created
    /// SIGNALED; outside the reset->submit window (no early-outs in between)
    /// it is always signaled again once the GPU drains, so any
    /// device_wait_idle leaves it signaled.
    copy_fence: vk::Fence,
    /// Which offscreen slot the in-flight copy reads, if any. Rendering to
    /// that slot again must wait `copy_fence` first (rare, sub-millisecond).
    copy_slot: Option<usize>,

    slot: usize,
    frame_no: u64,

    vsync: bool,
    msaa: u32,
    needs_recreate: bool,
    /// Resolution scale for the 3D/UI render target relative to the window
    /// (0.25..=2.0). The present copy becomes a filtered blit when != 1.
    render_scale: f32,
    pending_render_scale: Option<f32>,
    /// The offscreen/depth/MSAA extent: swapchain extent * render_scale.
    render_extent: vk::Extent2D,
    /// Present pacing for the vsync-off path: presents are attempted at the
    /// display's refresh cadence so queue_present never has to wait for a
    /// drawable; frames in between render unthrottled.
    last_present: std::time::Instant,
    present_interval: std::time::Duration,
    timing: FrameTiming,
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
        render_scale: f32,
    ) -> Self {
        let render_scale = render_scale.clamp(0.25, 2.0);
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
        let allocator = unsafe { GpuAllocator::new(&instance.instance, device.physical) };
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
        let render_extent = scaled_extent(swapchain.extent, render_scale);
        let targets = RenderTargets::new(
            &instance.instance,
            &device.device,
            device.physical,
            render_extent,
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

        // Default 1x1 white block texture array (before Pipelines::new: its
        // persistent set layout feeds layout_3d).
        let block_tex = BlockTextures::new_default(
            &instance.instance,
            &device.device,
            device.physical,
            device.graphics_queue,
            device.command_pool,
        );
        let (block_set_layout, block_pool, block_set) =
            block_textures::create_descriptor(&device.device);
        block_textures::write_descriptor(&device.device, block_set, &block_tex);

        let (offsets_set_layout, offsets_pool, offsets_sets) =
            buffers::create_offsets_descriptors(&device.device, FRAMES_IN_FLIGHT as u32);

        let pipeline_cache = create_pipeline_cache(&device.device);
        let pipelines = Pipelines::new(
            &device.device,
            pipeline_cache,
            swapchain.format,
            targets.depth_format,
            targets.samples,
            atlas.set_layout,
            offsets_set_layout,
            block_set_layout,
        );

        // Per-slot command buffers plus one extra for the present copy.
        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32 + 1);
        let mut cmds = unsafe {
            device
                .device
                .allocate_command_buffers(&cmd_info)
                .expect("Failed to allocate command buffers")
        };
        let copy_cmd = cmds.pop().expect("command buffer allocation");
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
                    render_done: device
                        .device
                        .create_semaphore(&semaphore_info, None)
                        .expect("Failed to create semaphore"),
                }
            })
            .collect();
        let copy_fence = unsafe {
            device
                .device
                .create_fence(&fence_info, None)
                .expect("Failed to create copy fence")
        };

        let present_semaphores = create_present_semaphores(&device.device, swapchain.images.len());
        let present_interval = display_refresh_interval(&window);
        let imm = (0..FRAMES_IN_FLIGHT)
            .map(|_| HostBuffer::new(vk::BufferUsageFlags::VERTEX_BUFFER))
            .collect();
        let offsets = (0..FRAMES_IN_FLIGHT)
            .map(|_| HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER))
            .collect();
        let indirect = (0..FRAMES_IN_FLIGHT)
            .map(|_| HostBuffer::new(vk::BufferUsageFlags::INDIRECT_BUFFER))
            .collect();

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
            pipeline_cache,
            atlas,
            block_textures: block_tex,
            block_set_layout,
            block_pool,
            block_set,
            frames,
            present_semaphores,
            imm,
            offsets,
            indirect,
            offsets_set_layout,
            offsets_pool,
            offsets_sets,
            draw_scratch: Vec::new(),
            draw_offsets_data: Vec::new(),
            draw_commands: Vec::new(),
            draw_runs: Vec::new(),
            copy_cmd,
            copy_fence,
            copy_slot: None,
            slot: 0,
            frame_no: FRAMES_IN_FLIGHT, // so nothing is "completed" before the first real frame
            vsync,
            msaa,
            needs_recreate: false,
            pending_vsync: None,
            pending_msaa: None,
            render_scale,
            pending_render_scale: None,
            render_extent,
            last_present: std::time::Instant::now(),
            present_interval,
            timing: FrameTiming::new(),
        }
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    pub fn request_recreate(&mut self) {
        self.needs_recreate = true;
    }

    pub fn set_vsync(&mut self, on: bool) {
        // Compare against the EFFECTIVE value (pending included) so a change
        // can also be cancelled back to the current state before it applies.
        if on != self.vsync() {
            self.pending_vsync = Some(on);
            self.needs_recreate = true;
        }
    }

    pub fn vsync(&self) -> bool {
        self.pending_vsync.unwrap_or(self.vsync)
    }

    pub fn set_msaa(&mut self, samples: u32) -> u32 {
        let clamped = clamp_msaa(samples, self.device.max_msaa());
        if clamped != self.msaa() {
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

    /// Requests a render-resolution scale; returns the clamped value that
    /// will apply at the next frame boundary.
    pub fn set_render_scale(&mut self, scale: f32) -> f32 {
        let clamped = scale.clamp(0.25, 2.0);
        if (clamped - self.render_scale()).abs() > f32::EPSILON {
            self.pending_render_scale = Some(clamped);
            self.needs_recreate = true;
        }
        clamped
    }

    pub fn render_scale(&self) -> f32 {
        self.pending_render_scale.unwrap_or(self.render_scale)
    }

    pub fn upload_mesh(&mut self, data: &MeshData) -> Option<MeshHandle> {
        unsafe {
            self.meshes.upload(
                &self.device.device,
                &mut self.allocator,
                data,
                self.frame_no,
            )
        }
    }

    pub fn free_mesh(&mut self, handle: MeshHandle) {
        self.meshes.free(handle, self.frame_no);
    }

    pub fn mesh_aabb(&self, handle: MeshHandle) -> Option<(glam::Vec3, glam::Vec3)> {
        self.meshes.get(handle).map(|m| (m.aabb_min, m.aabb_max))
    }

    /// Replaces the block texture array (RGBA8, `layers.len()` images of
    /// `size*size*4` bytes each). Rare operation: waits for the GPU to go
    /// idle, uploads the new array, and rewrites the persistent descriptor
    /// set — pipelines are untouched.
    pub fn set_block_textures(&mut self, size: u32, layers: &[Vec<u8>]) {
        unsafe {
            self.device
                .device
                .device_wait_idle()
                .expect("device_wait_idle failed");
            self.block_textures.destroy(&self.device.device);
        }
        self.block_textures = BlockTextures::upload(
            &self.instance.instance,
            &self.device.device,
            self.device.physical,
            self.device.graphics_queue,
            self.device.command_pool,
            size,
            layers,
        );
        block_textures::write_descriptor(&self.device.device, self.block_set, &self.block_textures);
        log::debug!(
            "block textures swapped: {} layers of {}x{}",
            self.block_textures.layers,
            self.block_textures.size,
            self.block_textures.size,
        );
    }

    /// Records and submits one frame from the recorded draw lists, and
    /// presents it when the presentation engine can keep up (manual
    /// mailbox: frames that outrun presentation are rendered but dropped).
    ///
    /// Frame anatomy, top-down (each phase is its own helper below and its
    /// own [`FrameTiming`] span):
    /// 1. [`Self::wait_slot_and_reclaim`] — frame fence + deferred frees
    /// 2. [`Self::decide_present`]        — copy-fence check + acquire
    /// 3. [`Self::write_immediates`]      — pack cube/line/2D verts
    /// 4. [`Self::record_render`]         — barriers, rendering, draws
    /// 5. [`Self::submit_render`]         — render queue submit (fence)
    /// 6. [`Self::present`]               — copy submit + queue_present
    pub fn draw_frame(&mut self, lists: &DrawLists) {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            // Minimized: no rendering, but the game keeps running (remote
            // edits keep remeshing chunks), so uploads and frees must not
            // accumulate unboundedly until restore.
            unsafe { self.reclaim_while_idle() };
            return;
        }
        if self.needs_recreate {
            unsafe { self.apply_pending() };
            if self.swapchain.extent.width == 0 || self.swapchain.extent.height == 0 {
                return;
            }
        }

        let slot = self.slot;

        let t0 = std::time::Instant::now();
        self.wait_slot_and_reclaim(slot);
        self.timing.fence = t0.elapsed();

        let t0 = std::time::Instant::now();
        let present_target = self.decide_present(slot);
        self.timing.acquire = t0.elapsed();

        let t0 = std::time::Instant::now();
        let offsets = self.write_immediates(slot, lists);
        self.prepare_mesh_draws(slot, lists);
        self.record_render(slot, lists, offsets);
        self.timing.record = t0.elapsed();

        let t0 = std::time::Instant::now();
        self.submit_render(slot, present_target.is_some());
        self.timing.submit = t0.elapsed();

        let t0 = std::time::Instant::now();
        self.present(slot, present_target);
        self.timing.present = t0.elapsed();

        self.timing.tick();

        self.frame_no += 1;
        self.slot = (self.slot + 1) % self.frames.len();
    }

    /// Waits the slot's frame fence (the GPU is done with this slot's
    /// command buffer and immediate buffer), then reclaims retired GPU
    /// memory whose last possible use has completed.
    fn wait_slot_and_reclaim(&mut self, slot: usize) {
        let device = &self.device.device;
        unsafe {
            device
                .wait_for_fences(&[self.frames[slot].fence], true, u64::MAX)
                .expect("fence wait failed");
            // The in-flight present copy may still be reading this slot's
            // offscreen image, which the render below overwrites. Rare (the
            // copy usually retires well within the two-frame slot cycle)
            // and sub-millisecond when it happens.
            if self.copy_slot == Some(slot) {
                device
                    .wait_for_fences(&[self.copy_fence], true, u64::MAX)
                    .expect("copy fence wait failed");
                self.copy_slot = None;
            }
            self.meshes.collect(&mut self.allocator, self.frame_no);
            self.allocator.shrink_staging(device);
        }
    }

    /// Present eligibility, decided before the render submit so that
    /// render_done is only signaled when the copy submit will consume it
    /// (see FrameSlot::render_done). Strict ordering: the copy_fence
    /// status check comes first, the acquire is only attempted once we
    /// know a copy can be submitted, and a successful acquire is ALWAYS
    /// followed by the copy + present in [`Self::present`] — never skipped.
    fn decide_present(&mut self, slot: usize) -> Option<u32> {
        // With vsync off, pace PRESENTS to the display's refresh: presenting
        // faster than drawables recycle makes queue_present block for
        // milliseconds, throttling the whole loop. At refresh cadence a
        // drawable is essentially always free, so presents cost ~nothing and
        // every frame between them renders unthrottled.
        let present_due =
            self.vsync || self.last_present.elapsed() >= self.present_interval.mul_f32(0.9);
        let mut present_target = None;
        unsafe {
            // Previous copy still in flight? Skip presenting this frame:
            // it is rendered, just never shown (the mailbox drop).
            let copy_ready = present_due
                && self
                    .device
                    .device
                    .get_fence_status(self.copy_fence)
                    .expect("fence status failed");
            if copy_ready {
                // vsync on: block for an image so a copy is ALWAYS submitted
                // and the copy-fence wait below actually paces the loop at
                // refresh. vsync off: never wait — drop the present instead.
                let timeout = if self.vsync { u64::MAX } else { 0 };
                match self.swapchain.loader.acquire_next_image(
                    self.swapchain.swapchain,
                    timeout,
                    self.frames[slot].image_available,
                    vk::Fence::null(),
                ) {
                    Ok((image_index, suboptimal)) => {
                        if suboptimal {
                            self.needs_recreate = true;
                        }
                        present_target = Some(image_index);
                    }
                    // No swapchain image immediately free: drop the present.
                    Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {}
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_recreate = true,
                    Err(err) => panic!("acquire_next_image failed: {err:?}"),
                }
            }
        }
        present_target
    }

    /// Packs this frame's immediate geometry (cubes, lines, 2D overlay) into
    /// one host buffer and returns the section offsets. Also runs the
    /// buffer's capacity decay — safe point: the slot's fence was just
    /// waited in [`Self::wait_slot_and_reclaim`].
    fn write_immediates(&mut self, slot: usize, lists: &DrawLists) -> ImmOffsets {
        let cube_bytes: &[u8] = bytemuck::cast_slice(&lists.cube_verts);
        let line_bytes: &[u8] = bytemuck::cast_slice(&lists.line_verts);
        let d2_bytes: &[u8] = bytemuck::cast_slice(&lists.verts_2d);
        let line = (cube_bytes.len() as u64).next_multiple_of(16);
        let d2 = (line + line_bytes.len() as u64).next_multiple_of(16);
        let total = d2 + d2_bytes.len() as u64;
        let imm = &mut self.imm[slot];
        unsafe {
            imm.maintain(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                total,
            );
            if total > 0 {
                imm.write(0, cube_bytes);
                imm.write(line, line_bytes);
                imm.write(d2, d2_bytes);
            }
        }
        ImmOffsets { line, d2 }
    }

    /// Resolves this frame's mesh draws, groups them by arena buffer, and
    /// writes the per-draw offsets SSBO + indirect command buffer for the
    /// slot, so `record_render` can issue ONE indirect call per arena
    /// instead of a bind+push+draw per mesh (the MoltenVK command-encode
    /// cost in queue_submit scaled with per-draw commands).
    ///
    /// SSBO layout: slot 0 = zero vec4 (immediate cubes/lines draw with
    /// first_instance 0), slots 1.. = mesh offsets, indexed by each draw's
    /// first_instance via the shader's raw InstanceIndex builtin.
    ///
    /// Same safe point as `write_immediates` for buffer growth/decay; a
    /// recreated SSBO also gets its per-slot descriptor set rewritten here
    /// (the slot's fence was waited, so no pending commands use the set).
    fn prepare_mesh_draws(&mut self, slot: usize, lists: &DrawLists) {
        use ash::vk::Handle;

        self.draw_scratch.clear();
        self.draw_offsets_data.clear();
        self.draw_commands.clear();
        self.draw_runs.clear();

        if lists.has_3d {
            self.draw_offsets_data.push([0.0; 4]); // reserved zero slot
            for &(handle, off) in &lists.mesh_draws {
                let Some(mesh) = self.meshes.get(handle) else {
                    continue;
                };
                self.draw_scratch.push(DrawEntry {
                    buffer: mesh.buffer(),
                    index_count: mesh.index_count,
                    first_index: mesh.first_index,
                    vertex_offset: mesh.vertex_offset,
                    offset: off,
                });
            }
            // Group by arena buffer so each arena becomes one contiguous
            // command range. Order within the mesh pass is free to change:
            // opaque, depth-tested geometry.
            self.draw_scratch
                .sort_unstable_by_key(|entry| entry.buffer.as_raw());

            for entry in &self.draw_scratch {
                let ssbo_slot = self.draw_offsets_data.len() as u32;
                let command_index = self.draw_commands.len() as u32;
                self.draw_offsets_data
                    .push([entry.offset.x, entry.offset.y, entry.offset.z, 0.0]);
                self.draw_commands.push(DrawIndexedIndirect {
                    index_count: entry.index_count,
                    instance_count: 1,
                    first_index: entry.first_index,
                    vertex_offset: entry.vertex_offset,
                    first_instance: ssbo_slot,
                });
                match self.draw_runs.last_mut() {
                    Some(run) if run.buffer == entry.buffer => run.count += 1,
                    _ => self.draw_runs.push(DrawRun {
                        buffer: entry.buffer,
                        first: command_index,
                        count: 1,
                    }),
                }
            }
        }

        let offsets_bytes: &[u8] = bytemuck::cast_slice(&self.draw_offsets_data);
        let indirect_bytes: &[u8] = bytemuck::cast_slice(&self.draw_commands);
        unsafe {
            let ssbo = &mut self.offsets[slot];
            let recreated = ssbo.maintain(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                offsets_bytes.len() as u64,
            );
            if !offsets_bytes.is_empty() {
                ssbo.write(0, offsets_bytes);
            }
            // Recreate-on-grow (or decay): retarget this slot's descriptor.
            // The set is only ever referenced by this slot's command buffer,
            // whose fence was just waited.
            if recreated && ssbo.buffer != vk::Buffer::null() {
                buffers::write_offsets_descriptor(
                    &self.device.device,
                    self.offsets_sets[slot],
                    ssbo.buffer,
                );
            }

            let indirect = &mut self.indirect[slot];
            indirect.maintain(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                indirect_bytes.len() as u64,
            );
            if !indirect_bytes.is_empty() {
                indirect.write(0, indirect_bytes);
            }
        }
    }

    /// Resets and records the slot's command buffer: staged mesh copies,
    /// attachment layout transitions, the dynamic-rendering pass with all
    /// draws, and the hand-off barrier to the present copy.
    fn record_render(&mut self, slot: usize, lists: &DrawLists, offsets: ImmOffsets) {
        let device = &self.device.device;
        let frame = &self.frames[slot];
        // Rendering happens at the (possibly scaled) offscreen resolution;
        // 2D coordinates stay in window pixels (NDC is resolution-free).
        let extent = self.render_extent;
        let window_extent = self.swapchain.extent;
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

            // Layout transitions for this frame's attachments; old contents
            // are always discarded. offscreen[slot] needs no sync against
            // the present copies here: a copy still reading THIS slot was
            // host-waited via copy_fence above, and the other slot's
            // in-flight copy must NOT be waited on (its src stage is COPY,
            // deliberately excluded) or presentation backpressure would leak
            // back into rendering.
            // Fixed-size barrier array: no per-frame heap allocation in the
            // steady-state draw path.
            let offscreen_image = self.targets.offscreen_images[slot];
            let mut image_barriers = [vk::ImageMemoryBarrier2::default(); 3];
            image_barriers[0] = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(offscreen_image)
                .subresource_range(color_range());
            image_barriers[1] = vk::ImageMemoryBarrier2::default()
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
                .subresource_range(depth_range());
            let mut barrier_count = 2;
            if self.targets.multisampled() {
                image_barriers[2] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(self.targets.msaa_image)
                    .subresource_range(color_range());
                barrier_count = 3;
            }
            device.cmd_pipeline_barrier2(
                frame.cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&image_barriers[..barrier_count]),
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
            let offscreen_view = self.targets.offscreen_views[slot];
            let mut color_attachment = if self.targets.multisampled() {
                vk::RenderingAttachmentInfo::default()
                    .image_view(self.targets.msaa_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                    .resolve_image_view(offscreen_view)
                    .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
            } else {
                // offscreen[slot] IS the color target; STORE so the contents
                // survive for the present copy.
                vk::RenderingAttachmentInfo::default()
                    .image_view(offscreen_view)
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
                // Every 3D draw uses both sets: 0 = per-draw offsets SSBO
                // (immediates read the zero slot), 1 = block texture array
                // (layer 0 = white). Bound once for the whole 3D pass.
                device.cmd_bind_descriptor_sets(
                    frame.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines.layout_3d,
                    0,
                    &[self.offsets_sets[slot], self.block_set],
                    &[],
                );

                if !self.draw_runs.is_empty() {
                    device.cmd_bind_pipeline(
                        frame.cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.mesh3d,
                    );
                    // One vertex + index bind per arena buffer, both at
                    // offset 0: MESH_ALIGN keeps suballocations exact in
                    // whole vertices (command vertex_offset) and indices
                    // (absolute first_index). Then a single indirect call
                    // draws the arena's whole contiguous command range —
                    // with per-draw offsets fetched from the SSBO by
                    // first_instance, nothing is pushed between draws.
                    let indirect_buffer = self.indirect[slot].buffer;
                    const STRIDE: u64 = std::mem::size_of::<DrawIndexedIndirect>() as u64;
                    for run in &self.draw_runs {
                        device.cmd_bind_index_buffer(
                            frame.cmd,
                            run.buffer,
                            0,
                            vk::IndexType::UINT32,
                        );
                        device.cmd_bind_vertex_buffers(frame.cmd, 0, &[run.buffer], &[0]);
                        if self.device.multi_draw_indirect
                            && self.device.draw_indirect_first_instance
                        {
                            device.cmd_draw_indexed_indirect(
                                frame.cmd,
                                indirect_buffer,
                                run.first as u64 * STRIDE,
                                run.count,
                                STRIDE as u32,
                            );
                        } else if self.device.draw_indirect_first_instance {
                            // No multiDrawIndirect: drawCount=1 indirect
                            // calls need no feature and still skip the
                            // per-draw binds/pushes.
                            for i in run.first..run.first + run.count {
                                device.cmd_draw_indexed_indirect(
                                    frame.cmd,
                                    indirect_buffer,
                                    i as u64 * STRIDE,
                                    1,
                                    STRIDE as u32,
                                );
                            }
                        } else {
                            // No drawIndirectFirstInstance: direct draws may
                            // always carry firstInstance, so replay the
                            // commands CPU-side (still bind/push-free).
                            let range = run.first as usize..(run.first + run.count) as usize;
                            for c in &self.draw_commands[range] {
                                device.cmd_draw_indexed(
                                    frame.cmd,
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
                    device.cmd_bind_vertex_buffers(frame.cmd, 0, &[imm_buffer], &[offsets.line]);
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
                let pixels_to_ndc = [
                    2.0 / window_extent.width as f32,
                    2.0 / window_extent.height as f32,
                ];
                device.cmd_push_constants(
                    frame.cmd,
                    self.pipelines.layout_2d,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&pixels_to_ndc),
                );
                device.cmd_bind_vertex_buffers(frame.cmd, 0, &[imm_buffer], &[offsets.d2]);
                device.cmd_draw(frame.cmd, lists.verts_2d.len() as u32, 1, 0, 0);
            }

            device.cmd_end_rendering(frame.cmd);

            // Hand offscreen[slot] to the present copy (this frame's, or —
            // via same-queue submission order — any later frame's).
            let to_copy_src = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                // ALL_TRANSFER, not COPY: the present step is a BLIT when
                // render_scale != 1, and sync2's COPY stage does not cover it.
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(offscreen_image)
                .subresource_range(color_range())];
            device.cmd_pipeline_barrier2(
                frame.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_copy_src),
            );

            device
                .end_command_buffer(frame.cmd)
                .expect("end command buffer failed");
        }
    }

    /// Submits the recorded command buffer, signaling the slot's fence and —
    /// only on frames that will present — render_done (a binary-semaphore
    /// signal with no consumer would make the slot's next signal invalid).
    fn submit_render(&mut self, slot: usize, will_present: bool) {
        let frame = &self.frames[slot];
        // Render submit: no waits (nothing renders into the swapchain
        // anymore); the frame fence keeps its pacing role unchanged.
        let signal_info = [vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.render_done)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.cmd)];
        let mut submit_info = vk::SubmitInfo2::default().command_buffer_infos(&cmd_info);
        if will_present {
            submit_info = submit_info.signal_semaphore_infos(&signal_info);
        }
        let submit = [submit_info];
        unsafe {
            self.device
                .device
                .queue_submit2(self.device.graphics_queue, &submit, frame.fence)
                .expect("queue submit failed");
        }
    }

    /// Copies the finished frame into the acquired swapchain image (when one
    /// was acquired in [`Self::decide_present`]) and queues the present.
    fn present(&mut self, slot: usize, present_target: Option<u32>) {
        if let Some(image_index) = present_target {
            unsafe { self.submit_present_copy(slot, image_index) };
            self.last_present = std::time::Instant::now();
        }
        if self.vsync {
            // Presentation backpressure paces the loop at the display
            // refresh, preserving the classic vsync feel (and its power
            // savings). With vsync off this fence is only unsignaled for the
            // microseconds the copy takes, so the loop stays uncapped.
            unsafe {
                self.device
                    .device
                    .wait_for_fences(&[self.copy_fence], true, u64::MAX)
                    .expect("copy fence wait failed");
            }
        }
    }

    /// Records and submits the offscreen[slot] -> swapchain copy, then
    /// queues the present. Caller guarantees `copy_fence` is signaled and
    /// the image was just acquired with `frames[slot].image_available`.
    unsafe fn submit_present_copy(&mut self, slot: usize, image_index: u32) {
        let device = &self.device.device;
        let extent = self.swapchain.extent;
        let swap_image = self.swapchain.images[image_index as usize];
        unsafe {
            device
                .reset_fences(&[self.copy_fence])
                .expect("copy fence reset failed");
            device
                .reset_command_buffer(self.copy_cmd, vk::CommandBufferResetFlags::empty())
                .expect("command buffer reset failed");
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.copy_cmd, &begin)
                .expect("begin command buffer failed");

            // Swapchain image to TRANSFER_DST; src stage COPY / access NONE
            // chains with the acquire semaphore wait (also at COPY). Old
            // contents are discarded.
            let to_dst = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(swap_image)
                .subresource_range(color_range())];
            device.cmd_pipeline_barrier2(
                self.copy_cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_dst),
            );

            let layers = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let src_extent = self.render_extent;
            if src_extent == extent {
                let region = vk::ImageCopy::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    });
                device.cmd_copy_image(
                    self.copy_cmd,
                    self.targets.offscreen_images[slot],
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    swap_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            } else {
                // Render scale != 1: filtered blit up/down to the window.
                let blit = vk::ImageBlit::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: src_extent.width as i32,
                            y: src_extent.height as i32,
                            z: 1,
                        },
                    ])
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: extent.width as i32,
                            y: extent.height as i32,
                            z: 1,
                        },
                    ]);
                device.cmd_blit_image(
                    self.copy_cmd,
                    self.targets.offscreen_images[slot],
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    swap_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            let to_present = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(swap_image)
                .subresource_range(color_range())];
            device.cmd_pipeline_barrier2(
                self.copy_cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_present),
            );
            device
                .end_command_buffer(self.copy_cmd)
                .expect("end command buffer failed");

            let wait_info = [
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.frames[slot].image_available)
                    .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER),
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.frames[slot].render_done)
                    .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER),
            ];
            let signal_info = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.present_semaphores[image_index as usize])
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(self.copy_cmd)];
            let submit = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_info)
                .command_buffer_infos(&cmd_info)
                .signal_semaphore_infos(&signal_info)];
            device
                .queue_submit2(self.device.graphics_queue, &submit, self.copy_fence)
                .expect("copy submit failed");
            self.copy_slot = Some(slot);

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
    }

    /// While no frames are being submitted (minimized window): waits out the
    /// in-flight fences, flushes any staged mesh copies with a standalone
    /// submit, and frees the whole retire queue.
    unsafe fn reclaim_while_idle(&mut self) {
        if !self.meshes.has_pending() && !self.meshes.has_garbage() {
            return;
        }
        let device = &self.device.device;
        unsafe {
            let mut fences: Vec<vk::Fence> = self.frames.iter().map(|f| f.fence).collect();
            fences.push(self.copy_fence);
            device
                .wait_for_fences(&fences, true, u64::MAX)
                .expect("fence wait failed");
            self.copy_slot = None;

            if self.meshes.has_pending() {
                // Reuse slot 0's command buffer — its fence is signaled and
                // nothing else records until the next real frame.
                let cmd = self.frames[0].cmd;
                device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .expect("command buffer reset failed");
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device
                    .begin_command_buffer(cmd, &begin)
                    .expect("begin command buffer failed");
                self.meshes.flush_copies(device, cmd, self.frame_no);
                device
                    .end_command_buffer(cmd)
                    .expect("end command buffer failed");
                let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
                let submit = [vk::SubmitInfo2::default().command_buffer_infos(&cmd_info)];
                device
                    .queue_submit2(self.device.graphics_queue, &submit, vk::Fence::null())
                    .expect("queue submit failed");
                device
                    .queue_wait_idle(self.device.graphics_queue)
                    .expect("queue wait failed");
            }

            // GPU idle + copies flushed: everything retired is reclaimable.
            self.meshes.collect_all(&mut self.allocator);
        }
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
            if let Some(scale) = self.pending_render_scale.take() {
                self.render_scale = scale;
            }
            self.render_extent = scaled_extent(self.swapchain.extent, self.render_scale);

            self.targets.destroy(&self.device.device);
            self.targets = RenderTargets::new(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                self.render_extent,
                self.swapchain.format,
                self.msaa,
            );
            // The offscreen images were just recreated; forget any copy
            // tracking. copy_fence needs no reset: it is only ever
            // unsignaled between its reset and the copy submit retiring, so
            // the device_wait_idle above left it signaled.
            self.copy_slot = None;

            if new_msaa.is_some() || format_changed {
                self.pipelines.destroy(&self.device.device);
                self.pipelines = Pipelines::new(
                    &self.device.device,
                    self.pipeline_cache,
                    self.swapchain.format,
                    self.targets.depth_format,
                    self.targets.samples,
                    self.atlas.set_layout,
                    self.offsets_set_layout,
                    self.block_set_layout,
                );
            }

            for &sem in &self.present_semaphores {
                self.device.device.destroy_semaphore(sem, None);
            }
            self.present_semaphores =
                create_present_semaphores(&self.device.device, self.swapchain.images.len());

            self.present_interval = display_refresh_interval(&self.window);
            self.needs_recreate = false;
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        log::debug!("GPU memory at shutdown: {:?}", self.allocator.stats());
        unsafe {
            let device = &self.device.device;
            let _ = device.device_wait_idle();

            self.pipelines.destroy(device);
            save_pipeline_cache(device, self.pipeline_cache);
            device.destroy_pipeline_cache(self.pipeline_cache, None);
            self.atlas.destroy(device);
            device.destroy_descriptor_pool(self.block_pool, None);
            device.destroy_descriptor_set_layout(self.block_set_layout, None);
            device.destroy_descriptor_pool(self.offsets_pool, None);
            device.destroy_descriptor_set_layout(self.offsets_set_layout, None);
            self.block_textures.destroy(device);
            self.targets.destroy(device);
            for buffer in self
                .imm
                .iter_mut()
                .chain(&mut self.offsets)
                .chain(&mut self.indirect)
            {
                buffer.destroy(device);
            }
            self.meshes.destroy_all(&mut self.allocator);
            self.allocator.destroy(device);
            for &sem in &self.present_semaphores {
                device.destroy_semaphore(sem, None);
            }
            device.destroy_fence(self.copy_fence, None);
            for frame in &self.frames {
                device.destroy_semaphore(frame.image_available, None);
                device.destroy_semaphore(frame.render_done, None);
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

/// Where the pipeline cache lives on disk. The OS temp dir was chosen over
/// "next to the executable" because it is per-user writable everywhere (an
/// installed game may sit in a read-only location) and the cache is a
/// machine-local driver artifact, not user data — losing it on a temp clean
/// only costs one warm-up compile.
fn pipeline_cache_path() -> std::path::PathBuf {
    std::env::temp_dir().join("voxel_engine_pipeline.cache")
}

/// Creates the renderer's pipeline cache, seeded from disk when a previous
/// run saved one. Stale or corrupt data is harmless: the driver validates
/// the blob's header (UUID etc.) and falls back to an empty cache, and any
/// create error falls back to an empty cache too.
fn create_pipeline_cache(device: &ash::Device) -> vk::PipelineCache {
    let data = std::fs::read(pipeline_cache_path()).unwrap_or_default();
    if !data.is_empty() {
        let info = vk::PipelineCacheCreateInfo::default().initial_data(&data);
        if let Ok(cache) = unsafe { device.create_pipeline_cache(&info, None) } {
            log::debug!("pipeline cache loaded ({} bytes)", data.len());
            return cache;
        }
        log::warn!("saved pipeline cache rejected; starting empty");
    }
    unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }
        .expect("Failed to create pipeline cache")
}

/// Best-effort write-back of the pipeline cache; a failure only costs the
/// next run's warm start.
fn save_pipeline_cache(device: &ash::Device, cache: vk::PipelineCache) {
    match unsafe { device.get_pipeline_cache_data(cache) } {
        Ok(data) if !data.is_empty() => {
            if let Err(err) = std::fs::write(pipeline_cache_path(), &data) {
                log::debug!("pipeline cache not saved: {err}");
            }
        }
        Ok(_) => {}
        Err(err) => log::debug!("pipeline cache data unavailable: {err:?}"),
    }
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

/// One display refresh period, from the window's current monitor
/// (fallback: 120 Hz, the fastest common case — undershooting only means a
/// few wasted present attempts, never a stall).
fn display_refresh_interval(window: &winit::window::Window) -> std::time::Duration {
    let millihertz = window
        .current_monitor()
        .and_then(|m| m.refresh_rate_millihertz())
        .filter(|&mhz| mhz > 0) // Some(0) = unknown on some X11/VM backends
        .unwrap_or(120_000);
    std::time::Duration::from_secs_f64(1000.0 / millihertz as f64)
}

fn scaled_extent(extent: vk::Extent2D, scale: f32) -> vk::Extent2D {
    vk::Extent2D {
        width: ((extent.width as f32 * scale) as u32).max(1),
        height: ((extent.height as f32 * scale) as u32).max(1),
    }
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

/// Env-gated (`VOXEL_ENGINE_TIMING=1`) per-phase frame timers, logged every
/// couple of seconds. Costs two `Instant::now()` calls per frame when off.
/// Phases: fence = frame-fence wait (plus the rare same-slot copy wait);
/// acquire = copy_fence status check + zero-timeout acquire; record = render
/// command recording; submit = the render queue_submit2; present = copy
/// record + copy submit + queue_present, and under vsync the presentation
/// backpressure wait.
struct FrameTiming {
    enabled: bool,
    fence: std::time::Duration,
    acquire: std::time::Duration,
    record: std::time::Duration,
    submit: std::time::Duration,
    present: std::time::Duration,
    sum: [std::time::Duration; 5],
    frames: u32,
}

impl FrameTiming {
    fn new() -> Self {
        Self {
            enabled: std::env::var("VOXEL_ENGINE_TIMING").is_ok_and(|v| v != "0"),
            fence: Default::default(),
            acquire: Default::default(),
            record: Default::default(),
            submit: Default::default(),
            present: Default::default(),
            sum: Default::default(),
            frames: 0,
        }
    }

    fn tick(&mut self) {
        if !self.enabled {
            return;
        }
        self.sum[0] += self.fence;
        self.sum[1] += self.acquire;
        self.sum[2] += self.record;
        self.sum[3] += self.submit;
        self.sum[4] += self.present;
        self.frames += 1;
        if self.frames >= 240 {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / self.frames as f64;
            log::info!(
                "frame phases avg: fence {:.3}ms acquire {:.3}ms record {:.3}ms submit {:.3}ms present {:.3}ms",
                ms(self.sum[0]),
                ms(self.sum[1]),
                ms(self.sum[2]),
                ms(self.sum[3]),
                ms(self.sum[4]),
            );
            self.sum = Default::default();
            self.frames = 0;
        }
    }
}
