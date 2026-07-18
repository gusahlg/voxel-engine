/// The Vulkan renderer: instance, device, swapchain, render targets, pipelines,
/// GPU memory, and frame loop. Vulkan 1.3 with dynamic rendering + synchronization2;
/// 2 frames in flight; reversed-Z depth; optional MSAA with resolve.
///
/// Rendering and presentation decouple: frames render into offscreen images and
/// present only when a swapchain image is available (mailbox). On macOS, vsync
/// paces at refresh via presentation backpressure; vsync off uncaps the loop.
pub(crate) mod alloc;
pub(crate) mod block_textures;
pub(crate) mod bloom;
pub(crate) mod buffers;
pub(crate) mod cull;
pub(crate) mod device;
pub(crate) mod exposure;
pub(crate) mod image;
pub(crate) mod image_upload;
pub(crate) mod instance;
pub(crate) mod minimap;
pub(crate) mod pass;
pub(crate) mod pipeline;
pub(crate) mod render_client;
pub(crate) mod shadow;
pub(crate) mod swapchain;
pub(crate) mod taa;
pub(crate) mod targets;
pub(crate) mod texture;
pub(crate) mod timeline;
pub(crate) mod transfer;
pub(crate) mod uniforms;
pub(crate) mod vertex_input;
pub(crate) mod vrs;

use std::num::NonZeroU32;
use std::sync::mpsc::Sender;

use ash::{khr, vk};

use crate::frame::DrawLists;
#[cfg(test)]
use crate::frame::Scene3D;
use crate::mesh::Pass;
use crate::skeleton::{FrameSlot, PerSlot};
use block_textures::BlockTextures;
use buffers::{DrawIndexedIndirect, FRAMES_IN_FLIGHT, GpuResident, HostBuffer, MeshResidency};
use device::Device;
use instance::InstanceBundle;
use minimap::MinimapTexture;
use pipeline::Pipelines;
use render_client::{Capture, DeviceCaps, DeviceLeftovers, InitReply, RenderConfig, RenderReturn};
use swapchain::Swapchain;
use targets::RenderTargets;
use texture::FontAtlas;
use timeline::{
    BinarySemaphore, RenderCompletion, RenderSubmit, Timeline, TimelineValue, acquire_next_image,
    queue_present,
};
use transfer::TransferLane;

/// Recoverable environmental events raised by acquire/present. `OutOfDate`
/// and `SurfaceLost` drive the existing swapchain-recreate flow. `DeviceLost`
/// is classified here for completeness but has NO recovery path — it is
/// surfaced only so callers panic on it explicitly rather than swallowing it.
enum Env {
    OutOfDate,
    SurfaceLost,
    DeviceLost,
}

impl Env {
    /// Classifies an acquire/present error. `None` for errors that are unrecoverable.
    fn classify(err: vk::Result) -> Option<Env> {
        match err {
            vk::Result::ERROR_OUT_OF_DATE_KHR => Some(Env::OutOfDate),
            vk::Result::ERROR_SURFACE_LOST_KHR => Some(Env::SurfaceLost),
            vk::Result::ERROR_DEVICE_LOST => Some(Env::DeviceLost),
            _ => None,
        }
    }
}

struct SlotState {
    cmd: vk::CommandBuffer,
    /// Signaled by acquire, waited by present copy. Reused only after the
    /// previous copy retires. Binary (WSI doesn't support timeline semaphores).
    image_available: BinarySemaphore,
    /// Timeline value the slot's render submit signals; waited before reusing
    /// the slot. Seeded to `TimelineValue::START` so frame 0 doesn't block.
    render_value: TimelineValue,
    /// Timeline value the slot's present copy signals; waited before rendering
    /// into the slot the copy reads. Seeded to `TimelineValue::START`.
    copy_value: TimelineValue,
    imm: HostBuffer,
    indirect: HostBuffer,
    /// Depth valid with scene fingerprint; gates VRS reuse.
    vrs_ready: Option<u64>,
    /// Shadow map cleared to all-lit; gates shadow pass skip.
    shadow_lit_ready: bool,
    /// Which image holds the final HDR (offscreen or TAA history).
    hdr_source: HdrSource,
}

/// Token returned by `acquire_slot` proving the slot is safe to render into
/// (its copy hazard is resolved).
struct SlotGuard(usize);

/// Byte offsets within a frame's packed immediate buffer.
#[derive(Clone, Copy)]
struct ImmOffsets {
    line: u64,
    shadow: u64,
    d2: u64,
    d2_tex: u64,
}

/// Witness that HDR image is ready for present.
#[must_use = "the offscreen HDR must be finalized to SHADER_READ before present"]
pub(crate) struct HdrReadable {
    slot: usize,
}

impl HdrReadable {
    /// Mint the witness (only finalizers can create this).
    pub(in crate::vk) fn new(slot: usize) -> Self {
        HdrReadable { slot }
    }
}

/// 2D overlay draw parameters for present pass.
#[derive(Clone, Copy)]
struct OverlayPresent {
    d2_offset: u64,
    d2_count: u32,
    d2_tex_offset: u64,
    d2_tex_count: u32,
}

/// Resolved mesh draw (one direction-run or whole mesh), pre-sort scratch.
/// Placement/style live in the persistent record SSBOs, reached through
/// `slot`; this carries only what the CPU sort/batch needs.
#[derive(Clone, Copy)]
struct DrawEntry {
    buffer: vk::Buffer,
    pass: Pass,
    first: u32,
    count: u32,
    vertex_offset: i32,
    /// Mesh slot: the emitted command's `first_instance`, indexing the
    /// record/dyn SSBOs in the vertex shader.
    slot: u32,
    /// Squared distance to AABB center (monotonic; back-to-front sort key).
    dist2: f32,
}

/// Contiguous indirect commands sharing one buffer and pass.
#[derive(Clone, Copy)]
struct DrawRun {
    buffer: vk::Buffer,
    pass: Pass,
    first: u32,
    count: u32,
}

/// Applies sub-pixel jitter to the view-proj matrix. Jitter only exists at
/// record time; the returned matrix is consumed immediately and never stored.
fn jittered_clip(clean: glam::Mat4, jitter_px: glam::Vec2, extent: vk::Extent2D) -> glam::Mat4 {
    let ox = 2.0 * jitter_px.x / extent.width.max(1) as f32;
    let oy = -2.0 * jitter_px.y / extent.height.max(1) as f32;
    let t = glam::Mat4::from_cols(
        glam::Vec4::X,
        glam::Vec4::Y,
        glam::Vec4::Z,
        glam::Vec4::new(ox, oy, 0.0, 1.0),
    );
    t * clean
}

/// Get frame's sun direction, defaulting to up if absent.
fn sun_dir(lists: &DrawLists) -> glam::DVec3 {
    lists
        .scene
        .as_ref()
        .map(|s| s.frame_uniforms)
        .map(|u| {
            glam::DVec3::new(
                u.sun_dir_elev[0] as f64,
                u.sun_dir_elev[1] as f64,
                u.sun_dir_elev[2] as f64,
            )
        })
        .filter(|d| d.length_squared() > 1e-6)
        .unwrap_or(glam::DVec3::Y)
}

/// Hash depth-affecting inputs (view, visibility, draws). Gates VRS reuse.
fn scene_fingerprint(lists: &DrawLists, draws: &[DrawEntry], visible_mask: &[u32]) -> u64 {
    use ash::vk::Handle;
    use std::hash::{Hash, Hasher};

    let mut h = std::collections::hash_map::DefaultHasher::new();
    lists.scene.is_some().hash(&mut h);
    // Hash visibility mask (flipped bits invalidate reused depth).
    visible_mask.hash(&mut h);
    if let Some(scene) = &lists.scene {
        for c in scene.view_proj.to_cols_array() {
            c.to_bits().hash(&mut h);
        }
    }
    for d in draws {
        d.buffer.as_raw().hash(&mut h);
        // Slot represents placement (records are slot-tied).
        (d.pass as u8, d.first, d.count, d.vertex_offset, d.slot).hash(&mut h);
    }
    // Debug cubes affect depth, not color.
    for v in &lists.cube_verts {
        for c in v.pos {
            c.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

/// Minimap texture edge length in texels.
pub(crate) const MINIMAP_SIZE: u32 = 256;

pub(crate) struct Renderer {
    instance: InstanceBundle,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    device: Device,

    /// Mesh residency mirrors (render-side).
    mesh_res: MeshResidency,
    /// Freed allocation channel.
    ret: Sender<RenderReturn>,
    /// Swapchain size.
    size: vk::Extent2D,

    swapchain: Swapchain,
    targets: RenderTargets,
    pipelines: Pipelines,
    /// Pipeline cache.
    pipeline_cache: vk::PipelineCache,
    atlas: FontAtlas,
    block_textures: BlockTextures,
    /// Retired textures.
    retired_textures: buffers::RetireQueue<BlockTextures>,
    /// Minimap texture.
    minimap: MinimapTexture,

    /// Per-slot state (command buffer, sync, readiness bits).
    slots: PerSlot<SlotState>,
    /// Copy submit semaphores.
    /// Binary because the WSI rejects timeline semaphores.
    present_semaphores: Vec<BinarySemaphore>,
    /// Persistent per-mesh record/dyn SSBOs (slot-indexed via `first_instance`).
    pub(crate) records: buffers::RecordTable,
    /// The record/dyn/arena buffers flushed for the recording slot this frame;
    /// the mesh passes' descriptor pushes read them. `None` while no mesh exists.
    record_buffers: Option<buffers::RecordBuffers>,
    /// Dirty cache for shadow depth.
    shadow_cache: shadow::ShadowCache,
    /// GPU draw-command emission.
    cull: cull::CullState,
    /// Arena registry with live counts.
    arena_dir: cull::ArenaDirectory,
    /// Cull output for this frame; GPU sources opaque/cutout/shadow from here.
    cull_frame: Option<cull::CullFrame>,
    /// App visibility mask (gates GPU and CPU cull).
    visible_mask: Vec<u32>,
    /// Shared quad index buffer.
    quad_ibo: buffers::QuadIbo,
    /// 3D pipeline descriptor layout.
    mesh3d_set_layout: vk::DescriptorSetLayout,
    /// Per-frame uniforms.
    ubo_ring: uniforms::UboRing,
    /// Shadow pass.
    shadow: shadow::ShadowPass,
    /// Exposure metering.
    exposure: exposure::ExposureState,
    /// Bloom pipelines.
    bloom: bloom::BloomState,
    /// TAA state.
    taa: taa::TaaState,

    /// Resolved Blend draws (CPU path scratch).
    draw_scratch: Vec<DrawEntry>,
    /// Blend indirect commands.
    draw_commands: Vec<DrawIndexedIndirect>,
    /// Blend draw runs.
    draw_runs: Vec<DrawRun>,

    /// Feature flags.
    flags: crate::engine::RenderFlags,

    /// Present copy command buffer.
    copy_cmd: vk::CommandBuffer,
    /// Render and present timeline.
    timeline: Timeline,
    /// Transfer queue for staging copies.
    transfer_lane: TransferLane,
    /// Highest transfer-lane value submitted this frame.
    pending_transfer_wait: Option<TimelineValue>,
    /// Last present copy timeline value.
    last_copy_value: TimelineValue,
    /// Last render timeline value.
    last_render_value: TimelineValue,
    /// Offscreen slot for in-flight copy.
    copy_slot: Option<usize>,

    /// Pending screenshot/capture.
    pending_capture: Option<Capture>,

    slot: usize,

    /// Current scene fingerprint.
    scene_fingerprint: u64,

    vsync: Pending<bool>,
    msaa: Pending<SampleCount>,
    needs_recreate: bool,
    /// Render target scale relative to window.
    render_scale: Pending<f32>,
    /// Offscreen render extent.
    render_extent: vk::Extent2D,
    /// Last present time (for vsync-off pacing).
    last_present: std::time::Instant,
    present_interval: std::time::Duration,
    gpu_timer: GpuTimer,
}

/// Host-visible buffer for screenshot readback.
struct Readback {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl Renderer {
    /// Builds the renderer ON the render thread from the main-created instance +
    /// surface (a `!Send` window handle never crosses). Returns the renderer and
    /// the [`InitReply`] main uses to build its allocator. The window itself
    /// stays on main.
    pub(crate) fn build(
        instance: InstanceBundle,
        surface_loader: khr::surface::Instance,
        surface: vk::SurfaceKHR,
        cfg: RenderConfig,
        ret: Sender<RenderReturn>,
    ) -> (Self, InitReply) {
        let RenderConfig {
            vsync,
            msaa,
            render_scale,
            size: win_size,
            present_interval,
            flags,
        } = cfg;
        let render_scale = Scale::new(render_scale).as_f32();

        let device = Device::new(&instance.instance, &surface_loader, surface);
        let mut transfer_lane = unsafe {
            TransferLane::new(
                &device.device,
                device.transfer_family,
                device.transfer_queue,
                device.transfer_tier,
            )
        };
        log::info!(
            "transfer lane: {:?} (family {})",
            transfer_lane.tier(),
            transfer_lane.family(),
        );
        let mesh_res = MeshResidency::new();

        let size = vk::Extent2D {
            width: win_size.width,
            height: win_size.height,
        };
        let swapchain = Swapchain::new(
            &instance.instance,
            &device,
            &surface_loader,
            surface,
            size,
            vsync,
            vk::SwapchainKHR::null(),
        );

        let msaa = resolve_msaa(msaa, device.max_msaa(), "requested");
        let render_extent = scaled_extent(swapchain.extent, render_scale);
        let targets = RenderTargets::new(
            &instance.instance,
            &device.device,
            device.physical,
            render_extent,
            msaa,
            device.fragment_shading_rate.as_ref(),
        );

        let atlas = FontAtlas::new(
            &instance.instance,
            &device.device,
            device.physical,
            device.graphics_queue,
            device.graphics_family,
            device.command_pool,
            &mut transfer_lane,
        );

        // Default 1x1 white block texture array (before Pipelines::new: its
        // persistent set layout feeds layout_3d).
        let block_tex = BlockTextures::new_default(
            &instance.instance,
            &device.device,
            device.physical,
            device.graphics_queue,
            device.graphics_family,
            device.command_pool,
            &mut transfer_lane,
            device.anisotropy,
        );
        let mesh3d_set_layout = buffers::create_mesh3d_set_layout(
            &device.device,
            device.dynamic_rendering_local_read,
        );

        let minimap = MinimapTexture::new(
            &instance.instance,
            &device.device,
            device.physical,
            device.graphics_queue,
            device.command_pool,
            MINIMAP_SIZE,
            crate::color::Color::BLACK,
        );

        let pipeline_cache = create_pipeline_cache(&device.device);
        let pipelines = Pipelines::new(
            &device.device,
            pipeline_cache,
            targets.color_format,
            swapchain.format,
            targets.depth_format,
            targets.samples,
            atlas.set_layout,
            mesh3d_set_layout,
            device.fragment_shading_rate.as_ref(),
            device.dynamic_rendering_local_read,
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
        let timeline = unsafe { Timeline::new(&device.device) };
        let mut cmds = cmds.into_iter();
        let slots = PerSlot::new(std::array::from_fn(|_| SlotState {
            cmd: cmds.next().expect("per-slot command buffer"),
            image_available: unsafe { BinarySemaphore::new(&device.device) },
            render_value: TimelineValue::START,
            copy_value: TimelineValue::START,
            imm: HostBuffer::new(vk::BufferUsageFlags::VERTEX_BUFFER),
            indirect: HostBuffer::new(vk::BufferUsageFlags::INDIRECT_BUFFER),
            vrs_ready: None,
            shadow_lit_ready: false,
            hdr_source: HdrSource::Offscreen,
        }));

        let present_semaphores = create_present_semaphores(&device.device, swapchain.images.len());
        let ubo_ring = uniforms::UboRing::new(&instance.instance, &device.device, device.physical);

        let shadow = shadow::ShadowPass::new(
            &instance.instance,
            &device.device,
            device.physical,
            pipeline_cache,
            pipelines.layout_3d,
            pipelines.layout_debug,
        );
        let memory_props = unsafe {
            instance
                .instance
                .get_physical_device_memory_properties(device.physical)
        };
        let exposure = exposure::ExposureState::new(
            &device.device,
            &memory_props,
            render_extent,
            pipeline_cache,
        );
        let taa = taa::TaaState::new(&device.device, &memory_props, render_extent, pipeline_cache);
        let bloom = bloom::BloomState::new(&device.device, pipeline_cache);

        let gpu_timer = GpuTimer::new(
            &device.device,
            device.timestamps_supported,
            device.timestamp_period_ns,
        );

        let caps = DeviceCaps {
            max_msaa: device.max_msaa(),
            max_texture_layers: device.max_image_array_layers,
        };
        let reply = InitReply {
            instance: instance.instance.clone(),
            physical: device.physical,
            memory_budget: device.memory_budget,
            device: device.device.clone(),
            caps,
            exposure: exposure.shared(),
        };

        let cull = cull::CullState::new(&device.device, pipeline_cache);
        // GPU-driven emission: opaque/cutout/shadow draws are always emitted
        // by the cull dispatch, so the device must support drawIndirectCount.
        // Device selection enforces this; this assert makes mis-selection fail
        // loudly here rather than silently mis-render.
        assert!(
            device.draw_indirect_count,
            "GPU cull requires drawIndirectCount; device selection must require it"
        );
        let renderer = Self {
            instance,
            surface_loader,
            surface,
            device,
            mesh_res,
            ret,
            size,
            swapchain,
            targets,
            pipelines,
            pipeline_cache,
            atlas,
            block_textures: block_tex,
            retired_textures: buffers::RetireQueue::new(),
            minimap,
            slots,
            present_semaphores,
            records: buffers::RecordTable::new(),
            record_buffers: None,
            shadow_cache: shadow::ShadowCache::new(),
            cull,
            arena_dir: cull::ArenaDirectory::new(),
            cull_frame: None,
            visible_mask: Vec::new(),
            quad_ibo: buffers::QuadIbo::new(),
            mesh3d_set_layout,
            ubo_ring,
            shadow,
            exposure,
            bloom,
            taa,
            draw_scratch: Vec::new(),
            flags,
            draw_commands: Vec::new(),
            draw_runs: Vec::new(),
            copy_cmd,
            timeline,
            transfer_lane,
            pending_transfer_wait: None,
            last_copy_value: TimelineValue::START,
            last_render_value: TimelineValue::START,
            copy_slot: None,
            pending_capture: None,
            slot: 0,
            scene_fingerprint: 0,
            vsync: Pending::new(vsync),
            msaa: Pending::new(msaa),
            needs_recreate: false,
            render_scale: Pending::new(render_scale),
            render_extent,
            last_present: std::time::Instant::now(),
            present_interval,
            gpu_timer,
        };
        (renderer, reply)
    }

    /// Handle window resize and flag swapchain rebuild.
    pub(crate) fn on_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.size = vk::Extent2D {
            width: size.width,
            height: size.height,
        };
        self.needs_recreate = true;
    }

    // Setters driven by RenderCmd; getters cached main-side in RenderClient.

    pub fn set_vsync(&mut self, on: bool) {
        if self.vsync.set(on) {
            self.needs_recreate = true;
        }
    }

    pub fn set_msaa(&mut self, samples: u32) -> u32 {
        let resolved = resolve_msaa(samples, self.device.max_msaa(), "set_msaa");
        if self.msaa.set(resolved) {
            self.needs_recreate = true;
        }
        resolved.as_u32()
    }

    /// Get mesh3d pipeline for the pass.
    fn mesh_pipeline_for(&self, pass: Pass) -> vk::Pipeline {
        self.pipelines.pipeline_for(pass)
    }

    /// Replace feature flags (safe mid-run).
    pub fn set_flags(&mut self, flags: crate::engine::RenderFlags) {
        // Flag transitions reset temporal state to avoid stale cached values.
        if self.flags.exposure && !flags.exposure {
            self.exposure.reset();
        } else if !self.flags.exposure && flags.exposure {
            self.exposure.rearm();
        }
        if self.flags.taa != flags.taa {
            self.taa.invalidate_history();
        }
        self.flags = flags;
    }

    /// Set render scale; returns clamped value.
    pub fn set_render_scale(&mut self, scale: f32) -> f32 {
        let clamped = Scale::new(scale).get();
        if (clamped - self.render_scale.effective()).abs() > f32::EPSILON {
            self.render_scale.queue(clamped);
            self.needs_recreate = true;
        }
        clamped
    }

    /// Installs a main-built mesh resident into the residency mirror (from the
    /// ordered command stream). Identity/meta live on main; the mirror carries
    /// only the device buffer + staged copy.
    pub(crate) fn apply_upload_mesh(
        &mut self,
        slot: u32,
        generation: NonZeroU32,
        quads: u32,
        resident: GpuResident,
        record: buffers::MeshRecord,
    ) {
        // Grow the shared quad IBO to index this mesh before its draws record.
        self.quad_ibo.require(quads);
        self.arena_dir
            .note_upload(slot, generation, resident.buffer(), record.pass());
        self.mesh_res.apply_upload(slot, generation, resident);
        self.records.install(slot, record);
    }

    /// Set one word of the visibility mask.
    pub(crate) fn set_visible_word(&mut self, word: u32, bits: u32) {
        let i = word as usize;
        if self.visible_mask.len() <= i {
            self.visible_mask.resize(i + 1, 0);
        }
        self.visible_mask[i] = bits;
    }

    /// Retire a freed mesh resident.
    pub(crate) fn apply_free_mesh(&mut self, slot: u32, generation: NonZeroU32) {
        self.arena_dir.note_free(slot, generation);
        self.records.clear_arena(slot);
        self.mesh_res
            .apply_free(slot, generation, self.last_render_value);
    }

    /// Queue screenshot capture to path.
    pub fn request_capture(&mut self, capture: Capture) {
        // Surface without TRANSFER_SRC cannot screenshot.
        if !self.swapchain.screenshot_capable {
            log::error!("screenshot refused: surface lacks TRANSFER_SRC swapchain usage");
            if let Some(reply) = capture.reply {
                let _ = reply.send(Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "surface does not support screenshot copies",
                )));
            }
            return;
        }
        if let Some(prev) = self.pending_capture.replace(capture) {
            if let Some(reply) = prev.reply {
                let _ = reply.send(Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "capture superseded by a newer request",
                )));
            }
        }
    }

    /// Replace block texture array; old one retired through timeline.
    pub fn set_block_textures(&mut self, size: u32, layers: &[Vec<u8>]) {
        // Clamp to device's max image array layers.
        let cap = self.device.max_image_array_layers as usize;
        let layers = if layers.len() > cap {
            log::error!(
                "set_block_textures: {} layers exceeds the device cap of {cap}; truncating",
                layers.len()
            );
            &layers[..cap]
        } else {
            layers
        };
        // Build before swap to avoid double-free on panic.
        let new_textures = BlockTextures::upload(
            &self.instance.instance,
            &self.device.device,
            self.device.physical,
            self.device.graphics_queue,
            self.device.graphics_family,
            self.device.command_pool,
            &mut self.transfer_lane,
            self.device.anisotropy,
            size,
            layers,
        );
        let old_textures = std::mem::replace(&mut self.block_textures, new_textures);
        // Old array may be sampled by in-flight frames; retire past max timeline.
        let done_at = self.timeline.last_reserved();
        self.retired_textures.push(done_at, old_textures);
        log::debug!(
            "block textures swapped: {} layers of {}x{}",
            self.block_textures.layers,
            self.block_textures.size,
            self.block_textures.size,
        );
    }

    /// Uploads minimap pixels to staging buffer (synced per-slot).
    pub fn update_minimap(&mut self, rgba: &[u8]) {
        self.minimap.update(rgba);
    }

    /// Records and submits one frame from the recorded draw lists, and
    /// presents it when the presentation engine can keep up (manual
    /// mailbox: frames that outrun presentation are rendered but dropped).
    ///
    /// Frame anatomy, top-down (each phase is its own helper below and its
    /// own [`crate::profile`] meter):
    /// 1. [`Self::wait_slot_and_reclaim`] — frame fence + deferred frees
    /// 2. [`Self::decide_present`]        — copy-fence check + acquire
    /// 3. [`Self::write_immediates`]      — pack cube/line/2D verts
    /// 4. [`Self::record_render`]         — barriers, rendering, draws
    /// 5. [`Self::submit_render`]         — render queue submit (fence)
    /// 6. [`Self::present`]               — copy submit + queue_present
    pub(crate) fn draw_frame(&mut self, lists: &DrawLists) {
        let size = self.size;
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
        use crate::profile::{Meter, scope};

        {
            let _p = scope(Meter::Fence);
            self.wait_slot_and_reclaim(slot);
        }

        let present_target;
        let guard = {
            let _p = scope(Meter::Acquire);
            present_target = self.decide_present(slot);
            self.acquire_slot(slot)
        };

        let offsets = {
            let _p = scope(Meter::Pack);
            let offsets = self.write_immediates(slot, lists);
            self.prepare_mesh_draws(slot, lists);
            offsets
        };

        // Per-frame UBO (set 0, binding 2). A 3D scene always carries lighting
        // (`Frame::begin_3d` takes it as a required `Lighting` argument, so
        // `Scene3D::frame_uniforms` is never optional). This `None` branch is
        // therefore reached ONLY by pure-2D frames, where the mesh shaders never
        // sample the block; the full-bright filler just keeps the binding live
        // and validated.
        {
            let mut u = lists
                .scene
                .as_ref()
                .map(|s| s.frame_uniforms)
                .unwrap_or_else(crate::skeleton::FrameUniformsGpu::full_bright);
            // Debug-flat: claim the `extras` lane as [r, g, b, enabled] —
            // sRGB-encoded key channels + an enable flag. mesh3d.frag linearises rgb
            // (as it does every CPU colour) and outputs it flat while depth writes.
            // Overwriting `extras.x` (the stars gain) is safe: the lane's only
            // other consumer is sky.frag, and the app never draws the sky in the
            // debug-flat (TerrainKey) view.
            if let Some(c) = lists.debug_flat {
                u.extras = [
                    c.r as f32 / 255.0,
                    c.g as f32 / 255.0,
                    c.b as f32 / 255.0,
                    1.0,
                ];
            }
            self.ubo_ring.write(FrameSlot::new(slot), &u);
        }
        let (rs, hdr_readable) = {
            let _p = scope(Meter::Record);
            self.record_render(&guard, lists, offsets)
        };

        {
            let _p = scope(Meter::Submit);
            self.submit_render(rs, slot);
        }

        {
            let _p = scope(Meter::Present);
            // In wide-FOV the overlay was skipped in the scene pass; draw it here,
            // after the tonemap resample, so it stays crisp and unwarped.
            let overlay = OverlayPresent {
                d2_offset: offsets.d2,
                d2_count: lists.verts_2d.len() as u32,
                d2_tex_offset: offsets.d2_tex,
                d2_tex_count: lists.tex_verts_2d.len() as u32,
            };
            // Project the sun to presented uv for the tonemap godray march.
            // Computed here (not in the copy submit) so it rides the same camera +
            // frame-uniform snapshot the scene was drawn from. `project` returns a
            // strength-0 no-op when godrays are off, the sun is behind the camera,
            // or there is no 3D camera this frame.
            let godray = match lists.scene.as_ref() {
                Some(scene) => {
                    let cam = scene.camera;
                    let u = scene.frame_uniforms;
                    let (sun_dir, tint) = (
                        glam::Vec3::new(u.sun_dir_elev[0], u.sun_dir_elev[1], u.sun_dir_elev[2]),
                        [u.light[0], u.light[1], u.light[2]],
                    );
                    crate::camera::Godray::project(
                        // The tonemap shader samples single-sample depth; under
                        // MSAA that is the resolve target (`sampleable_depth`), so
                        // godrays are gated only on the feature flag now.
                        self.flags.godrays,
                        sun_dir,
                        tint,
                        &cam,
                        self.size.width as f32,
                        self.size.height as f32,
                        [
                            scene.jitter.0.x / self.render_extent.width as f32,
                            scene.jitter.0.y / self.render_extent.height as f32,
                        ],
                    )
                }
                None => crate::camera::Godray::OFF,
            };
            let warp_map = lists
                .scene
                .as_ref()
                .map_or(crate::camera::WarpMap::Identity, |s| s.warp_map);
            self.present(
                slot,
                present_target,
                warp_map,
                overlay,
                hdr_readable,
                godray,
            );
        }

        self.slot = (self.slot + 1) % FRAMES_IN_FLIGHT as usize;
    }

    /// Tracks which offscreen slot the current copy is reading from.
    fn track_copy(&mut self, slot: usize) {
        self.copy_slot = Some(slot);
    }

    /// Forgets any tracked copy hazard: the copy has been waited to
    /// completion, or the offscreen images it read no longer exist.
    fn clear_copy(&mut self) {
        self.copy_slot = None;
    }

    /// Waits until the slot's last render has completed (GPU is done with its
    /// command buffer and immediate buffer), then reclaims retired GPU memory
    /// whose last possible use the timeline has reached.
    /// The layout the scene-pass depth image lives in for the WHOLE frame.
    /// With the water-absorption path active it is `RENDERING_LOCAL_READ` —
    /// the one layout valid simultaneously as depth attachment and as the
    /// blend pass's input attachment (mid-pass transitions are illegal, so a
    /// single frame-wide layout is the only coherent design). Every depth
    /// barrier and attachment info reads this ONE function, so the two
    /// configurations cannot drift apart.
    fn depth_pass_layout(&self) -> vk::ImageLayout {
        if self.pipelines.mesh3d_transparent_absorb.is_some() {
            vk::ImageLayout::RENDERING_LOCAL_READ_KHR
        } else {
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
        }
    }

    fn wait_slot_and_reclaim(&mut self, slot: usize) {
        let device = &self.device.device;
        unsafe {
            self.timeline
                .wait(device, self.slots[FrameSlot::new(slot)].render_value);
            let current = self.timeline.counter(device);
            // Retired allocations return to the main-owned allocator freelist;
            // staging-block shrink happens main-side after it reclaims them.
            let ret = &self.ret;
            self.mesh_res
                .collect(current, &mut |a| drop(ret.send(RenderReturn::FreeAlloc(a))));
            self.retired_textures
                .collect(current, |mut tex| tex.destroy(device));
            // Superseded quad IBO buffers are render-owned raw buffers (not
            // allocator suballocations), so destroy them here rather than shipping
            // them back to main's freelist.
            self.quad_ibo.collect(device, current);
            // Staging submitted on a separate transfer queue is
            // reclaimed against the LANE's own timeline, not the render
            // one — a non-blocking probe (never waited: nothing here may
            // stall this reclaim pass on the transfer queue's progress).
            if let Some(transfer_current) = self.transfer_lane.counter(device) {
                self.mesh_res.collect_transfer(transfer_current, &mut |a| {
                    drop(ret.send(RenderReturn::FreeAlloc(a)))
                });
                self.quad_ibo.collect_transfer(device, transfer_current);
            }
        }
    }

    /// Resolves the copy hazard on `slot` before it is rendered into: the
    /// in-flight present copy may still be reading this slot's offscreen
    /// image, which the render below overwrites. Rare (the copy usually
    /// retires well within the two-frame slot cycle) and sub-millisecond.
    /// Returns a guard proving the slot is safe to record into.
    fn acquire_slot(&mut self, slot: usize) -> SlotGuard {
        if self.copy_slot == Some(slot) {
            let device = &self.device.device;
            unsafe {
                self.timeline
                    .wait(device, self.slots[FrameSlot::new(slot)].copy_value)
            };
            self.copy_slot = None;
        }
        SlotGuard(slot)
    }

    /// Present eligibility, decided before the render submit. Strict ordering:
    /// the previous copy's completion is probed first (non-blocking, so the
    /// mailbox drop never stalls), the acquire is only attempted once we know a
    /// copy can be submitted, and a successful acquire is ALWAYS followed by
    /// the copy + present in [`Self::present`] — never skipped.
    fn decide_present(&mut self, slot: usize) -> Option<u32> {
        // With vsync off, throttle presents to refresh cadence to avoid blocking
        // on drawable availability; instead render frames in between unthrottled.
        // Present slightly ahead of the refresh interval so scheduling jitter
        // never pushes a present past the drawable's availability window.
        const PRESENT_THROTTLE: f32 = 0.9;
        // A pending capture makes this present MANDATORY: dropping it (throttle,
        // copy-in-flight mailbox skip, or a non-blocking acquire miss) would
        // silently discard the frame the caller asked to capture. So a forced
        // present ignores the throttle, WAITS for the prior copy to retire
        // instead of skipping, and blocks for a drawable.
        let force = self.pending_capture.is_some();
        let present_due = force
            || self.vsync.current()
            || self.last_present.elapsed() >= self.present_interval.mul_f32(PRESENT_THROTTLE);
        let mut present_target = None;
        let device = &self.device.device;
        unsafe {
            if force {
                // Wait out the prior copy rather than treating it as a drop.
                self.timeline.wait(device, self.last_copy_value);
            }
            // Skip present if previous copy still in flight (mailbox drop).
            let copy_ready =
                force || (present_due && self.timeline.probe(device).reached(self.last_copy_value));
            if copy_ready {
                // With vsync or a forced capture: wait for an image. Plain
                // vsync-off: never wait, allow drop.
                let timeout = if self.vsync.current() || force {
                    u64::MAX
                } else {
                    0
                };
                match acquire_next_image(
                    &self.swapchain.loader,
                    self.swapchain.swapchain,
                    timeout,
                    self.slots[FrameSlot::new(slot)].image_available,
                ) {
                    Ok((image_index, suboptimal)) => {
                        if suboptimal {
                            self.needs_recreate = true;
                        }
                        present_target = Some(image_index);
                    }
                    // No image available; drop the present.
                    Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {}
                    // OUT_OF_DATE/SURFACE_LOST: environmental, recreate next frame.
                    // Other errors are unrecoverable.
                    Err(err) => match Env::classify(err) {
                        Some(Env::OutOfDate | Env::SurfaceLost) => self.needs_recreate = true,
                        _ => panic!("acquire_next_image failed: {err:?}"),
                    },
                }
            }
        }
        present_target
    }

    /// Packs frame immediates (cubes, lines, 2D) into host buffer and returns offsets.
    fn write_immediates(&mut self, slot: usize, lists: &DrawLists) -> ImmOffsets {
        let cube_bytes: &[u8] = bytemuck::cast_slice(&lists.cube_verts);
        let line_bytes: &[u8] = bytemuck::cast_slice(&lists.line_verts);
        let shadow_bytes: &[u8] = bytemuck::cast_slice(&lists.shadow_verts);
        let d2_bytes: &[u8] = bytemuck::cast_slice(&lists.verts_2d);
        let d2_tex_bytes: &[u8] = bytemuck::cast_slice(&lists.tex_verts_2d);
        let line = (cube_bytes.len() as u64).next_multiple_of(16);
        let shadow = (line + line_bytes.len() as u64).next_multiple_of(16);
        let d2 = (shadow + shadow_bytes.len() as u64).next_multiple_of(16);
        let d2_tex = (d2 + d2_bytes.len() as u64).next_multiple_of(16);
        let total = d2_tex + d2_tex_bytes.len() as u64;
        let imm = &mut self.slots[FrameSlot::new(slot)].imm;
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
                imm.write(shadow, shadow_bytes);
                imm.write(d2, d2_bytes);
                imm.write(d2_tex, d2_tex_bytes);
            }
        }
        ImmOffsets {
            line,
            shadow,
            d2,
            d2_tex,
        }
    }

    /// Prepares this frame's two draw sources from the persistent state.
    ///
    /// - GPU cull (opaque/cutout/shadow): exact partitions from the arena live
    ///   counts, params (camera + cascade frusta, eye split), and the persistent
    ///   `visible_mask` as the dispatch's visibility input. The mask carries what
    ///   only the app knows (LOD selection, quadrant masks, occlusion): a
    ///   resident-but-hidden slot must not draw just because its record is live.
    ///   The cull shader frustum-tests every visible slot and appends
    ///   `first_instance = slot` commands the graphics side draws indirect-count.
    ///
    /// - CPU Blend re-source: transparency needs exact far→near ordering the GPU
    ///   cull does not provide, so Blend is the ONE pass still resolved CPU-side.
    ///   It is sourced from the SAME persistent state — records + arena directory
    ///   + `visible_mask`, NOT a per-frame draw list — by iterating the resident,
    ///   visible, Blend-pass slots, frustum-culling, sorting by distance, and
    ///   emitting whole-mesh indirect commands (also `first_instance = slot`, so
    ///   placement/style come from the record/dyn SSBOs, never rebuilt per frame).
    fn prepare_mesh_draws(&mut self, slot: usize, lists: &DrawLists) {
        use ash::vk::Handle;

        self.draw_scratch.clear();
        self.draw_commands.clear();
        self.draw_runs.clear();

        // Flush record/dyn patches into this slot's copies (post-fence, same
        // discipline as the HostBuffer maintains below).
        self.record_buffers = unsafe {
            self.records.flush(
                slot,
                &self.arena_dir,
                &self.mesh_res,
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
            )
        };

        // The cull dispatch owns the shadow set, so these cascade frusta feed
        // its params.
        let shadow_frusta = lists
            .scene
            .as_ref()
            .filter(|_| self.flags.shadows)
            .map(|scene| {
                let cfg = crate::skeleton::ShadowCfg::PROVISIONAL;
                let sun = sun_dir(lists);
                [
                    crate::skeleton::Cascade::Near,
                    crate::skeleton::Cascade::Far,
                ]
                .map(|c| {
                    crate::camera::Frustum::from_view_proj(
                        &shadow::fit(scene.eye, sun, c, &cfg).view_proj.0,
                    )
                })
            });

        // GPU cull prep: the persistent visibility mask IS the dispatch's
        // visibility input, grown to cover every live slot (zero = hidden).
        self.cull_frame = if let Some(scene) = &lists.scene {
            let camera = crate::camera::Frustum::from_view_proj(&scene.view_proj);
            let eye = pipeline::EyeSplit::of(scene.eye);
            if let Some(records) = self.record_buffers {
                let need = records.slots.div_ceil(32) as usize;
                if self.visible_mask.len() < need {
                    self.visible_mask.resize(need, 0);
                }
                unsafe {
                    self.cull.prepare(
                        slot,
                        &self.instance.instance,
                        &self.device.device,
                        self.device.physical,
                        &self.arena_dir,
                        records,
                        &camera,
                        shadow_frusta.as_ref(),
                        eye,
                        &self.visible_mask,
                    )
                }
            } else {
                None
            }
        } else {
            None
        };

        // CPU Blend re-source: walk the resident, visible, Blend-pass records.
        if let Some(scene) = &lists.scene {
            let camera = crate::camera::Frustum::from_view_proj(&scene.view_proj);
            let eye = pipeline::EyeSplit::of(scene.eye);
            let slot_count = self.record_buffers.map_or(0, |r| r.slots);
            for s in 0..slot_count {
                // Arena word (0 = not resident) is the arena index + 1, giving
                // the vertex buffer without a residency-handle lookup. Gated
                // on `is_arrived` too: a budget-deferred copy is registered in
                // `arena_dir` (capacity/ref-count bookkeeping happens at
                // upload) before its bytes actually land — reading it here
                // early would source the CPU Blend draw from uninitialized
                // arena memory, same hazard `RecordTable::flush` guards for
                // the GPU cull path.
                let arena = self.arena_dir.arena_word(s as usize);
                if arena == 0 || !self.mesh_res.is_arrived(s) {
                    continue;
                }
                // The same persistent mask the GPU cull reads.
                let visible = self
                    .visible_mask
                    .get((s >> 5) as usize)
                    .is_some_and(|w| w & (1 << (s & 31)) != 0);
                if !visible {
                    continue;
                }
                let Some(rec) = self.records.record(s) else {
                    continue;
                };
                if rec.pass() != Pass::Blend {
                    continue;
                }
                // Camera-relative placement reconstructed exactly as the vertex
                // shader does (integer block minus camera block, then the
                // fractional remainder), so the CPU sort/cull agrees with the GPU
                // draw. `detail_scale` decodes the BIASED detail field — never
                // decode `detail_pass` here by hand (it carries a to_gpu_bits offset).
                let scale = rec.detail_scale();
                let offset = glam::Vec3::new(
                    (rec.block[0] - eye.block[0]) as f32 - eye.frac[0] + rec.local_off[0],
                    (rec.block[1] - eye.block[1]) as f32 - eye.frac[1] + rec.local_off[1],
                    (rec.block[2] - eye.block[2]) as f32 - eye.frac[2] + rec.local_off[2],
                );
                let amin = glam::Vec3::from(rec.aabb_min);
                let amax = glam::Vec3::from(rec.aabb_max);
                if !camera.intersects_aabb(amin * scale + offset, amax * scale + offset) {
                    continue;
                }
                let center = offset + (amin + amax) * 0.5 * scale;
                let dist2 = (center - scene.cam_pos).length_squared();
                self.draw_scratch.push(DrawEntry {
                    buffer: self.arena_dir.arena_buffer((arena - 1) as usize),
                    pass: Pass::Blend,
                    first: 0,
                    count: rec.index_count,
                    vertex_offset: rec.vertex_offset,
                    slot: s,
                    dist2,
                });
            }
            // Blend far→near for correct back-to-front alpha compositing.
            self.draw_scratch.sort_unstable_by(|a, b| {
                b.dist2
                    .total_cmp(&a.dist2)
                    // Deterministic tiebreak; keeps equidistant same-arena draws batched.
                    .then_with(|| a.buffer.as_raw().cmp(&b.buffer.as_raw()))
            });

            for entry in &self.draw_scratch {
                let command_index = self.draw_commands.len() as u32;
                self.draw_commands.push(DrawIndexedIndirect {
                    index_count: entry.count,
                    instance_count: 1,
                    first_index: entry.first,
                    vertex_offset: entry.vertex_offset,
                    first_instance: entry.slot,
                });
                match self.draw_runs.last_mut() {
                    Some(run) if run.buffer == entry.buffer && run.pass == entry.pass => {
                        run.count += 1
                    }
                    _ => self.draw_runs.push(DrawRun {
                        buffer: entry.buffer,
                        pass: entry.pass,
                        first: command_index,
                        count: 1,
                    }),
                }
            }
        }
        // The fingerprint exists only so VRS can tell whether a slot's stored
        // depth still matches the scene it will classify; with VRS off nothing
        // reads it, so skip the per-frame hash.
        self.scene_fingerprint = if self.targets.vrs.is_some() {
            scene_fingerprint(lists, &self.draw_scratch, &self.visible_mask)
        } else {
            0
        };

        let indirect_bytes: &[u8] = bytemuck::cast_slice(&self.draw_commands);
        unsafe {
            let indirect = &mut self.slots[FrameSlot::new(slot)].indirect;
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
        crate::profile::gauge(
            crate::profile::Gauge::DrawsPacked,
            self.draw_commands.len() as u64,
        );
    }

    /// Records the command buffer: mesh copies, render pass, and transitions.
    fn record_render(
        &mut self,
        guard: &SlotGuard,
        lists: &DrawLists,
        offsets: ImmOffsets,
    ) -> (RenderSubmit, HdrReadable) {
        let slot = guard.0;
        let cmd = self.slots[FrameSlot::new(slot)].cmd;
        // Read the prior render-pass GPU time for this slot before its queries
        // are reset below (the slot's fence was already waited this frame).
        let profiling = crate::profile::is_enabled();
        if profiling {
            let mut passes = [0.0f64; GpuPass::COUNT];
            if unsafe {
                self.gpu_timer
                    .read_into(&self.device.device, slot, &mut passes)
            }
            .is_some()
            {
                for pass in GpuPass::ALL {
                    crate::profile::add_ms(pass.meter(), passes[pass as usize]);
                }
            }
        }
        // Begin render submission; this gets the timeline value to stamp mesh copies.
        let rs = self.timeline.begin_render(cmd);
        let done_at = rs.value();
        unsafe {
            let device = &self.device.device;
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("command buffer reset failed");
            device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .expect("begin command buffer failed");

            self.pending_transfer_wait = self.mesh_res.flush_copies(
                device,
                &mut self.transfer_lane,
                cmd,
                self.device.graphics_family,
                done_at,
            );
            // Slots whose copy just submitted: their arena word was gated to
            // 0 in every RecordTable copy until now (see `RecordTable::flush`);
            // re-mark them dirty so the NEXT prepare_mesh_draws re-reads
            // `is_arrived` and exposes the real word (this frame's cull dispatch
            // already ran, upstream of this flush — a one-frame-late reveal,
            // never early).
            let arrived = self.mesh_res.take_arrived();
            self.records.mark_arrived(&arrived);
            // Grow the shared quad IBO (if a bigger mesh arrived) before any draw
            // indexes it; ordered either by its own barrier (same-queue tiers)
            // or the lane wait folded into `pending_transfer_wait` below —
            // exactly like the mesh staging copies above.
            let quad_wait = self.quad_ibo.ensure(
                &self.instance.instance,
                device,
                self.device.physical,
                &mut self.transfer_lane,
                cmd,
                self.device.graphics_family,
                done_at,
            );
            self.pending_transfer_wait = match (self.pending_transfer_wait, quad_wait) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            // Upload this slot's minimap texture (if its version is stale) on the
            // live frame command buffer, before the render pass begins.
            self.minimap.sync(device, cmd, slot);
            if profiling {
                self.gpu_timer.begin(&self.device.device, cmd, slot);
            }
        }

        // GPU cull: emit this frame's opaque/cutout/shadow draw commands from
        // the persistent record set, BEFORE any pass that consumes them (the
        // shadow occluders below and the mesh passes). Outside any rendering
        // scope; its trailing barrier orders the writes against DRAW_INDIRECT.
        if lists.scene.is_some()
            && let Some(frame) = &self.cull_frame
            && let Some(records) = self.record_buffers
        {
            let _g = crate::profile::scope(crate::profile::Meter::Pack);
            unsafe {
                self.cull.record(
                    &self.device.device,
                    &self.device.push_descriptor,
                    cmd,
                    slot,
                    records,
                    frame,
                );
            }
        }

        // Cascaded shadows: fit both cascades around this frame's frustum,
        // publish the binding-3 uniforms the receiver samples, and render the
        // occluders into the shadow map BEFORE the color pass (it leaves the map
        // in SHADER_READ_ONLY_OPTIMAL for mesh3d.frag's PCF). The mesh pass always
        // samples binding 4, so this must run whenever a 3D scene exists.
        if let Some(scene) = &lists.scene {
            // With shadows disabled the map holds a constant fully-lit clear and
            // its cascade UBO a constant (SHADOW_LIMIT=∞) block, so once this
            // slot has primed both there is nothing left to re-record: skip the
            // per-frame clears + barriers (the bulk of the off-path cost). Any
            // shadowed frame re-runs the pass and re-arms every slot.
            let cfg = crate::skeleton::ShadowCfg::PROVISIONAL;
            let sun = sun_dir(lists);
            // Off: the map holds a constant fully-lit clear; prime each slot once
            // then skip forever (unchanged fast path). The cached generation is
            // meaningless while off, so re-render both slots on re-enable.
            // On: additionally gate the re-render on the dirty cache — regenerate
            // only when the sun/eye-snap/occluders actually shifted the depth.
            // Avatar boxes cast into the shadow map; hash their geometry so the
            // cache regenerates on any motion (and stays cached when still).
            let caster_verts = lists.cube_verts.len() as u32;
            let casters = {
                use std::hash::{Hash, Hasher};
                let mut h = std::hash::DefaultHasher::new();
                bytemuck::cast_slice::<_, u8>(&lists.cube_verts).hash(&mut h);
                h.finish()
            };
            let render = if self.flags.shadows {
                let key = shadow::ShadowKey::of(
                    scene.eye,
                    sun,
                    self.records.occluder_rev(),
                    casters,
                    &cfg,
                );
                self.shadow_cache.take_render(slot, key, &cfg)
            } else {
                self.shadow_cache.invalidate();
                !self.slots[FrameSlot::new(slot)].shadow_lit_ready
            };
            if render {
                let _g = crate::profile::scope(crate::profile::Meter::RecShadow);
                // Write the UBO only on a regenerating frame so the receiver's
                // sampled cascade matrices always match the depth actually in the
                // (possibly cached-from-an-earlier-frame) image.
                let cu = self.shadow_uniforms(scene.eye, sun, &cfg);
                self.shadow.write_uniforms(slot, &cu);
                self.record_shadow_pass(cmd, slot, scene.eye, sun, &cfg, caster_verts);
                self.slots[FrameSlot::new(slot)].shadow_lit_ready = !self.flags.shadows;
            }
        }

        // VRS generation needs a validly-written, single-sampled depth image to
        // classify. MSAA depth would need multisample sampling (skipped), and a
        // slot's depth is only readable once it has been rendered at least once.
        let do_vrs = lists.scene.is_some()
            && self.flags.vrs
            && self.targets.vrs.is_some()
            && self.slots[FrameSlot::new(slot)].vrs_ready == Some(self.scene_fingerprint);

        let device = &self.device.device;
        let stamp = |p| {
            if profiling {
                unsafe { self.gpu_timer.mark(device, cmd, slot, p) };
            }
        };
        // A fresh scene render makes the offscreen the frame's HDR again; the
        // TAA pass overrides this if it runs (set before `begin` borrows self
        // shared for the whole pass).
        self.slots[FrameSlot::new(slot)].hdr_source = HdrSource::Offscreen;
        let pass = {
            let _g = crate::profile::scope(crate::profile::Meter::RecTransitions);
            unsafe { RenderPass::begin(self, cmd, slot, lists, offsets, do_vrs) }
        };
        if lists.scene.is_some() {
            use crate::profile::{Meter, scope};
            // Transparency forces an interleave: all opaque geometry (mesh runs
            // AND opaque debug cubes/lines) writes depth before any transparent
            // mesh run tests against it.
            unsafe {
                {
                    let _g = scope(Meter::RecMesh);
                    pass.record_mesh_indirect(Pass::Opaque);
                    // Cutout writes depth like opaque, so it belongs in the opaque
                    // prefix (before sky). Dormant until a block emits it.
                    pass.record_mesh_indirect(Pass::Cutout);
                }
                stamp(GpuPass::Opaque);
                // Sky fills the background (uncovered pixels) right after opaque
                // depth is laid down. It must precede the immediate debug
                // cubes/lines: the highlight lines are depth read-only (no depth
                // write), so a line silhouetted against the background leaves the
                // depth cleared there — drawing sky afterward would overpaint it.
                // Debug geometry and transparent water both composite over the sky.
                {
                    let _g = scope(Meter::RecSky);
                    if self.flags.sky {
                        pass.record_sky();
                    }
                }
                stamp(GpuPass::Sky);
                {
                    let _g = scope(Meter::RecImmediate);
                    pass.record_immediate_cubes();
                    stamp(GpuPass::Cubes);
                    pass.record_lines();
                    stamp(GpuPass::Lines);
                    // Contact shadows: translucent, blended over the opaque terrain
                    // depth just laid down, before transparent water.
                    pass.record_shadows();
                }
                stamp(GpuPass::Shadows);
                {
                    let _g = scope(Meter::RecMesh);
                    pass.record_mesh_indirect(Pass::Blend);
                }
                stamp(GpuPass::Transparent);
            }
        }
        // The offscreen HDR must reach SHADER_READ_ONLY before the tonemap
        // present copy samples it. `end` performs that COLOR_ATTACHMENT→
        // SHADER_READ barrier UNLESS a later offscreen writer runs after it: the
        // TAA resolve and exposure metering both write the offscreen *after*
        // `end`, so when either is active `end` must NOT transition (the barrier
        // would race their writes) — the deferred finalization below owns it
        // instead. With both disabled (the common path) `end` transitions.
        let taa = lists.scene.is_some() && self.flags.taa;
        let exposure = lists.scene.is_some() && self.flags.exposure;
        let deferred = taa || exposure;
        // Overlay composited post-tonemap so warp/TAA don't affect the HUD.
        stamp(GpuPass::Overlay);
        // Finalize the offscreen to SHADER_READ_ONLY exactly once and obtain the
        // [`HdrReadable`] proof the tonemap present-copy requires. The branches
        // are exhaustive and each ends with the offscreen sampled: (a) not
        // deferred → the render pass transitions; (b) deferred + exposure →
        // metering owns the transition; (c) deferred + !exposure (TAA-on,
        // exposure-off) → TAA left COLOR_ATTACHMENT, so transition explicitly.
        // Producing the proof only inside these paths is what makes "nobody
        // finalized the layout" fail to compile at `present` rather than trip the
        // validation layer (the exact bug from the exposure-default-off change).
        let readable: HdrReadable = {
            let _g = crate::profile::scope(crate::profile::Meter::RecTransitions);
            if deferred {
                unsafe { pass.end_deferred() };
                // TAA resolve runs AFTER the HDR resolve and BEFORE exposure, so
                // exposure meters the stabilized image. It reads the current HDR +
                // reprojected history, writes the resolved HDR back, and leaves it
                // COLOR_ATTACHMENT for the exposure pass. Reprojection uses the
                // un-jittered view-proj; a false `flags.taa` never reaches here.
                if taa {
                    // `taa` is `lists.scene.is_some() && self.flags.taa` (above).
                    let scene = lists.scene.as_ref().expect("taa true implies a 3D scene");
                    self.record_taa_pass(
                        cmd,
                        FrameSlot::new(slot),
                        scene.view_proj,
                        scene.eye,
                        scene.jitter.0,
                    );
                }
                if exposure {
                    // Reduce the frame HDR to per-tile mean log2-luma, publish
                    // the smoothed exposure, and finalize the HDR in SHADER_READ.
                    self.record_exposure_pass(cmd, FrameSlot::new(slot))
                } else if self.slots[FrameSlot::new(slot)].hdr_source != HdrSource::Offscreen {
                    // TAA published its output as the frame HDR and already
                    // left it (and the offscreen) sampled: nothing to record.
                    HdrReadable::new(slot)
                } else {
                    unsafe { self.transition_offscreen_to_sampled(cmd, slot) }
                }
            } else {
                // Common path (TAA + exposure both off): the render pass finalizes.
                unsafe { pass.end_sampled() }
            }
        };
        // Close the resolve/finalize segment (MSAA resolve + transitions +
        // TAA/exposure) before bloom records, so the report splits them.
        if profiling {
            unsafe {
                self.gpu_timer
                    .mark(&self.device.device, cmd, slot, GpuPass::Resolve)
            };
        }
        // Bloom: threshold + downsample the finalized HDR into this slot's
        // mip chain; the tonemap present-copy composites it. Recorded here so the
        // render→present semaphore makes the pyramid visible to the tonemap sample,
        // exactly as it does for the offscreen. A pure function of this frame.
        self.record_bloom_pass(cmd, FrameSlot::new(slot));
        // Close the tail: without this stamp the TAA/exposure/bloom work
        // recorded above ends after the last boundary and never reaches the
        // report. (The tonemap/present copy runs on the copy command buffer
        // and remains unmetered — tracked in structural opportunity #15.)
        if profiling {
            unsafe {
                self.gpu_timer
                    .mark(&self.device.device, cmd, slot, GpuPass::Post)
            };
        }
        self.gpu_timer.finish(slot);
        // The main pass just wrote (and stored) this slot's depth, so a later
        // cycle reusing this slot may read it for VRS classification. Stamp
        // ready-and-fingerprint in one write (they can never disagree).
        self.slots[FrameSlot::new(slot)].vrs_ready = Some(self.scene_fingerprint);

        unsafe {
            self.device
                .device
                .end_command_buffer(cmd)
                .expect("end command buffer failed");
        }
        (rs, readable)
    }

    /// The image+view holding slot `slot`'s FINAL HDR (see `hdr_source`).
    fn hdr_of(&self, slot: usize) -> (vk::Image, vk::ImageView) {
        match self.slots[FrameSlot::new(slot)].hdr_source {
            HdrSource::Offscreen => (
                self.targets.offscreen[slot].image(),
                self.targets.offscreen[slot].view(),
            ),
            HdrSource::TaaHistory(i) => self.taa.history_image(i),
        }
    }

    /// Transitions slot `slot`'s offscreen HDR from `COLOR_ATTACHMENT_OPTIMAL` to
    /// `SHADER_READ_ONLY_OPTIMAL` for the tonemap present-copy, minting the
    /// [`HdrReadable`] proof. Used on the TAA-on/exposure-off path, where the TAA
    /// resolve left the offscreen in `COLOR_ATTACHMENT` and no metering pass owns
    /// the transition.
    unsafe fn transition_offscreen_to_sampled(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
    ) -> HdrReadable {
        let to_sampled = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.targets.offscreen[slot].image())
            .subresource_range(color_range())];
        unsafe {
            self.device.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        }
        HdrReadable::new(slot)
    }

    /// Records the VRS classifier: samples this slot's depth (from two cycles
    /// ago) and writes its rate image, leaving depth back in
    /// `DEPTH_ATTACHMENT_OPTIMAL` for the geometry pass. Returns the attachment
    /// the caller binds. Only called when `do_vrs`, so both `vrs` and
    /// `vrs_compute` are present. `cmd` must be recording, outside a render pass.
    unsafe fn record_vrs_generate(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        d_threshold: f32,
    ) -> vrs::RateAttachment {
        let device = &self.device.device;
        let vrs = self.targets.vrs.as_ref().expect("do_vrs implies vrs");
        let compute = self
            .pipelines
            .vrs_compute
            .as_ref()
            .expect("do_vrs implies vrs_compute");
        // MSAA: the classifier samples the single-sample resolve of this slot's
        // depth from two cycles ago; single-sampled it is that depth directly.
        let depth = self.targets.sampleable_depth(slot);
        let tiles = vrs.tiles();
        unsafe {
            // depth: read for sampling; rate: write target.
            let pre = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(self.depth_pass_layout())
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(depth.image())
                    .subresource_range(depth_range()),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(vrs.image(slot))
                    .subresource_range(color_range()),
            ];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&pre),
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, compute.pipeline);
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(compute.depth_sampler)
                .image_view(depth.view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let rate_info = [vk::DescriptorImageInfo::default()
                .image_view(vrs.view(slot))
                .image_layout(vk::ImageLayout::GENERAL)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&rate_info),
            ];
            self.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                compute.layout,
                0,
                &writes,
            );

            let push = vrs::VrsPush {
                d_threshold,
                texel_w: vrs.texel_size.width,
                texel_h: vrs.texel_size.height,
            };
            device.cmd_push_constants(
                cmd,
                compute.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(cmd, tiles.width.div_ceil(8), tiles.height.div_ceil(8), 1);

            // rate → shading-rate attachment; depth → back to attachment layout.
            let post = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR)
                    .dst_access_mask(vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR)
                    .image(vrs.image(slot))
                    .subresource_range(color_range()),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(self.depth_pass_layout())
                    .image(depth.image())
                    .subresource_range(depth_range()),
            ];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&post),
            );
        }

        vrs::RateAttachment {
            view: vrs.view(slot),
            texel_size: vrs.texel_size,
        }
    }

    /// Submits the recorded command buffer and advances the timeline. Waits
    /// on the transfer lane's semaphore too when this frame's `flush_copies`/
    /// `quad_ibo.ensure` submitted on a separate queue — a cross-queue
    /// dependency needs a semaphore wait; the in-command-buffer barrier used
    /// otherwise only orders work within one queue.
    fn submit_render(&mut self, rs: RenderSubmit, slot: usize) {
        let extra_wait = self
            .pending_transfer_wait
            .take()
            .map(|value| (self.transfer_lane.semaphore(), value));
        let completion = unsafe {
            rs.submit(
                &self.device.device,
                self.device.graphics_queue,
                &self.timeline,
                extra_wait,
            )
        };
        self.slots[FrameSlot::new(slot)].render_value = completion.value();
        self.last_render_value = completion.value();
    }

    /// Copies the finished frame into the acquired swapchain image (when one
    /// was acquired in [`Self::decide_present`]) and queues the present. The
    /// [`HdrReadable`] proof is REQUIRED (not merely passed): the present copy's
    /// tonemap samples the offscreen, so this signature makes it impossible to
    /// present a frame whose offscreen was never finalized to
    /// `SHADER_READ_ONLY_OPTIMAL`.
    fn present(
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
        if self.vsync.current() {
            // Wait for copy to pace at display refresh.
            unsafe {
                self.timeline
                    .wait(&self.device.device, self.last_copy_value);
            }
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
            {
                let depth_to_read = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(self.depth_pass_layout())
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
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(self.depth_pass_layout())
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
                        self.needs_recreate = true;
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

    /// While no frames are being submitted (minimized window): waits out the
    /// in-flight fences, flushes any staged mesh copies with a standalone
    /// submit, and frees the whole retire queue.
    unsafe fn reclaim_while_idle(&mut self) {
        if !self.mesh_res.has_pending() && !self.mesh_res.has_garbage() {
            return;
        }
        let device = &self.device.device;
        unsafe {
            // Wait for all in-flight submits to complete.
            self.timeline.wait(device, self.timeline.last_reserved());
            self.copy_slot = None;

            if self.mesh_res.has_pending() {
                // Reuse slot 0's command buffer. Always real and valid: even
                // under a separate transfer queue, the `DedicatedFamily` tier
                // needs a real graphics-side command buffer to record its
                // ownership-transfer ACQUIRE barrier into (see
                // `MeshResidency::flush_copies`).
                let cmd = self.slots[FrameSlot::new(0)].cmd;
                device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .expect("command buffer reset failed");
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device
                    .begin_command_buffer(cmd, &begin)
                    .expect("begin command buffer failed");
                let transfer_wait = self.mesh_res.flush_copies(
                    device,
                    &mut self.transfer_lane,
                    cmd,
                    self.device.graphics_family,
                    self.last_render_value,
                );
                device
                    .end_command_buffer(cmd)
                    .expect("end command buffer failed");
                let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
                let wait_info = transfer_wait.map(|value| {
                    [vk::SemaphoreSubmitInfo::default()
                        .semaphore(self.transfer_lane.semaphore())
                        .value(value.raw())
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)]
                });
                let mut submit = vk::SubmitInfo2::default().command_buffer_infos(&cmd_info);
                if let Some(wait_info) = &wait_info {
                    submit = submit.wait_semaphore_infos(wait_info);
                }
                device
                    .queue_submit2(self.device.graphics_queue, &[submit], vk::Fence::null())
                    .expect("queue submit failed");
                // Waiting graphics idle here transitively proves the transfer
                // queue's copy completed too: this submission waited on its
                // semaphore before executing, so graphics cannot have
                // finished without that wait already being satisfied.
                device
                    .queue_wait_idle(self.device.graphics_queue)
                    .expect("queue wait failed");
                // Same reveal step as the live-frame path (see `draw_frame`):
                // without this, a mesh uploaded just before minimizing stays
                // gated at arena word 0 forever once the window is restored.
                let arrived = self.mesh_res.take_arrived();
                self.records.mark_arrived(&arrived);
            }

            // GPU idle + copies flushed: everything retired returns to main.
            let ret = &self.ret;
            self.mesh_res
                .collect_all(&mut |a| drop(ret.send(RenderReturn::FreeAlloc(a))));
        }
    }

    /// Applies pending vsync/MSAA changes and rebuilds swapchain-sized state.
    unsafe fn apply_pending(&mut self) {
        unsafe {
            self.device
                .device
                .device_wait_idle()
                .expect("device_wait_idle failed");

            let size = self.size;
            if size.width == 0 || size.height == 0 {
                // Still minimized: can't rebuild swapchain yet.
                return;
            }

            // Commit pending changes; vsync must apply before swapchain rebuild.
            self.vsync.commit();

            let new_swapchain = Swapchain::new(
                &self.instance.instance,
                &self.device,
                &self.surface_loader,
                self.surface,
                size,
                self.vsync.effective(),
                self.swapchain.swapchain,
            );
            self.swapchain.destroy(&self.device.device);
            let format_changed = new_swapchain.format != self.swapchain.format;
            self.swapchain = new_swapchain;

            let msaa_changed = self.msaa.commit();
            self.render_scale.commit();
            self.render_extent =
                scaled_extent(self.swapchain.extent, self.render_scale.effective());

            self.targets.destroy(&self.device.device);
            self.targets = RenderTargets::new(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                self.render_extent,
                self.msaa.effective(),
                self.device.fragment_shading_rate.as_ref(),
            );
            // Exposure's tile grid tracks the render extent: rebuild its GPU
            // resources in place (the published `ExposureShared` cell the main
            // thread holds is preserved, so `compose()` keeps reading it).
            let memory_props = self
                .instance
                .instance
                .get_physical_device_memory_properties(self.device.physical);
            self.exposure
                .recreate(&self.device.device, &memory_props, self.render_extent);
            // History is extent-sized; recreate discards it (reconverges).
            self.taa
                .recreate(&self.device.device, &memory_props, self.render_extent);

            // Offscreen images recreated; clear copy tracking.
            self.clear_copy();
            // Depth images recreated (layout UNDEFINED): VRS must re-prime.
            // Shadow map recreated (layout UNDEFINED): re-prime the lit clear.
            for slot in 0..FRAMES_IN_FLIGHT as usize {
                let s = &mut self.slots[FrameSlot::new(slot)];
                s.vrs_ready = None;
                s.shadow_lit_ready = false;
            }
            // Shadow images are UNDEFINED after recreate: force both slots to
            // re-render rather than sample a stale/garbage cached depth.
            self.shadow_cache.invalidate();

            if msaa_changed || format_changed {
                self.pipelines.destroy(&self.device.device);
                self.pipelines = Pipelines::new(
                    &self.device.device,
                    self.pipeline_cache,
                    self.targets.color_format,
                    self.swapchain.format,
                    self.targets.depth_format,
                    self.targets.samples,
                    self.atlas.set_layout,
                    self.mesh3d_set_layout,
                    self.device.fragment_shading_rate.as_ref(),
                    self.device.dynamic_rendering_local_read,
                );
            }

            for &sem in &self.present_semaphores {
                sem.destroy(&self.device.device);
            }
            self.present_semaphores =
                create_present_semaphores(&self.device.device, self.swapchain.images.len());

            self.needs_recreate = false;
        }
    }
}

/// Manages dynamic rendering for one frame. Must call `end()` explicitly.
struct RenderPass<'a> {
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
    unsafe fn begin(
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
        unsafe {
            // Generate the rate map first: it samples this slot's depth (leaving
            // it in DEPTH_ATTACHMENT_OPTIMAL, ready for the pass below) and
            // returns the only valid `RateAttachment`. Done before the color
            // barriers so the compute dispatch overlaps nothing it depends on.
            let rate = do_vrs.then(|| {
                let scene = lists.scene.as_ref().expect("do_vrs implies a 3D scene");
                let focal_px = 0.5 * extent.height as f32 / scene.fovy_tan_half.max(1e-4);
                let d_threshold = crate::camera::Z_NEAR / focal_px;
                r.record_vrs_generate(cmd, slot, d_threshold)
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
            r.targets.shadow[FrameSlot::new(self.slot)].sampler,
            r.targets.shadow[FrameSlot::new(self.slot)].sample_view,
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
    unsafe fn record_mesh_indirect(&self, pass: Pass) {
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
    /// one `vkCmdDrawIndexedIndirectCount` per non-empty (pass, arena)
    /// partition, consuming the commands the cull dispatch emitted earlier in
    /// this command buffer. Never called for Blend (CPU-sorted path).
    unsafe fn record_mesh_indirect_count(&self, pass: Pass) {
        let Some(frame) = &self.r.cull_frame else {
            return; // nothing live to draw
        };
        let group = match pass {
            Pass::Opaque => 0usize,
            Pass::Cutout => 1,
            Pass::Blend => unreachable!("Blend stays on the CPU path"),
        };
        let base = group * frame.arena_count;
        if frame.partitions[base..base + frame.arena_count]
            .iter()
            .all(|p| p.capacity == 0)
        {
            return;
        }
        unsafe { self.bind_mesh3d_state() };
        let device = &self.r.device.device;
        unsafe {
            device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.r.mesh_pipeline_for(pass),
            );
            let quad_ibo = self
                .r
                .quad_ibo
                .bound()
                .expect("live records imply the quad IBO is allocated");
            device.cmd_bind_index_buffer(self.cmd, quad_ibo, 0, vk::IndexType::UINT32);
            for arena in 0..frame.arena_count {
                let part = frame.partitions[base + arena];
                if part.capacity == 0 {
                    continue;
                }
                device.cmd_bind_vertex_buffers(
                    self.cmd,
                    0,
                    &[self.r.arena_dir.arena_buffer(arena)],
                    &[0],
                );
                device.cmd_draw_indexed_indirect_count(
                    self.cmd,
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
    unsafe fn record_immediate_cubes(&self) {
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
    unsafe fn record_shadows(&self) {
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
    unsafe fn record_lines(&self) {
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

    /// The procedural sky background pass (sky pipeline: geometry push constant
    /// + the shared per-frame `FrameUniforms` at set 0 binding 2, no vertex
    /// buffer). A single fullscreen triangle at the reversed-Z far plane; the
    /// read-only depth test rejects it wherever terrain wrote closer depth, so
    /// it shades only background pixels. Skipped unless the frame set a sky
    /// palette.
    unsafe fn record_sky(&self) {
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
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&params),
            );
            // Push ONLY binding 2 (the per-frame UBO): the sky fragment reads its
            // colours from the same linear `FrameUniforms` the terrain fog does
            // Reuses the mesh layout. The layout's other bindings (offsets SSBO, textures,
            // shadow map) are unused by this pass, so they stay unwritten — valid
            // for a push-descriptor set when the shader never accesses them.
            let ubo = self.r.ubo_ring.buffer(FrameSlot::new(self.slot));
            let ubo_infos = [vk::DescriptorBufferInfo::default()
                .buffer(ubo)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let writes = [vk::WriteDescriptorSet::default()
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&ubo_infos)];
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

    /// Ends dynamic rendering and transitions the offscreen image to be sampled
    /// by the tonemap pass. Ordering across the render/tonemap submits is
    /// enforced by the timeline; this barrier only owns the layout + visibility.
    /// Ends dynamic rendering. When `transition_offscreen` the offscreen is left
    /// in `SHADER_READ_ONLY_OPTIMAL` for the present-copy tonemap; the exposure
    /// metering pass (which itself transitions COLOR_ATTACHMENT→SHADER_READ)
    /// passes `false` so it owns that barrier instead.
    /// Ends dynamic rendering AND transitions the offscreen to
    /// `SHADER_READ_ONLY_OPTIMAL`, returning the [`HdrReadable`] proof. Use when
    /// no later pass writes the offscreen (the common path: TAA and exposure both
    /// off), so this pass owns the finalization.
    unsafe fn end_sampled(self) -> HdrReadable {
        let slot = self.slot;
        unsafe { self.end(true) };
        HdrReadable::new(slot)
    }

    /// Ends dynamic rendering WITHOUT the sampled transition: a later offscreen
    /// writer (TAA resolve / exposure metering) runs after this, and one of them
    /// owns the finalization instead (its barrier would otherwise race their
    /// writes). Yields no proof — the deferred finalizer produces it.
    unsafe fn end_deferred(self) {
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

impl Renderer {
    /// Destroys every render-owned resource (GPU idle first) and hands the
    /// device/instance/surface back to main, which destroys the allocator
    /// buffers and then `vkDestroyDevice` in the correct order. Consuming `self`
    /// (rather than `Drop`) is what lets those fields move out to main.
    pub(crate) fn teardown(mut self) -> DeviceLeftovers {
        unsafe {
            let device = &self.device.device;
            let _ = device.device_wait_idle();

            self.pipelines.destroy(device);
            save_pipeline_cache(device, self.pipeline_cache);
            device.destroy_pipeline_cache(self.pipeline_cache, None);
            self.atlas.destroy(device);
            self.minimap.destroy(device);
            device.destroy_descriptor_set_layout(self.mesh3d_set_layout, None);
            self.ubo_ring.destroy(device);
            self.shadow.destroy(device);
            self.exposure.destroy(device);
            self.bloom.destroy(device);
            self.taa.destroy(device);
            self.block_textures.destroy(device);
            self.retired_textures
                .collect_all(|mut tex| tex.destroy(device));
            self.gpu_timer.destroy(device);
            self.targets.destroy(device);
            self.records.destroy(device);
            self.cull.destroy(device);
            self.quad_ibo.destroy(device);
            // The residents' allocations belong to the main-owned allocator
            // (destroyed there after this returns); just drop them — no Vulkan
            // calls, GPU already idle.
            self.mesh_res.destroy_all(&mut |_a| {});
            for &sem in &self.present_semaphores {
                sem.destroy(device);
            }
            self.timeline.destroy(device);
            self.transfer_lane.destroy(device);
            for slot in 0..FRAMES_IN_FLIGHT as usize {
                let s = &mut self.slots[FrameSlot::new(slot)];
                s.imm.destroy(device);
                s.indirect.destroy(device);
                s.image_available.destroy(device);
            }
            self.swapchain.destroy(device);
        }
        DeviceLeftovers {
            instance: self.instance,
            surface_loader: self.surface_loader,
            surface: self.surface,
            device: self.device,
        }
    }
}

/// Clamp range for render-resolution scale (0.25x to 2.0x). Re-exported from
/// crate root so settings UI and renderer stay in sync.
pub const RENDER_SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.25..=2.0;

/// Render-resolution scale relative to the window, clamped to
/// [`RENDER_SCALE_RANGE`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scale(f32);

impl Scale {
    /// Clamps `value` into the supported [`RENDER_SCALE_RANGE`].
    pub fn new(value: f32) -> Self {
        Scale(value.clamp(*RENDER_SCALE_RANGE.start(), *RENDER_SCALE_RANGE.end()))
    }

    /// The clamped scale factor.
    pub fn get(self) -> f32 {
        self.0
    }

    /// Alias for [`Scale::get`].
    pub fn as_f32(self) -> f32 {
        self.0
    }
}

/// A value plus an optional change queued to apply at the next frame boundary.
/// Reads go through [`effective`](Self::effective), so a getter can never
/// forget to account for a pending change — the footgun of parallel
/// `current`/`pending` fields.
struct Pending<T> {
    current: T,
    pending: Option<T>,
}

impl<T: Copy + PartialEq> Pending<T> {
    fn new(current: T) -> Self {
        Self {
            current,
            pending: None,
        }
    }

    /// The value including any queued change.
    fn effective(&self) -> T {
        self.pending.unwrap_or(self.current)
    }

    /// The currently-applied value, ignoring any queued change. Use where the
    /// live GPU state (not the requested one) matters, e.g. present pacing
    /// before the swapchain is rebuilt.
    fn current(&self) -> T {
        self.current
    }

    /// Queues `next` unconditionally (caller owns the change test).
    fn queue(&mut self, next: T) {
        self.pending = Some(next);
    }

    /// Queues `next` if it differs from the effective value; returns whether
    /// it did, so callers can flag a recreate.
    fn set(&mut self, next: T) -> bool {
        let changed = next != self.effective();
        if changed {
            self.pending = Some(next);
        }
        changed
    }

    /// Applies any queued change; returns whether one was applied.
    fn commit(&mut self) -> bool {
        match self.pending.take() {
            Some(v) => {
                self.current = v;
                true
            }
            None => false,
        }
    }
}

/// A validated MSAA sample count (powers of two only: 1, 2, 4, 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleCount {
    X1,
    X2,
    X4,
    X8,
}

impl SampleCount {
    /// The sample count as a `u32` (1, 2, 4, or 8).
    pub fn as_u32(self) -> u32 {
        match self {
            SampleCount::X1 => 1,
            SampleCount::X2 => 2,
            SampleCount::X4 => 4,
            SampleCount::X8 => 8,
        }
    }

    /// The corresponding Vulkan sample-count flag.
    pub fn as_flags(self) -> vk::SampleCountFlags {
        match self {
            SampleCount::X1 => vk::SampleCountFlags::TYPE_1,
            SampleCount::X2 => vk::SampleCountFlags::TYPE_2,
            SampleCount::X4 => vk::SampleCountFlags::TYPE_4,
            SampleCount::X8 => vk::SampleCountFlags::TYPE_8,
        }
    }

    /// Rounds an arbitrary `u32` DOWN to the nearest valid {1,2,4,8} bucket.
    fn bucket(value: u32) -> SampleCount {
        match value {
            0 | 1 => SampleCount::X1,
            2..=3 => SampleCount::X2,
            4..=7 => SampleCount::X4,
            _ => SampleCount::X8,
        }
    }

    /// Finds the largest supported count <= max, clamped to {1,2,4,8}. Returns
    /// (count, changed) so callers can log downgrades.
    pub fn nearest_supported(requested: u32, max: u32) -> (SampleCount, bool) {
        let mut count = SampleCount::bucket(requested);
        let cap = SampleCount::bucket(max);
        if count.as_u32() > cap.as_u32() {
            count = cap;
        }
        (count, count.as_u32() != requested)
    }
}

/// Resolves an MSAA request to a supported {1,2,4,8} count, logging any
/// downgrade. `context` labels the log line.
fn resolve_msaa(requested: u32, max: u32, context: &str) -> SampleCount {
    let (count, changed) = SampleCount::nearest_supported(requested, max);
    if changed {
        log::debug!(
            "MSAA ({context}): requested {requested}x -> using {}x (max {max}x)",
            count.as_u32(),
        );
    }
    count
}

/// Pipeline cache path: OS temp dir for per-user write access.
fn pipeline_cache_path() -> std::path::PathBuf {
    std::env::temp_dir().join("voxel_engine_pipeline.cache")
}

/// Creates pipeline cache, seeded from disk if available. Invalid data falls back to empty.
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

fn create_present_semaphores(device: &ash::Device, count: usize) -> Vec<BinarySemaphore> {
    (0..count)
        .map(|_| unsafe { BinarySemaphore::new(device) })
        .collect()
}

/// Clamps an MSAA request to a supported {1,2,4,8} sample count (as a `u32`),
/// mirroring [`Renderer::set_msaa`] so the client can clamp locally.
pub(crate) fn clamp_msaa(requested: u32, max: u32) -> u32 {
    SampleCount::nearest_supported(requested, max).0.as_u32()
}

/// Display refresh interval; falls back to 60 Hz if unavailable.
pub(crate) fn display_refresh_interval(window: &winit::window::Window) -> std::time::Duration {
    let millihertz = window
        .current_monitor()
        .and_then(|m| m.refresh_rate_millihertz())
        .filter(|&mhz| mhz > 0) // Some(0) = unknown on some X11/VM backends
        .unwrap_or(60_000);
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

/// See `SlotState::hdr_source`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HdrSource {
    /// The scene render's offscreen target (TAA off).
    Offscreen,
    /// The TAA history image at this index (TAA on: the resolve output).
    TaaHistory(usize),
}

/// A GPU render pass boundary, in record order. The variant ordinal indexes the
/// per-pass accumulator (matches the tracking in [`crate::profile::Meter`]).
#[derive(Clone, Copy, PartialEq)]
enum GpuPass {
    Opaque,
    Sky,
    Cubes,
    Lines,
    Shadows,
    Transparent,
    Overlay,
    /// End of the scene pass: `cmd_end_rendering` (where the MSAA color
    /// resolve executes), offscreen finalize transitions, TAA, and exposure.
    Resolve,
    /// The bloom chain — the render-command tail after the resolve. Both tail
    /// stamps used to not exist (no closing timestamp), silently hiding the
    /// whole post stack from the report.
    Post,
}

impl GpuPass {
    const ALL: [GpuPass; 9] = [
        GpuPass::Opaque,
        GpuPass::Sky,
        GpuPass::Cubes,
        GpuPass::Lines,
        GpuPass::Shadows,
        GpuPass::Transparent,
        GpuPass::Overlay,
        GpuPass::Resolve,
        GpuPass::Post,
    ];
    const COUNT: usize = Self::ALL.len();

    fn meter(self) -> crate::profile::Meter {
        use crate::profile::Meter;
        match self {
            GpuPass::Opaque => Meter::GpuOpaque,
            GpuPass::Sky => Meter::GpuSky,
            GpuPass::Cubes => Meter::GpuCubes,
            GpuPass::Lines => Meter::GpuLines,
            GpuPass::Shadows => Meter::GpuShadows,
            GpuPass::Transparent => Meter::GpuTransparent,
            GpuPass::Overlay => Meter::GpuOverlay,
            GpuPass::Resolve => Meter::GpuResolve,
            GpuPass::Post => Meter::GpuPost,
        }
    }
}

/// One start timestamp plus one boundary per pass.
const GPU_STAMPS: usize = GpuPass::COUNT + 1;

/// Per-pass GPU timing via a timestamp query pool: a start timestamp plus one
/// after each recorded pass. Only the passes that actually run write a stamp,
/// and the label written alongside each stamp keeps deltas attributable even
/// when a frame skips passes (no 3D, VRS off). A slot's results are read one
/// cycle later, after its fence is waited, so the read never stalls.
///
/// `count`/`label` are [`Cell`]s so a mark needs only `&self`: the render pass
/// holds an immutable `&Renderer` while recording, and all timer state is
/// touched on the single render thread. A null pool (hardware without timestamp
/// support) makes every method a no-op.
struct GpuTimer {
    pool: vk::QueryPool,
    /// Nanoseconds per tick (`limits.timestampPeriod`).
    period_ns: f32,
    /// Whether each slot holds completed timestamps to read back.
    primed: [bool; FRAMES_IN_FLIGHT as usize],
    /// Stamps written for each slot's most recent recording (incl. the start).
    count: [std::cell::Cell<u32>; FRAMES_IN_FLIGHT as usize],
    /// The pass that ended at each stamp (index `i` labels the span `i-1..i`).
    label: [[std::cell::Cell<GpuPass>; GPU_STAMPS]; FRAMES_IN_FLIGHT as usize],
}

impl GpuTimer {
    fn new(device: &ash::Device, supported: bool, period_ns: f32) -> Self {
        let pool = if supported {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(GPU_STAMPS as u32 * FRAMES_IN_FLIGHT as u32);
            unsafe {
                device
                    .create_query_pool(&info, None)
                    .expect("Failed to create timestamp query pool")
            }
        } else {
            vk::QueryPool::null()
        };
        Self {
            pool,
            period_ns,
            primed: [false; FRAMES_IN_FLIGHT as usize],
            count: std::array::from_fn(|_| std::cell::Cell::new(0)),
            label: std::array::from_fn(|_| {
                std::array::from_fn(|_| std::cell::Cell::new(GpuPass::Opaque))
            }),
        }
    }

    fn enabled(&self) -> bool {
        self.pool != vk::QueryPool::null()
    }

    /// Reads `slot`'s prior render-pass per-pass durations (ms), adding each to
    /// `sink`. The caller must have waited `slot`'s fence, so the result is
    /// ready without a GPU stall. Returns the summed render-pass time (ms).
    unsafe fn read_into(&self, device: &ash::Device, slot: usize, sink: &mut [f64]) -> Option<f64> {
        if !self.enabled() || !self.primed[slot] {
            return None;
        }
        let n = self.count[slot].get() as usize;
        if n < 2 {
            return None;
        }
        let mut ts = [0u64; GPU_STAMPS];
        unsafe {
            device.get_query_pool_results(
                self.pool,
                slot as u32 * GPU_STAMPS as u32,
                &mut ts[..n],
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .ok()?;
        let mut total = 0.0;
        for i in 1..n {
            let ms = ts[i].wrapping_sub(ts[i - 1]) as f64 * self.period_ns as f64 / 1.0e6;
            sink[self.label[slot][i].get() as usize] += ms;
            total += ms;
        }
        Some(total)
    }

    /// Resets `slot`'s queries and writes the start timestamp. Must be recorded
    /// outside any render pass.
    unsafe fn begin(&self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        if !self.enabled() {
            return;
        }
        let base = slot as u32 * GPU_STAMPS as u32;
        unsafe {
            device.cmd_reset_query_pool(cmd, self.pool, base, GPU_STAMPS as u32);
            device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, self.pool, base);
        }
        self.count[slot].set(1);
    }

    /// Writes a boundary timestamp closing `pass` for `slot`. Recorded inside
    /// the render pass; needs only `&self` (interior-mutable bookkeeping).
    unsafe fn mark(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        pass: GpuPass,
    ) {
        if !self.enabled() {
            return;
        }
        let i = self.count[slot].get();
        if i as usize >= GPU_STAMPS {
            return;
        }
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.pool,
                slot as u32 * GPU_STAMPS as u32 + i,
            );
        }
        self.label[slot][i as usize].set(pass);
        self.count[slot].set(i + 1);
    }

    /// Marks `slot` readable next cycle. Call after the render pass ends.
    fn finish(&mut self, slot: usize) {
        if self.enabled() {
            self.primed[slot] = true;
        }
    }

    unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.enabled() {
            unsafe { device.destroy_query_pool(self.pool, None) };
            self.pool = vk::QueryPool::null();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vrs_scene_fingerprint_tracks_view_and_depth_geometry() {
        let mut lists = DrawLists::new();
        let base = scene_fingerprint(&lists, &[], &[]);

        lists.scene = Some(Scene3D::test_stub());
        let with_3d = scene_fingerprint(&lists, &[], &[]);
        assert_ne!(base, with_3d);

        lists.scene.as_mut().unwrap().view_proj = glam::Mat4::from_rotation_y(0.25);
        let turned = scene_fingerprint(&lists, &[], &[]);
        assert_ne!(with_3d, turned);

        lists.cube_verts.push(crate::mesh::DebugVertex {
            pos: [1.0, 2.0, 3.0],
            color: [255; 4],
        });
        assert_ne!(turned, scene_fingerprint(&lists, &[], &[]));

        // A flipped visibility bit (streamed-in / LOD-settled terrain) must also
        // invalidate the reused depth — the mask is now the opaque draw source.
        let masked = scene_fingerprint(&lists, &[], &[0b1]);
        assert_ne!(scene_fingerprint(&lists, &[], &[]), masked);
    }

    #[test]
    fn scale_clamps_into_range() {
        assert_eq!(Scale::new(1.0).get(), 1.0);
        assert_eq!(Scale::new(0.5).as_f32(), 0.5);
        // Below the floor and above the ceiling clamp to the bounds.
        assert_eq!(Scale::new(0.0).get(), 0.25);
        assert_eq!(Scale::new(-5.0).get(), 0.25);
        assert_eq!(Scale::new(10.0).get(), 2.0);
        assert_eq!(Scale::new(2.0).get(), 2.0);
        assert_eq!(Scale::new(0.25).get(), 0.25);
    }

    #[test]
    fn sample_count_conversions() {
        for (count, n, flag) in [
            (SampleCount::X1, 1, vk::SampleCountFlags::TYPE_1),
            (SampleCount::X2, 2, vk::SampleCountFlags::TYPE_2),
            (SampleCount::X4, 4, vk::SampleCountFlags::TYPE_4),
            (SampleCount::X8, 8, vk::SampleCountFlags::TYPE_8),
        ] {
            assert_eq!(count.as_u32(), n);
            assert_eq!(count.as_flags(), flag);
        }
    }

    #[test]
    fn nearest_supported_exact_values_are_unchanged() {
        for n in [1, 2, 4, 8] {
            let (count, changed) = SampleCount::nearest_supported(n, 8);
            assert_eq!(count.as_u32(), n);
            assert!(!changed, "{n} is exact and within cap");
        }
    }

    #[test]
    fn nearest_supported_rounds_odd_down_and_flags_change() {
        // Odd / non-power-of-two values round DOWN to the nearest bucket and
        // report changed = true (the log-worthy downgrade case).
        for (req, expected) in [(0, 1), (3, 2), (5, 4), (7, 4), (16, 8), (100, 8)] {
            let (count, changed) = SampleCount::nearest_supported(req, 8);
            assert_eq!(count.as_u32(), expected, "requested {req}");
            assert!(changed, "requested {req} was downgraded");
        }
    }

    #[test]
    fn nearest_supported_caps_at_max_and_flags_change() {
        // Hardware cap: 8x requested but only 4x supported -> 4x, changed.
        let (count, changed) = SampleCount::nearest_supported(8, 4);
        assert_eq!(count.as_u32(), 4);
        assert!(changed);

        // Cap of 1x (no MSAA support) forces X1.
        let (count, changed) = SampleCount::nearest_supported(4, 1);
        assert_eq!(count.as_u32(), 1);
        assert!(changed);

        // Requested already at the cap: unchanged.
        let (count, changed) = SampleCount::nearest_supported(4, 4);
        assert_eq!(count.as_u32(), 4);
        assert!(!changed);
    }
}
