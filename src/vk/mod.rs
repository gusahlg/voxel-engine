/// The Vulkan renderer: instance, device, swapchain, render targets, pipelines,
/// GPU memory, and frame loop. Vulkan 1.3 with dynamic rendering + synchronization2;
/// 2 frames in flight; reversed-Z depth; optional MSAA with resolve.
///
/// Rendering and presentation decouple: frames render into offscreen images and
/// present only when a swapchain image is available (mailbox). On macOS, vsync
/// paces at refresh via presentation backpressure; vsync off uncaps the loop.
pub(crate) mod alloc;
pub(crate) mod block_textures;
pub(crate) mod buffers;
pub(crate) mod device;
pub(crate) mod image_upload;
pub(crate) mod instance;
pub(crate) mod minimap;
pub(crate) mod pipeline;
pub(crate) mod render_client;
pub(crate) mod swapchain;
pub(crate) mod targets;
pub(crate) mod texture;
pub(crate) mod timeline;
pub(crate) mod vertex_input;
pub(crate) mod vrs;

use std::num::NonZeroU32;
use std::sync::mpsc::Sender;

use ash::{khr, vk};

use crate::frame::DrawLists;
use crate::mesh::Pass;
use block_textures::BlockTextures;
use buffers::{
    DrawIndexedIndirect, DrawOffset, FRAMES_IN_FLIGHT, GpuResident, HostBuffer, MeshResidency,
    SurfaceResidency,
};
use device::Device;
use render_client::{DeviceCaps, DeviceLeftovers, InitReply, RenderConfig, RenderReturn};
use instance::InstanceBundle;
use minimap::MinimapTexture;
use pipeline::Pipelines;
use swapchain::Swapchain;
use targets::RenderTargets;
use texture::FontAtlas;
use timeline::{
    BinarySemaphore, RenderCompletion, RenderSubmit, Timeline, TimelineValue, acquire_next_image,
    queue_present,
};

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

struct FrameSlot {
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
}

/// Token returned by `acquire_slot` proving the slot is safe to render into
/// (its copy hazard is resolved).
struct SlotGuard(usize);

/// Byte offsets of the line and 2D sections inside a frame's packed
/// immediate buffer (cube verts start at offset 0).
#[derive(Clone, Copy)]
struct ImmOffsets {
    line: u64,
    shadow: u64,
    d2: u64,
    d2_tex: u64,
}

/// The 2D overlay draw parameters carried to the post-tonemap present pass when
/// wide-FOV is active (so the HUD/minimap render in swapchain space, unwarped).
#[derive(Clone, Copy)]
struct OverlayPresent {
    d2_offset: u64,
    d2_count: u32,
    d2_tex_offset: u64,
    d2_tex_count: u32,
}

/// One resolved mesh draw (one direction-run of one mesh), pre-sort scratch for
/// [`Renderer::prepare_mesh_draws`]. `first`/`count` are the index sub-range for
/// this run (the whole mesh when face-culling is off).
#[derive(Clone, Copy)]
struct DrawEntry {
    buffer: vk::Buffer,
    pass: Pass,
    /// Draw through the depth-biased opaque pipeline (`mesh3d_biased`). Only
    /// meaningful for [`Pass::Opaque`]; an orthogonal per-draw axis, not a
    /// mesh-intrinsic pass.
    biased: bool,
    first: u32,
    count: u32,
    vertex_offset: i32,
    offset: glam::Vec3,
    scale: f32,
    /// Squared distance from the camera to the mesh's world-space AABB center.
    /// Sort key for depth ordering; squared avoids a sqrt (monotonic).
    dist2: f32,
}

/// One resolved surface draw (handle → GpuSurface), copied out so the
/// `surfaces` borrow is released before the shared command/offset buffers are
/// mutated. Always opaque, one index range, camera-relative offset + scale.
#[derive(Clone, Copy)]
struct SurfaceEntry {
    buffer: vk::Buffer,
    first: u32,
    count: u32,
    vertex_offset: i32,
    offset: glam::Vec3,
    scale: f32,
}

/// A contiguous range of indirect commands sharing one arena buffer AND one
/// pass: one pipeline bind (on pass change) + one vertex/index bind + one
/// indirect call each.
#[derive(Clone, Copy)]
struct DrawRun {
    buffer: vk::Buffer,
    pass: Pass,
    /// All entries in a run share one pipeline; `biased` distinguishes
    /// `mesh3d` from `mesh3d_biased` within [`Pass::Opaque`].
    biased: bool,
    first: u32,
    count: u32,
}

/// Which of the six face directions (indexed by `Normal`: `0=+X 1=-X 2=+Y
/// 3=-Y 4=+Z 5=-Z`) can face the camera, given the camera position in the
/// mesh's own frame and the mesh's world-space (already scaled) AABB. Inside an
/// axis's extent both of that axis's faces show; outside, only the near one.
fn visible_dirs(cam_local: glam::Vec3, aabb_min: glam::Vec3, aabb_max: glam::Vec3) -> [bool; 6] {
    [
        cam_local.x > aabb_min.x, // +X
        cam_local.x < aabb_max.x, // -X
        cam_local.y > aabb_min.y, // +Y
        cam_local.y < aabb_max.y, // -Y
        cam_local.z > aabb_min.z, // +Z
        cam_local.z < aabb_max.z, // -Z
    ]
}

/// Maximal contiguous runs of visible directions as `[start, end)` index pairs
/// over `0..6`, returned in a fixed array with a count. At most three runs (six
/// slots can alternate visible/hidden at most three times). Adjacent visible
/// directions coalesce into one run so their contiguous index buckets draw as a
/// single command.
fn contiguous_runs(vis: [bool; 6]) -> ([(u8, u8); 3], usize) {
    let mut runs = [(0u8, 0u8); 3];
    let mut n = 0;
    let mut i = 0u8;
    while i < 6 {
        if vis[i as usize] {
            let start = i;
            while i < 6 && vis[i as usize] {
                i += 1;
            }
            runs[n] = (start, i);
            n += 1;
        } else {
            i += 1;
        }
    }
    (runs, n)
}

/// Fingerprint every input that shapes a slot's stored depth image. Attachment
/// VRS reuses that depth two frames later to classify shading rate, but the
/// reuse is only valid while the view and the depth-writing draw set are
/// unchanged; a moved camera or altered draw list would otherwise let stale sky
/// pixels coarsen newly visible edges. Color, sort distance, and other
/// shading-only inputs are excluded — they cannot alter the depth buffer.
fn scene_fingerprint(lists: &DrawLists, draws: &[DrawEntry], surfaces: &[SurfaceEntry]) -> u64 {
    use ash::vk::Handle;
    use std::hash::{Hash, Hasher};

    let mut h = std::collections::hash_map::DefaultHasher::new();
    lists.has_3d.hash(&mut h);
    for c in lists.view_proj.to_cols_array() {
        c.to_bits().hash(&mut h);
    }
    for d in draws {
        d.buffer.as_raw().hash(&mut h);
        (d.pass as u8, d.biased, d.first, d.count, d.vertex_offset).hash(&mut h);
        for c in d.offset.to_array() {
            c.to_bits().hash(&mut h);
        }
        d.scale.to_bits().hash(&mut h);
    }
    // Zone-3 far-skin surfaces lay opaque depth before the VRS source is stored.
    for s in surfaces {
        s.buffer.as_raw().hash(&mut h);
        (s.first, s.count, s.vertex_offset).hash(&mut h);
        for c in s.offset.to_array() {
            c.to_bits().hash(&mut h);
        }
        s.scale.to_bits().hash(&mut h);
    }
    // Immediate debug cubes (avatars/highlights) also write depth first; every
    // position matters, their colors never do.
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

    /// Render-side residency mirrors (device buffers, staged copies, retire).
    /// Fed by the ordered command stream; freed allocations return to main.
    mesh_res: MeshResidency,
    /// Retained colored surfaces (Zone-3 far grey skin), drawn through the
    /// `surface3d` lane. Shares the offsets SSBO + indirect buffers with meshes.
    surface_res: SurfaceResidency,
    /// Returns freed allocations (freelist recycling) to the main-owned
    /// allocator. The render loop uses its own clone for frame-buffer recycling.
    ret: Sender<RenderReturn>,
    /// Window/present size, updated by `RenderCmd::Resize` — replaces the old
    /// `window.inner_size()` reads now the window lives on main.
    size: vk::Extent2D,

    swapchain: Swapchain,
    targets: RenderTargets,
    pipelines: Pipelines,
    /// Disk-backed pipeline cache (see [`pipeline_cache_path`]): loaded at
    /// init, passed to every pipeline build, written back on Drop.
    pipeline_cache: vk::PipelineCache,
    atlas: FontAtlas,
    block_textures: BlockTextures,
    /// Old block-texture arrays retired on palette growth, destroyed once the
    /// timeline proves every in-flight frame that could sample them is done —
    /// avoids a load-time `device_wait_idle` stall (mirrors the mesh retire path).
    retired_textures: buffers::RetireQueue<BlockTextures>,
    /// The minimap texture (per-slot double buffer), sampled by `tris2d_tex`.
    minimap: MinimapTexture,

    frames: Vec<FrameSlot>,
    /// One per swapchain image: signaled by the copy submit, waited by present.
    /// Binary because the WSI rejects timeline semaphores.
    present_semaphores: Vec<BinarySemaphore>,
    imm: [HostBuffer; FRAMES_IN_FLIGHT as usize],
    /// Per-slot per-draw offsets SSBO (one [`DrawOffset`] per command, in
    /// lockstep with the indirect commands so `first_instance == command_index`).
    /// Immediate debug geometry uses the descriptor-set-free debug pipelines, so
    /// there is no reserved slot.
    offsets: [HostBuffer; FRAMES_IN_FLIGHT as usize],
    /// Per-slot indirect command buffer (20-byte VkDrawIndexedIndirectCommand
    /// per mesh draw, instance_count 1, first_instance = SSBO slot).
    indirect: [HostBuffer; FRAMES_IN_FLIGHT as usize],
    /// Single push-descriptor layout for the 3D pipeline: set 0 binding 0 =
    /// per-draw offsets SSBO (vertex), binding 1 = block texture array
    /// (fragment). Vulkan permits one push set per layout, so both share it.
    mesh3d_set_layout: vk::DescriptorSetLayout,

    /// Frame scratch (persistent capacity): resolved draws, sorted by (pass, arena).
    draw_scratch: Vec<DrawEntry>,
    /// Frame scratch: the offsets SSBO contents (one per command).
    draw_offsets_data: Vec<DrawOffset>,
    /// Frame scratch: indirect commands, (pass, arena)-contiguous.
    draw_commands: Vec<DrawIndexedIndirect>,
    /// Frame scratch: one entry per (pass, arena) group with visible draws.
    draw_runs: Vec<DrawRun>,
    /// Frame scratch: resolved surface draws (handle → GpuSurface), in the order
    /// the app recorded them. Cleared each frame.
    surface_scratch: Vec<SurfaceEntry>,
    /// Frame scratch: contiguous runs of surface commands keyed by surface buffer.
    surface_runs: Vec<DrawRun>,

    /// Opt-in six-way face culling. Off (default) → one draw per mesh, byte
    /// identical to a single coalesced run. On → per-mesh, the ≤3 maximal
    /// contiguous runs of camera-facing direction buckets.
    cull_faces: bool,

    /// Command buffer for the offscreen->swapchain present copy. A single
    /// one suffices: at most one copy is ever in flight, gated by the previous
    /// copy's timeline value being reached before the next is submitted.
    copy_cmd: vk::CommandBuffer,
    /// The one monotonic timeline: render submits and present copies both
    /// signal it, replacing the old per-slot render fence + render_done
    /// semaphore + global copy fence.
    timeline: Timeline,
    /// Timeline value of the last present copy submitted; the mailbox present
    /// decision probes this non-blockingly, and vsync pacing waits on it.
    last_copy_value: TimelineValue,
    /// Timeline value of the last render submitted; freed meshes are stamped
    /// with it.
    last_render_value: TimelineValue,
    /// Which offscreen slot the in-flight copy reads, if any. Rendering to
    /// that slot again must wait its `copy_value` first (rare, sub-millisecond).
    copy_slot: Option<usize>,

    /// Set by [`Self::request_screenshot`]; consumed by the next present copy,
    /// which reads the swapchain image back to a host buffer and writes a PNG.
    pending_screenshot: Option<std::path::PathBuf>,

    slot: usize,

    /// Per-slot: has this slot's depth image been rendered at least once since
    /// (re)creation? The VRS compute pass reads it, so it must skip a slot until
    /// its depth is validly written and in `DEPTH_ATTACHMENT_OPTIMAL`.
    vrs_depth_ready: [bool; FRAMES_IN_FLIGHT as usize],
    /// Scene fingerprint that produced each slot's reusable depth image; VRS may
    /// reuse a slot only while it still matches [`Self::scene_fingerprint`].
    vrs_scene_fingerprint: [Option<u64>; FRAMES_IN_FLIGHT as usize],
    /// Fingerprint of the scene being recorded in the current frame.
    scene_fingerprint: u64,

    vsync: Pending<bool>,
    msaa: Pending<SampleCount>,
    needs_recreate: bool,
    /// Resolution scale for the 3D/UI render target relative to the window
    /// (0.25..=2.0). The present copy becomes a filtered blit when != 1.
    render_scale: Pending<f32>,
    /// The offscreen/depth/MSAA extent: swapchain extent * render_scale.
    render_extent: vk::Extent2D,
    /// Present pacing for the vsync-off path: presents are attempted at the
    /// display's refresh cadence so queue_present never has to wait for a
    /// drawable; frames in between render unthrottled.
    last_present: std::time::Instant,
    present_interval: std::time::Duration,
    gpu_timer: GpuTimer,
}

/// A one-shot host-visible buffer holding a screenshot's pixels, copied from
/// the swapchain image and freed once encoded.
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
        } = cfg;
        let render_scale = Scale::new(render_scale).as_f32();

        let device = Device::new(&instance.instance, &surface_loader, surface);
        // The block-suballocating allocator lives on MAIN; the render side only
        // installs pre-built residents. Mesh uploads bypass staging on unified
        // memory (logged main-side is fine, but keep the hint here too).
        let mesh_res = MeshResidency::new();
        let surface_res = SurfaceResidency::new();

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
            device.anisotropy,
        );
        let mesh3d_set_layout = buffers::create_mesh3d_set_layout(&device.device);

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
        let frames = cmds
            .into_iter()
            .map(|cmd| FrameSlot {
                cmd,
                image_available: unsafe { BinarySemaphore::new(&device.device) },
                render_value: TimelineValue::START,
                copy_value: TimelineValue::START,
            })
            .collect();

        let present_semaphores = create_present_semaphores(&device.device, swapchain.images.len());
        let imm = std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::VERTEX_BUFFER));
        let offsets =
            std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER));
        let indirect =
            std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::INDIRECT_BUFFER));

        let gpu_timer = GpuTimer::new(
            &device.device,
            device.timestamps_supported,
            device.timestamp_period_ns,
        );

        let caps = DeviceCaps {
            max_msaa: device.max_msaa(),
        };
        let reply = InitReply {
            instance: instance.instance.clone(),
            physical: device.physical,
            memory_budget: device.memory_budget,
            device: device.device.clone(),
            caps,
        };

        let renderer = Self {
            instance,
            surface_loader,
            surface,
            device,
            mesh_res,
            surface_res,
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
            frames,
            present_semaphores,
            imm,
            offsets,
            indirect,
            mesh3d_set_layout,
            draw_scratch: Vec::new(),
            draw_offsets_data: Vec::new(),
            cull_faces: false,
            draw_commands: Vec::new(),
            draw_runs: Vec::new(),
            surface_scratch: Vec::new(),
            surface_runs: Vec::new(),
            copy_cmd,
            timeline,
            last_copy_value: TimelineValue::START,
            last_render_value: TimelineValue::START,
            copy_slot: None,
            pending_screenshot: None,
            slot: 0,
            vrs_depth_ready: [false; FRAMES_IN_FLIGHT as usize],
            vrs_scene_fingerprint: [None; FRAMES_IN_FLIGHT as usize],
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

    /// Updates the present size from a resize event and flags a swapchain rebuild.
    pub(crate) fn on_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.size = vk::Extent2D {
            width: size.width,
            height: size.height,
        };
        self.needs_recreate = true;
    }

    // Setters are driven by the ordered command stream (`RenderCmd`); the
    // matching getters live on `RenderClient`, which caches the requested state
    // main-side, so the renderer exposes no getters of its own.

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

    pub fn set_cull_faces(&mut self, on: bool) {
        self.cull_faces = on;
    }

    /// Requests a render-resolution scale; returns the clamped value that
    /// will apply at the next frame boundary.
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
        resident: GpuResident,
    ) {
        self.mesh_res.apply_upload(slot, generation, resident);
    }

    /// Retires a mesh resident past the latest submitted timeline value. The
    /// render thread stamps `done_at` here (main has no timeline to read).
    pub(crate) fn apply_free_mesh(&mut self, slot: u32, generation: NonZeroU32) {
        self.mesh_res
            .apply_free(slot, generation, self.last_render_value);
    }

    pub(crate) fn apply_upload_surface(
        &mut self,
        slot: u32,
        generation: NonZeroU32,
        resident: GpuResident,
    ) {
        self.surface_res.apply_upload(slot, generation, resident);
    }

    pub(crate) fn apply_free_surface(&mut self, slot: u32, generation: NonZeroU32) {
        self.surface_res
            .apply_free(slot, generation, self.last_render_value);
    }

    /// Requests that the next presented frame be saved to `path` as a PNG.
    /// The capture happens inside the next present copy (the exact image shown
    /// on screen). A pending request is overwritten if called again before it
    /// is serviced.
    pub fn request_screenshot(&mut self, path: std::path::PathBuf) {
        self.pending_screenshot = Some(path);
    }

    /// Replaces the block texture array (RGBA8, `layers.len()` images of
    /// `size*size*4` bytes each). Rare operation: the new array is uploaded and
    /// swapped in immediately, and the old array is retired through the timeline
    /// (destroyed once every in-flight frame that could sample it is done),
    /// avoiding a load-time `device_wait_idle` stall — pipelines and descriptors
    /// untouched, since the current texture is pushed afresh each frame.
    pub fn set_block_textures(&mut self, size: u32, layers: &[Vec<u8>]) {
        // Build new array before swapping out old to avoid double-free on panic.
        let new_textures = BlockTextures::upload(
            &self.instance.instance,
            &self.device.device,
            self.device.physical,
            self.device.graphics_queue,
            self.device.command_pool,
            self.device.anisotropy,
            size,
            layers,
        );
        let old_textures = std::mem::replace(&mut self.block_textures, new_textures);
        // The old array may still be sampled by frames already submitted; retire
        // it past the highest reserved timeline value so it outlives them.
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
        let rs = {
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
            self.present(slot, present_target, lists.warp_map, overlay);
        }

        self.slot = (self.slot + 1) % self.frames.len();
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
    fn wait_slot_and_reclaim(&mut self, slot: usize) {
        let device = &self.device.device;
        unsafe {
            self.timeline.wait(device, self.frames[slot].render_value);
            let current = self.timeline.counter(device);
            // Retired allocations return to the main-owned allocator freelist;
            // staging-block shrink happens main-side after it reclaims them.
            let ret = &self.ret;
            self.mesh_res
                .collect(current, &mut |a| drop(ret.send(RenderReturn::FreeAlloc(a))));
            self.surface_res
                .collect(current, &mut |a| drop(ret.send(RenderReturn::FreeAlloc(a))));
            self.retired_textures
                .collect(current, |mut tex| tex.destroy(device));
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
            unsafe { self.timeline.wait(device, self.frames[slot].copy_value) };
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
        let present_due = self.vsync.current()
            || self.last_present.elapsed() >= self.present_interval.mul_f32(PRESENT_THROTTLE);
        let mut present_target = None;
        let device = &self.device.device;
        unsafe {
            // Skip present if previous copy still in flight (mailbox drop).
            let copy_ready =
                present_due && self.timeline.probe(device).reached(self.last_copy_value);
            if copy_ready {
                // With vsync: wait for image. Without: never wait, allow drop.
                let timeout = if self.vsync.current() { u64::MAX } else { 0 };
                match acquire_next_image(
                    &self.swapchain.loader,
                    self.swapchain.swapchain,
                    timeout,
                    self.frames[slot].image_available,
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

    /// Resolves frame mesh draws into indirect commands, offsets, and runs.
    ///
    /// Each visible mesh emits one [`DrawEntry`] per direction-run (one run —
    /// the whole mesh — unless `cull_faces` splits it). Entries are sorted by
    /// `(pass, arena)` so opaque draws form a prefix and transparent a suffix,
    /// and same-arena same-pass draws stay contiguous for batching. A
    /// [`DrawOffset`] is pushed per command, in lockstep with the commands, so
    /// `first_instance == command_index` — the shader reads `draw_offsets`
    /// through the raw InstanceIndex.
    fn prepare_mesh_draws(&mut self, slot: usize, lists: &DrawLists) {
        use ash::vk::Handle;

        self.draw_scratch.clear();
        self.draw_offsets_data.clear();
        self.draw_commands.clear();
        self.draw_runs.clear();
        self.surface_scratch.clear();
        self.surface_runs.clear();

        if lists.has_3d {
            for d in &lists.mesh_draws {
                // Gen-checked resolve against the residency mirror: Option-skip a
                // stale snapshot referencing a since-freed/realloc'd slot.
                let Some(buffer) = self.mesh_res.resolve(d) else {
                    continue;
                };
                let (offset, scale, biased) = (d.offset, d.scale, d.biased);
                let (pass, vertex_offset) = (d.pass, d.vertex_offset);
                let (amin, amax) = (d.aabb_min, d.aabb_max);
                let bounds = d.bounds;
                let center = offset + (amin + amax) * 0.5 * scale;
                let dist2 = (center - lists.cam_pos).length_squared();
                let mut emit = |range: std::ops::Range<u32>| {
                    if range.is_empty() {
                        return;
                    }
                    self.draw_scratch.push(DrawEntry {
                        buffer,
                        pass,
                        biased,
                        first: range.start,
                        count: range.end - range.start,
                        vertex_offset,
                        offset,
                        scale,
                        dist2,
                    });
                };
                if self.cull_faces {
                    // Compare the camera against the mesh's world-space (scaled)
                    // AABB, in the mesh's own frame (camera − offset).
                    let cam_local = lists.cam_pos - offset;
                    let vis = visible_dirs(cam_local, amin * scale, amax * scale);
                    let (runs, n) = contiguous_runs(vis);
                    for &(start, end) in &runs[..n] {
                        emit(bounds[start as usize]..bounds[end as usize]);
                    }
                } else {
                    emit(bounds[0]..bounds[6]);
                }
            }
            self.draw_scratch.sort_unstable_by(|a, b| {
                a.pass
                    .cmp(&b.pass) // opaque prefix, transparent suffix
                    // Non-biased opaque (full-res chunks) before biased (far-LOD
                    // tiles): chunks fill depth first so the tile backdrop is
                    // early-Z rejected where it's occluded.
                    .then_with(|| a.biased.cmp(&b.biased))
                    .then_with(|| match a.pass {
                        // Opaque/cutout near→far: reversed-Z early-Z rejects occluded
                        // fragments (both write depth).
                        Pass::Opaque | Pass::Cutout => a.dist2.total_cmp(&b.dist2),
                        // Blend far→near: correct back-to-front alpha compositing.
                        Pass::Blend => b.dist2.total_cmp(&a.dist2),
                    })
                    // Deterministic tiebreak; keeps equidistant same-arena draws batched.
                    .then_with(|| a.buffer.as_raw().cmp(&b.buffer.as_raw()))
            });

            for entry in &self.draw_scratch {
                let ssbo_slot = self.draw_offsets_data.len() as u32;
                let command_index = self.draw_commands.len() as u32;
                debug_assert_eq!(ssbo_slot, command_index, "offset slot must track command");
                self.draw_offsets_data.push(DrawOffset {
                    offset: entry.offset.to_array(),
                    scale: entry.scale,
                });
                self.draw_commands.push(DrawIndexedIndirect {
                    index_count: entry.count,
                    instance_count: 1,
                    first_index: entry.first,
                    vertex_offset: entry.vertex_offset,
                    first_instance: ssbo_slot,
                });
                match self.draw_runs.last_mut() {
                    Some(run)
                        if run.buffer == entry.buffer
                            && run.pass == entry.pass
                            && run.biased == entry.biased =>
                    {
                        run.count += 1
                    }
                    _ => self.draw_runs.push(DrawRun {
                        buffer: entry.buffer,
                        pass: entry.pass,
                        biased: entry.biased,
                        first: command_index,
                        count: 1,
                    }),
                }
            }

            // Resolve the surface lane (Zone-3 far skin). Frustum culling already
            // happened app-side; here just resolve handle→GpuSurface (skip if
            // missing) and copy out the fields so the `surfaces` borrow is
            // released before the shared command/offset buffers are mutated.
            for d in &lists.surface_draws {
                let Some(buffer) = self.surface_res.resolve(d) else {
                    continue;
                };
                self.surface_scratch.push(SurfaceEntry {
                    buffer,
                    first: d.index_first,
                    count: d.index_count,
                    vertex_offset: d.vertex_offset,
                    offset: d.offset,
                    scale: d.scale,
                });
            }
            // Append surface offsets AND commands to the SAME host buffers AFTER
            // the voxel ones so the addressing scheme stays consistent. Runs are keyed by surface buffer.
            for entry in &self.surface_scratch {
                let ssbo_slot = self.draw_offsets_data.len() as u32;
                let command_index = self.draw_commands.len() as u32;
                debug_assert_eq!(ssbo_slot, command_index, "offset slot must track command");
                self.draw_offsets_data.push(DrawOffset {
                    offset: entry.offset.to_array(),
                    scale: entry.scale,
                });
                self.draw_commands.push(DrawIndexedIndirect {
                    index_count: entry.count,
                    instance_count: 1,
                    first_index: entry.first,
                    vertex_offset: entry.vertex_offset,
                    first_instance: ssbo_slot,
                });
                match self.surface_runs.last_mut() {
                    Some(run) if run.buffer == entry.buffer => run.count += 1,
                    _ => self.surface_runs.push(DrawRun {
                        buffer: entry.buffer,
                        pass: Pass::Opaque,
                        biased: false,
                        first: command_index,
                        count: 1,
                    }),
                }
            }
        }
        self.scene_fingerprint =
            scene_fingerprint(lists, &self.draw_scratch, &self.surface_scratch);

        let offsets_bytes: &[u8] = bytemuck::cast_slice(&self.draw_offsets_data);
        let indirect_bytes: &[u8] = bytemuck::cast_slice(&self.draw_commands);
        unsafe {
            let ssbo = &mut self.offsets[slot];
            ssbo.maintain(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                offsets_bytes.len() as u64,
            );
            if !offsets_bytes.is_empty() {
                ssbo.write(0, offsets_bytes);
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

    /// Records the command buffer: mesh copies, render pass, and transitions.
    fn record_render(
        &mut self,
        guard: &SlotGuard,
        lists: &DrawLists,
        offsets: ImmOffsets,
    ) -> RenderSubmit {
        let slot = guard.0;
        let cmd = self.frames[slot].cmd;
        // Read the prior render-pass GPU time for this slot before its queries
        // are reset below (the slot's fence was already waited this frame).
        let profiling = crate::profile::is_enabled();
        if profiling {
            let mut passes = [0.0f64; GpuPass::COUNT];
            if unsafe { self.gpu_timer.read_into(&self.device.device, slot, &mut passes) }.is_some()
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

            self.mesh_res.flush_copies(device, cmd, done_at);
            self.surface_res.flush_copies(device, cmd, done_at);
            // Upload this slot's minimap texture (if its version is stale) on the
            // live frame command buffer, before the render pass begins.
            self.minimap.sync(device, cmd, slot);
            if profiling {
                self.gpu_timer.begin(device, cmd, slot);
            }
        }

        // VRS generation needs a validly-written, single-sampled depth image to
        // classify. MSAA depth would need multisample sampling (skipped), and a
        // slot's depth is only readable once it has been rendered at least once.
        let do_vrs = lists.has_3d
            && self.targets.vrs.is_some()
            && self.targets.msaa.is_none()
            && self.vrs_depth_ready[slot]
            && self.vrs_scene_fingerprint[slot] == Some(self.scene_fingerprint);

        let device = &self.device.device;
        let stamp = |p| {
            if profiling {
                unsafe { self.gpu_timer.mark(device, cmd, slot, p) };
            }
        };
        let pass = unsafe { RenderPass::begin(self, cmd, slot, lists, offsets, do_vrs) };
        if lists.has_3d {
            // Transparency forces an interleave: all opaque geometry (mesh runs
            // AND opaque debug cubes/lines) writes depth before any transparent
            // mesh run tests against it.
            unsafe {
                pass.record_mesh_indirect(Pass::Opaque);
                // Cutout writes depth like opaque, so it belongs in the opaque
                // prefix (before sky). Dormant until a block emits it.
                pass.record_mesh_indirect(Pass::Cutout);
                stamp(GpuPass::Opaque);
                // Zone-3 far skin: after chunks/tiles laid opaque depth (they
                // early-Z the hidden parts), before sky fills behind the edge.
                pass.record_surface_indirect();
                // Sky fills the background (uncovered pixels) right after opaque
                // depth is laid down. It must precede the immediate debug
                // cubes/lines: the highlight lines are depth read-only (no depth
                // write), so a line silhouetted against the background leaves the
                // depth cleared there — drawing sky afterward would overpaint it.
                // Debug geometry and transparent water both composite over the sky.
                pass.record_sky();
                stamp(GpuPass::Sky);
                pass.record_immediate_cubes();
                stamp(GpuPass::Cubes);
                pass.record_lines();
                stamp(GpuPass::Lines);
                // Contact shadows: translucent, blended over the opaque terrain
                // depth just laid down, before transparent water.
                pass.record_shadows();
                stamp(GpuPass::Shadows);
                pass.record_mesh_indirect(Pass::Blend);
                stamp(GpuPass::Transparent);
            }
        }
        unsafe {
            // In wide-FOV the overlay is drawn AFTER the tonemap resample (swapchain
            // space) so the warp never bends it; here it would be compressed. In
            // rectilinear it stays in the offscreen pass exactly as before.
            if lists.warp_map.is_identity() {
                pass.record_2d();
            }
            stamp(GpuPass::Overlay);
            pass.end();
        }
        self.gpu_timer.finish(slot);
        // The main pass just wrote (and stored) this slot's depth, so a later
        // cycle reusing this slot may read it for VRS classification.
        self.vrs_depth_ready[slot] = true;
        self.vrs_scene_fingerprint[slot] = Some(self.scene_fingerprint);

        unsafe {
            self.device
                .device
                .end_command_buffer(cmd)
                .expect("end command buffer failed");
        }
        rs
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
        let depth = &self.targets.depth[slot];
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
                    .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(depth.image)
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
                .image_view(depth.view)
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
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .image(depth.image)
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

    /// Submits the recorded command buffer and advances the timeline.
    fn submit_render(&mut self, rs: RenderSubmit, slot: usize) {
        let completion = unsafe {
            rs.submit(
                &self.device.device,
                self.device.graphics_queue,
                &self.timeline,
            )
        };
        self.frames[slot].render_value = completion.value();
        self.last_render_value = completion.value();
    }

    /// Copies the finished frame into the acquired swapchain image (when one
    /// was acquired in [`Self::decide_present`]) and queues the present.
    fn present(
        &mut self,
        slot: usize,
        present_target: Option<u32>,
        warp_map: crate::camera::WarpMap,
        overlay: OverlayPresent,
    ) {
        if let Some(image_index) = present_target {
            unsafe { self.submit_present_copy(slot, image_index, warp_map, overlay) };
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
                device.cmd_bind_vertex_buffers(cmd, 0, &[self.imm[slot].buffer], &[overlay.d2_offset]);
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
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.imm[slot].buffer],
                    &[overlay.d2_tex_offset],
                );
                device.cmd_draw(cmd, overlay.d2_tex_count, 1, 0, 0);
            }
        }
    }

    /// Records and submits the offscreen[slot] -> swapchain copy, then
    /// queues the present. Caller guarantees the previous copy has retired
    /// (its value reached) and the image was just acquired with
    /// `frames[slot].image_available`.
    unsafe fn submit_present_copy(
        &mut self,
        slot: usize,
        image_index: u32,
        warp_map: crate::camera::WarpMap,
        overlay: OverlayPresent,
    ) {
        // A pending screenshot piggybacks on this copy: after the tonemap draw,
        // the swapchain image is read back into `readback` instead of going
        // straight to PRESENT. Allocate the host buffer before borrowing
        // `device` so the read-back path adds no &mut-self conflicts below.
        let capture = self.pending_screenshot.take();
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
        const EXPOSURE: f32 = 1.0;
        let tonemap_push = warp_map.push(EXPOSURE);
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
            image_upload::push_combined_image_sampler(
                &self.device.push_descriptor,
                self.copy_cmd,
                self.pipelines.layout_tonemap,
                0,
                self.pipelines.tonemap_sampler,
                self.targets.offscreen[slot].view,
            );
            device.cmd_push_constants(
                self.copy_cmd,
                self.pipelines.layout_tonemap,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&tonemap_push),
            );
            device.cmd_draw(self.copy_cmd, 3, 1, 0, 0);
            // Wide-FOV only: composite the 2D overlay onto the tonemapped swapchain
            // (skipped in the offscreen scene pass) so the warp never bends it. Uses
            // a GL-style negative-height viewport, matching what tris2d.vert expects.
            if !warp_map.is_identity() {
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
            }
            device.cmd_end_rendering(self.copy_cmd);

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
                self.frames[slot].image_available,
                RenderCompletion::from_value(self.frames[slot].render_value),
                self.present_semaphores[image_index as usize],
            );
            self.frames[slot].copy_value = value;
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
        if let (Some(path), Some(rb)) = (capture, readback) {
            unsafe { self.finish_screenshot(rb, extent, path) };
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
    unsafe fn finish_screenshot(
        &self,
        rb: Readback,
        extent: vk::Extent2D,
        path: std::path::PathBuf,
    ) {
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
            match crate::screenshot::write_png(&path, width, height, &pixels) {
                Ok(()) => log::info!("screenshot saved: {}", path.display()),
                Err(e) => log::error!("screenshot encode failed ({}): {e}", path.display()),
            }
        });
    }

    /// While no frames are being submitted (minimized window): waits out the
    /// in-flight fences, flushes any staged mesh copies with a standalone
    /// submit, and frees the whole retire queue.
    unsafe fn reclaim_while_idle(&mut self) {
        if !self.mesh_res.has_pending()
            && !self.mesh_res.has_garbage()
            && !self.surface_res.has_pending()
            && !self.surface_res.has_garbage()
        {
            return;
        }
        let device = &self.device.device;
        unsafe {
            // Wait for all in-flight submits to complete.
            self.timeline.wait(device, self.timeline.last_reserved());
            self.copy_slot = None;

            if self.mesh_res.has_pending() || self.surface_res.has_pending() {
                // Reuse slot 0's command buffer.
                let cmd = self.frames[0].cmd;
                device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .expect("command buffer reset failed");
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device
                    .begin_command_buffer(cmd, &begin)
                    .expect("begin command buffer failed");
                self.mesh_res
                    .flush_copies(device, cmd, self.last_render_value);
                self.surface_res
                    .flush_copies(device, cmd, self.last_render_value);
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

            // GPU idle + copies flushed: everything retired returns to main.
            let ret = &self.ret;
            self.mesh_res
                .collect_all(&mut |a| drop(ret.send(RenderReturn::FreeAlloc(a))));
            self.surface_res
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
            // Offscreen images recreated; clear copy tracking.
            self.clear_copy();
            // Depth images recreated (layout UNDEFINED): VRS must re-prime.
            self.vrs_depth_ready = [false; FRAMES_IN_FLIGHT as usize];
            self.vrs_scene_fingerprint = [None; FRAMES_IN_FLIGHT as usize];

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
    // 2D coordinates stay in window pixels (NDC is resolution-free), while
    // rendering happens at the (possibly scaled) offscreen resolution.
    window_extent: vk::Extent2D,
    offscreen_image: vk::Image,
    ended: bool,
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
        let window_extent = r.swapchain.extent;
        let offscreen_image = r.targets.offscreen[slot].image;
        unsafe {
            // Generate the rate map first: it samples this slot's depth (leaving
            // it in DEPTH_ATTACHMENT_OPTIMAL, ready for the pass below) and
            // returns the only valid `RateAttachment`. Done before the color
            // barriers so the compute dispatch overlaps nothing it depends on.
            let rate = do_vrs.then(|| {
                let focal_px = 0.5 * extent.height as f32 / lists.fovy_tan_half.max(1e-4);
                let d_threshold = crate::camera::Z_NEAR / focal_px;
                r.record_vrs_generate(cmd, slot, d_threshold)
            });

            // Transition attachments to render targets; old contents discarded.
            // Depth is transitioned by the VRS pass above when `do_vrs`.
            let mut image_barriers = [vk::ImageMemoryBarrier2::default(); 3];
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
            if !do_vrs {
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
                    .image(r.targets.depth[slot].image)
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
                    .image(msaa.image)
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
                        lists.clear.r as f32 / 255.0,
                        lists.clear.g as f32 / 255.0,
                        lists.clear.b as f32 / 255.0,
                        1.0,
                    ],
                },
            };
            let offscreen_view = r.targets.offscreen[slot].view;
            let mut color_attachment = if let Some(msaa) = &r.targets.msaa {
                vk::RenderingAttachmentInfo::default()
                    .image_view(msaa.view)
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

            // Reversed-Z: clear depth to 0.0, GREATER_OR_EQUAL test. Stored so a
            // later cycle reusing this slot can classify it for VRS.
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(r.targets.depth[slot].view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });

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
            window_extent,
            offscreen_image,
            ended: false,
        }
    }

    /// Pushes the `layout_3d` constants (view_proj + sky lighting/fog) and the
    /// push descriptors (per-draw offsets SSBO at binding 0, block-texture array
    /// at binding 1) shared by both mesh passes. Called at the head of each mesh
    /// pass rather than once up front, because interleaved passes bind
    /// incompatible layouts that disturb this state. Only sound when at least
    /// one mesh run exists (else the offsets SSBO can be a null buffer).
    unsafe fn bind_mesh3d_state(&self) {
        let r = self.r;
        let cmd = self.cmd;
        let push = pipeline::Mesh3dPush {
            view_proj: self.lists.view_proj,
            sky: self.lists.sky_light,
        };
        unsafe {
            r.device.device.cmd_push_constants(
                cmd,
                r.pipelines.layout_3d,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            buffers::push_mesh3d_descriptors(
                &r.device.push_descriptor,
                cmd,
                r.pipelines.layout_3d,
                r.offsets[self.slot].buffer,
                r.block_textures.sampler,
                r.block_textures.view,
            );
        }
    }

    /// Issues indirect mesh draws for one pass, using the best available
    /// feature level and falling back from multi-draw to single-draw indirect
    /// as needed. Runs are sorted so a pass's runs are contiguous; the pass
    /// pipeline binds once, before the first matching run. Only called when
    /// `lists.has_3d`.
    unsafe fn record_mesh_indirect(&self, pass: Pass) {
        if !self.r.draw_runs.iter().any(|run| run.pass == pass) {
            return;
        }
        // Interleaved debug/sky/2D passes bind pipelines with layouts that are
        // not push-compatible with `layout_3d`, which per Vulkan's layout-
        // compatibility rules disturbs this layout's push constants and push
        // descriptors. Re-establish them at the head of every mesh pass so the
        // transparent pass (recorded after sky) draws with valid state.
        unsafe { self.bind_mesh3d_state() };
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            let indirect_buffer = self.r.indirect[self.slot].buffer;
            const STRIDE: u64 = std::mem::size_of::<DrawIndexedIndirect>() as u64;
            // Opaque runs may carry a `biased` flag selecting `mesh3d_biased`
            // (far-LOD tiles); rebind only when the selected pipeline changes.
            let mut bound: Option<vk::Pipeline> = None;
            for run in self.r.draw_runs.iter().filter(|run| run.pass == pass) {
                let pipeline = if run.biased {
                    self.r.pipelines.mesh3d_biased
                } else {
                    self.r.pipelines.pipeline_for(pass)
                };
                if bound != Some(pipeline) {
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    bound = Some(pipeline);
                }
                device.cmd_bind_index_buffer(cmd, run.buffer, 0, vk::IndexType::UINT32);
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
    }

    /// Issues the retained colored-surface draws (Zone-3 far skin) through the
    /// `surface3d` pipeline. Reuses `layout_3d` state via [`bind_mesh3d_state`]
    /// (valid because `surface3d` uses `layout_3d`); the surface commands were
    /// appended to the shared indirect/offset buffers after the mesh ones, so
    /// their `first_instance` still indexes the offsets SSBO correctly. Only
    /// called when `lists.has_3d`. Recorded after opaque meshes/tiles (they
    /// early-Z the hidden parts) and before sky (which fills behind the edge).
    unsafe fn record_surface_indirect(&self) {
        if self.r.surface_runs.is_empty() {
            return;
        }
        // Re-establish layout_3d push constants + descriptors (the opaque mesh
        // pass bound the same layout, but keep this self-contained/robust to
        // recording order).
        unsafe { self.bind_mesh3d_state() };
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            let indirect_buffer = self.r.indirect[self.slot].buffer;
            const STRIDE: u64 = std::mem::size_of::<DrawIndexedIndirect>() as u64;
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.r.pipelines.surface3d,
            );
            // Surface-specific push: carry the skin clip radius in the otherwise
            // unused `sun_light.w` lane (mesh passes ignore it). The fragment
            // discards skin within this horizontal radius so the far grey skin
            // renders only BEYOND the near zones, never poking through them.
            let mut push = pipeline::Mesh3dPush {
                view_proj: self.lists.view_proj,
                sky: self.lists.sky_light,
            };
            push.sky.sun_light[3] = self.lists.skin_clip;
            device.cmd_push_constants(
                cmd,
                self.r.pipelines.layout_3d,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            for run in &self.r.surface_runs {
                // Index and vertex data share one VkBuffer: indices bound at
                // offset 0 (first_index in the command), vertices via the
                // command's vertex_offset — mirroring MeshRegistry's layout.
                device.cmd_bind_index_buffer(cmd, run.buffer, 0, vk::IndexType::UINT32);
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
    }

    /// Pushes `view_proj` to `layout_debug` for the immediate debug geometry.
    /// Done per debug pass because the mesh passes bind `layout_3d`, whose
    /// incompatible push-constant range disturbs this value.
    unsafe fn push_debug_view_proj(&self) {
        let push = pipeline::DebugPush {
            view_proj: self.lists.view_proj,
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
    /// offset 0). Only issued when `lists.has_3d`.
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
                self.push_debug_view_proj();
                device.cmd_bind_vertex_buffers(cmd, 0, &[self.r.imm[self.slot].buffer], &[0]);
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
                self.push_debug_view_proj();
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.r.imm[self.slot].buffer],
                    &[self.offsets.shadow],
                );
                device.cmd_draw(cmd, self.lists.shadow_verts.len() as u32, 1, 0, 0);
            }
        }
    }

    /// The immediate-mode debug lines (debug_lines pipeline). Only issued when
    /// `lists.has_3d`.
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
                self.push_debug_view_proj();
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.r.imm[self.slot].buffer],
                    &[self.offsets.line],
                );
                device.cmd_draw(cmd, self.lists.line_verts.len() as u32, 1, 0, 0);
            }
        }
    }

    /// The procedural sky background pass (sky pipeline, push-constant only, no
    /// vertex buffer or descriptor set). A single fullscreen triangle at the
    /// reversed-Z far plane; the read-only depth test rejects it wherever
    /// terrain wrote closer depth, so it shades only background pixels. Skipped
    /// unless the frame set a sky palette.
    unsafe fn record_sky(&self) {
        let Some(desc) = self.lists.sky else {
            return;
        };
        let device = &self.r.device.device;
        let cmd = self.cmd;
        let params = pipeline::SkyParams::compose(self.lists.view_proj.inverse(), &desc);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.r.pipelines.sky);
            device.cmd_push_constants(
                cmd,
                self.r.pipelines.layout_sky,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&params),
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    /// The 2D overlay pass (tris2d pipeline, its own atlas descriptor set and
    /// pixels-to-NDC push constant). Issued regardless of `has_3d`.
    unsafe fn record_2d(&self) {
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            if !self.lists.verts_2d.is_empty() {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.r.pipelines.tris2d,
                );
                self.r.atlas.push_descriptor(
                    &self.r.device.push_descriptor,
                    cmd,
                    self.r.pipelines.layout_2d,
                    0,
                );
                let pixels_to_ndc = [
                    2.0 / self.window_extent.width as f32,
                    2.0 / self.window_extent.height as f32,
                ];
                device.cmd_push_constants(
                    cmd,
                    self.r.pipelines.layout_2d,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&pixels_to_ndc),
                );
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.r.imm[self.slot].buffer],
                    &[self.offsets.d2],
                );
                device.cmd_draw(cmd, self.lists.verts_2d.len() as u32, 1, 0, 0);
            }

            // Minimap pass: own descriptor set for the texture.
            if self.r.minimap.ready() && !self.lists.tex_verts_2d.is_empty() {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.r.pipelines.tris2d_tex,
                );
                self.r.minimap.push_descriptor(
                    &self.r.device.push_descriptor,
                    cmd,
                    self.r.pipelines.layout_2d,
                    self.slot,
                );
                let pixels_to_ndc = [
                    2.0 / self.window_extent.width as f32,
                    2.0 / self.window_extent.height as f32,
                ];
                device.cmd_push_constants(
                    cmd,
                    self.r.pipelines.layout_2d,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&pixels_to_ndc),
                );
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.r.imm[self.slot].buffer],
                    &[self.offsets.d2_tex],
                );
                device.cmd_draw(cmd, self.lists.tex_verts_2d.len() as u32, 1, 0, 0);
            }
        }
    }

    /// Ends dynamic rendering and transitions the offscreen image to be sampled
    /// by the tonemap pass. Ordering across the render/tonemap submits is
    /// enforced by the timeline; this barrier only owns the layout + visibility.
    unsafe fn end(mut self) {
        let device = &self.r.device.device;
        let cmd = self.cmd;
        unsafe {
            device.cmd_end_rendering(cmd);

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
        self.ended = true;
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
            self.block_textures.destroy(device);
            self.retired_textures
                .collect_all(|mut tex| tex.destroy(device));
            self.gpu_timer.destroy(device);
            self.targets.destroy(device);
            for buffer in self
                .imm
                .iter_mut()
                .chain(&mut self.offsets)
                .chain(&mut self.indirect)
            {
                buffer.destroy(device);
            }
            // The residents' allocations belong to the main-owned allocator
            // (destroyed there after this returns); just drop them — no Vulkan
            // calls, GPU already idle.
            self.mesh_res.destroy_all(&mut |_a| {});
            self.surface_res.destroy_all(&mut |_a| {});
            for &sem in &self.present_semaphores {
                sem.destroy(device);
            }
            self.timeline.destroy(device);
            for frame in &self.frames {
                frame.image_available.destroy(device);
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
}

impl GpuPass {
    const ALL: [GpuPass; 7] = [
        GpuPass::Opaque,
        GpuPass::Sky,
        GpuPass::Cubes,
        GpuPass::Lines,
        GpuPass::Shadows,
        GpuPass::Transparent,
        GpuPass::Overlay,
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
    fn visible_dirs_over_the_27_camera_regions() {
        // Unit box at origin; camera swept over the 27 relative regions.
        let (min, max) = (glam::Vec3::ZERO, glam::Vec3::splat(1.0));
        for sx in [-1i32, 0, 1] {
            for sy in [-1, 0, 1] {
                for sz in [-1, 0, 1] {
                    // Region centers: below min (-1), inside (0.5), above max (2).
                    let coord = |s: i32| match s {
                        -1 => -1.0,
                        0 => 0.5,
                        _ => 2.0,
                    };
                    let cam = glam::Vec3::new(coord(sx), coord(sy), coord(sz));
                    let vis = visible_dirs(cam, min, max);
                    // Per axis: inside → both faces; else only the near one.
                    let expect = |s: i32| match s {
                        -1 => (false, true), // below min: only the −face
                        0 => (true, true),   // inside: both
                        _ => (true, false),  // above max: only the +face
                    };
                    assert_eq!((vis[0], vis[1]), expect(sx), "X at {sx}");
                    assert_eq!((vis[2], vis[3]), expect(sy), "Y at {sy}");
                    assert_eq!((vis[4], vis[5]), expect(sz), "Z at {sz}");
                    // At least three faces always show (never fewer, never all six
                    // unless the camera is strictly inside every extent).
                    let count = vis.iter().filter(|&&b| b).count();
                    assert!((3..=6).contains(&count));
                }
            }
        }
    }

    #[test]
    fn contiguous_runs_coalesce_and_cap_at_three() {
        let runs = |vis| {
            let (r, n) = contiguous_runs(vis);
            r[..n].to_vec()
        };
        // All visible → one coalesced run over the whole 0..6 range.
        assert_eq!(runs([true; 6]), vec![(0, 6)]);
        // None visible → no runs.
        assert_eq!(runs([false; 6]), Vec::<(u8, u8)>::new());
        // Adjacent trues coalesce; a gap splits.
        assert_eq!(
            runs([true, true, false, true, false, false]),
            vec![(0, 2), (3, 4)]
        );
        // Maximal alternation → three runs (the cap).
        assert_eq!(
            runs([true, false, true, false, true, false]),
            vec![(0, 1), (2, 3), (4, 5)]
        );
        assert_eq!(
            runs([false, true, false, true, false, true]),
            vec![(1, 2), (3, 4), (5, 6)]
        );
    }

    #[test]
    fn vrs_scene_fingerprint_tracks_view_and_depth_geometry() {
        let mut lists = DrawLists::new();
        let base = scene_fingerprint(&lists, &[], &[]);

        lists.has_3d = true;
        let with_3d = scene_fingerprint(&lists, &[], &[]);
        assert_ne!(base, with_3d);

        lists.view_proj = glam::Mat4::from_rotation_y(0.25);
        let turned = scene_fingerprint(&lists, &[], &[]);
        assert_ne!(with_3d, turned);

        lists.cube_verts.push(crate::mesh::DebugVertex {
            pos: [1.0, 2.0, 3.0],
            color: [255; 4],
        });
        assert_ne!(turned, scene_fingerprint(&lists, &[], &[]));
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

