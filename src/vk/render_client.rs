//! Render-thread decoupling: the main thread records into pooled [`DrawLists`]
//! snapshots and drives resource lifetime through a [`RenderClient`], while a
//! spawned render thread owns the [`Renderer`](super::Renderer) and presents at
//! its own vsync cadence. The two communicate over ordered channels; every
//! value that crosses is `Send` by construction.
//!
//! Ownership boundary (strict):
//! - **Main** owns the window (in `Engine`), the [`GpuAllocator`] (never sent),
//!   handle identity + culling metadata ([`MeshHandles`]), and a cloned
//!   [`ash::Device`] used only for alloc/map/write.
//! - **Render thread** owns the `Renderer` (residency mirror, swapchain,
//!   targets, pipelines, per-frame `HostBuffer`s). The `Renderer` is *born* on
//!   the thread — never moved to it — because its `HostBuffer`s hold a raw
//!   `*mut u8` that is `!Send`.
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{
    Receiver, RecvError, Sender, SyncSender, TryRecvError, channel, sync_channel,
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ash::{khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use super::alloc::{Allocation, DEVICE_SHRINK_SETTLE_TICKS, GpuAllocator};
use super::buffers::{
    DrawDyn, GpuResident, MeshHandles, MeshMeta, MeshRecord, PlacementState, build_mesh_resident,
    build_mesh_resident_staged,
};
use super::device::{Device, MemoryBudget};
use super::image::{AllocError, render_target_oom_message};
use super::instance::InstanceBundle;
use super::mesh_staging::{MeshStager, MeshStaging, MeshStagingPool};
use super::{Renderer, Scale, clamp_msaa, display_refresh_interval};
use crate::engine::Config;
use crate::frame::DrawLists;
use crate::mesh::{Detail, MeshData, MeshHandle, MeshPlacement, Pass};

/// Device capabilities cached on main for local clamp and [`crate::GpuCaps`].
#[derive(Clone)]
pub(crate) struct DeviceCaps {
    pub max_msaa: u32,
    /// Block-texture array layer ceiling (`limits.maxImageArrayLayers`).
    pub max_texture_layers: u32,
    pub device_name: String,
    pub device_local_bytes: u64,
    pub supports_vrs: bool,
    /// Attachment shading-rate texel size when VRS is available.
    pub vrs_texel_size: Option<(u32, u32)>,
    pub supports_pipeline_stats: bool,
}

/// Ordered command stream from main to render thread.
pub(crate) enum RenderCmd {
    UploadMesh {
        slot: u32,
        generation: NonZeroU32,
        /// Quad count (`bounds[6]/6`) so the render thread grows the shared quad
        /// IBO to index this mesh before its draws record.
        quads: u32,
        resident: GpuResident,
        /// Persistent GPU mesh record.
        record: MeshRecord,
    },
    /// Patch a mover's record (ordering prevents staleness).
    SetRecord {
        slot: u32,
        record: MeshRecord,
    },
    /// Patch the dynamic style lane (ordered like SetRecord).
    SetDrawDyn {
        slot: u32,
        dyn_lane: DrawDyn,
    },
    /// Patch a 32-slot word of the visibility mask (word-granular batching).
    SetVisible {
        word: u32,
        bits: u32,
    },
    /// Free a mesh slot (render thread stamps done_at).
    FreeMesh {
        slot: u32,
        generation: NonZeroU32,
    },
    SetBlockTextures {
        size: u32,
        layers: Box<[Vec<u8>]>,
    },
    AppendBlockTextures {
        layers: Box<[Vec<u8>]>,
    },
    UpdateMinimap(Box<[u8]>),
    UpdateMinimapRect {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        pixels: Box<[u8]>,
    },
    Capture(Capture),
    Resize(PhysicalSize<u32>),
    SetVsync(bool),
    /// Replaces the render thread's feature-flag copy (see [`crate::RenderFlags`]).
    SetFlags(crate::RenderFlags),
    /// GPU face-run culling; applied at the next cull prepare.
    SetCullFaces(bool),
    /// Pre-clamped against device caps.
    SetMsaa(u32),
    SetRenderScale(Scale),
    Frame(Box<DrawLists>),
    Shutdown,
}

/// A capture request. A pending capture makes the next present *mandatory* (the
/// pacer never drops it), so the frame the caller wants is guaranteed to reach
/// the readback — a capture is a correctness obligation, not a paceable frame.
/// `reply`, when present, carries the real encode/write outcome back to a
/// blocking caller ([`crate::screenshot_to`]); the interactive path leaves it
/// `None` and consumes the capture best-effort.
pub(crate) struct Capture {
    pub path: std::path::PathBuf,
    pub reply: Option<Sender<std::io::Result<()>>>,
}

/// Render thread → main: recycled frame buffers and freed allocations.
pub(crate) enum RenderReturn {
    Frame(Box<DrawLists>),
    FreeAlloc(Allocation),
}

/// One-shot handshake from render thread.
pub(crate) struct InitReply {
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub memory_budget: Option<MemoryBudget>,
    pub device: ash::Device,
    pub caps: DeviceCaps,
    /// Render thread's published exposure cell for Engine's compose().
    pub exposure: super::exposure::ExposureShared,
    /// Shared with the render thread; workers clone [`MeshStager`] from it.
    pub mesh_staging: Arc<MeshStagingPool>,
}

/// Render-thread build parameters.
pub(crate) struct RenderConfig {
    pub vsync: bool,
    pub msaa: u32,
    pub render_scale: f32,
    pub size: PhysicalSize<u32>,
    pub present_interval: Duration,
    /// CPU-side feature flags for the render thread (see [`crate::RenderFlags`]).
    pub flags: crate::RenderFlags,
}

/// The device/instance/surface handed back from the render thread at shutdown so
/// main can destroy them in the correct order (allocator buffers first, then
/// `vkDestroyDevice`), with no concurrent device access.
pub(crate) struct DeviceLeftovers {
    pub instance: InstanceBundle,
    pub surface_loader: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub device: Device,
}

/// Recording snapshots in circulation between main and the render thread.
///
/// The render thread's own slot ring ([`FRAMES_IN_FLIGHT`]) is what keeps the
/// GPU fed. Main only needs **one** queued frame so the render thread never
/// starves, plus the box it is recording into and [`FramePool::last_drawn`].
/// Hence 3, independent of `FRAMES_IN_FLIGHT`.
///
/// A larger pool lets main run more than one frame ahead. With vsync on, the
/// render loop coalesces queued [`RenderCmd::Frame`]s to the newest so a slow
/// present path skips stale snapshots. Vsync off is treated as uncapped (the
/// render thread does not know a game FPS cap) and does not coalesce.
const FRAME_POOL_SIZE: usize = 3;

/// Pooled [`DrawLists`] boxes plus the most recently completed snapshot, used
/// so a blocking capture can re-present the last scene without cloning it
/// every frame. Boxes match [`RenderCmd::Frame`] so a pop is a pointer move.
#[allow(clippy::vec_box)]
struct FramePool {
    idle: Vec<Box<DrawLists>>,
    last_drawn: Option<Box<DrawLists>>,
    in_flight: u32,
}

impl FramePool {
    fn with_boxes(n: usize) -> Self {
        Self {
            idle: (0..n).map(|_| Box::new(DrawLists::new())).collect(),
            last_drawn: None,
            in_flight: 0,
        }
    }

    fn pop_idle(&mut self) -> Option<Box<DrawLists>> {
        self.idle.pop()
    }

    fn note_submit(&mut self) {
        self.in_flight += 1;
    }

    fn on_returned(&mut self, b: Box<DrawLists>) {
        self.in_flight = self.in_flight.saturating_sub(1);
        if let Some(prev) = self.last_drawn.replace(b) {
            self.idle.push(prev);
        }
    }
}

/// The main-thread half. Facade signatures match the old `Renderer` methods so
/// `Engine` and every app caller are untouched.
pub(crate) struct RenderClient {
    tx: SyncSender<RenderCmd>,
    ret_rx: Receiver<RenderReturn>,
    /// Idle snapshots to record into plus the last completed scene.
    frames: FramePool,
    /// Empty box reused by the submit-first handoff so that path does not
    /// allocate a fresh [`DrawLists`] when the idle pool is empty.
    spare: Option<Box<DrawLists>>,
    /// Remaining [`GpuAllocator::shrink_device`] ticks after a free. Armed at
    /// [`DEVICE_SHRINK_SETTLE_TICKS`] so empty device blocks can settle without
    /// scanning every frame that did not free anything.
    shrink_ticks: u32,
    mesh_ids: MeshHandles,
    /// Main's copy of the visibility mask and changed words (delta batching).
    visible: Vec<u32>,
    visible_dirty: std::collections::BTreeSet<u32>,
    mesh_alloc: GpuAllocator,
    mesh_staging: Arc<MeshStagingPool>,
    device: ash::Device,
    caps: DeviceCaps,
    size: PhysicalSize<u32>,
    render_scale: Scale,
    vsync: bool,
    msaa: u32,
    cull_faces: bool,
    /// The render thread's published exposure cell, cloned into `Engine`.
    exposure: super::exposure::ExposureShared,
    /// `None` once joined (shutdown is idempotent).
    join: Option<JoinHandle<Option<DeviceLeftovers>>>,
    /// Render-thread completed `draw_frame` calls (monotonic).
    frames_rendered: Arc<AtomicU64>,
    /// Frames dropped by render-loop coalescing (monotonic).
    frames_coalesced: Arc<AtomicU64>,
}

impl RenderClient {
    /// Create window, spawn render thread, build main-side allocator.
    ///
    /// Render-target allocation failure is an `Err` with a readable message
    /// (logged here) rather than a render-thread panic + `RecvError`.
    pub(crate) fn spawn(
        event_loop: &ActiveEventLoop,
        config: &Config,
    ) -> Result<(Window, RenderClient), String> {
        let mut attrs = winit::window::WindowAttributes::default()
            .with_title(&config.title)
            .with_inner_size(winit::dpi::LogicalSize::new(config.width, config.height))
            .with_resizable(config.resizable);
        if config.fullscreen {
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

        let size = window.inner_size();
        let present_interval = display_refresh_interval(&window);
        let cfg = RenderConfig {
            vsync: config.vsync,
            msaa: config.msaa,
            render_scale: config.render_scale,
            size,
            present_interval,
            flags: config.flags,
        };

        let (cmd_tx, cmd_rx) = sync_channel::<RenderCmd>(1024);
        let (ret_tx, ret_rx) = channel::<RenderReturn>();
        let (init_tx, init_rx) = channel::<Result<InitReply, AllocError>>();
        let ret_for_renderer = ret_tx.clone();
        let frames_rendered = Arc::new(AtomicU64::new(0));
        let frames_coalesced = Arc::new(AtomicU64::new(0));
        let rendered_for_loop = Arc::clone(&frames_rendered);
        let coalesced_for_loop = Arc::clone(&frames_coalesced);
        let join = std::thread::Builder::new()
            .name("render".into())
            .spawn(move || {
                match Renderer::build(instance, surface_loader, surface, cfg, ret_for_renderer) {
                    Ok((renderer, reply)) => {
                        let _ = init_tx.send(Ok(reply));
                        Some(render_loop(
                            renderer,
                            cmd_rx,
                            ret_tx,
                            rendered_for_loop,
                            coalesced_for_loop,
                        ))
                    }
                    Err(err) => {
                        let _ = init_tx.send(Err(err));
                        None
                    }
                }
            })
            .expect("Failed to spawn render thread");

        let reply = match init_rx.recv() {
            Ok(Ok(reply)) => reply,
            Ok(Err(err)) => {
                let msg = render_target_oom_message(&err);
                log::error!("{msg}");
                let _ = join.join();
                return Err(msg);
            }
            Err(_) => {
                let msg = "renderer: render thread failed during initialization".to_string();
                log::error!("{msg}");
                let _ = join.join();
                return Err(msg);
            }
        };
        let mesh_alloc =
            unsafe { GpuAllocator::new(&reply.instance, reply.physical, reply.memory_budget) };
        if mesh_alloc.unified_memory() {
            log::info!("Unified memory detected: mesh uploads bypass staging");
        }
        let msaa = clamp_msaa(config.msaa, reply.caps.max_msaa);

        let client = RenderClient {
            tx: cmd_tx,
            ret_rx,
            frames: FramePool::with_boxes(FRAME_POOL_SIZE),
            spare: None,
            shrink_ticks: 0,
            mesh_ids: MeshHandles::new(),
            visible: Vec::new(),
            visible_dirty: std::collections::BTreeSet::new(),
            mesh_alloc,
            mesh_staging: reply.mesh_staging,
            device: reply.device,
            caps: reply.caps,
            size,
            render_scale: Scale::new(config.render_scale),
            vsync: config.vsync,
            msaa,
            cull_faces: true,
            exposure: reply.exposure,
            join: Some(join),
            frames_rendered,
            frames_coalesced,
        };
        Ok((window, client))
    }

    /// The render thread's published exposure cell, for `Engine`'s compose path.
    pub(crate) fn exposure(&self) -> super::exposure::ExposureShared {
        self.exposure.clone()
    }

    // ---- meshes ----

    /// Cheap `Clone` handle workers use to acquire staging regions.
    pub(crate) fn mesh_stager(&self) -> MeshStager {
        self.mesh_staging.stager()
    }

    /// Legacy upload: placement is recovered from each draw's offset
    /// ([`PlacementState::Tracked`]). Movers and demo geometry.
    pub(crate) fn upload_mesh(&mut self, data: &MeshData) -> Option<MeshHandle> {
        self.upload(data, None)
    }

    /// Placed upload: the placement is pinned at upload and draws never patch
    /// it. The terrain path.
    pub(crate) fn upload_mesh_placed(
        &mut self,
        data: &MeshData,
        placement: MeshPlacement,
    ) -> Option<MeshHandle> {
        self.upload(data, Some(placement))
    }

    /// Install a worker-written staging region as a placed mesh.
    pub(crate) fn upload_mesh_staged(
        &mut self,
        staging: MeshStaging,
        quads: [u32; 6],
        pass: Pass,
        placement: MeshPlacement,
    ) -> Option<MeshHandle> {
        let (meta, resident) = unsafe {
            build_mesh_resident_staged(
                &self.device,
                &mut self.mesh_alloc,
                &self.mesh_staging,
                staging,
                quads,
                pass,
            )
        }?;
        self.install(meta, resident, Some(placement))
    }

    /// Explicit release of a stale staging region; same as drop.
    pub(crate) fn release_mesh_staging(&self, staging: MeshStaging) {
        staging.release();
    }

    fn upload(&mut self, data: &MeshData, placement: Option<MeshPlacement>) -> Option<MeshHandle> {
        // Legacy CPU-side MeshData never uses the staging ring: workers that
        // already wrote into a region go through `upload_mesh_staged`.
        let (meta, resident) =
            unsafe { build_mesh_resident(&self.device, &mut self.mesh_alloc, data)? };
        self.install(meta, resident, placement)
    }

    fn install(
        &mut self,
        mut meta: MeshMeta,
        resident: GpuResident,
        placement: Option<MeshPlacement>,
    ) -> Option<MeshHandle> {
        if placement.is_some() {
            meta.placement = PlacementState::Pinned;
        }
        let record = MeshRecord::compose(
            &meta,
            placement.unwrap_or(MeshPlacement::terrain(glam::IVec3::ZERO, Detail::FULL)),
        );
        let quads = meta.bounds[6] / 6;
        let handle = self.mesh_ids.alloc_slot(meta);
        self.set_visible(handle.slot, true);
        let _ = self.tx.send(RenderCmd::UploadMesh {
            slot: handle.slot,
            generation: handle.generation,
            quads,
            resident,
            record,
        });
        Some(handle)
    }

    /// Patch the dynamic style lane (gen-checked, silent if unchanged).
    pub(crate) fn set_mesh_style(&mut self, handle: MeshHandle, dyn_lane: DrawDyn) {
        let Some(meta) = self.mesh_ids.meta_mut(handle) else {
            return;
        };
        if meta.dyn_lane != dyn_lane {
            meta.dyn_lane = dyn_lane;
            let _ = self.tx.send(RenderCmd::SetDrawDyn {
                slot: handle.slot,
                dyn_lane,
            });
        }
    }

    /// Patch a mover's world placement (no-op for pinned terrain/LOD).
    pub(crate) fn set_mesh_placement(&mut self, handle: MeshHandle, placement: MeshPlacement) {
        let Some(meta) = self.mesh_ids.meta_mut(handle) else {
            return;
        };
        if let PlacementState::Tracked(cached) = &mut meta.placement {
            if cached.is_none_or(|prev| placement.supersedes(&prev)) {
                *cached = Some(placement);
                let record = MeshRecord::compose(meta, placement);
                let _ = self.tx.send(RenderCmd::SetRecord {
                    slot: handle.slot,
                    record,
                });
            }
        }
    }

    /// Patch a slot's visibility bit (app uses for LOD, not terrain/movers).
    pub(crate) fn set_visible(&mut self, slot: u32, on: bool) {
        let word = (slot >> 5) as usize;
        if self.visible.len() <= word {
            self.visible.resize(word + 1, 0);
        }
        let bit = 1u32 << (slot & 31);
        let before = self.visible[word];
        self.visible[word] = if on { before | bit } else { before & !bit };
        if self.visible[word] != before {
            self.visible_dirty.insert(word as u32);
        }
    }

    /// Send visibility patches for changed words.
    fn flush_visible(&mut self) {
        for word in std::mem::take(&mut self.visible_dirty) {
            let _ = self.tx.send(RenderCmd::SetVisible {
                word,
                bits: self.visible[word as usize],
            });
        }
    }

    pub(crate) fn free_mesh(&mut self, handle: MeshHandle) {
        if self.mesh_ids.free_slot(handle) {
            // A dead slot draws nothing: clearing here (the sole free
            // chokepoint) is what keeps a recycled slot from inheriting the
            // previous tenant's visibility.
            self.set_visible(handle.slot, false);
            let _ = self.tx.send(RenderCmd::FreeMesh {
                slot: handle.slot,
                generation: handle.generation,
            });
        }
    }

    // ---- textures / minimap / screenshot ----

    pub(crate) fn set_block_textures(&mut self, size: u32, layers: &[Vec<u8>]) {
        let _ = self.tx.send(RenderCmd::SetBlockTextures {
            size,
            layers: layers.to_vec().into_boxed_slice(),
        });
    }

    pub(crate) fn append_block_textures(&mut self, layers: &[Vec<u8>]) {
        let _ = self.tx.send(RenderCmd::AppendBlockTextures {
            layers: layers.to_vec().into_boxed_slice(),
        });
    }

    pub(crate) fn update_minimap(&mut self, rgba: &[u8]) {
        let _ = self
            .tx
            .send(RenderCmd::UpdateMinimap(rgba.to_vec().into_boxed_slice()));
    }

    pub(crate) fn update_minimap_owned(&mut self, rgba: Box<[u8]>) {
        let _ = self.tx.send(RenderCmd::UpdateMinimap(rgba));
    }

    pub(crate) fn update_minimap_rect(&mut self, x: u32, y: u32, w: u32, h: u32, rgba: &[u8]) {
        let _ = self.tx.send(RenderCmd::UpdateMinimapRect {
            x,
            y,
            w,
            h,
            pixels: rgba.to_vec().into_boxed_slice(),
        });
    }

    pub(crate) fn minimap_size(&self) -> (u32, u32) {
        let n = super::MINIMAP_SIZE;
        (n, n)
    }

    pub(crate) fn request_capture(&mut self, capture: Capture) {
        let _ = self.tx.send(RenderCmd::Capture(capture));
    }

    // ---- settings (cached on main; getters read the cache) ----

    pub(crate) fn set_vsync(&mut self, on: bool) {
        if self.vsync == on {
            return;
        }
        self.vsync = on;
        let _ = self.tx.send(RenderCmd::SetVsync(on));
    }

    pub(crate) fn vsync(&self) -> bool {
        self.vsync
    }

    pub(crate) fn set_msaa(&mut self, samples: u32) -> u32 {
        let resolved = clamp_msaa(samples, self.caps.max_msaa);
        if self.msaa == resolved {
            return resolved;
        }
        self.msaa = resolved;
        let _ = self.tx.send(RenderCmd::SetMsaa(resolved));
        resolved
    }

    pub(crate) fn msaa(&self) -> u32 {
        self.msaa
    }

    /// Attachment shading-rate texel size, if the device has one.
    pub(crate) fn vrs_texel_size(&self) -> Option<(u32, u32)> {
        self.caps.vrs_texel_size
    }

    /// Offscreen pixel count after render scale (same formula as `scaled_extent`).
    pub(crate) fn render_pixels(&self) -> u32 {
        let scale = self.render_scale.get();
        let w = ((self.size.width as f32 * scale) as u32).max(1);
        let h = ((self.size.height as f32 * scale) as u32).max(1);
        w.saturating_mul(h)
    }

    pub(crate) fn max_msaa(&self) -> u32 {
        self.caps.max_msaa
    }

    pub(crate) fn max_texture_layers(&self) -> u32 {
        self.caps.max_texture_layers
    }

    pub(crate) fn gpu_caps(&self) -> crate::GpuCaps {
        crate::GpuCaps {
            device_name: self.caps.device_name.clone(),
            device_local_bytes: self.caps.device_local_bytes,
            max_texture_array_layers: self.caps.max_texture_layers,
            max_msaa: self.caps.max_msaa,
            supports_vrs: self.caps.supports_vrs,
            supports_pipeline_stats: self.caps.supports_pipeline_stats,
        }
    }

    /// GPU face-run culling. On by default; `false` is an explicit opt-out.
    /// Ships [`RenderCmd::SetCullFaces`] so the render thread follows; a change
    /// takes effect at the next frame boundary.
    pub(crate) fn set_cull_faces(&mut self, on: bool) {
        if self.cull_faces == on {
            return;
        }
        self.cull_faces = on;
        let _ = self.tx.send(RenderCmd::SetCullFaces(on));
    }

    pub(crate) fn cull_faces(&self) -> bool {
        self.cull_faces
    }

    pub(crate) fn set_flags(&mut self, flags: crate::RenderFlags) {
        let _ = self.tx.send(RenderCmd::SetFlags(flags));
    }

    pub(crate) fn set_render_scale(&mut self, scale: f32) -> f32 {
        let s = Scale::new(scale);
        if self.render_scale == s {
            return s.get();
        }
        self.render_scale = s;
        let _ = self.tx.send(RenderCmd::SetRenderScale(s));
        s.get()
    }

    pub(crate) fn render_scale(&self) -> f32 {
        self.render_scale.get()
    }

    // ---- window size (authority on main; shipped to render on resize) ----

    pub(crate) fn resize(&mut self, size: PhysicalSize<u32>) {
        self.size = size;
        let _ = self.tx.send(RenderCmd::Resize(size));
    }

    pub(crate) fn frames_rendered(&self) -> u64 {
        self.frames_rendered.load(Ordering::Relaxed)
    }

    pub(crate) fn frames_coalesced(&self) -> u64 {
        self.frames_coalesced.load(Ordering::Relaxed)
    }

    pub(crate) fn screen_width(&self) -> i32 {
        self.size.width as i32
    }

    pub(crate) fn screen_height(&self) -> i32 {
        self.size.height as i32
    }

    // ---- frame handoff / present pacing ----

    fn handle_return(&mut self, r: RenderReturn) {
        match r {
            RenderReturn::Frame(b) => self.frames.on_returned(b),
            RenderReturn::FreeAlloc(a) => {
                unsafe {
                    self.mesh_alloc.free(a);
                    self.mesh_alloc.shrink_staging(&self.device);
                }
                self.shrink_ticks = DEVICE_SHRINK_SETTLE_TICKS;
            }
        }
    }

    fn shrink_device_if_due(&mut self) {
        if self.shrink_ticks == 0 {
            return;
        }
        unsafe {
            self.mesh_alloc.shrink_device(&self.device);
        }
        self.shrink_ticks -= 1;
    }

    /// Drains render→main returns each cycle: recycled frame buffers back to the
    /// pool, freed allocations back to the allocator freelist. Allocator shrink
    /// scans run only after a free (and for the device-block settle window).
    pub(crate) fn drain_returns(&mut self) {
        while let Ok(r) = self.ret_rx.try_recv() {
            self.handle_return(r);
        }
        self.shrink_device_if_due();
    }

    /// Non-blocking pop of an idle recording snapshot.
    pub(crate) fn pop_idle_frame(&mut self) -> Option<Box<DrawLists>> {
        self.frames.pop_idle()
    }

    /// Empty box used only as a swap slot so the caller can submit before a
    /// blocking [`Self::take_frame`].
    pub(crate) fn take_placeholder(&mut self) -> Box<DrawLists> {
        self.spare
            .take()
            .unwrap_or_else(|| Box::new(DrawLists::new()))
    }

    pub(crate) fn stash_placeholder(&mut self, b: Box<DrawLists>) {
        if self.spare.is_none() {
            self.spare = Some(b);
        }
    }

    /// Pops an idle snapshot to record into; blocks on the return channel when
    /// the pool is empty. That block IS the present-pacing (the render thread
    /// returns a buffer only after it has consumed one). Never takes
    /// [`FramePool::last_drawn`]: that box still holds the last completed
    /// scene for [`Self::wait_last_drawn`].
    ///
    /// `spin` is true only when the engine is uncapped (vsync off and no FPS
    /// cap): poll the return channel briefly before parking so a 20 µs frame
    /// does not pay a futex wake every hand-off.
    pub(crate) fn take_frame(&mut self, spin: bool) -> Box<DrawLists> {
        loop {
            if let Some(b) = self.frames.pop_idle() {
                return b;
            }
            if self.frames.in_flight == 0 {
                // Nothing in flight to wait for. Allocate rather than steal
                // `last_drawn` (the blocking capture path needs that snapshot).
                return Box::new(DrawLists::new());
            }
            // Empty pool: this recv is the present-pacing wait. Timed as wait, not
            // main-thread work; the non-blocking pop above is not a wait.
            let r = {
                let _p = crate::profile::scope(crate::profile::Meter::WaitFrame);
                recv_spin(&self.ret_rx, spin)
            };
            match r {
                Ok(r) => self.handle_return(r),
                // Render thread gone: unblock with a throwaway (shutdown path).
                Err(_) => return Box::new(DrawLists::new()),
            }
        }
    }

    /// Waits until every submitted snapshot has returned and yields the most
    /// recently completed one (data still intact — it is reset only when
    /// reused for recording). Used by the blocking capture path.
    pub(crate) fn wait_last_drawn(&mut self) -> Box<DrawLists> {
        loop {
            if self.frames.in_flight == 0 {
                return self
                    .frames
                    .last_drawn
                    .take()
                    .unwrap_or_else(|| Box::new(DrawLists::new()));
            }
            match self.ret_rx.recv() {
                Ok(r) => self.handle_return(r),
                Err(_) => return Box::new(DrawLists::new()),
            }
        }
    }

    /// Submits a recorded snapshot to the render thread.
    pub(crate) fn submit_frame(&mut self, lists: Box<DrawLists>) {
        self.flush_visible();
        if self.tx.send(RenderCmd::Frame(lists)).is_ok() {
            self.frames.note_submit();
        }
    }

    /// Stops the render thread and destroys the device/instance/surface in the
    /// correct order (allocator buffers first). Idempotent via `join.take()`.
    pub(crate) fn shutdown(&mut self) {
        let _ = self.tx.send(RenderCmd::Shutdown);
        if let Some(join) = self.join.take()
            && let Ok(Some(mut lo)) = join.join()
        {
            log::debug!("GPU memory at shutdown: {:?}", self.mesh_alloc.stats());
            unsafe {
                // GPU is idle (the thread's teardown waited it) and stopped.
                self.mesh_alloc.destroy(&lo.device.device);
                lo.device.destroy();
                lo.surface_loader.destroy_surface(lo.surface, None);
                lo.instance.destroy();
            }
        }
    }
}

impl Drop for RenderClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Outcome of one drain of the command stream.
enum Drain {
    Shutdown,
    Frame(Option<Box<DrawLists>>),
}

/// Drain `first` plus whatever `try_next` yields.
///
/// Non-frame commands are applied in order. With `coalesce` (vsync on), queued
/// [`RenderCmd::Frame`]s collapse to the newest and `recycle` returns dropped
/// snapshots. Without it (vsync off / uncapped), stop at the first `Frame` so
/// later frames stay in the channel for the next iteration — otherwise a faster
/// main thread drops every other game frame. Vsync-off is treated as uncapped
/// because the render thread cannot see a game FPS cap.
fn drain_cmds(
    first: RenderCmd,
    mut try_next: impl FnMut() -> Option<RenderCmd>,
    coalesce: bool,
    mut apply: impl FnMut(RenderCmd),
    mut recycle: impl FnMut(Box<DrawLists>),
) -> Drain {
    let mut latest_frame: Option<Box<DrawLists>> = None;
    let mut cmd = Some(first);
    while let Some(c) = cmd.take().or_else(&mut try_next) {
        match c {
            RenderCmd::Frame(f) => {
                if let Some(old) = latest_frame.replace(f) {
                    recycle(old);
                }
                if !coalesce {
                    break;
                }
            }
            RenderCmd::Shutdown => return Drain::Shutdown,
            other => apply(other),
        }
    }
    Drain::Frame(latest_frame)
}

/// The render thread's loop: block when idle, drain the command stream applying
/// resource commands in order (coalescing frames only when vsync is on), draw
/// once, then recycle retired allocations. Returns the device leftovers for
/// main to finish teardown.
/// Same idea as [`super::timeline::Timeline::wait_spin`] / `FENCE_SPIN_BUDGET`
/// in `frame_loop`: spin a short, bounded window so a producer that is already
/// on the way does not pay a futex park. Never spin when vsync or an FPS cap
/// is pacing the loop — there the sleep is the point.
const CHANNEL_SPIN_BUDGET: Duration = Duration::from_micros(50);

fn recv_spin<T>(rx: &Receiver<T>, spin: bool) -> Result<T, RecvError> {
    if spin {
        let start = Instant::now();
        loop {
            match rx.try_recv() {
                Ok(v) => return Ok(v),
                Err(TryRecvError::Disconnected) => return Err(RecvError),
                Err(TryRecvError::Empty) => {
                    if start.elapsed() >= CHANNEL_SPIN_BUDGET {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        }
    }
    rx.recv()
}

fn render_loop(
    mut renderer: Renderer,
    rx: Receiver<RenderCmd>,
    ret: Sender<RenderReturn>,
    frames_rendered: Arc<AtomicU64>,
    frames_coalesced: Arc<AtomicU64>,
) -> DeviceLeftovers {
    loop {
        let spin = !renderer.vsync.effective();
        let Ok(first) = recv_spin(&rx, spin) else {
            break;
        };
        // Vsync-off: render every queued Frame. Vsync-on: coalesce to newest.
        let coalesce = renderer.vsync.effective();
        match drain_cmds(
            first,
            || rx.try_recv().ok(),
            coalesce,
            |c| match c {
                RenderCmd::UploadMesh {
                    slot,
                    generation,
                    quads,
                    resident,
                    record,
                } => renderer.apply_upload_mesh(slot, generation, quads, resident, record),
                RenderCmd::FreeMesh { slot, generation } => {
                    renderer.apply_free_mesh(slot, generation)
                }
                RenderCmd::SetRecord { slot, record } => renderer.apply_set_record(slot, record),
                RenderCmd::SetVisible { word, bits } => renderer.set_visible_word(word, bits),
                RenderCmd::SetDrawDyn { slot, dyn_lane } => {
                    renderer.records.set_dyn(slot, dyn_lane)
                }
                RenderCmd::SetBlockTextures { size, layers } => {
                    renderer.set_block_textures(size, &layers)
                }
                RenderCmd::AppendBlockTextures { layers } => {
                    renderer.append_block_textures(&layers)
                }
                RenderCmd::UpdateMinimap(px) => renderer.update_minimap(&px),
                RenderCmd::UpdateMinimapRect { x, y, w, h, pixels } => {
                    renderer.update_minimap_rect(x, y, w, h, &pixels)
                }
                RenderCmd::Capture(capture) => renderer.request_capture(capture),
                RenderCmd::Resize(size) => renderer.on_resize(size),
                RenderCmd::SetVsync(v) => renderer.set_vsync(v),
                RenderCmd::SetFlags(f) => renderer.set_flags(f),
                RenderCmd::SetCullFaces(on) => renderer.set_cull_faces(on),
                RenderCmd::SetMsaa(m) => {
                    renderer.set_msaa(m);
                }
                RenderCmd::SetRenderScale(s) => {
                    renderer.set_render_scale(s.get());
                }
                RenderCmd::Frame(_) | RenderCmd::Shutdown => unreachable!(),
            },
            |old| {
                frames_coalesced.fetch_add(1, Ordering::Relaxed);
                crate::profile::count(crate::profile::Counter::Coalesced);
                let _ = ret.send(RenderReturn::Frame(old));
            },
        ) {
            Drain::Shutdown => return renderer.teardown(),
            Drain::Frame(Some(frame)) => {
                renderer.draw_frame(&frame);
                frames_rendered.fetch_add(1, Ordering::Relaxed);
                let _ = ret.send(RenderReturn::Frame(frame));
            }
            Drain::Frame(None) => {}
        }
    }
    // Sender dropped without a Shutdown (main gone): tear down anyway.
    renderer.teardown()
}

#[cfg(test)]
mod tests {
    use super::super::buffers::FRAMES_IN_FLIGHT;
    use super::{
        CHANNEL_SPIN_BUDGET, Drain, FRAME_POOL_SIZE, FramePool, RenderCmd, drain_cmds, recv_spin,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn frame_pool_keeps_main_at_most_one_ahead() {
        assert_eq!(FRAME_POOL_SIZE, 3);
        // Recording + one queued/being-rendered + last_drawn. Must not grow
        // with FRAMES_IN_FLIGHT: extra slack lets main queue ahead of render.
        // Vsync coalesces; vsync-off leaves later Frames in the channel.
        assert_ne!(FRAME_POOL_SIZE, FRAMES_IN_FLIGHT as usize + 1);
    }

    #[test]
    fn returned_frames_keep_the_latest_and_park_the_previous_in_idle() {
        let mut p = FramePool::with_boxes(0);
        p.note_submit();
        p.note_submit();
        p.on_returned(Box::new(super::DrawLists::new()));
        p.on_returned(Box::new(super::DrawLists::new()));
        assert_eq!(p.in_flight, 0);
        assert!(p.last_drawn.is_some());
        assert_eq!(p.idle.len(), 1);
        assert!(p.pop_idle().is_some());
        assert!(p.pop_idle().is_none());
        assert!(p.last_drawn.take().is_some());
    }

    #[test]
    fn recv_spin_returns_immediately_when_queued() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(7).unwrap();
        let start = Instant::now();
        assert_eq!(recv_spin(&rx, true).unwrap(), 7);
        assert!(
            start.elapsed() < CHANNEL_SPIN_BUDGET,
            "already-queued value must not wait out the spin budget"
        );
    }

    #[test]
    fn recv_spin_falls_back_to_blocking_after_budget() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            std::thread::sleep(CHANNEL_SPIN_BUDGET + Duration::from_millis(2));
            tx.send(1).unwrap();
        });
        let start = Instant::now();
        assert_eq!(recv_spin(&rx, true).unwrap(), 1);
        assert!(
            start.elapsed() >= CHANNEL_SPIN_BUDGET,
            "empty channel must spin the budget then block until the value arrives"
        );
    }

    fn flag_cmd() -> RenderCmd {
        RenderCmd::SetFlags(crate::engine::RenderFlags::default())
    }

    #[test]
    fn uncapped_drain_stops_at_the_first_frame() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(flag_cmd()).unwrap();
        tx.send(RenderCmd::Frame(Box::new(super::DrawLists::new())))
            .unwrap();
        tx.send(RenderCmd::Frame(Box::new(super::DrawLists::new())))
            .unwrap();
        let first = rx.recv().unwrap();
        let mut applied = 0u32;
        let mut recycled = 0u32;
        let Drain::Frame(frame) = drain_cmds(
            first,
            || rx.try_recv().ok(),
            false,
            |_| applied += 1,
            |_| recycled += 1,
        ) else {
            panic!("expected a frame drain");
        };
        assert!(frame.is_some());
        assert_eq!(applied, 1, "non-frame cmds before the first Frame apply");
        assert_eq!(
            recycled, 0,
            "uncapped must not coalesce; frames_coalesced stays 0"
        );
        assert!(
            matches!(rx.try_recv(), Ok(RenderCmd::Frame(_))),
            "later Frames stay in the channel"
        );
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn vsync_drain_coalesces_queued_frames() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(RenderCmd::Frame(Box::new(super::DrawLists::new())))
            .unwrap();
        tx.send(flag_cmd()).unwrap();
        tx.send(RenderCmd::Frame(Box::new(super::DrawLists::new())))
            .unwrap();
        let first = rx.recv().unwrap();
        let mut applied = 0u32;
        let mut recycled = 0u32;
        let Drain::Frame(frame) = drain_cmds(
            first,
            || rx.try_recv().ok(),
            true,
            |_| applied += 1,
            |_| recycled += 1,
        ) else {
            panic!("expected a frame drain");
        };
        assert!(frame.is_some());
        assert_eq!(applied, 1);
        assert_eq!(recycled, 1);
        assert!(rx.try_recv().is_err());
    }
}
