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
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::thread::JoinHandle;
use std::time::Duration;

use ash::{khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use super::alloc::{Allocation, GpuAllocator};
use super::buffers::{
    DrawDyn, FRAMES_IN_FLIGHT, GpuResident, MeshHandles, MeshRecord, PlacementState,
    build_mesh_resident,
};
use super::device::{Device, MemoryBudget};
use super::instance::InstanceBundle;
use super::{Renderer, Scale, clamp_msaa, display_refresh_interval};
use crate::engine::Config;
use crate::frame::DrawLists;
use crate::mesh::{Detail, MeshData, MeshHandle, MeshPlacement};

/// Device capabilities cached on main for local clamp.
#[derive(Clone, Copy)]
pub(crate) struct DeviceCaps {
    pub max_msaa: u32,
    /// Block-texture array layer ceiling (`limits.maxImageArrayLayers`).
    pub max_texture_layers: u32,
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
    UpdateMinimap(Box<[u8]>),
    Capture(Capture),
    Resize(PhysicalSize<u32>),
    SetVsync(bool),
    /// Replaces the render thread's feature-flag copy (see [`crate::RenderFlags`]).
    SetFlags(crate::RenderFlags),
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

/// The main-thread half. Facade signatures match the old `Renderer` methods so
/// `Engine` and every app caller are untouched.
pub(crate) struct RenderClient {
    tx: SyncSender<RenderCmd>,
    ret_rx: Receiver<RenderReturn>,
    /// Idle snapshots to record into (cap = FRAMES_IN_FLIGHT); blocking to pop
    /// one IS the present-pacing that replaced the old spin `pace()`.
    frame_pool: Vec<Box<DrawLists>>,
    mesh_ids: MeshHandles,
    /// Main's copy of the visibility mask and changed words (delta batching).
    visible: Vec<u32>,
    visible_dirty: std::collections::BTreeSet<u32>,
    mesh_alloc: GpuAllocator,
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
    join: Option<JoinHandle<DeviceLeftovers>>,
}

impl RenderClient {
    /// Create window, spawn render thread, build main-side allocator.
    pub(crate) fn spawn(event_loop: &ActiveEventLoop, config: &Config) -> (Window, RenderClient) {
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
        let (init_tx, init_rx) = channel::<InitReply>();
        let ret_for_renderer = ret_tx.clone();
        let join = std::thread::Builder::new()
            .name("render".into())
            .spawn(move || {
                let (renderer, reply) =
                    Renderer::build(instance, surface_loader, surface, cfg, ret_for_renderer);
                let _ = init_tx.send(reply);
                render_loop(renderer, cmd_rx, ret_tx)
            })
            .expect("Failed to spawn render thread");

        let reply = init_rx.recv().expect("render thread init failed");
        let mesh_alloc =
            unsafe { GpuAllocator::new(&reply.instance, reply.physical, reply.memory_budget) };
        if mesh_alloc.unified_memory() {
            log::info!("Unified memory detected: mesh uploads bypass staging");
        }
        let msaa = clamp_msaa(config.msaa, reply.caps.max_msaa);
        let frame_pool = (0..FRAMES_IN_FLIGHT)
            .map(|_| Box::new(DrawLists::new()))
            .collect();

        let client = RenderClient {
            tx: cmd_tx,
            ret_rx,
            frame_pool,
            mesh_ids: MeshHandles::new(),
            visible: Vec::new(),
            visible_dirty: std::collections::BTreeSet::new(),
            mesh_alloc,
            device: reply.device,
            caps: reply.caps,
            size,
            render_scale: Scale::new(config.render_scale),
            vsync: config.vsync,
            msaa,
            cull_faces: false,
            exposure: reply.exposure,
            join: Some(join),
        };
        (window, client)
    }

    /// The render thread's published exposure cell, for `Engine`'s compose path.
    pub(crate) fn exposure(&self) -> super::exposure::ExposureShared {
        self.exposure.clone()
    }

    // ---- meshes ----

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

    fn upload(
        &mut self,
        data: &MeshData,
        placement: Option<MeshPlacement>,
    ) -> Option<MeshHandle> {
        let (mut meta, resident) =
            unsafe { build_mesh_resident(&self.device, &mut self.mesh_alloc, data) }?;
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

    pub(crate) fn update_minimap(&mut self, rgba: &[u8]) {
        let _ = self
            .tx
            .send(RenderCmd::UpdateMinimap(rgba.to_vec().into_boxed_slice()));
    }

    pub(crate) fn request_capture(&mut self, capture: Capture) {
        let _ = self.tx.send(RenderCmd::Capture(capture));
    }

    // ---- settings (cached on main; getters read the cache) ----

    pub(crate) fn set_vsync(&mut self, on: bool) {
        self.vsync = on;
        let _ = self.tx.send(RenderCmd::SetVsync(on));
    }

    pub(crate) fn vsync(&self) -> bool {
        self.vsync
    }

    pub(crate) fn set_msaa(&mut self, samples: u32) -> u32 {
        let resolved = clamp_msaa(samples, self.caps.max_msaa);
        self.msaa = resolved;
        let _ = self.tx.send(RenderCmd::SetMsaa(resolved));
        resolved
    }

    pub(crate) fn msaa(&self) -> u32 {
        self.msaa
    }

    pub(crate) fn max_msaa(&self) -> u32 {
        self.caps.max_msaa
    }

    pub(crate) fn max_texture_layers(&self) -> u32 {
        self.caps.max_texture_layers
    }

    /// Cached-only since the GPU cull became unconditional: both the
    /// GPU opaque/cutout emission and the CPU Blend re-source draw whole-mesh
    /// index ranges, so per-face splitting has no live consumer. Retained so
    /// the app's settings toggle still round-trips; INERT until per-face
    /// partitioning is taught to the cull shader (or the setting is retired).
    pub(crate) fn set_cull_faces(&mut self, on: bool) {
        self.cull_faces = on;
    }

    pub(crate) fn cull_faces(&self) -> bool {
        self.cull_faces
    }

    pub(crate) fn set_flags(&mut self, flags: crate::RenderFlags) {
        let _ = self.tx.send(RenderCmd::SetFlags(flags));
    }

    pub(crate) fn set_render_scale(&mut self, scale: f32) -> f32 {
        let s = Scale::new(scale);
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

    pub(crate) fn screen_width(&self) -> i32 {
        self.size.width as i32
    }

    pub(crate) fn screen_height(&self) -> i32 {
        self.size.height as i32
    }

    // ---- frame handoff / present pacing ----

    /// Drains render→main returns each cycle: recycled frame buffers back to the
    /// pool, freed allocations back to the allocator freelist.
    pub(crate) fn drain_returns(&mut self) {
        while let Ok(r) = self.ret_rx.try_recv() {
            match r {
                RenderReturn::Frame(b) => self.frame_pool.push(b),
                RenderReturn::FreeAlloc(a) => unsafe { self.mesh_alloc.free(a) },
            }
        }
        unsafe {
            self.mesh_alloc.shrink_staging(&self.device);
            self.mesh_alloc.shrink_device(&self.device);
        }
    }

    /// Pops an idle snapshot to record into; blocks on the return channel when
    /// the pool is empty. That block IS the present-pacing (the render thread
    /// returns a buffer only after it has consumed one).
    pub(crate) fn take_frame(&mut self) -> Box<DrawLists> {
        if let Some(b) = self.frame_pool.pop() {
            return b;
        }
        loop {
            match self.ret_rx.recv() {
                Ok(RenderReturn::Frame(b)) => return b,
                Ok(RenderReturn::FreeAlloc(a)) => unsafe { self.mesh_alloc.free(a) },
                // Render thread gone: unblock with a throwaway (shutdown path).
                Err(_) => return Box::new(DrawLists::new()),
            }
        }
    }

    /// Submits a recorded snapshot to the render thread.
    pub(crate) fn submit_frame(&mut self, lists: Box<DrawLists>) {
        self.flush_visible();
        let _ = self.tx.send(RenderCmd::Frame(lists));
    }

    /// Stops the render thread and destroys the device/instance/surface in the
    /// correct order (allocator buffers first). Idempotent via `join.take()`.
    pub(crate) fn shutdown(&mut self) {
        let _ = self.tx.send(RenderCmd::Shutdown);
        if let Some(join) = self.join.take() {
            if let Ok(mut lo) = join.join() {
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
}

impl Drop for RenderClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The render thread's loop: block when idle, greedily drain the command stream
/// applying resource commands in order and coalescing frames to the latest, draw
/// once, then recycle retired allocations. Returns the device leftovers for
/// main to finish teardown.
fn render_loop(
    mut renderer: Renderer,
    rx: Receiver<RenderCmd>,
    ret: Sender<RenderReturn>,
) -> DeviceLeftovers {
    while let Ok(first) = rx.recv() {
        let mut latest_frame: Option<Box<DrawLists>> = None;
        let mut cmd = Some(first);
        while let Some(c) = cmd.take().or_else(|| rx.try_recv().ok()) {
            match c {
                RenderCmd::Frame(f) => {
                    // Coalesce: keep only the newest, recycle the dropped one.
                    if let Some(old) = latest_frame.replace(f) {
                        let _ = ret.send(RenderReturn::Frame(old));
                    }
                }
                RenderCmd::Shutdown => return renderer.teardown(),
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
                RenderCmd::SetRecord { slot, record } => renderer.records.set_record(slot, record),
                RenderCmd::SetVisible { word, bits } => renderer.set_visible_word(word, bits),
                RenderCmd::SetDrawDyn { slot, dyn_lane } => {
                    renderer.records.set_dyn(slot, dyn_lane)
                }
                RenderCmd::SetBlockTextures { size, layers } => {
                    renderer.set_block_textures(size, &layers)
                }
                RenderCmd::UpdateMinimap(px) => renderer.update_minimap(&px),
                RenderCmd::Capture(capture) => renderer.request_capture(capture),
                RenderCmd::Resize(size) => renderer.on_resize(size),
                RenderCmd::SetVsync(v) => renderer.set_vsync(v),
                RenderCmd::SetFlags(f) => renderer.set_flags(f),
                RenderCmd::SetMsaa(m) => {
                    renderer.set_msaa(m);
                }
                RenderCmd::SetRenderScale(s) => {
                    renderer.set_render_scale(s.get());
                }
            }
        }
        if let Some(frame) = latest_frame {
            renderer.draw_frame(&frame);
            let _ = ret.send(RenderReturn::Frame(frame));
        }
    }
    // Sender dropped without a Shutdown (main gone): tear down anyway.
    renderer.teardown()
}
