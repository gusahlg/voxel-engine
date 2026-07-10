//! Render-thread decoupling: the main thread records into pooled [`DrawLists`]
//! snapshots and drives resource lifetime through a [`RenderClient`], while a
//! spawned render thread owns the [`Renderer`](super::Renderer) and presents at
//! its own vsync cadence. The two communicate over ordered channels; every
//! value that crosses is `Send` by construction.
//!
//! Ownership boundary (strict):
//! - **Main** owns the window (in `Engine`), the [`GpuAllocator`] (never sent),
//!   handle identity + culling metadata ([`MeshHandles`]/[`SurfaceHandles`]),
//!   and a cloned [`ash::Device`] used only for alloc/map/write.
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
    FRAMES_IN_FLIGHT, GpuResident, MeshHandles, MeshMeta, SurfaceHandles, SurfaceMeta,
    build_mesh_resident, build_surface_resident,
};
use super::device::{Device, MemoryBudget};
use super::instance::InstanceBundle;
use super::{Renderer, Scale, clamp_msaa, display_refresh_interval};
use crate::engine::Config;
use crate::frame::DrawLists;
use crate::mesh::{MeshData, MeshHandle};
use crate::surface::{SurfaceData, SurfaceHandle};

/// Device capabilities the main thread caches so settings clamp locally (no
/// round-trip to the render thread).
#[derive(Clone, Copy)]
pub(crate) struct DeviceCaps {
    pub max_msaa: u32,
}

/// The ordered command stream, main → render. Every payload is `Send`
/// ([`GpuResident`] because [`Allocation`] is, [`DrawLists`] because it is fully
/// resolved POD).
pub(crate) enum RenderCmd {
    UploadMesh {
        slot: u32,
        generation: NonZeroU32,
        resident: GpuResident,
    },
    /// The render thread stamps `done_at` when it applies this (the timeline is
    /// render-side state; main has no timeline to read).
    FreeMesh {
        slot: u32,
        generation: NonZeroU32,
    },
    UploadSurface {
        slot: u32,
        generation: NonZeroU32,
        resident: GpuResident,
    },
    FreeSurface {
        slot: u32,
        generation: NonZeroU32,
    },
    SetBlockTextures {
        size: u32,
        layers: Box<[Vec<u8>]>,
    },
    UpdateMinimap(Box<[u8]>),
    Screenshot(std::path::PathBuf),
    Resize(PhysicalSize<u32>),
    SetVsync(bool),
    /// Pre-clamped on main against cached caps.
    SetMsaa(u32),
    SetCullFaces(bool),
    SetRenderScale(Scale),
    Frame(Box<DrawLists>),
    Shutdown,
}

/// Render → main: recycled frame buffers AND freed allocations for main's
/// allocator freelist. One channel keeps ordering simple.
pub(crate) enum RenderReturn {
    Frame(Box<DrawLists>),
    FreeAlloc(Allocation),
}

/// One-shot handshake so `RenderClient` can build its main-side [`GpuAllocator`]
/// from the exact `GpuAllocator::new(instance, physical, memory_budget)` args.
pub(crate) struct InitReply {
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub memory_budget: Option<MemoryBudget>,
    pub device: ash::Device,
    pub caps: DeviceCaps,
}

/// The render-thread build parameters (window-free slice of [`Config`] + the
/// initial size and refresh interval queried on main).
pub(crate) struct RenderConfig {
    pub vsync: bool,
    pub msaa: u32,
    pub render_scale: f32,
    pub size: PhysicalSize<u32>,
    pub present_interval: Duration,
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
    surface_ids: SurfaceHandles,
    mesh_alloc: GpuAllocator,
    device: ash::Device,
    caps: DeviceCaps,
    size: PhysicalSize<u32>,
    render_scale: Scale,
    vsync: bool,
    msaa: u32,
    cull_faces: bool,
    /// `None` once joined (shutdown is idempotent).
    join: Option<JoinHandle<DeviceLeftovers>>,
}

impl RenderClient {
    /// Creates the window + instance + surface on main, spawns the render
    /// thread (which builds the `Renderer` and replies), then builds the
    /// main-side allocator. Returns the window (kept by `Engine`) and the client.
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
            surface_ids: SurfaceHandles::new(),
            mesh_alloc,
            device: reply.device,
            caps: reply.caps,
            size,
            render_scale: Scale::new(config.render_scale),
            vsync: config.vsync,
            msaa,
            cull_faces: false,
            join: Some(join),
        };
        (window, client)
    }

    // ---- meshes ----

    pub(crate) fn upload_mesh(&mut self, data: &MeshData) -> Option<MeshHandle> {
        let (meta, resident) =
            unsafe { build_mesh_resident(&self.device, &mut self.mesh_alloc, data) }?;
        let handle = self.mesh_ids.alloc_slot(meta);
        let _ = self.tx.send(RenderCmd::UploadMesh {
            slot: handle.slot,
            generation: handle.generation,
            resident,
        });
        Some(handle)
    }

    pub(crate) fn free_mesh(&mut self, handle: MeshHandle) {
        if self.mesh_ids.free_slot(handle) {
            let _ = self.tx.send(RenderCmd::FreeMesh {
                slot: handle.slot,
                generation: handle.generation,
            });
        }
    }

    pub(crate) fn mesh_meta(&self, handle: MeshHandle) -> Option<MeshMeta> {
        self.mesh_ids.meta(handle)
    }

    // ---- surfaces ----

    pub(crate) fn upload_surface(&mut self, data: &SurfaceData) -> Option<SurfaceHandle> {
        let (meta, resident) =
            unsafe { build_surface_resident(&self.device, &mut self.mesh_alloc, data) }?;
        let handle = self.surface_ids.alloc_slot(meta);
        let _ = self.tx.send(RenderCmd::UploadSurface {
            slot: handle.slot,
            generation: handle.generation,
            resident,
        });
        Some(handle)
    }

    pub(crate) fn free_surface(&mut self, handle: SurfaceHandle) {
        if self.surface_ids.free_slot(handle) {
            let _ = self.tx.send(RenderCmd::FreeSurface {
                slot: handle.slot,
                generation: handle.generation,
            });
        }
    }

    pub(crate) fn surface_meta(&self, handle: SurfaceHandle) -> Option<SurfaceMeta> {
        self.surface_ids.meta(handle)
    }

    pub(crate) fn surface_aabb(&self, handle: SurfaceHandle) -> Option<(glam::Vec3, glam::Vec3)> {
        self.surface_ids.meta(handle).map(|s| s.aabb())
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

    pub(crate) fn request_screenshot(&mut self, path: std::path::PathBuf) {
        let _ = self.tx.send(RenderCmd::Screenshot(path));
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

    pub(crate) fn set_cull_faces(&mut self, on: bool) {
        self.cull_faces = on;
        let _ = self.tx.send(RenderCmd::SetCullFaces(on));
    }

    pub(crate) fn cull_faces(&self) -> bool {
        self.cull_faces
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
        unsafe { self.mesh_alloc.shrink_staging(&self.device) };
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
                    resident,
                } => renderer.apply_upload_mesh(slot, generation, resident),
                RenderCmd::FreeMesh { slot, generation } => {
                    renderer.apply_free_mesh(slot, generation)
                }
                RenderCmd::UploadSurface {
                    slot,
                    generation,
                    resident,
                } => renderer.apply_upload_surface(slot, generation, resident),
                RenderCmd::FreeSurface { slot, generation } => {
                    renderer.apply_free_surface(slot, generation)
                }
                RenderCmd::SetBlockTextures { size, layers } => {
                    renderer.set_block_textures(size, &layers)
                }
                RenderCmd::UpdateMinimap(px) => renderer.update_minimap(&px),
                RenderCmd::Screenshot(path) => renderer.request_screenshot(path),
                RenderCmd::Resize(size) => renderer.on_resize(size),
                RenderCmd::SetVsync(v) => renderer.set_vsync(v),
                RenderCmd::SetMsaa(m) => {
                    renderer.set_msaa(m);
                }
                RenderCmd::SetCullFaces(c) => renderer.set_cull_faces(c),
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
