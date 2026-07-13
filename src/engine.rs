/// The public engine facade and the winit event loop driver.
///
/// `run(config, |eng| { ... })` calls the closure once per frame after input
/// collection; the closure draws via `eng.begin_frame(..)` and returns
/// `false` to exit. This mirrors a raylib-style polling main loop on top of
/// winit 0.30's callback model.
use std::time::{Duration, Instant};

use glam::{Vec2, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use crate::camera::{self, Camera3D};
use crate::color::LinearRgb;
use crate::font;
use crate::frame::{DrawLists, Frame};
use crate::input::{InputState, Key, MouseButton};
use crate::mesh::{MeshData, MeshHandle};
use crate::vk::render_client::{Capture, RenderClient};

#[derive(Clone)]
pub struct Config {
    pub title: String,
    pub width: u32,
    pub height: u32,
    /// 0 = uncapped.
    pub target_fps: u32,
    /// Rendering is decoupled from presentation (manual mailbox): every
    /// frame renders offscreen and is only copied to the screen when the
    /// presentation engine can take it. With vsync on, presentation
    /// backpressure paces the frame loop at the display refresh (the classic
    /// vsync feel, tear-free). With vsync off, the loop is fully uncapped:
    /// frames that outrun the display are rendered but dropped, and
    /// `target_fps` is the only pacing.
    pub vsync: bool,
    pub msaa: u32,
    /// Render-resolution scale relative to the window (0.25..=2.0).
    pub render_scale: f32,
    pub resizable: bool,
    pub fullscreen: bool,
    /// CPU-side render feature flags (app is the single source; see [`RenderFlags`]).
    pub flags: RenderFlags,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            title: "voxel_engine".into(),
            width: 1280,
            height: 720,
            target_fps: 0,
            vsync: true,
            msaa: 1,
            render_scale: 1.0,
            resizable: true,
            fullscreen: false,
            flags: RenderFlags::default(),
        }
    }
}

/// CPU-side render feature flags, carried in [`Config`] and threaded to both the
/// main thread ([`Engine`]) and the render thread ([`crate::vk::Renderer`]) at
/// construction — the app is the single source (no ambient env state). All gates
/// are CPU-side: they neutralize a `FrameUniforms` lane (`frame::gate_uniforms`,
/// applied to `Lighting::Composed`) or skip a pass's work (vk/mod.rs,
/// vk/shadow.rs) — no shader variants.
#[derive(Clone, Copy)]
pub struct RenderFlags {
    /// Camera jitter + TAA resolve, always coupled.
    pub taa: bool,
    /// Distance fog (`horizon.w` density).
    pub fog: bool,
    /// Torch/candle light (`candle.rgb`).
    pub blocklight: bool,
    /// Omni ambient floor (`candle.w`) — off means black caves.
    pub ambient: bool,
    /// Sun/skylight (`light.rgb`), also kills the sky halo glow.
    pub sunlight: bool,
    /// Auto-exposure metering; off pins exposure at 1.0.
    pub exposure: bool,
    /// HDR bloom (threshold + downsample compute → tonemap composite). Off skips
    /// the dispatch and clears the bloom target so the tonemap add is a no-op.
    pub bloom: bool,
    /// Screen-space godrays: volumetric sun rays in tonemap. Off disables them.
    pub godrays: bool,
    /// Cascade occluder draws + far-field fallback; off is fully lit.
    pub shadows: bool,
    /// Procedural sky background pass; off shows the clear colour.
    pub sky: bool,
    /// Radial vignette darkening in the tonemap pass.
    pub vignette: bool,
}

impl Default for RenderFlags {
    /// The shipped defaults (formerly the `WATT_*` unset-defaults).
    fn default() -> Self {
        Self {
            taa: false,
            fog: false,
            blocklight: false,
            ambient: false,
            sunlight: true,
            exposure: false,
            bloom: true,
            godrays: true,
            shadows: false,
            sky: true,
            vignette: false,
        }
    }
}

pub struct Engine {
    pub(crate) client: RenderClient,
    /// The render thread's published exposure, read by
    /// [`Engine::exposure_for_compose`] each frame.
    pub(crate) exposure_shared: crate::vk::exposure::ExposureShared,
    /// The window lives on the main thread; only the `Renderer` moved to the
    /// render thread. Window-touching methods read this directly.
    pub(crate) window: winit::window::Window,
    pub(crate) input: InputState,
    /// The frame being recorded; swapped with a pooled buffer each frame.
    pub(crate) lists: Box<DrawLists>,
    /// CPU-side feature flags for the main/record thread (fog/blocklight/etc gate
    /// `gate_uniforms`; `taa` gates jitter injection). The render thread holds its
    /// own copy on `Renderer`. Both are set from `Config::flags` at construction.
    pub(crate) flags: RenderFlags,
    /// The most recently submitted frame's draw lists, retained so a blocking
    /// deterministic capture ([`crate::screenshot_to`]) can re-present the same
    /// scene until the readback PNG lands instead of a blank frame.
    pub(crate) last_lists: Box<DrawLists>,

    target_fps: u32,
    frame_start: Instant,
    dt: f32,
    fps_window_start: Instant,
    fps_window_frames: u32,
    fps_cached: i32,

    should_close: bool,
}

impl Engine {
    fn new(window: winit::window::Window, mut client: RenderClient, config: &Config) -> Self {
        let lists = client.take_frame();
        let exposure_shared = client.exposure();
        Self {
            client,
            exposure_shared,
            window,
            input: InputState::new(),
            lists,
            flags: config.flags,
            last_lists: Box::new(DrawLists::new()),
            target_fps: config.target_fps,
            frame_start: Instant::now(),
            dt: 0.0,
            fps_window_start: Instant::now(),
            fps_window_frames: 0,
            fps_cached: 0,
            should_close: false,
        }
    }

    // ---- window / timing ----

    pub fn screen_width(&self) -> i32 {
        self.client.screen_width()
    }

    pub fn screen_height(&self) -> i32 {
        self.client.screen_height()
    }

    /// Seconds the previous frame took (including pacing sleep).
    pub fn frame_time(&self) -> f32 {
        self.dt
    }

    /// Measured frames per second, averaged over a short window.
    pub fn fps(&self) -> i32 {
        self.fps_cached
    }

    pub fn set_target_fps(&mut self, fps: u32) {
        self.target_fps = fps;
    }

    pub fn target_fps(&self) -> u32 {
        self.target_fps
    }

    /// True once the OS asked the window to close (close button). The game
    /// decides when to actually stop by returning `false` from the frame
    /// callback.
    pub fn should_close(&self) -> bool {
        self.should_close
    }

    // ---- graphics settings ----

    pub fn set_fullscreen(&mut self, on: bool) {
        if on == self.fullscreen() {
            return;
        }
        let mode = on.then(|| winit::window::Fullscreen::Borderless(None));
        self.window.set_fullscreen(mode);
        // A fullscreen toggle changes the window size; the ensuing Resized event
        // ships the new size + recreate to the render thread.
    }

    pub fn fullscreen(&self) -> bool {
        self.window.fullscreen().is_some()
    }

    pub fn set_vsync(&mut self, on: bool) {
        self.client.set_vsync(on);
    }

    pub fn vsync(&self) -> bool {
        self.client.vsync()
    }

    /// Requests an MSAA sample count; returns the value actually applied
    /// (clamped to hardware support).
    pub fn set_msaa(&mut self, samples: u32) -> u32 {
        self.client.set_msaa(samples)
    }

    pub fn msaa(&self) -> u32 {
        self.client.msaa()
    }

    pub fn max_msaa(&self) -> u32 {
        self.client.max_msaa()
    }

    /// Enables opt-in six-way face culling: each mesh submits only its
    /// camera-facing direction buckets. Off by default (one draw per mesh);
    /// earns its keep only under heavy vertex load.
    pub fn set_cull_faces(&mut self, on: bool) {
        self.client.set_cull_faces(on);
    }

    pub fn cull_faces(&self) -> bool {
        self.client.cull_faces()
    }

    /// Live-swap the CPU-side render feature flags on both threads. The main
    /// thread's copy gates `gate_uniforms`/jitter; the render thread's copy
    /// gates each pass. Idempotent — cheap to call every frame.
    pub fn set_flags(&mut self, flags: RenderFlags) {
        self.flags = flags;
        self.client.set_flags(flags);
    }

    /// Requests a render-resolution scale (0.25..=2.0); returns the value
    /// that will apply. The 3D scene and UI rasterize at the scaled
    /// resolution and are blitted to the window with linear filtering.
    pub fn set_render_scale(&mut self, scale: f32) -> f32 {
        self.client.set_render_scale(scale)
    }

    pub fn render_scale(&self) -> f32 {
        self.client.render_scale()
    }

    // ---- input ----

    pub fn is_key_down(&self, key: Key) -> bool {
        self.input.is_key_down(key)
    }

    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.input.is_key_pressed(key)
    }

    pub fn get_char_pressed(&self) -> Option<char> {
        self.input.get_char_pressed()
    }

    pub fn mouse_delta(&self) -> Vec2 {
        self.input.mouse_delta()
    }

    /// Vertical scroll accumulated this frame, positive scrolling up/away, in
    /// line units (a mouse notch is ~1.0).
    pub fn mouse_wheel(&self) -> f32 {
        self.input.mouse_wheel()
    }

    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.input.is_mouse_button_pressed(button)
    }

    pub fn is_mouse_button_down(&self, button: MouseButton) -> bool {
        self.input.is_mouse_button_down(button)
    }

    /// Captures the cursor: hidden, locked to the window, relative deltas
    /// keep flowing.
    pub fn disable_cursor(&mut self) {
        use winit::window::CursorGrabMode;
        let window = &self.window;
        if window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            .is_err()
        {
            log::warn!("cursor grab not supported on this platform");
        }
        window.set_cursor_visible(false);
    }

    pub fn enable_cursor(&mut self) {
        let window = &self.window;
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
        window.set_cursor_visible(true);
    }

    // ---- meshes ----

    /// Uploads a mesh; drawable in the same frame. Returns None for empty data.
    pub fn upload_mesh(&mut self, data: &MeshData) -> Option<MeshHandle> {
        self.client.upload_mesh(data)
    }

    /// Frees a mesh. Safe while the GPU still uses it (deferred internally).
    pub fn free_mesh(&mut self, handle: MeshHandle) {
        self.client.free_mesh(handle);
    }

    // ---- screenshots ----

    /// Captures the next presented frame (exactly what is shown) to a
    /// timestamped PNG under a cwd-relative `screenshots/` directory, never
    /// overwriting an existing file. Returns the path that will be written, or
    /// `None` if the directory can't be created.
    pub fn screenshot(&mut self) -> Option<std::path::PathBuf> {
        let path = crate::screenshot::next_path()?;
        // Interactive path: no reply awaited (best-effort, fire-and-forget).
        self.client.request_capture(Capture { path: path.clone(), reply: None });
        Some(path)
    }

    // ---- textures ----

    /// Replaces the block texture array sampled by all 3D geometry
    /// ([`MeshVertex`](crate::MeshVertex)'s `layer` field selects the layer).
    /// `layers` are
    /// RGBA8 images of `size*size*4` bytes each; the engine builds mip chains
    /// (box filter) CPU-side and uploads a fresh device-local texture array.
    ///
    /// Rare operation (world load / palette growth): waits for the GPU to go
    /// idle. Contract: layer 0 must render pure white — the engine's
    /// immediate cubes/wires always draw with layer 0. Before the first call
    /// a default 1x1 all-white single-layer array is bound.
    pub fn set_block_textures(&mut self, size: u32, layers: &[Vec<u8>]) {
        self.client.set_block_textures(size, layers);
    }

    /// Uploads minimap pixels (synced per-slot, version-gated).
    pub fn update_minimap(&mut self, rgba: &[u8]) {
        self.client.update_minimap(rgba);
    }

    // ---- text / math ----

    pub fn measure_text(&self, text: &str, font_size: i32) -> i32 {
        font::measure_text(text, font_size)
    }

    /// Projects a world point to screen pixels with the same matrices used
    /// for rendering. Callers filter points behind the camera (raylib parity).
    pub fn world_to_screen(&self, p: Vec3, cam: &Camera3D) -> Vec2 {
        camera::world_to_screen(
            p,
            cam,
            self.client.screen_width().max(1) as f32,
            self.client.screen_height().max(1) as f32,
        )
    }

    // ---- drawing ----

    pub fn begin_frame(&mut self, clear: LinearRgb) -> Frame<'_> {
        self.lists.clear = clear;
        Frame { eng: self }
    }

    pub(crate) fn finish_frame(&mut self) {
        // Recycle returned buffers/allocations, then swap the recorded snapshot
        // for a fresh pooled one (this take_frame blocks when the pool is empty,
        // which is the present-pacing) and submit the recorded one.
        self.client.drain_returns();
        let next = self.client.take_frame();
        let filled = std::mem::replace(&mut self.lists, next);
        self.lists.reset();
        // Retain a copy for deterministic capture before the scene crosses to
        // the render thread (cheap: once-per-frame clone reusing the retained
        // Vec capacities).
        // Deref so `DrawLists::clone_from` reuses the retained Vec capacities
        // rather than `Box::clone_from` reallocating each frame.
        (*self.last_lists).clone_from(&filled);
        self.client.submit_frame(filled);
    }

    /// Re-submits the last presented scene ([`Self::last_lists`]) so a pending
    /// screenshot request latches a real frame. Mirrors [`Self::finish_frame`]'s
    /// present-pacing (`take_frame` blocks until the render thread returns a
    /// buffer) but does not touch the in-progress `lists`. Used only by the
    /// blocking capture path.
    pub(crate) fn present_last(&mut self) {
        self.client.drain_returns();
        let mut next = self.client.take_frame();
        (*next).clone_from(&self.last_lists);
        self.client.submit_frame(next);
    }

    /// Requests a forced capture of the next presented frame to `path`, with a
    /// `reply` channel that receives the real write outcome. The present is
    /// mandatory (never dropped by the pacer); see [`crate::screenshot_to`] for
    /// the blocking wrapper that drives and awaits it.
    pub(crate) fn request_capture(
        &mut self,
        path: std::path::PathBuf,
    ) -> std::sync::mpsc::Receiver<std::io::Result<()>> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.client.request_capture(Capture { path, reply: Some(tx) });
        rx
    }

    fn tick_timing(&mut self) {
        let now = Instant::now();
        self.dt = now.duration_since(self.frame_start).as_secs_f32();
        self.frame_start = now;

        self.fps_window_frames += 1;
        let window = now.duration_since(self.fps_window_start).as_secs_f64();
        if window >= 0.5 {
            self.fps_cached = (self.fps_window_frames as f64 / window).round() as i32;
            self.fps_window_frames = 0;
            self.fps_window_start = now;
        }
    }

    /// Event-driven cadence: the deadline at which the next sim frame must run
    /// if no OS event wakes us first. `None` when uncapped (run every cycle).
    /// `frame_start` was stamped in [`Self::tick_timing`] this cycle.
    fn next_deadline(&self) -> Option<Instant> {
        if self.target_fps == 0 {
            return None;
        }
        let budget = Duration::from_secs_f64(1.0 / self.target_fps as f64);
        Some(self.frame_start + budget)
    }
}

/// Runs the engine's event loop until the callback returns `false` (or the
/// process is asked to quit and the callback honors `should_close`).
pub fn run(config: Config, frame_callback: impl FnMut(&mut Engine) -> bool) {
    // The engine reports everything through `log`; give binaries that never
    // set up a logger a working RUST_LOG path (no-op if one exists).
    let _ = env_logger::try_init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = EngineApp {
        config,
        engine: None,
        callback: frame_callback,
        finished: false,
        ran_this_cycle: false,
    };
    event_loop.run_app(&mut app).expect("Event loop failed");
}

struct EngineApp<F> {
    config: Config,
    engine: Option<Engine>,
    callback: F,
    /// Set once the callback returns false; queued events after `exit()` must
    /// not run another frame (or the last frame's output would repeat).
    finished: bool,
    /// One frame per event-loop cycle: an OS-delivered RedrawRequested
    /// (expose, live-resize) and about_to_wait must not both run a frame.
    ran_this_cycle: bool,
}

impl<F: FnMut(&mut Engine) -> bool> EngineApp<F> {
    /// One full game frame: timing, callback (which draws), input reset,
    /// pacing. Driven from `about_to_wait` — every poll iteration — because
    /// `RedrawRequested` is throttled to the display refresh on macOS, which
    /// would cap an uncapped game at ~60-120 fps regardless of present mode.
    fn run_frame(&mut self, event_loop: &ActiveEventLoop) {
        if self.finished || self.ran_this_cycle {
            return;
        }
        self.ran_this_cycle = true;
        let Some(engine) = self.engine.as_mut() else {
            return;
        };
        engine.tick_timing();
        if !(self.callback)(engine) {
            self.finished = true;
            event_loop.exit();
            return;
        }
        // Reset edges/chars/delta AFTER the game consumed them; new events
        // accumulate for the next frame (raylib poll model).
        engine.input.begin_frame();
        crate::profile::frame_end();
        // Event-first wake: sleep until the sim deadline, but any window/device
        // event wakes the loop immediately (input cadence is event-driven, not
        // frame-capped). Uncapped → Poll (run every cycle).
        match engine.next_deadline() {
            Some(deadline) => event_loop.set_control_flow(ControlFlow::WaitUntil(deadline)),
            None => event_loop.set_control_flow(ControlFlow::Poll),
        }
    }
}

impl<F: FnMut(&mut Engine) -> bool> ApplicationHandler for EngineApp<F> {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        self.ran_this_cycle = false;
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.engine.is_some() {
            return;
        }
        // Window + instance + surface are created on main; the render thread is
        // spawned and builds the Renderer, then replies so the client can build
        // its allocator. The window stays on main (in `Engine`).
        let (window, client) = RenderClient::spawn(event_loop, &self.config);
        self.engine = Some(Engine::new(window, client, &self.config));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(engine) = self.engine.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => engine.should_close = true,
            WindowEvent::Resized(size) => engine.client.resize(size),
            // Frames are driven from about_to_wait; the OS-requested redraw
            // (expose, live-resize) still renders so the window never shows
            // stale content mid-drag.
            WindowEvent::RedrawRequested => self.run_frame(event_loop),
            other => engine.input.on_window_event(&other),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(engine) = self.engine.as_mut() {
            engine.input.on_device_event(&event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.run_frame(event_loop);
    }
}
