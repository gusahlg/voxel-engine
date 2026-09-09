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
    /// Camera jitter + TAA resolve at present time (output/swapchain
    /// resolution). Always coupled: jitter is injected every rendered frame;
    /// the resolve (and history write) run only on presented frames.
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
    /// HDR bloom (threshold + downsample compute → quarter-res spill). Off skips
    /// the pyramid dispatch and clears the bloom target so the spill bloom term
    /// is zero; with godrays also off the spill dispatch is skipped entirely.
    pub bloom: bool,
    /// Screen-space godrays: volumetric sun rays in the quarter-res spill pass.
    /// Off zeroes that term; with bloom also off the spill dispatch is skipped.
    pub godrays: bool,
    /// Cascade occluder draws + far-field fallback; off is fully lit.
    pub shadows: bool,
    /// Procedural sky background pass; off shows the clear colour.
    pub sky: bool,
    /// Variable-rate shading: the depth-classified rate image that coarsens
    /// fragment shading on distant/flat/sky tiles. Off skips both the classify
    /// dispatch and the rate attachment (full-rate shading everywhere). No-op
    /// when the device lacks attachment fragment shading rate.
    ///
    /// Default off: measured on an RTX 4060 at 1080p the classify pass plus
    /// the shading-rate attachment cost more than the coarse shading saves
    /// (3361 vs 5064 FPS). VRS pays off only at 4K-class render extents, so
    /// it is opt-in.
    pub vrs: bool,
    /// Water surface animation (`anim` lane time). Off freezes the phase:
    /// water renders, tinted and reflective, but still — the cheapest frame
    /// for water-heavy scenes.
    pub water_anim: bool,
    /// Radial vignette darkening in the tonemap pass.
    pub vignette: bool,
    /// Night starfield in the sky pass (`extras.x` gain). Off skips the
    /// per-pixel hash-grid star evaluation entirely.
    pub stars: bool,
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
            shadows: true,
            sky: true,
            vrs: false,
            water_anim: true,
            vignette: false,
            stars: true,
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

    target_fps: u32,
    frame_start: Instant,
    dt: f32,
    fps_window_start: Instant,
    fps_window_frames: u32,
    fps_cached: i32,

    should_close: bool,
}

impl Engine {
    fn new(event_loop: &ActiveEventLoop, config: &Config) -> Result<Self, String> {
        let (window, mut client) = RenderClient::spawn(event_loop, config)?;
        let lists = client.take_frame();
        let exposure_shared = client.exposure();
        Ok(Self {
            client,
            exposure_shared,
            window,
            input: InputState::new(),
            lists,
            flags: config.flags,
            target_fps: config.target_fps,
            frame_start: Instant::now(),
            dt: 0.0,
            fps_window_start: Instant::now(),
            fps_window_frames: 0,
            fps_cached: 0,
            should_close: false,
        })
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

    /// Frames the render thread completed (`draw_frame` returned). Monotonic.
    ///
    /// A benchmark must count rendered frames, not game frames: the render loop
    /// coalesces queued snapshots to the newest, so the game FPS counter can
    /// run ahead of what actually reached the GPU.
    pub fn frames_rendered(&self) -> u64 {
        self.client.frames_rendered()
    }

    /// Frames dropped by render-loop coalescing (kept only the newest queued
    /// `RenderCmd::Frame`). Monotonic.
    pub fn frames_coalesced(&self) -> u64 {
        self.client.frames_coalesced()
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

    /// Replaces the render feature flags at runtime (settings menu / console).
    /// Updates both CPU copies: this thread's gate set and, via the ordered
    /// command stream, the render thread's — so the change lands atomically at
    /// the next frame boundary.
    pub fn set_flags(&mut self, flags: RenderFlags) {
        self.flags = flags;
        self.client.set_flags(flags);
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

    /// The device's block-texture array layer ceiling
    /// (`limits.maxImageArrayLayers`). The app clamps how many block texture
    /// layers it builds against this; ids past it wrap (see the game's mesher).
    pub fn max_texture_array_layers(&self) -> u32 {
        self.client.max_texture_layers()
    }

    /// GPU per-direction face-run culling: the cull shader emits contiguous
    /// camera-facing quad runs instead of a whole-mesh draw.
    ///
    /// On by default (`Config` has no field). Safe to toggle at runtime — the
    /// change is sent on the render-thread command stream and lands at the next
    /// frame boundary. `false` is an explicit opt-out (whole-mesh draws).
    pub fn set_cull_faces(&mut self, on: bool) {
        self.client.set_cull_faces(on);
    }

    pub fn cull_faces(&self) -> bool {
        self.client.cull_faces()
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

    /// Upload a tracked mesh; placement recovered from draw offset (movers).
    /// Static geometry should use [`upload_mesh_placed`](Self::upload_mesh_placed).
    pub fn upload_mesh(&mut self, data: &MeshData) -> Option<MeshHandle> {
        self.client.upload_mesh(data)
    }

    /// Upload terrain mesh with pinned placement; draws only gate visibility.
    pub fn upload_mesh_placed(
        &mut self,
        data: &MeshData,
        placement: crate::mesh::MeshPlacement,
    ) -> Option<MeshHandle> {
        self.client.upload_mesh_placed(data, placement)
    }

    /// Frees a mesh. Safe while the GPU still uses it (deferred internally).
    pub fn free_mesh(&mut self, handle: MeshHandle) {
        self.client.free_mesh(handle);
    }

    /// Gate mesh visibility for GPU culling (coarse "app wants drawn").
    pub fn set_visible(&mut self, handle: MeshHandle, on: bool) {
        self.client.set_visible(handle.slot, on);
    }

    /// Set per-draw style (FadeStyle + flat sRGB RGBA8); only changed values patch.
    pub fn set_mesh_style(
        &mut self,
        handle: MeshHandle,
        style: crate::frame::FadeStyle,
        flat_rgba: u32,
    ) {
        self.client.set_mesh_style(
            handle,
            crate::vk::buffers::DrawDyn {
                mode: style.bits(),
                flat_rgba,
            },
        );
    }

    /// Update mover mesh placement (avatars only; terrain ignores this).
    pub fn set_mesh_placement(
        &mut self,
        handle: MeshHandle,
        placement: crate::mesh::MeshPlacement,
    ) {
        self.client.set_mesh_placement(handle, placement);
    }

    // ---- screenshots ----

    /// Captures the next presented frame (exactly what is shown) to a
    /// timestamped PNG under a cwd-relative `screenshots/` directory, never
    /// overwriting an existing file. Returns the path that will be written, or
    /// `None` if the directory can't be created.
    pub fn screenshot(&mut self) -> Option<std::path::PathBuf> {
        let path = crate::screenshot::next_path()?;
        // Interactive path: no reply awaited (best-effort, fire-and-forget).
        self.client.request_capture(Capture {
            path: path.clone(),
            reply: None,
        });
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
        // Recycle returned buffers/allocations, submit the recorded snapshot,
        // then take a fresh pooled one to record into. Submit first so the
        // render thread can start (and recycle a box) while we wait. The pool
        // is sized so main stays at most one frame ahead; `take_frame` parks
        // when every box is in flight.
        self.client.drain_returns();
        if let Some(next) = self.client.pop_idle_frame() {
            let filled = std::mem::replace(&mut self.lists, next);
            self.client.submit_frame(filled);
        } else {
            let filled = std::mem::replace(&mut self.lists, self.client.take_placeholder());
            self.client.submit_frame(filled);
            let dummy = std::mem::replace(&mut self.lists, self.client.take_frame());
            self.client.stash_placeholder(dummy);
        }
        self.lists.reset();
    }

    /// Re-submits the last completed scene so a pending screenshot request
    /// latches a real frame. Waits for the just-submitted snapshot to return
    /// (its vertex data is still intact) instead of cloning draw lists every
    /// frame. Does not touch the in-progress `lists`.
    pub(crate) fn present_last(&mut self) {
        self.client.drain_returns();
        let last = self.client.wait_last_drawn();
        self.client.submit_frame(last);
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
        self.client.request_capture(Capture {
            path,
            reply: Some(tx),
        });
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
    let event_loop = EventLoop::new().expect(
        "Failed to create event loop: set DISPLAY or WAYLAND_DISPLAY, and a \
         windowing library must be loadable",
    );
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = EngineApp {
        config,
        engine: None,
        callback: frame_callback,
        finished: false,
        ran_this_cycle: false,
        init_failed: false,
    };
    if let Err(err) = event_loop.run_app(&mut app) {
        log::error!(
            "event loop failed: {err}; the windowing connection was lost — the \
             compositor drops clients whose main thread stalls for seconds; \
             check for long synchronous work in the frame callback"
        );
        // Keep `run` as `()` so `voxel_engine::run(config, |eng| ...)` callers
        // (the game and the demo) stay source-compatible. `process::exit` skips
        // remaining destructors, so drop the app first: `Engine`/`RenderClient`
        // join the render thread and destroy Vulkan objects the same way a
        // normal `finished` exit does.
        drop(app);
        std::process::exit(1);
    }
    if app.init_failed {
        drop(app);
        std::process::exit(1);
    }
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
    /// Renderer construction failed (logged); `run` exits non-zero after the
    /// event loop returns so this is not a panic and not a silent close.
    init_failed: bool,
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
        match Engine::new(event_loop, &self.config) {
            Ok(engine) => self.engine = Some(engine),
            Err(_) => {
                // `RenderClient::spawn` already logged the readable error.
                self.init_failed = true;
                event_loop.exit();
            }
        }
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
