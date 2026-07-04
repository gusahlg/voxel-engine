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
use crate::color::Color;
use crate::font;
use crate::frame::{DrawLists, Frame};
use crate::input::{InputState, Key, MouseButton};
use crate::mesh::{MeshData, MeshHandle};
use crate::vk::Renderer;

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
        }
    }
}

pub struct Engine {
    pub(crate) renderer: Renderer,
    pub(crate) input: InputState,
    pub(crate) lists: DrawLists,

    target_fps: u32,
    frame_start: Instant,
    dt: f32,
    fps_window_start: Instant,
    fps_window_frames: u32,
    fps_cached: i32,

    should_close: bool,
    fullscreen: bool,
    cursor_disabled: bool,
}

impl Engine {
    fn new(renderer: Renderer, config: &Config) -> Self {
        Self {
            renderer,
            input: InputState::new(),
            lists: DrawLists::new(),
            target_fps: config.target_fps,
            frame_start: Instant::now(),
            dt: 0.0,
            fps_window_start: Instant::now(),
            fps_window_frames: 0,
            fps_cached: 0,
            should_close: false,
            fullscreen: config.fullscreen,
            cursor_disabled: false,
        }
    }

    // ---- window / timing ----

    pub fn screen_width(&self) -> i32 {
        self.renderer.extent().width as i32
    }

    pub fn screen_height(&self) -> i32 {
        self.renderer.extent().height as i32
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
        if on == self.fullscreen {
            return;
        }
        self.fullscreen = on;
        let mode = on.then(|| winit::window::Fullscreen::Borderless(None));
        self.renderer.window.set_fullscreen(mode);
        self.renderer.request_recreate();
    }

    pub fn fullscreen(&self) -> bool {
        self.fullscreen
    }

    pub fn set_vsync(&mut self, on: bool) {
        self.renderer.set_vsync(on);
    }

    pub fn vsync(&self) -> bool {
        self.renderer.vsync()
    }

    /// Requests an MSAA sample count; returns the value actually applied
    /// (clamped to hardware support).
    pub fn set_msaa(&mut self, samples: u32) -> u32 {
        self.renderer.set_msaa(samples)
    }

    pub fn msaa(&self) -> u32 {
        self.renderer.msaa()
    }

    pub fn max_msaa(&self) -> u32 {
        self.renderer.max_msaa()
    }

    /// Requests a render-resolution scale (0.25..=2.0); returns the value
    /// that will apply. The 3D scene and UI rasterize at the scaled
    /// resolution and are blitted to the window with linear filtering.
    pub fn set_render_scale(&mut self, scale: f32) -> f32 {
        self.renderer.set_render_scale(scale)
    }

    pub fn render_scale(&self) -> f32 {
        self.renderer.render_scale()
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
        let window = &self.renderer.window;
        if window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            .is_err()
        {
            log::warn!("cursor grab not supported on this platform");
        }
        window.set_cursor_visible(false);
        self.cursor_disabled = true;
    }

    pub fn enable_cursor(&mut self) {
        let window = &self.renderer.window;
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
        window.set_cursor_visible(true);
        self.cursor_disabled = false;
    }

    // ---- meshes ----

    /// Uploads a mesh; drawable in the same frame. Returns None for empty data.
    pub fn upload_mesh(&mut self, data: &MeshData) -> Option<MeshHandle> {
        self.renderer.upload_mesh(data)
    }

    /// Frees a mesh. Safe while the GPU still uses it (deferred internally).
    pub fn free_mesh(&mut self, handle: MeshHandle) {
        self.renderer.free_mesh(handle);
    }

    // ---- textures ----

    /// Replaces the block texture array sampled by all 3D geometry
    /// ([`Vertex`](crate::Vertex) `color.a` selects the layer). `layers` are
    /// RGBA8 images of `size*size*4` bytes each; the engine builds mip chains
    /// (box filter) CPU-side and uploads a fresh device-local texture array.
    ///
    /// Rare operation (world load / palette growth): waits for the GPU to go
    /// idle. Contract: layer 0 must render pure white — the engine's
    /// immediate cubes/wires always draw with layer 0. Before the first call
    /// a default 1x1 all-white single-layer array is bound.
    pub fn set_block_textures(&mut self, size: u32, layers: &[Vec<u8>]) {
        self.renderer.set_block_textures(size, layers);
    }

    // ---- text / math ----

    pub fn measure_text(&self, text: &str, font_size: i32) -> i32 {
        font::measure_text(text, font_size)
    }

    /// Projects a world point to screen pixels with the same matrices used
    /// for rendering. Callers filter points behind the camera (raylib parity).
    pub fn world_to_screen(&self, p: Vec3, cam: &Camera3D) -> Vec2 {
        let extent = self.renderer.extent();
        camera::world_to_screen(
            p,
            cam,
            extent.width.max(1) as f32,
            extent.height.max(1) as f32,
        )
    }

    // ---- drawing ----

    pub fn begin_frame(&mut self, clear: Color) -> Frame<'_> {
        self.lists.clear = clear;
        Frame { eng: self }
    }

    pub(crate) fn finish_frame(&mut self) {
        self.renderer.draw_frame(&self.lists);
        self.lists.reset();
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

    /// raylib-style frame cap: sleep most of the remainder, spin the tail.
    fn pace(&self) {
        if self.target_fps == 0 {
            return;
        }
        let budget = Duration::from_secs_f64(1.0 / self.target_fps as f64);
        let elapsed = self.frame_start.elapsed();
        if elapsed >= budget {
            return;
        }
        let remaining = budget - elapsed;
        if remaining > Duration::from_millis(2) {
            std::thread::sleep(remaining - Duration::from_millis(1));
        }
        while self.frame_start.elapsed() < budget {
            std::hint::spin_loop();
        }
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
        engine.pace();
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
        let renderer = Renderer::new(
            event_loop,
            &self.config.title,
            self.config.width,
            self.config.height,
            self.config.resizable,
            self.config.fullscreen,
            self.config.vsync,
            self.config.msaa,
            self.config.render_scale,
        );
        self.engine = Some(Engine::new(renderer, &self.config));
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
            WindowEvent::Resized(_) => engine.renderer.request_recreate(),
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
