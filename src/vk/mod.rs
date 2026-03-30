mod renderer;
use renderer::*;

use winit::event_loop::ActiveEventLoop;
use winit::event::WindowEvent;
use winit::window::WindowId;
use winit::application::ApplicationHandler;

/// Top-level application state. Implements ApplicationHandler to receive
/// events from winit. Delegates all Vulkan work to Renderer.
pub struct App {
    renderer: Option<Renderer>,
    title: &'static str,
}

impl App {
    pub fn new(title: &'static str) -> Self {
        Self { renderer: None, title }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }
        self.renderer = Some(Renderer::new(event_loop, self.title));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                log::info!("Resized to {}x{}", size.width, size.height);
            }

            // For drawing frames
            WindowEvent::RedrawRequested => {
                self.renderer.as_ref().expect("Renderer was not found at frame rendering!").draw_frame();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = &self.renderer {
            renderer.window.request_redraw();
        }
    }
}
