mod vk;
use vk::*;

fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new("Voxetect");
    event_loop.run_app(&mut app).expect("Event loop failed");
}
