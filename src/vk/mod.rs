/// This is the main file for the rendering, this includes it being the home for the app struct
/// holing the vulkan instance and is the gateway to all operations.

// This is the api for window creation through winit
mod window;
use window::*;

use ash::vk;
use std::ffi::CStr;

// impl ApplicationHandler for Window {
//     pub fn resumed(&mut self, event_loop: &ActiveEventLoop) {
//         if self.window.is_some() {
//             return;
//         }
// 
//         let attrs = WindowAttributes::default().with_title("voxel_engine");
//         let window = event_loop.create_window(attrs).unwrap();
//         self.window = Some(window);
//     }
// 
//     pub fn window_event(
//         &mut self,
//         event_loop: &ActiveEventLoop,
//         _window_id: WindowId,
//         event: WindowEvent,
//     ) {
//         match event {
//             WindowEvent::CloseRequested => event_loop.exit(),
// 
//             WindowEvent::Resized(size) => {
//                 log::info!("Resized to {}x{}", size.width, size.height);
//             }
// 
//             WindowEvent::RedrawRequested => {
//                 // Later: render a Vulkan frame here.
//                 // For now: no-op.
//             }
// 
//             _ => {}
//         }
//     }
// 
//     pub fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
//         // Continuous frames: request redraw each loop iteration.
//         if let Some(w) = &self.window {
//             w.request_redraw();
//         }
//     }
// }

// Supposed to hold instance and expose all functionality to outside 
pub struct App {
    vk_instance: ash::Instance,
    vk_entry: ash::Entry,
    window: Window,
}
impl App {
    pub fn new() -> Self {
        let app_info = vk::ApplicationInfo::default()
            .application_name(CStr::from_bytes_with_nul(b"voxel_engine\0").unwrap())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);  // or 1_2, 1_0, etc.

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let entry = unsafe { ash::Entry::load().expect("Failed to create Vulkan instance") };
            
        let instance = unsafe { entry.create_instance(&create_info, None).expect("Failed to create Vulkan instance") };

        Self { vk_instance: instance, 
               vk_entry: entry,
               window: Window::new(),
        }
    }
}

