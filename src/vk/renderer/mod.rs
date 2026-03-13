use winit::event_loop::ActiveEventLoop;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

mod device;
use device::Device;

// Holds all core Vulkan state and the window. Created in App::resumed() once
// the event loop is active and we can obtain platform display/window handles.
pub struct Renderer {
    vk_entry: ash::Entry,
    vk_instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: ash::vk::SurfaceKHR,
    pub window: winit::window::Window,

    // Abstraction that stores:
    //      * Graphics and present queue
    //      * Physical device
    //      * Logical device
    //      * Command pool
    device: Device,
}

impl Renderer {
    pub fn new(event_loop: &ActiveEventLoop, title: &str) -> Self {
        // Load libvulkan.so and get global function pointers
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan") };

        // Ask the platform which Vulkan extensions are needed for surface creation
        let display_handle = event_loop.display_handle().unwrap().as_raw();
        let extensions = ash_window::enumerate_required_extensions(display_handle)
            .expect("Failed to enumerate required extensions");

        // Create the Vulkan instance with the required surface extensions enabled
        let app_info = ash::vk::ApplicationInfo::default()
            .application_name(c"voxel_engine")
            .application_version(ash::vk::make_api_version(0, 0, 1, 0))
            .engine_version(ash::vk::make_api_version(0, 0, 1, 0))
            .api_version(ash::vk::API_VERSION_1_3);

        let create_info = ash::vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(extensions);

        let instance = unsafe {
            entry.create_instance(&create_info, None)
                .expect("Failed to create Vulkan instance")
        };

        // Load the VK_KHR_surface function pointers (destroy_surface, query capabilities, etc.)
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // Create the window
        let attrs = winit::window::WindowAttributes::default().with_title(title);
        let window = event_loop.create_window(attrs).expect("Failed to create window");

        // Create the Vulkan surface from the window's platform handles
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .expect("Failed to create Vulkan surface")
        };

        // Below is for finding the physical devices and then also adding on the abstraction of the
        // logical device to make the renderer ready to actually interact with the gpu.
        let device: Device = Device::new(&instance, &surface_loader, surface);

        Self {
            vk_entry: entry, 
            vk_instance: instance, 
            surface_loader: surface_loader, 
            surface: surface, 
            window: window, 
            device: device,
        }
    }
}

// Destroy in reverse creation order: surface first, then instance.
// Entry doesn't need explicit cleanup (it's just loaded function pointers).
impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
            self.vk_instance.destroy_instance(None);
        }
    }
}

// Helper functions 

