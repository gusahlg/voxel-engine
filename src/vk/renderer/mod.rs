use winit::event_loop::ActiveEventLoop;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

mod device;
use device::Device;

mod swapchain;
use swapchain::{acquire_swapchain, SwapchainInfo};

mod render_pass;
use render_pass::{acquire_render_pass};

mod rendering;
use rendering::*;

// Holds all core Vulkan state and the window. Created in App::resumed() once
// the event loop is active and we can obtain platform display/window handles.
pub struct Renderer {
    // Abstraction that stores:
    //      * Graphics and present queue
    //      * Physical device
    //      * Logical device
    //      * Command pool
    device: Device,

    vk_entry: ash::Entry,
    vk_instance: ash::Instance,

    swapchain_info: SwapchainInfo,
    render_pass: ash::vk::RenderPass,
    
    // For rendering
    pipeline_bundle: RenderingBundle,

    // For interacting with the screen
    surface_loader: ash::khr::surface::Instance,
    surface: ash::vk::SurfaceKHR,
    pub window: winit::window::Window,
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

        let size = window.inner_size();
        let window_extent = ash::vk::Extent2D {
            width: size.width,
            height: size.height,
        };

        // Create swapchain
        let swapchain_info = acquire_swapchain(
            &instance,
            device.physical_device,
            &device.logical_device,
            &surface_loader,
            surface,
            window_extent,
            device.graphics_queue_family,
            device.present_queue_family,
        );

        let render_pass = acquire_render_pass(&device.logical_device, &swapchain_info.format);

        let pipeline_bundle = RenderingBundle::new(
            &device.logical_device,
            render_pass,
            swapchain_info.extent,
            &swapchain_info.image_views,
        );

        Self {
            vk_entry: entry,
            vk_instance: instance,
            swapchain_info,
            render_pass,
            pipeline_bundle,
            surface_loader,
            surface,
            window,
            device,
        }
    }

    pub fn draw_frame(&self) {

    }
}

// Destroy in reverse creation order.
// Entry doesn't need explicit cleanup (it's just loaded function pointers).
impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.pipeline_bundle.destroy(&self.device.logical_device);

            for &view in &self.swapchain_info.image_views {
                self.device.logical_device.destroy_image_view(view, None);
            }
            self.swapchain_info.swapchain_loader
                .destroy_swapchain(self.swapchain_info.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.vk_instance.destroy_instance(None);
        }
    }
}
