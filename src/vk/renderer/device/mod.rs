/// The file that exposes all of the device functionality, is supposed to act as a thin and
/// practical api that keeps complicated details hidden.
mod physical_device;
use physical_device::*;

mod logical_device;
use logical_device::*;

struct Device {
    physical_device: ash::vk::PhysicalDevice,
    logical_device: ash::Device,

    graphics_queue: ash::vk::Queue,
    present_queue: ash::vk::Queue,

    command_pool: ash::vk::CommandPool,
}
impl Device {
    pub fn configure(&mut self, instance: &ash::Instance, surface_loader: &ash::khr::surface::Instance, surface: ash::vk::SurfaceKHR) {
        // Pick physical device
        let devices = instance.enumerate_physical_devices().expect("Could not find any physical devices!");

        // Get a compatible device and save it and its graphics and present available queue family
        // indices.
        let queue_indices: (QueueFamiliesIndices, ash::vk::PhysicalDevice) = devices.into_iter()
        .find_map(|device| {
            acquire_queue_families(instance, device, surface_loader, surface).map(|v| {
                (
                    QueueFamiliesIndices {
                        graphics: v.graphics,
                        present: v.present,
                    },
                    device,
                )
            })
        }).expect("No suitable physical device found!");

        let (indices, physical_device) = queue_indices;
        self.physical_device = physical_device;

        // Extensions
        let device_extensions = [
            ash::khr::swapchain::NAME.as_ptr(),
        ];

        // Device features
        let device_features = ash::vk::PhysicalDeviceFeatures::default();

        let queue_priorities = [1.0_f32];

        let logical_device = acquire_logical_device(instance, self.physical_device, indices, queue_priorities, device_extensions, device_features);
        self.logical_device = logical_device;

        let graphics_queue = unsafe {
            self.logical_device.get_device_queue(indices.graphics, 0)
        };

        let present_queue = unsafe {
            self.logical_device.get_device_queue(indices.present, 0)
        };
        
        self.graphics_queue = graphics_queue;
        self.present_queue = present_queue;
    }
}

