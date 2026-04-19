/// The file that exposes all of the device functionality, is supposed to act as a thin and
/// practical api that keeps complicated details hidden.
use ash::vk;

mod physical_device;
use physical_device::*;

mod logical_device;
use logical_device::*;

pub struct Device {
    pub physical_device: vk::PhysicalDevice,
    pub logical_device: ash::Device,

    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,

    pub graphics_queue_family: u32,
    pub present_queue_family: u32,

    pub command_pool: vk::CommandPool,
}
impl Device {
    pub fn new(instance: &ash::Instance, surface_loader: &ash::khr::surface::Instance, surface: vk::SurfaceKHR) -> Self {
        // Pick physical device
        let devices = unsafe { instance.enumerate_physical_devices().expect("Could not find any physical devices!") };

        // Get a compatible device and save it and its graphics and present available queue family
        // indices.
        let queue_indices: (QueueFamiliesIndices, vk::PhysicalDevice) = devices.into_iter()
        .find_map(|device| {
            if !supports_modern_features(instance, device) {
                return None;
            }

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

        // Extensions
        let device_extensions = [
            ash::khr::swapchain::NAME.as_ptr(),
        ];

        // Device features
        let device_features = vk::PhysicalDeviceFeatures::default();

        let queue_priorities = [1.0_f32];

        let logical_device = acquire_logical_device(
            instance,
            physical_device,
            indices.clone(),
            &queue_priorities,
            &device_extensions,
            &device_features,
        );

        let graphics_queue = unsafe {
            logical_device.get_device_queue(indices.graphics, 0)
        };

        let present_queue = unsafe {
            logical_device.get_device_queue(indices.present, 0)
        };

        // Command pool for allocating command buffers
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(indices.graphics)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            logical_device.create_command_pool(&pool_info, None)
                .expect("Failed to create command pool")
        };
        
        Self {
            physical_device,
            logical_device,
            graphics_queue,
            present_queue,
            graphics_queue_family: indices.graphics,
            present_queue_family: indices.present,
            command_pool,
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.destroy_command_pool(self.command_pool, None);
            self.logical_device.destroy_device(None);
        }
    }
}
