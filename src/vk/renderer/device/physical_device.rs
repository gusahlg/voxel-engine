/// This file is for setting up the physical device and should make all the checks associated with
/// doing so as well as holding additional metadata.
pub fn get_queues(instance: &ash::Instance, physical_device: ash::vk::PhysicalDevice) -> Vec<ash::vk::QueueFamilyProperties> {
    unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
}

// Move to mod.rs
pub fn configure_queue_families(instance: &ash::Instance, physical_device: ash::vk::PhysicalDevice, surface_loader: &ash::khr::surface::Instance, surface: ash::vk::SurfaceKHR) -> QueueFamiliesIndices {
    let queues = get_queues(instance, physical_device);

    // Store indices
    let graphics_queue_idx: u32;
    let present_queue_idx: u32;

    let supports_graphics: bool;
    let supports_present: bool;
    for (index, family) in queues.iter().enumerate() {
        let index = index as u32;

        supports_graphics = family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS);

        supports_present = unsafe {
            surface_loader.get_physical_device_surface_support(
                physical_device,
                index,
                surface,
            )?
        };
        
        if supports_graphics {
            graphics_queue_idx = index;
        }
        else if supports_present {
            present_queue_idx = index;
        }
        else if supports_graphics && supports_present {
            return QueueFamiliesIndices { graphics: graphics_queue_idx, present: present_queue_idx };
        }
    }
}

pub struct QueueFamiliesIndices {
    graphics: u32,
    present: u32,
}
