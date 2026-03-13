/// This file is for setting up the physical device and should make all the checks associated with
/// doing so as well as holding additional metadata.
fn get_queues(instance: &ash::Instance, physical_device: ash::vk::PhysicalDevice) -> Vec<ash::vk::QueueFamilyProperties> {
    unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
}

pub fn acquire_queue_families(instance: &ash::Instance, physical_device: ash::vk::PhysicalDevice, surface_loader: &ash::khr::surface::Instance, surface: ash::vk::SurfaceKHR) -> Option<QueueFamiliesIndices> {
    let queues = get_queues(instance, physical_device);

    let mut graphics_queue_idx: Option<u32> = None;
    let mut present_queue_idx: Option<u32> = None;

    for (index, family) in queues.iter().enumerate() {
        let index = index as u32;

        let supports_graphics = family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS);

        let supports_present = unsafe {
            surface_loader.get_physical_device_surface_support(
                physical_device,
                index,
                surface,
            ).unwrap_or(false)
        };

        if supports_graphics && graphics_queue_idx.is_none() {
            graphics_queue_idx = Some(index);
        }
        if supports_present && present_queue_idx.is_none() {
            present_queue_idx = Some(index);
        }

        if graphics_queue_idx.is_some() && present_queue_idx.is_some() {
            break;
        }
    }

    Some(QueueFamiliesIndices {
        graphics: graphics_queue_idx?,
        present: present_queue_idx?,
    })
}

#[derive(Clone)]
pub struct QueueFamiliesIndices {
    pub graphics: u32,
    pub present: u32,
}
