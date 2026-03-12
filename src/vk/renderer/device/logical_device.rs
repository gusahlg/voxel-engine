/// This file is suppossed to help with setting up the logical device used for explaining what
/// features from the physical device to use and in what way.
pub fn acquire_logical_device(
    instance: &ash::Instance, physical_device: ash::vk::PhysicalDevice,
    indices: QueueFamiliesIndices, queue_priorities: &[f32],
    device_extensions: &[*const i8], device_features: &ash::vk::PhysicalDeviceFeatures,
) -> ash::Device {
    if indices.graphics == indices.present {
        let queue_info = ash::vk::DeviceQueueCreateInfo::default()
            .queue_family_index(indices.graphics)
            .queue_priorities(&queue_priorities);
        let queue_infos = [queue_info];

        let device_create_info = ash::vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features);

        let logical_device = instance.create_device(
            physical_device,
            &device_create_info,
            None,
        ).expect("Failed to create logical device!");
        return logical_device;
    }

    else {
        let graphics_queue_info = ash::vk::DeviceQueueCreateInfo::default()
            .queue_family_index(indices.graphics)
            .queue_priorities(&queue_priorities);

        // Create info for present queue
        let present_queue_info = ash::vk::DeviceQueueCreateInfo::default()
            .queue_family_index(indices.present)
            .queue_priorities(&queue_priorities);
        let queue_infos = [graphics_queue_info, present_queue_info];

        let device_create_info = ash::vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features);

        let logical_device = instance.create_device(
            physical_device,
            &device_create_info,
            None,
        ).expect("Failed to create logical device!");
        return logical_device;
    }
}
