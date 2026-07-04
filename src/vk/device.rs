/// Physical device selection and logical device creation.
///
/// Selection requires Vulkan 1.3 (dynamic rendering + synchronization2), the
/// swapchain extension, and graphics+present queues; among suitable devices a
/// discrete GPU wins over an integrated one.
use ash::{khr, vk};

pub struct Device {
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub graphics_family: u32,
    pub present_family: u32,
    pub command_pool: vk::CommandPool,
    pub properties: vk::PhysicalDeviceProperties,
    /// Sample counts supported by BOTH color and depth framebuffer attachments.
    pub msaa_caps: vk::SampleCountFlags,
}

struct Candidate {
    physical: vk::PhysicalDevice,
    graphics_family: u32,
    present_family: u32,
    properties: vk::PhysicalDeviceProperties,
    score: u32,
}

impl Device {
    pub fn new(
        instance: &ash::Instance,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Self {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };

        let best = physical_devices
            .into_iter()
            .filter_map(|pd| evaluate(instance, pd, surface_loader, surface))
            .max_by_key(|c| c.score)
            .expect("No suitable Vulkan 1.3 GPU found (needs dynamic rendering + synchronization2 + swapchain)");

        log::info!(
            "Using GPU: {}",
            best.properties
                .device_name_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "<unknown>".into())
        );

        let mut device_extensions = vec![khr::swapchain::NAME.as_ptr()];
        #[cfg(target_os = "macos")]
        device_extensions.push(khr::portability_subset::NAME.as_ptr());

        // One queue per distinct family (graphics may equal present).
        let queue_priorities = [1.0_f32];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(best.graphics_family)
                .queue_priorities(&queue_priorities),
        ];
        if best.present_family != best.graphics_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(best.present_family)
                    .queue_priorities(&queue_priorities),
            );
        }

        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let device_features = vk::PhysicalDeviceFeatures::default();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features)
            .push_next(&mut vulkan_13_features);

        let device = unsafe {
            instance
                .create_device(best.physical, &device_create_info, None)
                .expect("Failed to create logical device")
        };

        let graphics_queue = unsafe { device.get_device_queue(best.graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(best.present_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(best.graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .expect("Failed to create command pool")
        };

        let msaa_caps = best.properties.limits.framebuffer_color_sample_counts
            & best.properties.limits.framebuffer_depth_sample_counts;

        Self {
            physical: best.physical,
            device,
            graphics_queue,
            present_queue,
            graphics_family: best.graphics_family,
            present_family: best.present_family,
            command_pool,
            properties: best.properties,
            msaa_caps,
        }
    }

    /// Largest supported sample count out of {1, 2, 4, 8}.
    pub fn max_msaa(&self) -> u32 {
        for (flag, n) in [
            (vk::SampleCountFlags::TYPE_8, 8),
            (vk::SampleCountFlags::TYPE_4, 4),
            (vk::SampleCountFlags::TYPE_2, 2),
        ] {
            if self.msaa_caps.contains(flag) {
                return n;
            }
        }
        1
    }

    pub unsafe fn destroy(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

fn evaluate(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Option<Candidate> {
    let properties = unsafe { instance.get_physical_device_properties(physical) };

    // Chaining a Vulkan13Features struct is only valid on API >= 1.3 devices.
    if vk::api_version_major(properties.api_version) < 1
        || (vk::api_version_major(properties.api_version) == 1
            && vk::api_version_minor(properties.api_version) < 3)
    {
        return None;
    }

    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan_13_features);
    unsafe { instance.get_physical_device_features2(physical, &mut features2) };
    if vulkan_13_features.dynamic_rendering != vk::TRUE
        || vulkan_13_features.synchronization2 != vk::TRUE
    {
        return None;
    }

    let extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical)
            .ok()?
    };
    let has_swapchain = extensions.iter().any(|ext| {
        ext.extension_name_as_c_str()
            .is_ok_and(|name| name == khr::swapchain::NAME)
    });
    if !has_swapchain {
        return None;
    }

    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let mut graphics_family = None;
    let mut present_family = None;
    for (index, family) in families.iter().enumerate() {
        let index = index as u32;
        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics_family.is_none() {
            graphics_family = Some(index);
        }
        let supports_present = unsafe {
            surface_loader
                .get_physical_device_surface_support(physical, index, surface)
                .unwrap_or_else(|err| {
                    log::warn!("surface support query failed for family {index}: {err:?}");
                    false
                })
        };
        if supports_present && present_family.is_none() {
            present_family = Some(index);
        }
    }

    let score = match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 100,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 20,
        _ => 10,
    };

    Some(Candidate {
        physical,
        graphics_family: graphics_family?,
        present_family: present_family?,
        properties,
        score,
    })
}
