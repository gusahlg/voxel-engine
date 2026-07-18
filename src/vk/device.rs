use ash::{ext, khr, vk};

/// Proof that `VK_EXT_memory_budget` is enabled.
#[derive(Clone, Copy)]
pub struct MemoryBudget {
    _priv: (),
}

#[derive(Clone, Copy)]
pub struct BudgetSnapshot {
    pub heap_budget: [u64; vk::MAX_MEMORY_HEAPS],
    pub heap_usage: [u64; vk::MAX_MEMORY_HEAPS],
}

impl MemoryBudget {
    unsafe fn assume_enabled() -> Self {
        Self { _priv: () }
    }

    pub unsafe fn query(
        self,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
    ) -> BudgetSnapshot {
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { instance.get_physical_device_memory_properties2(physical, &mut props2) };
        BudgetSnapshot {
            heap_budget: budget.heap_budget,
            heap_usage: budget.heap_usage,
        }
    }
}

/// Max anisotropy when supported.
#[derive(Clone, Copy)]
pub struct Anisotropy(f32);

impl Anisotropy {
    pub fn clamp(self, desired: f32) -> f32 {
        desired.clamp(1.0, self.0)
    }
}

pub struct FragmentShadingRate {
    pub texel_size: vk::Extent2D,
}

pub struct Device {
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub graphics_family: u32,
    pub present_family: u32,
    pub transfer_queue: vk::Queue,
    pub transfer_family: u32,
    pub transfer_tier: crate::vk::transfer::Tier,
    pub command_pool: vk::CommandPool,
    pub push_descriptor: khr::push_descriptor::Device,
    pub memory_budget: Option<MemoryBudget>,
    pub anisotropy: Option<Anisotropy>,
    pub fragment_shading_rate: Option<FragmentShadingRate>,
    pub dynamic_rendering_local_read: bool,
    pub local_read: Option<khr::dynamic_rendering_local_read::Device>,
    pub msaa_caps: vk::SampleCountFlags,
    pub multi_draw_indirect: bool,
    pub draw_indirect_first_instance: bool,
    /// Required for GPU-driven culling; kept as field for single-source enable.
    pub draw_indirect_count: bool,
    pub timestamp_period_ns: f32,
    pub timestamps_supported: bool,
    pub max_image_array_layers: u32,
}

struct Candidate {
    physical: vk::PhysicalDevice,
    graphics_family: u32,
    present_family: u32,
    /// Separate transfer family, if available.
    transfer_family: Option<u32>,
    graphics_queue_count: u32,
    properties: vk::PhysicalDeviceProperties,
    multi_draw_indirect: bool,
    draw_indirect_first_instance: bool,
    draw_indirect_count: bool,
    memory_budget: bool,
    max_anisotropy: Option<f32>,
    fragment_shading_rate: Option<vk::Extent2D>,
    dynamic_rendering_local_read: bool,
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
            .expect("No suitable Vulkan 1.3 GPU found (needs dynamic rendering + synchronization2 + drawIndirectCount + swapchain)");

        log::info!(
            "Using GPU: {}",
            best.properties
                .device_name_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "<unknown>".into())
        );

        let mut device_extensions = vec![
            khr::swapchain::NAME.as_ptr(),
            khr::push_descriptor::NAME.as_ptr(),
        ];
        #[cfg(target_os = "macos")]
        device_extensions.push(khr::portability_subset::NAME.as_ptr());
        if best.memory_budget {
            device_extensions.push(ext::memory_budget::NAME.as_ptr());
        }
        if best.fragment_shading_rate.is_some() {
            device_extensions.push(khr::fragment_shading_rate::NAME.as_ptr());
        }
        if best.dynamic_rendering_local_read {
            device_extensions.push(khr::dynamic_rendering_local_read::NAME.as_ptr());
        }

        // Pick transfer tier based on available queues.
        let transfer_tier = if best.transfer_family.is_some() {
            crate::vk::transfer::Tier::DedicatedFamily
        } else if best.graphics_queue_count > 1 {
            crate::vk::transfer::Tier::SecondQueueSameFamily
        } else {
            crate::vk::transfer::Tier::SameQueueFallback
        };

        // Create queues: one per family, plus second graphics queue if needed.
        let single_priority = [1.0_f32];
        let dual_priority = [1.0_f32, 1.0_f32];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(best.graphics_family)
                .queue_priorities(
                    if transfer_tier == crate::vk::transfer::Tier::SecondQueueSameFamily {
                        &dual_priority[..]
                    } else {
                        &single_priority[..]
                    },
                ),
        ];
        if best.present_family != best.graphics_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(best.present_family)
                    .queue_priorities(&single_priority),
            );
        }
        if let Some(family) = best.transfer_family
            && family != best.graphics_family
            && family != best.present_family
        {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&single_priority),
            );
        }

        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .draw_indirect_count(best.draw_indirect_count);
        let mut vulkan_11_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        let device_features = vk::PhysicalDeviceFeatures::default()
            .multi_draw_indirect(best.multi_draw_indirect)
            .draw_indirect_first_instance(best.draw_indirect_first_instance)
            .sampler_anisotropy(best.max_anisotropy.is_some());
        if !best.multi_draw_indirect || !best.draw_indirect_first_instance {
            log::info!(
                "indirect draw features: multiDrawIndirect={} drawIndirectFirstInstance={} (using fallback draw path)",
                best.multi_draw_indirect,
                best.draw_indirect_first_instance,
            );
        }

        let mut fsr_features = vk::PhysicalDeviceFragmentShadingRateFeaturesKHR::default()
            .attachment_fragment_shading_rate(true);
        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features)
            .push_next(&mut vulkan_13_features)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_11_features);
        if best.fragment_shading_rate.is_some() {
            device_create_info = device_create_info.push_next(&mut fsr_features);
        }
        let mut local_read_features =
            vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR::default()
                .dynamic_rendering_local_read(true);
        if best.dynamic_rendering_local_read {
            device_create_info = device_create_info.push_next(&mut local_read_features);
        }

        let device = unsafe {
            instance
                .create_device(best.physical, &device_create_info, None)
                .expect("Failed to create logical device")
        };

        let push_descriptor = khr::push_descriptor::Device::new(instance, &device);

        let graphics_queue = unsafe { device.get_device_queue(best.graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(best.present_family, 0) };

        let (transfer_family, transfer_queue) = match transfer_tier {
            crate::vk::transfer::Tier::DedicatedFamily => {
                let family = best
                    .transfer_family
                    .expect("DedicatedFamily tier implies a transfer family");
                (family, unsafe { device.get_device_queue(family, 0) })
            }
            crate::vk::transfer::Tier::SecondQueueSameFamily => (best.graphics_family, unsafe {
                device.get_device_queue(best.graphics_family, 1)
            }),
            crate::vk::transfer::Tier::SameQueueFallback => (best.graphics_family, graphics_queue),
        };

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

        let memory_budget = best
            .memory_budget
            .then(|| unsafe { MemoryBudget::assume_enabled() });

        let fragment_shading_rate = best
            .fragment_shading_rate
            .map(|texel_size| FragmentShadingRate { texel_size });

        let local_read = best
            .dynamic_rendering_local_read
            .then(|| khr::dynamic_rendering_local_read::Device::new(instance, &device));

        Self {
            physical: best.physical,
            device,
            graphics_queue,
            present_queue,
            graphics_family: best.graphics_family,
            present_family: best.present_family,
            transfer_queue,
            transfer_family,
            transfer_tier,
            command_pool,
            push_descriptor,
            memory_budget,
            anisotropy: best.max_anisotropy.map(Anisotropy),
            fragment_shading_rate,
            dynamic_rendering_local_read: best.dynamic_rendering_local_read,
            local_read,
            msaa_caps,
            multi_draw_indirect: best.multi_draw_indirect,
            draw_indirect_first_instance: best.draw_indirect_first_instance,
            draw_indirect_count: best.draw_indirect_count,
            timestamp_period_ns: best.properties.limits.timestamp_period,
            timestamps_supported: best.properties.limits.timestamp_compute_and_graphics == vk::TRUE,
            max_image_array_layers: best.properties.limits.max_image_array_layers,
        }
    }

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

    if vk::api_version_major(properties.api_version) < 1
        || (vk::api_version_major(properties.api_version) == 1
            && vk::api_version_minor(properties.api_version) < 3)
    {
        return None;
    }

    let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default();
    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut vulkan_13_features)
        .push_next(&mut vulkan_12_features);
    unsafe { instance.get_physical_device_features2(physical, &mut features2) };
    let multi_draw_indirect = features2.features.multi_draw_indirect == vk::TRUE;
    let draw_indirect_first_instance = features2.features.draw_indirect_first_instance == vk::TRUE;
    let max_anisotropy = (features2.features.sampler_anisotropy == vk::TRUE)
        .then_some(properties.limits.max_sampler_anisotropy);
    // Required for GPU-driven culling; reject devices that lack it.
    let draw_indirect_count = vulkan_12_features.draw_indirect_count == vk::TRUE;
    if vulkan_13_features.dynamic_rendering != vk::TRUE
        || vulkan_13_features.synchronization2 != vk::TRUE
        || !draw_indirect_count
    {
        return None;
    }

    let extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical)
            .ok()?
    };
    let has_extension = |name: &std::ffi::CStr| {
        extensions
            .iter()
            .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
    };
    if !has_extension(khr::swapchain::NAME) || !has_extension(khr::push_descriptor::NAME) {
        return None;
    }
    // Optional: budget-aware allocation degrades gracefully when absent.
    let memory_budget = has_extension(ext::memory_budget::NAME);

    // Optional: attachment-based variable-rate shading. Requires both the
    // extension and the attachment feature; the texel size (tile a rate entry
    // covers) comes from the properties chain. Largest tile = smallest rate
    // image, which is what we want for coarse far-field shading.
    let fragment_shading_rate = has_extension(khr::fragment_shading_rate::NAME)
        .then(|| {
            let mut fsr_features = vk::PhysicalDeviceFragmentShadingRateFeaturesKHR::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut fsr_features);
            unsafe { instance.get_physical_device_features2(physical, &mut f2) };
            (fsr_features.attachment_fragment_shading_rate == vk::TRUE).then(|| {
                let mut fsr_props = vk::PhysicalDeviceFragmentShadingRatePropertiesKHR::default();
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut fsr_props);
                unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
                fsr_props.max_fragment_shading_rate_attachment_texel_size
            })
        })
        .flatten();

    // Optional: same-scope depth input-attachment reads for water absorption.
    // Requires both the extension and the feature bit; absence ⇒ interim tint.
    let dynamic_rendering_local_read = has_extension(khr::dynamic_rendering_local_read::NAME) && {
        let mut lr_features = vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut lr_features);
        unsafe { instance.get_physical_device_features2(physical, &mut f2) };
        lr_features.dynamic_rendering_local_read == vk::TRUE
    };

    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let mut graphics_family = None;
    let mut present_family = None;
    // Look for dedicated TRANSFER-only family.
    let mut transfer_family = None;
    for (index, family) in families.iter().enumerate() {
        let index = index as u32;
        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics_family.is_none() {
            graphics_family = Some(index);
        }
        if family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && transfer_family.is_none()
        {
            transfer_family = Some(index);
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
    let graphics_queue_count = graphics_family
        .map(|f| families[f as usize].queue_count)
        .unwrap_or(0);

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
        transfer_family,
        graphics_queue_count,
        properties,
        multi_draw_indirect,
        draw_indirect_first_instance,
        draw_indirect_count,
        memory_budget,
        max_anisotropy,
        fragment_shading_rate,
        dynamic_rendering_local_read,
        score,
    })
}
