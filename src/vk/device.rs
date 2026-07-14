/// Physical device selection and logical device creation.
///
/// Selection requires Vulkan 1.3 (dynamic rendering + synchronization2), the
/// swapchain extension, and graphics+present queues; among suitable devices a
/// discrete GPU wins over an integrated one.
use ash::{ext, khr, vk};

/// Marker returned only when `VK_EXT_memory_budget` is enabled, so budget
/// queries can't be issued without it.
#[derive(Clone, Copy)]
pub struct MemoryBudget {
    _priv: (),
}

/// A snapshot of per-heap budget/usage from `VK_EXT_memory_budget`. Values are
/// driver best-effort and stale the moment they are read — treat them as an
/// admission hint, never a hard cap.
#[derive(Clone, Copy)]
pub struct BudgetSnapshot {
    pub heap_budget: [u64; vk::MAX_MEMORY_HEAPS],
    pub heap_usage: [u64; vk::MAX_MEMORY_HEAPS],
}

impl MemoryBudget {
    /// Called only in `Device::new`, once the extension name is in the enabled
    /// device-extension list.
    unsafe fn assume_enabled() -> Self {
        Self { _priv: () }
    }

    /// Query live per-heap budget/usage (requires token, only safe with extension).
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

/// Present only when `samplerAnisotropy` was enabled at device creation;
/// carries the device's max ratio so no caller can over-request. Minted only
/// by `Device::new`, so holding one proves the feature is on.
#[derive(Clone, Copy)]
pub struct Anisotropy(f32);

impl Anisotropy {
    /// Clamp a desired ratio into the device-supported range [1.0, max].
    pub fn clamp(self, desired: f32) -> f32 {
        desired.clamp(1.0, self.0)
    }
}

/// Present only when `VK_KHR_fragment_shading_rate` is enabled with
/// `attachmentFragmentShadingRate`. Attachment VRS drives shading through the
/// core dynamic-rendering pNext chain, so only the texel size (the tile one
/// rate-image entry covers) is needed. Minted only by `Device::new`, so holding
/// one proves the feature is on.
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
    pub command_pool: vk::CommandPool,
    /// Loader for `VK_KHR_push_descriptor` (a hard requirement): lets the
    /// renderer push single-binding sets at record time instead of owning
    /// pools/sets and running `vkUpdateDescriptorSets`.
    pub push_descriptor: khr::push_descriptor::Device,
    /// `Some` when `VK_EXT_memory_budget` is enabled, gating budget queries.
    pub memory_budget: Option<MemoryBudget>,
    /// `Some` when `samplerAnisotropy` is enabled; carries the max ratio.
    pub anisotropy: Option<Anisotropy>,
    /// `Some` when attachment-based variable-rate shading is available.
    pub fragment_shading_rate: Option<FragmentShadingRate>,
    /// `VK_KHR_dynamic_rendering_local_read` enabled: lets the blend pass read
    /// the same-scope depth attachment as an input attachment (true water
    /// depth-difference absorption). Optional — absent ⇒ the interim tint path.
    pub dynamic_rendering_local_read: bool,
    /// The extension's command table (`vkCmdSetRenderingInputAttachmentIndices`),
    /// present exactly when `dynamic_rendering_local_read` is.
    pub local_read: Option<khr::dynamic_rendering_local_read::Device>,
    /// Sample counts supported by BOTH color and depth framebuffer attachments.
    pub msaa_caps: vk::SampleCountFlags,
    /// `multiDrawIndirect` enabled; otherwise renderer loops single-draw calls.
    pub multi_draw_indirect: bool,
    /// `drawIndirectFirstInstance` enabled; otherwise uses direct draw calls.
    pub draw_indirect_first_instance: bool,
    /// Nanoseconds per timestamp-query tick (`limits.timestampPeriod`).
    pub timestamp_period_ns: f32,
    /// Whether all graphics/compute queues support timestamp queries.
    pub timestamps_supported: bool,
    /// `limits.maxImageArrayLayers` — the block-texture array's layer ceiling
    /// (spec minimum 256, commonly 2048 on desktop).
    pub max_image_array_layers: u32,
}

struct Candidate {
    physical: vk::PhysicalDevice,
    graphics_family: u32,
    present_family: u32,
    properties: vk::PhysicalDeviceProperties,
    multi_draw_indirect: bool,
    draw_indirect_first_instance: bool,
    memory_budget: bool,
    /// `Some(max_ratio)` when `samplerAnisotropy` is supported; optional, so
    /// absence never disqualifies a device.
    max_anisotropy: Option<f32>,
    /// `Some(texel_size)` when attachment VRS is supported; the tile size one
    /// rate-image texel covers. Optional, so absence never disqualifies.
    fragment_shading_rate: Option<vk::Extent2D>,
    /// `true` when both the extension and the `dynamicRenderingLocalRead` feature
    /// bit are present. Optional, so absence never disqualifies a device.
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
            .expect("No suitable Vulkan 1.3 GPU found (needs dynamic rendering + synchronization2 + swapchain)");

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
        // Only request the extension when the device advertises it; requesting
        // an unsupported extension makes create_device fail outright.
        if best.memory_budget {
            device_extensions.push(ext::memory_budget::NAME.as_ptr());
        }
        if best.fragment_shading_rate.is_some() {
            device_extensions.push(khr::fragment_shading_rate::NAME.as_ptr());
        }
        if best.dynamic_rendering_local_read {
            device_extensions.push(khr::dynamic_rendering_local_read::NAME.as_ptr());
        }

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
        let mut vulkan_12_features =
            vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        // The vertex-less sky triangle reads SV_VertexID, which Slang lowers via
        // the DrawParameters capability — core in Vulkan 1.1, but must be opted in.
        let mut vulkan_11_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        // Batched indirect draws want both; each is optional with a fallback
        // draw path in the renderer, so enable exactly what the device has.
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

        // SAFETY: the extension is in `device_extensions` above exactly when
        // `best.memory_budget` is true, so the token matches the enabled state.
        let memory_budget = best
            .memory_budget
            .then(|| unsafe { MemoryBudget::assume_enabled() });

        // SAFETY: the extension + feature are enabled above exactly when
        // `best.fragment_shading_rate` is `Some`, so the loader is valid to use.
        let fragment_shading_rate = best
            .fragment_shading_rate
            .map(|texel_size| FragmentShadingRate { texel_size });

        // Built before the struct literal moves `device` into place.
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
            timestamp_period_ns: best.properties.limits.timestamp_period,
            timestamps_supported: best.properties.limits.timestamp_compute_and_graphics == vk::TRUE,
            max_image_array_layers: best.properties.limits.max_image_array_layers,
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
    let multi_draw_indirect = features2.features.multi_draw_indirect == vk::TRUE;
    let draw_indirect_first_instance = features2.features.draw_indirect_first_instance == vk::TRUE;
    // Optional: anisotropic filtering; absence just falls back to plain mips.
    let max_anisotropy = (features2.features.sampler_anisotropy == vk::TRUE)
        .then_some(properties.limits.max_sampler_anisotropy);
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
                let mut fsr_props =
                    vk::PhysicalDeviceFragmentShadingRatePropertiesKHR::default();
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut fsr_props);
                unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
                fsr_props.max_fragment_shading_rate_attachment_texel_size
            })
        })
        .flatten();

    // Optional: same-scope depth input-attachment reads for water absorption.
    // Requires both the extension and the feature bit; absence ⇒ interim tint.
    let dynamic_rendering_local_read = has_extension(khr::dynamic_rendering_local_read::NAME) && {
        let mut lr_features =
            vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut lr_features);
        unsafe { instance.get_physical_device_features2(physical, &mut f2) };
        lr_features.dynamic_rendering_local_read == vk::TRUE
    };

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
        multi_draw_indirect,
        draw_indirect_first_instance,
        memory_budget,
        max_anisotropy,
        fragment_shading_rate,
        dynamic_rendering_local_read,
        score,
    })
}
