/// Render targets that live alongside the swapchain: the depth buffer, the
/// per-frame-slot offscreen color images that all rendering targets (the
/// swapchain image is only ever a copy destination at present time), and,
/// when MSAA is enabled, the multisampled color image that resolves into the
/// offscreen image. Recreated on resize and on MSAA changes.
use ash::vk;

use super::alloc::find_memory_type;
use super::buffers::FRAMES_IN_FLIGHT;

/// Linear-HDR format for the offscreen/MSAA color targets. Rendering,
/// lighting, and fog all happen here in linear space at float precision; a
/// later tonemap pass encodes to the LDR swapchain. `R16G16B16A16_SFLOAT` is a
/// mandatory-supported color-attachment + sampled + blit format in core Vulkan,
/// so this needs no capability query and no fallback.
pub const HDR_COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// One image + its backing memory + view, freed together.
pub(crate) struct ImageResources {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
}

impl ImageResources {
    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

pub struct RenderTargets {
    /// Per-slot so the VRS compute pass can sample this slot's depth from two
    /// cycles ago (fence-synchronised) while the other slot is in flight.
    pub(crate) depth: [ImageResources; FRAMES_IN_FLIGHT as usize],
    pub depth_format: vk::Format,
    /// `Some` only when multisampled; `None` is single-sampled (no MSAA image).
    pub(crate) msaa: Option<ImageResources>,
    /// Per-slot offscreen color targets (swapchain format/extent, single
    /// sampled): each frame draws — or MSAA-resolves — into `offscreen[slot]`,
    /// and presentation is a separate copy from it into a swapchain image.
    /// TRANSFER_SRC for that copy.
    pub(crate) offscreen: [ImageResources; FRAMES_IN_FLIGHT as usize],
    pub samples: vk::SampleCountFlags,
    /// The HDR format shared by `msaa` + `offscreen`; the geometry pipelines
    /// must be built with this same format. Never the swapchain format.
    pub color_format: vk::Format,
    /// `Some` when attachment VRS is active. Owns the per-slot rate images and
    /// their texel size as one consistent value — there is no way to have the
    /// images without the size or vice versa.
    pub(crate) vrs: Option<super::vrs::Vrs>,
}

impl RenderTargets {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        extent: vk::Extent2D,
        samples: super::SampleCount,
        fsr: Option<&super::device::FragmentShadingRate>,
    ) -> Self {
        let color_format = HDR_COLOR_FORMAT;
        let samples = samples.as_flags();
        let depth_format = pick_depth_format(instance, physical);
        // Queried once and shared by every render-target image below.
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        let depth = std::array::from_fn(|_| {
            create_image(
                device,
                &memory_props,
                extent,
                depth_format,
                samples,
                // SAMPLED so the VRS compute classifier can read it.
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::DEPTH,
            )
        });

        let msaa = (samples != vk::SampleCountFlags::TYPE_1).then(|| {
            create_image(
                device,
                &memory_props,
                extent,
                color_format,
                samples,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                vk::ImageAspectFlags::COLOR,
            )
        });

        let offscreen = std::array::from_fn(|_| {
            create_image(
                device,
                &memory_props,
                extent,
                color_format,
                vk::SampleCountFlags::TYPE_1,
                // SAMPLED (not TRANSFER_SRC): the tonemap pass reads this as a
                // texture; present is no longer a transfer copy.
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            )
        });

        let vrs = fsr.map(|f| super::vrs::Vrs::new(device, &memory_props, f, extent));

        Self {
            depth,
            depth_format,
            msaa,
            offscreen,
            samples,
            color_format,
            vrs,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for depth in &self.depth {
                depth.destroy(device);
            }
            if let Some(msaa) = &self.msaa {
                msaa.destroy(device);
            }
            for target in &self.offscreen {
                target.destroy(device);
            }
            if let Some(vrs) = &mut self.vrs {
                vrs.destroy(device);
            }
        }
    }
}


fn pick_depth_format(instance: &ash::Instance, physical: vk::PhysicalDevice) -> vk::Format {
    for format in [
        vk::Format::D32_SFLOAT,
        vk::Format::X8_D24_UNORM_PACK32,
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D16_UNORM,
    ] {
        let props = unsafe { instance.get_physical_device_format_properties(physical, format) };
        if props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return format;
        }
    }
    // Unreachable: Vulkan guarantees D16_UNORM (last candidate) supports
    // DEPTH_STENCIL_ATTACHMENT on every implementation.
    unreachable!("no depth format despite spec-guaranteed D16_UNORM support");
}

fn create_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    extent: vk::Extent2D,
    format: vk::Format,
    samples: vk::SampleCountFlags,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
) -> ImageResources {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(samples)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .expect("Failed to create render target image")
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(
        memory_props,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .expect("Failed to allocate render target memory")
    };
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .expect("Failed to bind render target memory");
    }

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .expect("Failed to create render target view")
    };

    ImageResources {
        image,
        memory,
        view,
    }
}
