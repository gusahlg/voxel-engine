/// Render targets that live alongside the swapchain: the depth buffer and,
/// when MSAA is enabled, the multisampled color image that resolves into the
/// swapchain. Recreated on resize and on MSAA changes.
use ash::vk;

pub struct RenderTargets {
    pub depth_image: vk::Image,
    pub depth_memory: vk::DeviceMemory,
    pub depth_view: vk::ImageView,
    pub depth_format: vk::Format,
    /// Present only when samples > 1.
    pub msaa_image: vk::Image,
    pub msaa_memory: vk::DeviceMemory,
    pub msaa_view: vk::ImageView,
    pub samples: vk::SampleCountFlags,
    pub extent: vk::Extent2D,
}

pub fn sample_count_flag(samples: u32) -> vk::SampleCountFlags {
    match samples {
        8 => vk::SampleCountFlags::TYPE_8,
        4 => vk::SampleCountFlags::TYPE_4,
        2 => vk::SampleCountFlags::TYPE_2,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}

impl RenderTargets {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        extent: vk::Extent2D,
        color_format: vk::Format,
        samples: u32,
    ) -> Self {
        let samples = sample_count_flag(samples);
        let depth_format = pick_depth_format(instance, physical);

        let (depth_image, depth_memory, depth_view) = create_image(
            instance,
            device,
            physical,
            extent,
            depth_format,
            samples,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
        );

        let (msaa_image, msaa_memory, msaa_view) = if samples != vk::SampleCountFlags::TYPE_1 {
            create_image(
                instance,
                device,
                physical,
                extent,
                color_format,
                samples,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                vk::ImageAspectFlags::COLOR,
            )
        } else {
            (vk::Image::null(), vk::DeviceMemory::null(), vk::ImageView::null())
        };

        Self {
            depth_image,
            depth_memory,
            depth_view,
            depth_format,
            msaa_image,
            msaa_memory,
            msaa_view,
            samples,
            extent,
        }
    }

    pub fn multisampled(&self) -> bool {
        self.samples != vk::SampleCountFlags::TYPE_1
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_memory, None);
            if self.multisampled() {
                device.destroy_image_view(self.msaa_view, None);
                device.destroy_image(self.msaa_image, None);
                device.free_memory(self.msaa_memory, None);
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
    panic!("No supported depth format");
}

fn create_image(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    extent: vk::Extent2D,
    format: vk::Format,
    samples: vk::SampleCountFlags,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
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
        instance,
        physical,
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

    (image, memory, view)
}

pub fn find_memory_type(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    for i in 0..memory_properties.memory_type_count {
        let suitable = (type_filter & (1 << i)) != 0;
        let has_props = memory_properties.memory_types[i as usize]
            .property_flags
            .contains(properties);
        if suitable && has_props {
            return i;
        }
    }
    panic!("No suitable memory type");
}
