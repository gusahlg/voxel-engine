use ash::vk;

pub struct SwapchainInfo {
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,

    // Needed for render pass initialization
    pub format: vk::Format,

    pub extent: vk::Extent2D,
}

pub fn acquire_swapchain(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    logical_device: &ash::Device,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    window_extent: vk::Extent2D,
    graphics_family: u32,
    present_family: u32,
) -> SwapchainInfo {
    // Query what the surface supports
    let surface_formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("Failed to get surface formats!")
    };

    let surface_format = surface_formats.iter().copied()
        .find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(surface_formats[0]);

    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
            .expect("Failed to get present modes!")
    };

    let present_mode = present_modes.iter().copied()
        .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    let capabilities = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .expect("Failed to get surface capabilities!")
    };

    // Clamp image count to the allowed range
    let mut image_count = capabilities.min_image_count + 1;
    if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
        image_count = capabilities.max_image_count;
    }

    // Resolve extent: if current_extent is u32::MAX the surface size isn't fixed (e.g. Wayland)
    // and we must clamp our desired size to the allowed range.
    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: window_extent.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: window_extent.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };

    let swapchain_loader = ash::khr::swapchain::Device::new(instance, logical_device);

    let queue_family_indices = [graphics_family, present_family];

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);

    // If graphics and present queues are from different families, images must be
    // shared between them. Otherwise exclusive access is fine (and faster).
    if graphics_family != present_family {
        swapchain_create_info = swapchain_create_info
            .image_sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&queue_family_indices);
    } else {
        swapchain_create_info = swapchain_create_info
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE);
    }

    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .expect("Failed to create swapchain!")
    };

    // Retrieve the images the swapchain created
    let images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("Failed to get swapchain images!")
    };

    // Create an image view for each swapchain image so the pipeline can use them
    let image_views: Vec<vk::ImageView> = images.iter().map(|&image| {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(surface_format.format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            logical_device.create_image_view(&view_info, None)
                .expect("Failed to create image view!")
        }
    }).collect();

    SwapchainInfo {
        swapchain_loader,
        swapchain,
        images,
        image_views,
        format: surface_format.format,
        extent,
    }
}
