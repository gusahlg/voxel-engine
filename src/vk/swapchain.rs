/// Swapchain creation and recreation.
///
/// The pipeline is decoupled from the swapchain (dynamic viewport/scissor),
/// so recreation only rebuilds the swapchain itself, its views, and the
/// per-image present semaphores owned by the renderer.
use ash::{khr, vk};

use super::device::Device;

pub struct Swapchain {
    pub loader: khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        instance: &ash::Instance,
        device: &Device,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        window_extent: vk::Extent2D,
        vsync: bool,
        old_swapchain: vk::SwapchainKHR,
    ) -> Self {
        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(device.physical, surface)
                .expect("Failed to get surface formats")
        };

        // UNORM preferred: the game was authored against raylib/GL with no
        // sRGB conversion on write, so UNORM passthrough reproduces its exact
        // colors. SRGB would double-encode.
        let surface_format = surface_formats
            .iter()
            .copied()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| {
                surface_formats.iter().copied().find(|f| {
                    f.format == vk::Format::R8G8B8A8_UNORM
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
            })
            .unwrap_or(surface_formats[0]);

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(device.physical, surface)
                .expect("Failed to get present modes")
        };
        // vsync-off prefers IMMEDIATE: on MoltenVK (and some Wayland stacks)
        // MAILBOX still syncs presentation to the display refresh, capping an
        // uncapped game at ~60-120 fps. IMMEDIATE is the only true uncap.
        let present_mode = if vsync {
            vk::PresentModeKHR::FIFO
        } else {
            [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::MAILBOX]
                .into_iter()
                .find(|m| present_modes.contains(m))
                .unwrap_or(vk::PresentModeKHR::FIFO)
        };

        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(device.physical, surface)
                .expect("Failed to get surface capabilities")
        };

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        // current_extent == u32::MAX means the surface size is driven by the
        // swapchain (e.g. Wayland); clamp our window size to the allowed range.
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

        let loader = khr::swapchain::Device::new(instance, &device.device);
        let queue_family_indices = [device.graphics_family, device.present_family];

        let mut create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            // TRANSFER_DST: presentation copies the offscreen render target
            // into the swapchain image (COLOR_ATTACHMENT — the only usage
            // guaranteed by the spec — is kept as a harmless fallback).
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        if device.graphics_family != device.present_family {
            create_info = create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices);
        } else {
            create_info = create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        }

        let swapchain = unsafe {
            loader
                .create_swapchain(&create_info, None)
                .expect("Failed to create swapchain")
        };
        log::info!(
            "swapchain: {:?} {}x{} x{} images, {:?}",
            surface_format.format,
            extent.width,
            extent.height,
            image_count,
            present_mode
        );

        let images = unsafe {
            loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images")
        };

        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    device
                        .device
                        .create_image_view(&view_info, None)
                        .expect("Failed to create swapchain image view")
                }
            })
            .collect();

        Self {
            loader,
            swapchain,
            images,
            image_views,
            format: surface_format.format,
            extent,
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
