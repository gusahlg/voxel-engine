//! Swapchain creation, format/mode selection, and recreation.

use ash::{khr, vk};

use super::device::Device;

pub struct Swapchain {
    pub loader: khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    /// Whether the swapchain images carry `TRANSFER_SRC` (screenshot copies).
    /// False on surfaces that don't support it — captures must refuse
    /// gracefully instead of recording an invalid copy.
    pub screenshot_capable: bool,
}

/// The 8-bit UNORM formats the tonemap output is correct in, in preference
/// order. The tonemap writes DISPLAY-ENCODED bytes; an `_SRGB` view would
/// re-encode them (visibly washed out), so sRGB is never silently selected.
const PREFERRED_FORMATS: [vk::Format; 3] = [
    vk::Format::B8G8R8A8_UNORM,
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::A8B8G8R8_UNORM_PACK32,
];

/// Choose the surface format: a preferred UNORM in sRGB-nonlinear colorspace,
/// then a preferred UNORM in any colorspace, then — only when the surface
/// offers no UNORM at all — its first format, flagged `false` so the caller
/// warns loudly (colors will be double-encoded on such a surface; making that
/// deliberate and visible is the portable behavior until the tonemap learns
/// to compensate).
fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> (vk::SurfaceFormatKHR, bool) {
    for want in PREFERRED_FORMATS {
        if let Some(f) = formats
            .iter()
            .copied()
            .find(|f| f.format == want && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        {
            return (f, true);
        }
    }
    for want in PREFERRED_FORMATS {
        if let Some(f) = formats.iter().copied().find(|f| f.format == want) {
            return (f, true);
        }
    }
    (*formats.first().expect("surface exposes no formats"), false)
}

/// Choose image usage from what the surface actually supports:
/// `COLOR_ATTACHMENT` is guaranteed for presentable surfaces; `TRANSFER_SRC`
/// (screenshots) is added only when offered. Returns the flags and whether
/// screenshots are possible.
fn choose_usage(supported: vk::ImageUsageFlags) -> (vk::ImageUsageFlags, bool) {
    let screenshots = supported.contains(vk::ImageUsageFlags::TRANSFER_SRC);
    let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    if screenshots {
        usage |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    (usage, screenshots)
}

/// Choose composite alpha: `OPAQUE` when offered (we render every pixel),
/// otherwise the first supported bit — never an unchecked constant.
fn choose_composite(supported: vk::CompositeAlphaFlagsKHR) -> vk::CompositeAlphaFlagsKHR {
    for candidate in [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::INHERIT,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
    ] {
        if supported.contains(candidate) {
            return candidate;
        }
    }
    // The spec requires at least one supported bit; unreachable in practice.
    vk::CompositeAlphaFlagsKHR::OPAQUE
}

/// Choose the present mode. Without vsync, prefer IMMEDIATE (MAILBOX still
/// syncs to refresh on some platforms); FIFO — the only mode the spec
/// guarantees — is the vsync path and the universal fallback.
fn choose_present_mode(vsync: bool, available: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if vsync {
        return vk::PresentModeKHR::FIFO;
    }
    [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::MAILBOX]
        .into_iter()
        .find(|m| available.contains(m))
        .unwrap_or(vk::PresentModeKHR::FIFO)
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

        let (surface_format, ideal) = choose_format(&surface_formats);
        if !ideal {
            log::warn!(
                "surface offers no UNORM format; using {:?}/{:?} — tonemap output will be \
                 re-encoded by the sRGB view on this surface",
                surface_format.format,
                surface_format.color_space
            );
        }

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(device.physical, surface)
                .expect("Failed to get present modes")
        };
        let present_mode = choose_present_mode(vsync, &present_modes);

        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(device.physical, surface)
                .expect("Failed to get surface capabilities")
        };
        let (usage, screenshot_capable) = choose_usage(capabilities.supported_usage_flags);
        if !screenshot_capable {
            log::warn!("surface does not support TRANSFER_SRC: screenshots unavailable");
        }
        let composite_alpha = choose_composite(capabilities.supported_composite_alpha);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        // If current_extent == u32::MAX (Wayland), clamp window size to supported range.
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
            .image_usage(usage)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
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
            screenshot_capable,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(format: vk::Format, color_space: vk::ColorSpaceKHR) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format,
            color_space,
        }
    }

    #[test]
    fn format_prefers_unorm_and_never_silently_picks_srgb() {
        // The common case: both flavors offered — UNORM wins.
        let (f, ideal) = choose_format(&[
            fmt(vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            fmt(
                vk::Format::B8G8R8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
        ]);
        assert_eq!(f.format, vk::Format::B8G8R8A8_UNORM);
        assert!(ideal);

        // A UNORM in an exotic colorspace still beats an sRGB format.
        let (f, ideal) = choose_format(&[
            fmt(vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            fmt(
                vk::Format::R8G8B8A8_UNORM,
                vk::ColorSpaceKHR::BT709_NONLINEAR_EXT,
            ),
        ]);
        assert_eq!(f.format, vk::Format::R8G8B8A8_UNORM);
        assert!(ideal);

        // sRGB-only surface: the fallback is taken but flagged NOT ideal, so
        // the caller warns — double-encoding is deliberate, never accidental.
        let (f, ideal) = choose_format(&[fmt(
            vk::Format::B8G8R8A8_SRGB,
            vk::ColorSpaceKHR::SRGB_NONLINEAR,
        )]);
        assert_eq!(f.format, vk::Format::B8G8R8A8_SRGB);
        assert!(!ideal);
    }

    #[test]
    fn usage_requests_transfer_src_only_when_supported() {
        let all = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
        assert_eq!(choose_usage(all), (all, true));

        let (usage, screenshots) = choose_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);
        assert_eq!(usage, vk::ImageUsageFlags::COLOR_ATTACHMENT);
        assert!(
            !screenshots,
            "no TRANSFER_SRC means screenshots must be refused"
        );
    }

    #[test]
    fn composite_alpha_falls_back_to_a_supported_bit() {
        assert_eq!(
            choose_composite(
                vk::CompositeAlphaFlagsKHR::OPAQUE | vk::CompositeAlphaFlagsKHR::INHERIT
            ),
            vk::CompositeAlphaFlagsKHR::OPAQUE
        );
        // A compositor offering only INHERIT (some Wayland stacks): honored,
        // not overridden with an unsupported OPAQUE.
        assert_eq!(
            choose_composite(vk::CompositeAlphaFlagsKHR::INHERIT),
            vk::CompositeAlphaFlagsKHR::INHERIT
        );
        assert_eq!(
            choose_composite(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED),
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
        );
    }

    #[test]
    fn present_mode_honors_vsync_and_availability() {
        let all = [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::FIFO,
        ];
        assert_eq!(choose_present_mode(true, &all), vk::PresentModeKHR::FIFO);
        assert_eq!(
            choose_present_mode(false, &all),
            vk::PresentModeKHR::IMMEDIATE
        );
        assert_eq!(
            choose_present_mode(
                false,
                &[vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::FIFO]
            ),
            vk::PresentModeKHR::MAILBOX
        );
        // FIFO-only surface: the guaranteed mode is the fallback.
        assert_eq!(
            choose_present_mode(false, &[vk::PresentModeKHR::FIFO]),
            vk::PresentModeKHR::FIFO
        );
    }
}
