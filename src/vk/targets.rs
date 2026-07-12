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

/// Cascaded-shadow-map depth format and per-cascade resolution. `D32_SFLOAT` is
/// a mandatory-supported depth-attachment + sampled format, so no capability
/// query is needed; it also gives the CSM the precision reversed-Z ortho wants.
/// 2048² per cascade, two cascades → two array layers of ONE image.
pub const SHADOW_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
pub const SHADOW_RESOLUTION: u32 = 2048;
// The shader's PCF derives texel size from the generated twin (no per-fragment
// GetDimensions); build.rs can't read this module, so the pair is pinned here.
const _: () = assert!(crate::genconst::SHADOW_RESOLUTION == SHADOW_RESOLUTION as f32);
/// Exactly two cascades (mirrors `skeleton::Cascade`), so exactly two layers.
pub const SHADOW_CASCADES: u32 = 2;

/// The cascaded shadow map: one D32 image with two array layers (one per
/// [`crate::skeleton::Cascade`]), each `SHADOW_RESOLUTION²`. `layer_views` are
/// per-cascade single-layer depth attachments the producer renders into;
/// `sample_view` is the whole 2D array the receiver samples through `sampler`
/// (a comparison sampler for hardware PCF). Persistent: its size is independent
/// of the swapchain, but it is recreated with the rest of `RenderTargets` on
/// resize so its lifetime is a single owner.
pub(crate) struct ShadowMap {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    /// 2D-array view over all cascades, bound at set 0 binding 4 for sampling.
    pub sample_view: vk::ImageView,
    /// One single-layer 2D view per cascade, used as the depth render target.
    pub layer_views: [vk::ImageView; SHADOW_CASCADES as usize],
    /// Depth-comparison sampler (reversed-Z: `GREATER_OR_EQUAL`) for PCF.
    pub sampler: vk::Sampler,
}

impl ShadowMap {
    fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        let extent = vk::Extent3D {
            width: SHADOW_RESOLUTION,
            height: SHADOW_RESOLUTION,
            depth: 1,
        };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(SHADOW_FORMAT)
            .extent(extent)
            .mip_levels(1)
            .array_layers(SHADOW_CASCADES)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            // Rendered into (occluder depth) and sampled by the receiver.
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("Failed to create shadow map image")
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
                .expect("Failed to allocate shadow map memory")
        };
        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("Failed to bind shadow map memory");
        }

        let sample_view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                        .format(SHADOW_FORMAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: SHADOW_CASCADES,
                        }),
                    None,
                )
                .expect("Failed to create shadow sample view")
        };

        let layer_views = std::array::from_fn(|i| unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(SHADOW_FORMAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: i as u32,
                            layer_count: 1,
                        }),
                    None,
                )
                .expect("Failed to create shadow layer view")
        });

        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        // Reversed-Z ortho: a receiver is lit when its depth is
                        // nearer-or-equal to the stored occluder depth.
                        .compare_enable(true)
                        .compare_op(vk::CompareOp::GREATER_OR_EQUAL),
                    None,
                )
                .expect("Failed to create shadow comparison sampler")
        };

        Self {
            image,
            memory,
            sample_view,
            layer_views,
            sampler,
        }
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            for view in &self.layer_views {
                device.destroy_image_view(*view, None);
            }
            device.destroy_image_view(self.sample_view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// One image + its backing memory + view, freed together.
pub(crate) struct ImageResources {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
}

impl ImageResources {
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Bloom mip chain (half-res HDR pyramid): compute threshold and downsample
/// passes feed the tonemap composite. Per-slot to avoid races between frames.
pub(crate) struct BloomChain {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub sample_view: vk::ImageView,
    pub mip_views: Vec<vk::ImageView>,
    pub mip_extents: Vec<vk::Extent2D>,
}

/// Maximum mip levels; pyramid reaches 1x1 or BLOOM_MAX_MIPS, whichever is shorter.
const BLOOM_MAX_MIPS: u32 = 6;

impl BloomChain {
    fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
    ) -> BloomChain {
        // Half-res base; each mip halves (rounding up) to a floor of 1 texel.
        let base = vk::Extent2D {
            width: extent.width.div_ceil(2).max(1),
            height: extent.height.div_ceil(2).max(1),
        };
        let mut mip_extents = Vec::new();
        let mut e = base;
        loop {
            mip_extents.push(e);
            if mip_extents.len() as u32 >= BLOOM_MAX_MIPS || (e.width == 1 && e.height == 1) {
                break;
            }
            e = vk::Extent2D {
                width: e.width.div_ceil(2).max(1),
                height: e.height.div_ceil(2).max(1),
            };
        }
        let levels = mip_extents.len() as u32;

        // RGBA16F is a mandatory storage-image + sampled + linear-filter format,
        // so the pyramid needs no capability query. STORAGE for the compute
        // read/write, SAMPLED for the tonemap composite.
        let image = unsafe {
            device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(HDR_COLOR_FORMAT)
                        .extent(vk::Extent3D {
                            width: base.width,
                            height: base.height,
                            depth: 1,
                        })
                        .mip_levels(levels)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        // TRANSFER_DST: when the bloom lane is off, the pass clears
                        // this to black instead of generating it, so the tonemap
                        // composite is a no-op with no shader/push-constant branch.
                        .usage(
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .expect("create bloom image")
        };
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(find_memory_type(
                            memory_props,
                            reqs.memory_type_bits,
                            vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        )),
                    None,
                )
                .expect("allocate bloom memory")
        };
        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("bind bloom memory");
        }

        let view = |base_mip: u32, count: u32| unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(HDR_COLOR_FORMAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: base_mip,
                            level_count: count,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .expect("create bloom image view")
        };
        let sample_view = view(0, levels);
        let mip_views = (0..levels).map(|m| view(m, 1)).collect();

        BloomChain {
            image,
            memory,
            sample_view,
            mip_views,
            mip_extents,
        }
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.sample_view, None);
            for v in &self.mip_views {
                device.destroy_image_view(*v, None);
            }
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
    /// The cascaded shadow map. One D32 array image, persistent but
    /// re-created with the rest of the targets on resize (its resolution is
    /// swapchain-independent, 2048² per cascade, so this only re-homes ownership).
    pub(crate) shadow: ShadowMap,
    /// Per-slot bloom mip chain. Extent-dependent, so recreated with the
    /// rest of the targets on resize.
    pub(crate) bloom: [BloomChain; FRAMES_IN_FLIGHT as usize],
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
                // SAMPLED so the VRS compute classifier can read it;
                // INPUT_ATTACHMENT so the blend pass can read it same-scope via
                // dynamic_rendering_local_read for water depth-difference
                // absorption (depth formats support input-attachment usage
                // wherever they support depth-stencil-attachment usage).
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::INPUT_ATTACHMENT,
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
                // SAMPLED: the tonemap + exposure passes read this as a texture.
                // TRANSFER_DST: the TAA resolve copies the stabilized HDR
                // back into the offscreen so exposure meters and tonemap reads it.
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::COLOR,
            )
        });

        // Variable-rate shading is opt-in: `VOXEL_VRS=1` enables it where the
        // hardware supports it. Otherwise the rate image is never allocated, so
        // `do_vrs` (which gates on `targets.vrs.is_some()`) stays false and the
        // full-rate path runs.
        let vrs = fsr
            .filter(|_| matches!(std::env::var("VOXEL_VRS").as_deref(), Ok("1")))
            .map(|f| super::vrs::Vrs::new(device, &memory_props, f, extent));

        let shadow = ShadowMap::new(device, &memory_props);

        let bloom = std::array::from_fn(|_| BloomChain::new(device, &memory_props, extent));

        Self {
            depth,
            depth_format,
            msaa,
            offscreen,
            samples,
            color_format,
            vrs,
            shadow,
            bloom,
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
            self.shadow.destroy(device);
            for chain in &self.bloom {
                chain.destroy(device);
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
