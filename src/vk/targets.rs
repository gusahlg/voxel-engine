/// Render targets that live alongside the swapchain: the depth buffer, the
/// per-frame-slot offscreen color images that all rendering targets (the
/// swapchain image is only ever a copy destination at present time), and,
/// when MSAA is enabled, the multisampled color image that resolves into the
/// offscreen image. Recreated on resize and on MSAA changes.
use ash::vk;

use super::buffers::FRAMES_IN_FLIGHT;
use super::image::{
    AllocError, ImageDesc, ImageResource, allocate_and_bind_image, create_image_array,
    image_purpose,
};

const SLOTS: usize = FRAMES_IN_FLIGHT as usize;

/// Mandatory linear-HDR format (`R16G16B16A16_SFLOAT`) for the bloom pyramid,
/// quarter-res spill, sky-cloud LUT, and the 1×1 black bloom fallback.
/// Color-attachment + sampled + storage + blit are required of this format in
/// core Vulkan, so those images need no capability query. The offscreen/MSAA
/// color target uses [`RenderTargets::color_format`], which is this format
/// unless the experimental `VOXEL_HDR_11BIT=1` switch selects packed 11-bit.
pub const HDR_COLOR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Packed RGB-only 11-bit unsigned float. Experimental HDR offscreen/MSAA
/// format behind `VOXEL_HDR_11BIT=1`; no alpha channel.
pub(crate) const HDR_11BIT_FORMAT: vk::Format = vk::Format::B10G11R11_UFLOAT_PACK32;

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

/// Quarter-res spill extent (1/`SPILL_FACTOR` of the render extent, floored to 1).
pub(crate) fn spill_extent(render: vk::Extent2D) -> vk::Extent2D {
    let f = crate::genconst::SPILL_FACTOR;
    vk::Extent2D {
        width: render.width.div_ceil(f).max(1),
        height: render.height.div_ceil(f).max(1),
    }
}

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
    ) -> Result<Self, AllocError> {
        let extent = vk::Extent3D {
            width: SHADOW_RESOLUTION,
            height: SHADOW_RESOLUTION,
            depth: 1,
        };
        let purpose = image_purpose(
            "shadow map",
            vk::Extent2D {
                width: SHADOW_RESOLUTION,
                height: SHADOW_RESOLUTION,
            },
            vk::SampleCountFlags::TYPE_1,
        );
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

        let memory = match allocate_and_bind_image(device, memory_props, image, &purpose) {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { device.destroy_image(image, None) };
                return Err(err);
            }
        };

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

        Ok(Self {
            image,
            memory,
            sample_view,
            layer_views,
            sampler,
        })
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

/// Bloom mip chain (half-res HDR pyramid): compute threshold and downsample
/// passes feed the quarter-res spill composite. Per-slot to avoid races between frames.
pub(crate) struct BloomChain {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub sample_view: vk::ImageView,
    pub mip_views: Vec<vk::ImageView>,
    pub mip_extents: Vec<vk::Extent2D>,
    /// True once a bloom-off clear has left this pyramid black in
    /// `SHADER_READ_ONLY`. Avoids re-clearing every frame.
    pub cleared: bool,
}

/// The spill pass samples only `BLOOM_SPIRAL_LOD`, so the pyramid stops there.
const BLOOM_MAX_MIPS: u32 = crate::genconst::BLOOM_MAX_MIPS;
const _: () = assert!(BLOOM_MAX_MIPS == crate::genconst::BLOOM_SPIRAL_LOD as u32 + 1);

impl BloomChain {
    fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
    ) -> Result<BloomChain, AllocError> {
        // Half-res base; each mip halves (rounding up) to a floor of 1 texel.
        let base = vk::Extent2D {
            width: extent.width.div_ceil(2).max(1),
            height: extent.height.div_ceil(2).max(1),
        };
        let purpose = image_purpose("bloom pyramid", base, vk::SampleCountFlags::TYPE_1);
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
        // read/write, SAMPLED for the spill-pass composite.
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
                        // this to black instead of generating it, so the spill
                        // bloom term is a no-op with no extra descriptor branch.
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
        let memory = match allocate_and_bind_image(device, memory_props, image, &purpose) {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { device.destroy_image(image, None) };
                return Err(err);
            }
        };

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

        Ok(BloomChain {
            image,
            memory,
            sample_view,
            mip_views,
            mip_extents,
            cleared: false,
        })
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
    pub(crate) depth: [ImageResource; FRAMES_IN_FLIGHT as usize],
    /// Per-slot single-sample MSAA depth resolve target; `Some` only when
    /// multisampled. The MS `depth` can't feed a `Sampler2D`, so the geometry
    /// pass resolves (SAMPLE_ZERO) into this and VRS/godrays/present TAA sample it.
    pub(crate) resolved_depth: [Option<ImageResource>; FRAMES_IN_FLIGHT as usize],
    pub depth_format: vk::Format,
    /// `Some` only when multisampled; `None` is single-sampled (no MSAA image).
    pub(crate) msaa: Option<ImageResource>,
    /// Per-slot offscreen color targets (swapchain format/extent, single
    /// sampled): each frame draws — or MSAA-resolves — into `offscreen[slot]`,
    /// and presentation is a separate copy from it into a swapchain image.
    /// TRANSFER_SRC for that copy.
    pub(crate) offscreen: [ImageResource; FRAMES_IN_FLIGHT as usize],
    pub samples: vk::SampleCountFlags,
    /// The HDR format shared by `msaa` + `offscreen`; the geometry pipelines
    /// must be built with this same format. Never the swapchain format. Default
    /// [`HDR_COLOR_FORMAT`]; packed 11-bit when `VOXEL_HDR_11BIT=1` is accepted.
    pub color_format: vk::Format,
    /// `Some` when the device supports attachment VRS. Owns the per-slot rate
    /// images, history, and mix readback. `RenderFlags::vrs` decides whether
    /// a frame actually classifies and binds the rate attachment.
    pub(crate) vrs: Option<super::vrs::Vrs>,
    /// Shared cascaded shadow map (both FIF slots sample the same image).
    /// Regenerated once per `ShadowKey`; see `shadow.rs` hazard analysis.
    pub(crate) shadow: ShadowMap,
    /// Per-slot bloom mip chain. Extent-dependent, so recreated with the
    /// rest of the targets on resize.
    pub(crate) bloom: [BloomChain; FRAMES_IN_FLIGHT as usize],
    /// Per-slot quarter-res RGBA16F spill (bloom composite + godrays). Written
    /// by compute on presented frames, sampled by the tonemap fragment. Recreated
    /// with the targets; new images begin UNDEFINED. One image per FIF slot so
    /// the in-flight present copy of the other slot can still sample its own.
    pub(crate) spill: [ImageResource; FRAMES_IN_FLIGHT as usize],
    /// Per-slot octahedral cloud LUT (RGBA16F). Size is a genconst, independent
    /// of the swapchain; still owned here so resize tears it down with everything else.
    pub(crate) sky_cloud: [ImageResource; FRAMES_IN_FLIGHT as usize],
}

/// In-progress `RenderTargets::new`. Drop destroys anything already created
/// unless [`TargetBuild::finish`] disarms it.
struct TargetBuild<'a> {
    device: &'a ash::Device,
    depth: [Option<ImageResource>; SLOTS],
    resolved_depth: [Option<ImageResource>; SLOTS],
    msaa: Option<ImageResource>,
    offscreen: [Option<ImageResource>; SLOTS],
    vrs: Option<super::vrs::Vrs>,
    shadow: Option<ShadowMap>,
    bloom: [Option<BloomChain>; SLOTS],
    spill: [Option<ImageResource>; SLOTS],
    sky_cloud: [Option<ImageResource>; SLOTS],
    live: bool,
}

impl Drop for TargetBuild<'_> {
    fn drop(&mut self) {
        if !self.live {
            return;
        }
        let device = self.device;
        unsafe {
            for img in self.depth.iter().flatten() {
                img.destroy(device);
            }
            for img in self.resolved_depth.iter().flatten() {
                img.destroy(device);
            }
            if let Some(msaa) = &self.msaa {
                msaa.destroy(device);
            }
            for img in self.offscreen.iter().flatten() {
                img.destroy(device);
            }
            if let Some(vrs) = &mut self.vrs {
                vrs.destroy(device);
            }
            if let Some(shadow) = &self.shadow {
                shadow.destroy(device);
            }
            for chain in self.bloom.iter().flatten() {
                chain.destroy(device);
            }
            for img in self.spill.iter().flatten() {
                img.destroy(device);
            }
            for img in self.sky_cloud.iter().flatten() {
                img.destroy(device);
            }
        }
    }
}

impl TargetBuild<'_> {
    fn finish(
        mut self,
        depth_format: vk::Format,
        samples: vk::SampleCountFlags,
        color_format: vk::Format,
    ) -> RenderTargets {
        self.live = false;
        RenderTargets {
            depth: take_filled(&mut self.depth),
            resolved_depth: self.resolved_depth.each_mut().map(Option::take),
            depth_format,
            msaa: self.msaa.take(),
            offscreen: take_filled(&mut self.offscreen),
            samples,
            color_format,
            vrs: self.vrs.take(),
            shadow: self.shadow.take().expect("shadow map"),
            bloom: take_filled(&mut self.bloom),
            spill: take_filled(&mut self.spill),
            sky_cloud: take_filled(&mut self.sky_cloud),
        }
    }
}

fn take_filled<T, const N: usize>(slots: &mut [Option<T>; N]) -> [T; N] {
    std::array::from_fn(|i| slots[i].take().expect("slot filled"))
}

impl RenderTargets {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        extent: vk::Extent2D,
        samples: super::SampleCount,
        fsr: Option<&super::device::FragmentShadingRate>,
    ) -> Result<Self, AllocError> {
        let color_format = pick_hdr_color_format(instance, physical);
        let samples = samples.as_flags();
        let depth_format = pick_depth_format(instance, physical);
        // Queried once and shared by every render-target image below.
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        let mut build = TargetBuild {
            device,
            depth: std::array::from_fn(|_| None),
            resolved_depth: std::array::from_fn(|_| None),
            msaa: None,
            offscreen: std::array::from_fn(|_| None),
            vrs: None,
            shadow: None,
            bloom: std::array::from_fn(|_| None),
            spill: std::array::from_fn(|_| None),
            sky_cloud: std::array::from_fn(|_| None),
            live: true,
        };

        let depth_purpose = image_purpose("depth", extent, samples);
        build.depth = create_image_array(device, || {
            ImageResource::create(
                device,
                &memory_props,
                &ImageDesc {
                    extent,
                    format: depth_format,
                    // Sampled by VRS + input attachment for water absorption.
                    usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                    layers: 1,
                    aspect: vk::ImageAspectFlags::DEPTH,
                    samples,
                },
                &depth_purpose,
            )
        })?
        .map(Some);

        if samples != vk::SampleCountFlags::TYPE_1 {
            let resolved_purpose =
                image_purpose("resolved depth", extent, vk::SampleCountFlags::TYPE_1);
            build.resolved_depth = create_image_array(device, || {
                ImageResource::create(
                    device,
                    &memory_props,
                    &ImageDesc {
                        extent,
                        format: depth_format,
                        // MSAA resolve target for post-pass sampling.
                        usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                            | vk::ImageUsageFlags::SAMPLED,
                        layers: 1,
                        aspect: vk::ImageAspectFlags::DEPTH,
                        samples: vk::SampleCountFlags::TYPE_1,
                    },
                    &resolved_purpose,
                )
            })?
            .map(Some);
            build.msaa = Some(ImageResource::create(
                device,
                &memory_props,
                &ImageDesc {
                    extent,
                    format: color_format,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                    layers: 1,
                    aspect: vk::ImageAspectFlags::COLOR,
                    samples,
                },
                &image_purpose("HDR colour", extent, samples),
            )?);
        }

        let offscreen_purpose = image_purpose("HDR colour", extent, vk::SampleCountFlags::TYPE_1);
        build.offscreen = create_image_array(device, || {
            ImageResource::create(
                device,
                &memory_props,
                &ImageDesc {
                    extent,
                    format: color_format,
                    // Sampled by tonemap (and bloom/exposure). TAA history is a
                    // separate swapchain-sized image.
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST,
                    layers: 1,
                    aspect: vk::ImageAspectFlags::COLOR,
                    samples: vk::SampleCountFlags::TYPE_1,
                },
                &offscreen_purpose,
            )
        })?
        .map(Some);

        // Rate images exist whenever the device supports attachment FSR.
        // `RenderFlags::vrs` (default off) is the runtime switch: off skips the
        // classify dispatch and the rate attachment, shading 1×1 everywhere.
        if let Some(f) = fsr {
            build.vrs = Some(super::vrs::Vrs::new(device, &memory_props, f, extent)?);
        }

        build.shadow = Some(ShadowMap::new(device, &memory_props)?);

        for chain in &mut build.bloom {
            *chain = Some(BloomChain::new(device, &memory_props, extent)?);
        }
        let spill_extent = spill_extent(extent);
        let spill_purpose = image_purpose("spill", spill_extent, vk::SampleCountFlags::TYPE_1);
        build.spill = create_image_array(device, || {
            ImageResource::create(
                device,
                &memory_props,
                &ImageDesc {
                    extent: spill_extent,
                    format: HDR_COLOR_FORMAT,
                    // STORAGE: spill compute write. SAMPLED: tonemap bilinear tap.
                    usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                    layers: 1,
                    aspect: vk::ImageAspectFlags::COLOR,
                    samples: vk::SampleCountFlags::TYPE_1,
                },
                &spill_purpose,
            )
        })?
        .map(Some);
        let lut = crate::genconst::SKY_CLOUD_LUT_SIZE;
        let lut_extent = vk::Extent2D {
            width: lut,
            height: lut,
        };
        let lut_purpose = image_purpose("sky cloud LUT", lut_extent, vk::SampleCountFlags::TYPE_1);
        build.sky_cloud = create_image_array(device, || {
            ImageResource::create(
                device,
                &memory_props,
                &ImageDesc {
                    extent: lut_extent,
                    format: HDR_COLOR_FORMAT,
                    usage: vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST,
                    layers: 1,
                    aspect: vk::ImageAspectFlags::COLOR,
                    samples: vk::SampleCountFlags::TYPE_1,
                },
                &lut_purpose,
            )
        })?
        .map(Some);

        Ok(build.finish(depth_format, samples, color_format))
    }

    /// The single-sample depth VRS/spill-godrays/present-TAA sample: the MSAA resolve target
    /// when multisampled, else the (already single-sample) `depth`. After the
    /// scene pass this image rests in [`super::SAMPLEABLE_DEPTH_REST_LAYOUT`];
    /// during the pass its write scope is [`super::sampleable_depth_attachment_state`].
    pub(crate) fn sampleable_depth(&self, slot: usize) -> &ImageResource {
        self.resolved_depth[slot]
            .as_ref()
            .unwrap_or(&self.depth[slot])
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for depth in &self.depth {
                depth.destroy(device);
            }
            for resolved in self.resolved_depth.iter().flatten() {
                resolved.destroy(device);
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
            for spill in &self.spill {
                spill.destroy(device);
            }
            for lut in &self.sky_cloud {
                lut.destroy(device);
            }
        }
    }
}

/// Optimal-tiling features the scene depth image actually uses.
///
/// Geometry writes it as a depth attachment (and MSAA `SAMPLE_ZERO` resolve
/// still needs `DEPTH_STENCIL_ATTACHMENT`). TAA reprojection, the quarter-res
/// spill/godray pass, and VRS classify sample it (`SAMPLED` usage;
/// [`super::SAMPLEABLE_DEPTH_REST_LAYOUT`]). Water's depth input attachment is
/// covered by `DEPTH_STENCIL_ATTACHMENT`. There is no transfer or blit of
/// scene depth.
/// Experimental `VOXEL_HDR_11BIT=1` switch. Read once at renderer creation
/// (first [`RenderTargets::new`]); not a public API. Any value other than
/// `"0"` enables the packed 11-bit offscreen attempt.
fn hdr_11bit_requested() -> bool {
    static REQUESTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REQUESTED.get_or_init(|| std::env::var("VOXEL_HDR_11BIT").is_ok_and(|v| v != "0"))
}

/// Optimal-tiling features the HDR offscreen (and MSAA color, when present)
/// actually use: rendered into, blended (transparent/water/debug/HUD), and
/// sampled with a linear filter (tonemap, bloom threshold, exposure).
///
/// Bloom/spill/VRS storage images are separate RGBA16F (or R8 rate) targets —
/// no pass binds the HDR offscreen as a storage image, so `STORAGE_IMAGE` is
/// not required (and would reject 11-bit on many devices).
fn hdr_offscreen_features() -> vk::FormatFeatureFlags {
    vk::FormatFeatureFlags::COLOR_ATTACHMENT
        | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
        | vk::FormatFeatureFlags::SAMPLED_IMAGE
        | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
}

/// Choose the HDR offscreen format. `Ok` is the format to use; `Err` is the
/// 11-bit bits that were missing (caller logs and falls back).
fn choose_hdr_color_format(
    want_11bit: bool,
    eleven_features: vk::FormatFeatureFlags,
) -> Result<vk::Format, vk::FormatFeatureFlags> {
    if !want_11bit {
        return Ok(HDR_COLOR_FORMAT);
    }
    let required = hdr_offscreen_features();
    if eleven_features.contains(required) {
        Ok(HDR_11BIT_FORMAT)
    } else {
        Err(required & !eleven_features)
    }
}

/// Experimental packed 11-bit HDR offscreen (`VOXEL_HDR_11BIT=1`). Not a public
/// API. Falls back to [`HDR_COLOR_FORMAT`] when the device is missing a
/// required optimal-tiling feature.
fn pick_hdr_color_format(instance: &ash::Instance, physical: vk::PhysicalDevice) -> vk::Format {
    if !hdr_11bit_requested() {
        return HDR_COLOR_FORMAT;
    }
    let props =
        unsafe { instance.get_physical_device_format_properties(physical, HDR_11BIT_FORMAT) };
    match choose_hdr_color_format(true, props.optimal_tiling_features) {
        Ok(format) => format,
        Err(missing) => {
            log::info!(
                "VOXEL_HDR_11BIT: {HDR_11BIT_FORMAT:?} missing {missing:?}; using {HDR_COLOR_FORMAT:?}"
            );
            HDR_COLOR_FORMAT
        }
    }
}

/// Whether a color attachment format has an alpha channel. Packed 11-bit HDR
/// is RGB-only; blend write masks must not include `A`.
pub(crate) fn color_format_has_alpha(format: vk::Format) -> bool {
    format != HDR_11BIT_FORMAT
}

fn depth_format_features() -> vk::FormatFeatureFlags {
    vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE
}

/// Candidate order: D32 first (reversed-Z precision), then packed D24, then
/// the spec-guaranteed D16 fallback.
const DEPTH_FORMAT_CANDIDATES: [vk::Format; 4] = [
    vk::Format::D32_SFLOAT,
    vk::Format::X8_D24_UNORM_PACK32,
    vk::Format::D24_UNORM_S8_UINT,
    vk::Format::D16_UNORM,
];

/// First candidate whose features contain [`depth_format_features`].
/// `Err` is the bits the last candidate was missing (D16 is last).
fn first_depth_format(
    candidates: impl IntoIterator<Item = (vk::Format, vk::FormatFeatureFlags)>,
) -> Result<vk::Format, vk::FormatFeatureFlags> {
    let required = depth_format_features();
    let mut missing = required;
    for (format, features) in candidates {
        if features.contains(required) {
            return Ok(format);
        }
        missing = required & !features;
    }
    Err(missing)
}

/// Pick a depth format the engine can render into **and** sample.
///
/// Vulkan requires `DEPTH_STENCIL_ATTACHMENT` for `D16_UNORM` and for (at
/// least one of) packed D24 / `D32_SFLOAT`, but `SAMPLED_IMAGE` is **not**
/// mandatory for `X8_D24_UNORM_PACK32` or `D24_UNORM_S8_UINT`. The depth
/// image is sampled (TAA, spill, VRS; see [`depth_format_features`]), so the
/// picker requires that full set. `D16_UNORM` is spec-guaranteed to provide
/// both bits, so falling through the list is unreachable; a driver that still
/// fails it panics naming the missing feature.
fn pick_depth_format(instance: &ash::Instance, physical: vk::PhysicalDevice) -> vk::Format {
    let queried = DEPTH_FORMAT_CANDIDATES.map(|format| {
        let props = unsafe { instance.get_physical_device_format_properties(physical, format) };
        (format, props.optimal_tiling_features)
    });
    match first_depth_format(queried) {
        Ok(format) => format,
        Err(missing) if missing.is_empty() => {
            // D16 reported the required set, so the loop must have returned it.
            unreachable!(
                "D16_UNORM reported {:?} but was not selected",
                depth_format_features()
            )
        }
        Err(missing) => panic!(
            "no depth format with {:?}; D16_UNORM is missing {missing:?}",
            depth_format_features()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_picker_requires_sampled_not_just_attachment() {
        let att = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
        let both = att | vk::FormatFeatureFlags::SAMPLED_IMAGE;
        assert_eq!(
            first_depth_format([(vk::Format::D32_SFLOAT, att)]),
            Err(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        );
        assert_eq!(
            first_depth_format([(vk::Format::D32_SFLOAT, both)]),
            Ok(vk::Format::D32_SFLOAT)
        );
    }

    #[test]
    fn depth_picker_keeps_d32_first() {
        let both = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::FormatFeatureFlags::SAMPLED_IMAGE;
        let att = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
        // D32 lacks sampled, later D16 has both → D16.
        assert_eq!(
            first_depth_format([
                (vk::Format::D32_SFLOAT, att),
                (vk::Format::X8_D24_UNORM_PACK32, att),
                (vk::Format::D24_UNORM_S8_UINT, att),
                (vk::Format::D16_UNORM, both),
            ]),
            Ok(vk::Format::D16_UNORM)
        );
        // Every candidate has the set → D32 wins (candidate order).
        assert_eq!(
            first_depth_format(DEPTH_FORMAT_CANDIDATES.map(|f| (f, both))),
            Ok(vk::Format::D32_SFLOAT)
        );
    }

    #[test]
    fn depth_picker_names_missing_sampled_on_d16() {
        let att = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
        assert_eq!(
            first_depth_format(DEPTH_FORMAT_CANDIDATES.map(|f| (f, att))),
            Err(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        );
    }

    #[test]
    fn hdr_11bit_stays_rgba16f_when_not_requested() {
        let none = vk::FormatFeatureFlags::empty();
        let all = hdr_offscreen_features();
        assert_eq!(choose_hdr_color_format(false, all), Ok(HDR_COLOR_FORMAT));
        assert_eq!(choose_hdr_color_format(false, none), Ok(HDR_COLOR_FORMAT));
    }

    #[test]
    fn hdr_11bit_requires_blend_and_sampled() {
        let all = hdr_offscreen_features();
        assert_eq!(choose_hdr_color_format(true, all), Ok(HDR_11BIT_FORMAT));
        let no_blend = all & !vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;
        assert_eq!(
            choose_hdr_color_format(true, no_blend),
            Err(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
        );
        let no_sampled = all & !vk::FormatFeatureFlags::SAMPLED_IMAGE;
        assert_eq!(
            choose_hdr_color_format(true, no_sampled),
            Err(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        );
        // Storage is not required of the offscreen (bloom/spill/VRS storage
        // images are separate).
        let no_storage = all & !vk::FormatFeatureFlags::STORAGE_IMAGE;
        assert_eq!(
            choose_hdr_color_format(true, no_storage),
            Ok(HDR_11BIT_FORMAT)
        );
    }

    #[test]
    fn packed_11bit_has_no_alpha() {
        assert!(!color_format_has_alpha(HDR_11BIT_FORMAT));
        assert!(color_format_has_alpha(HDR_COLOR_FORMAT));
        assert!(color_format_has_alpha(vk::Format::B8G8R8A8_UNORM));
    }
}
