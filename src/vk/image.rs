//! Shared creation and layout tracking for single-mip GPU images.

use std::fmt;

use ash::vk;

use super::alloc::find_memory_type;

/// GPU image memory allocation failed for a named render target.
#[derive(Debug, Clone)]
pub(crate) struct AllocError {
    size: u64,
    purpose: String,
    result: vk::Result,
}

impl AllocError {
    pub(crate) fn new(size: u64, purpose: impl Into<String>, result: vk::Result) -> Self {
        Self {
            size,
            purpose: purpose.into(),
            result,
        }
    }

    /// Allocation size in whole mebibytes (floored).
    pub(crate) fn size_mb(&self) -> u64 {
        self.size / (1024 * 1024)
    }
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({} MB): {:?}",
            self.purpose,
            self.size_mb(),
            self.result
        )
    }
}

impl std::error::Error for AllocError {}

/// Actionable message for a render-target allocation failure.
pub(crate) fn render_target_oom_message(err: &AllocError) -> String {
    format!(
        "renderer: could not allocate render targets ({} MB): {:?} — lower MSAA or render scale",
        err.size_mb(),
        err.result
    )
}

/// Human-readable target name, e.g. `HDR colour 6880x2880 8x MSAA`.
pub(crate) fn image_purpose(
    name: &str,
    extent: vk::Extent2D,
    samples: vk::SampleCountFlags,
) -> String {
    match sample_count(samples) {
        1 => format!("{name} {}x{}", extent.width, extent.height),
        n => format!("{name} {}x{} {n}x MSAA", extent.width, extent.height),
    }
}

fn sample_count(samples: vk::SampleCountFlags) -> u32 {
    if samples == vk::SampleCountFlags::TYPE_8 {
        8
    } else if samples == vk::SampleCountFlags::TYPE_4 {
        4
    } else if samples == vk::SampleCountFlags::TYPE_2 {
        2
    } else {
        1
    }
}

/// Device-local memory for `image`, named in any allocation error.
pub(crate) fn allocate_and_bind_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    image: vk::Image,
    purpose: &str,
) -> Result<vk::DeviceMemory, AllocError> {
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(
        memory_props,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    }
    .map_err(|result| AllocError::new(requirements.size, purpose, result))?;
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .expect("Failed to bind image memory");
    }
    Ok(memory)
}

/// Create `N` images, destroying any already-created on the first failure.
pub(crate) fn create_image_array<const N: usize>(
    device: &ash::Device,
    mut make: impl FnMut() -> Result<ImageResource, AllocError>,
) -> Result<[ImageResource; N], AllocError> {
    let mut acc: [Option<ImageResource>; N] = std::array::from_fn(|_| None);
    for slot in &mut acc {
        match make() {
            Ok(img) => *slot = Some(img),
            Err(err) => {
                for img in acc.iter().flatten() {
                    unsafe { img.destroy(device) };
                }
                return Err(err);
            }
        }
    }
    Ok(acc.map(|img| img.expect("image array filled")))
}

/// Description of a single-mip 2D image with one full-range view.
pub(crate) struct ImageDesc {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub layers: u32,
    pub aspect: vk::ImageAspectFlags,
    pub samples: vk::SampleCountFlags,
}

/// A destination this image is being transitioned TO. Variants are the
/// distinct (layout, stage, access) triples actually used at the sites
/// wired through `transition` (minimap upload, sky LUT, TAA history ping-pong)
/// — not a speculative catalogue of every Vulkan layout use.
pub(crate) enum LayoutUse {
    /// Upload target (minimap): copy destination.
    TransferDst,
    /// Sampled by the fragment shader right after a transfer write (minimap:
    /// RAW, waits on the copy's TRANSFER_WRITE).
    FragmentSampledAfterTransfer,
    /// Written by compute as storage; order-only prior-use (sky-cloud LUT
    /// write side).
    ComputeStorageWrite,
    /// Sampled by compute+fragment right after a compute storage write
    /// (sky-cloud LUT publish: RAW, waits on SHADER_STORAGE_WRITE).
    SampledAfterComputeWrite,
    /// Color-clear destination (`vkCmdClearColorImage`; sky-cloud LUT skip).
    TransferClear,
    /// Sampled by the fragment shader right after a color clear (LUT skip path).
    FragmentSampledAfterClear,
    /// Written as a color attachment (fused TAA history). Prior use is a
    /// fragment sampled read (last present's history) or UNDEFINED.
    ColorAttachmentWrite,
    /// Sampled by the fragment shader. Used to (a) publish a just-written TAA
    /// history (src color-attachment write) and (b) promote a fresh UNDEFINED
    /// history to SHADER_READ so the first present's unused read side is valid.
    FragmentSampled,
}

impl LayoutUse {
    fn dst(&self) -> (vk::ImageLayout, vk::PipelineStageFlags2, vk::AccessFlags2) {
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        match self {
            LayoutUse::TransferDst => (L::TRANSFER_DST_OPTIMAL, S::COPY, A::TRANSFER_WRITE),
            LayoutUse::FragmentSampledAfterTransfer => (
                L::SHADER_READ_ONLY_OPTIMAL,
                S::FRAGMENT_SHADER,
                A::SHADER_SAMPLED_READ,
            ),
            LayoutUse::ComputeStorageWrite => {
                (L::GENERAL, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE)
            }
            LayoutUse::SampledAfterComputeWrite => (
                L::SHADER_READ_ONLY_OPTIMAL,
                S::COMPUTE_SHADER | S::FRAGMENT_SHADER,
                A::SHADER_SAMPLED_READ,
            ),
            LayoutUse::TransferClear => (L::TRANSFER_DST_OPTIMAL, S::CLEAR, A::TRANSFER_WRITE),
            LayoutUse::FragmentSampledAfterClear => (
                L::SHADER_READ_ONLY_OPTIMAL,
                S::FRAGMENT_SHADER,
                A::SHADER_SAMPLED_READ,
            ),
            LayoutUse::ColorAttachmentWrite => (
                L::COLOR_ATTACHMENT_OPTIMAL,
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_WRITE,
            ),
            LayoutUse::FragmentSampled => (
                L::SHADER_READ_ONLY_OPTIMAL,
                S::FRAGMENT_SHADER,
                A::SHADER_SAMPLED_READ,
            ),
        }
    }

    /// Src stage/access when the image was already in use (not fresh from
    /// UNDEFINED) — fixed per variant because each is only ever reached from
    /// one prior state at its real call site (see the variant docs above;
    /// mirrors minimap.rs's inline match and taa.rs's `history_src`).
    fn src_when_used(&self) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        match self {
            LayoutUse::TransferDst => (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ),
            LayoutUse::FragmentSampledAfterTransfer => (S::COPY, A::TRANSFER_WRITE),
            LayoutUse::ComputeStorageWrite => (S::COMPUTE_SHADER | S::FRAGMENT_SHADER, A::NONE),
            LayoutUse::SampledAfterComputeWrite => (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
            LayoutUse::TransferClear => (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ),
            LayoutUse::FragmentSampledAfterClear => (S::CLEAR, A::TRANSFER_WRITE),
            LayoutUse::ColorAttachmentWrite => (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ),
            LayoutUse::FragmentSampled => (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
        }
    }
}

/// A single-mip image whose current layout is tracked with the resource.
pub(crate) struct ImageResource {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    layout: vk::ImageLayout,
    subresource: vk::ImageSubresourceRange,
}

impl ImageResource {
    pub(crate) fn create(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        desc: &ImageDesc,
        purpose: &str,
    ) -> Result<Self, AllocError> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(desc.format)
            .extent(vk::Extent3D {
                width: desc.extent.width,
                height: desc.extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(desc.layers)
            .samples(desc.samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("Failed to create image")
        };

        let memory = match allocate_and_bind_image(device, memory_props, image, purpose) {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { device.destroy_image(image, None) };
                return Err(err);
            }
        };

        let subresource = vk::ImageSubresourceRange {
            aspect_mask: desc.aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: desc.layers,
        };
        let view_type = if desc.layers > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(view_type)
                        .format(desc.format)
                        .subresource_range(subresource),
                    None,
                )
                .expect("Failed to create image view")
        };

        Ok(ImageResource {
            image,
            memory,
            view,
            layout: vk::ImageLayout::UNDEFINED,
            subresource,
        })
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
    }

    pub(crate) fn layout(&self) -> vk::ImageLayout {
        self.layout
    }

    pub(crate) fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.subresource
    }

    /// Barrier from `self.layout` to `to`; tracks the new layout on return.
    pub(crate) fn transition(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        to: LayoutUse,
    ) {
        self.barrier(device, cmd, to, false);
    }

    /// Declares oldLayout = UNDEFINED for targets being fully overwritten
    /// (but dependency still uses real prior state). Waits for prior reads before discarding.
    /// TAA batches via [`Self::barrier_to`]; kept for single-image discard transitions.
    #[allow(dead_code)]
    pub(crate) fn transition_discard(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        to: LayoutUse,
    ) {
        self.barrier(device, cmd, to, true);
    }

    /// Layout barrier for `to` (and the discard hint). Tracks the new layout so
    /// callers can batch several images into one `cmd_pipeline_barrier2`.
    /// `'static` because `p_next` is null — ash's lifetime only exists for the
    /// pNext chain, not the image handle.
    pub(crate) fn barrier_to(
        &mut self,
        to: LayoutUse,
        discard: bool,
    ) -> vk::ImageMemoryBarrier2<'static> {
        let (new_layout, dst_stage, dst_access) = to.dst();
        let (src_stage, src_access) = if self.layout == vk::ImageLayout::UNDEFINED {
            (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE)
        } else {
            to.src_when_used()
        };
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            // The discard hint only forces the layout half; the dependency
            // above still names the real prior use.
            .old_layout(if discard {
                vk::ImageLayout::UNDEFINED
            } else {
                self.layout
            })
            .new_layout(new_layout)
            .image(self.image)
            .subresource_range(self.subresource);
        self.layout = new_layout;
        // SAFETY: `p_next` is null; ash's lifetime only tracks the pNext chain.
        unsafe { std::mem::transmute(barrier) }
    }

    fn barrier(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        to: LayoutUse,
        discard: bool,
    ) {
        let barrier = [self.barrier_to(to, discard)];
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barrier),
            );
        }
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Barrier source reflects the actual prior use, not a pipeline assumption.
    #[test]
    fn src_when_used_matches_each_variant_real_prior_use() {
        assert_eq!(
            LayoutUse::TransferDst.src_when_used(),
            (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ
            )
        );
        assert_eq!(
            LayoutUse::FragmentSampledAfterTransfer.src_when_used(),
            (
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE
            )
        );
        assert_eq!(
            LayoutUse::ComputeStorageWrite.src_when_used(),
            (
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::NONE
            )
        );
        assert_eq!(
            LayoutUse::SampledAfterComputeWrite.src_when_used(),
            (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE
            )
        );
        assert_eq!(
            LayoutUse::TransferClear.src_when_used(),
            (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ
            )
        );
        assert_eq!(
            LayoutUse::FragmentSampledAfterClear.src_when_used(),
            (
                vk::PipelineStageFlags2::CLEAR,
                vk::AccessFlags2::TRANSFER_WRITE
            )
        );
        assert_eq!(
            LayoutUse::ColorAttachmentWrite.src_when_used(),
            (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ
            )
        );
        assert_eq!(
            LayoutUse::FragmentSampled.src_when_used(),
            (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            )
        );
    }

    /// Ping-pong states stay within covered layout transitions.
    #[test]
    fn history_dst_layouts_stay_within_the_tracked_states() {
        let mut layouts = [vk::ImageLayout::UNDEFINED; 2];
        let mut read_idx = 0usize;
        for _ in 0..8 {
            let r = read_idx;
            let w = 1 - r;
            assert_eq!(
                LayoutUse::ColorAttachmentWrite.dst().0,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            );
            layouts[r] = LayoutUse::FragmentSampled.dst().0;
            layouts[w] = LayoutUse::FragmentSampled.dst().0;
            read_idx = w;
        }
        assert_eq!(layouts[read_idx], vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(
            layouts[1 - read_idx],
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }

    #[test]
    fn image_purpose_names_extent_and_msaa() {
        let extent = vk::Extent2D {
            width: 6880,
            height: 2880,
        };
        assert_eq!(
            image_purpose("HDR colour", extent, vk::SampleCountFlags::TYPE_8),
            "HDR colour 6880x2880 8x MSAA"
        );
        assert_eq!(
            image_purpose("HDR colour", extent, vk::SampleCountFlags::TYPE_1),
            "HDR colour 6880x2880"
        );
    }

    #[test]
    fn alloc_error_formats_size_mb_and_purpose() {
        let purpose = image_purpose(
            "HDR colour",
            vk::Extent2D {
                width: 6880,
                height: 2880,
            },
            vk::SampleCountFlags::TYPE_8,
        );
        let err = AllocError::new(
            1211 * 1024 * 1024,
            purpose,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        );
        let formatted = err.to_string();
        assert!(
            formatted.contains("1211 MB"),
            "size in MB missing: {formatted}"
        );
        assert!(
            formatted.contains("HDR colour 6880x2880 8x MSAA"),
            "purpose missing: {formatted}"
        );
        assert!(
            formatted.contains("ERROR_OUT_OF_DEVICE_MEMORY"),
            "vk result missing: {formatted}"
        );

        assert_eq!(
            render_target_oom_message(&err),
            "renderer: could not allocate render targets (1211 MB): ERROR_OUT_OF_DEVICE_MEMORY — lower MSAA or render scale"
        );
    }
}
