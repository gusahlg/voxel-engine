//! Shared creation and layout tracking for single-mip GPU images.

use ash::vk;

use super::alloc::find_memory_type;

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
/// distinct (layout, stage, access) triples actually used at the two sites
/// wired through `transition` (minimap upload, TAA history ping-pong) — not
/// a speculative catalogue of every Vulkan layout use.
pub(crate) enum LayoutUse {
    /// Upload target (minimap): copy destination.
    TransferDst,
    /// Sampled by the fragment shader right after a transfer write (minimap:
    /// RAW, waits on the copy's TRANSFER_WRITE).
    FragmentSampledAfterTransfer,
    /// Sampled by compute; the prior use was itself a sampled read or the
    /// image is fresh (TAA history read side: order-only, nothing to wait on
    /// since the write that produced it was already made visible on entry).
    ComputeSampledRead,
    /// Written by compute as storage; same order-only prior-use as above
    /// (TAA history write side).
    ComputeStorageWrite,
    /// Sampled by compute+fragment right after a compute storage write (TAA
    /// history publish: RAW, waits on SHADER_STORAGE_WRITE).
    SampledAfterComputeWrite,
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
            LayoutUse::ComputeSampledRead => (
                L::SHADER_READ_ONLY_OPTIMAL,
                S::COMPUTE_SHADER,
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
            LayoutUse::ComputeSampledRead | LayoutUse::ComputeStorageWrite => {
                (S::COMPUTE_SHADER | S::FRAGMENT_SHADER, A::NONE)
            }
            LayoutUse::SampledAfterComputeWrite => (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
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
    ) -> Self {
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

        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type = find_memory_type(
            memory_props,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type),
                    None,
                )
                .expect("Failed to allocate image memory")
        };
        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("Failed to bind image memory");
        }

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

        ImageResource {
            image,
            memory,
            view,
            layout: vk::ImageLayout::UNDEFINED,
            subresource,
        }
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
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
    pub(crate) fn transition_discard(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        to: LayoutUse,
    ) {
        self.barrier(device, cmd, to, true);
    }

    fn barrier(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        to: LayoutUse,
        discard: bool,
    ) {
        let (new_layout, dst_stage, dst_access) = to.dst();
        let (src_stage, src_access) = if self.layout == vk::ImageLayout::UNDEFINED {
            (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE)
        } else {
            to.src_when_used()
        };
        let barrier = [vk::ImageMemoryBarrier2::default()
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
            .subresource_range(self.subresource)];
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barrier),
            );
        }
        self.layout = new_layout;
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
            LayoutUse::ComputeSampledRead.src_when_used(),
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
                LayoutUse::ComputeSampledRead.dst().0,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            );
            assert_eq!(
                LayoutUse::ComputeStorageWrite.dst().0,
                vk::ImageLayout::GENERAL
            );
            layouts[r] = LayoutUse::ComputeSampledRead.dst().0;
            layouts[w] = LayoutUse::SampledAfterComputeWrite.dst().0;
            read_idx = w;
        }
        assert_eq!(layouts[read_idx], vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(
            layouts[1 - read_idx],
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }
}
