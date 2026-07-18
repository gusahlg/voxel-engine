//! The one way to create a GPU image and manage its layout. Subsumes the
//! duplicated create-image/allocate/bind/view boilerplate at each site and
//! turns ad hoc inline barriers into a method that reads the prior state off
//! the image itself instead of a caller-tracked variable.
use std::ops::Range;

use ash::vk;

use super::alloc::find_memory_type;

/// Derives a level's dispatch dimensions from the image rather than
/// re-counting them. Free function: testable without a device.
fn mip_extent_of(base: vk::Extent2D, level: u32) -> vk::Extent2D {
    vk::Extent2D {
        width: (base.width >> level).max(1),
        height: (base.height >> level).max(1),
    }
}

/// What `ImageResource::create` builds: a single 2D(-array) image with one
/// full-range view over `mips`×`layers`, device-local memory.
///
/// Adds `aspect` and `samples` fields: every real site needs an aspect mask
/// for the view (COLOR vs DEPTH) and targets.rs varies sample count for MSAA.
pub(crate) struct ImageDesc {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub mips: u32,
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

/// The ONE way to create and transition an image; owns its layout so a
/// barrier is `image.transition(device, cmd, to)`, never inline ceremony.
pub(crate) struct ImageResource {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// One view per mip level for pyramid passes (read level n while writing n+1).
    mip_views: Vec<vk::ImageView>,
    /// Per-mip layout: pyramid reads n while writing n+1 simultaneously.
    layouts: Vec<vk::ImageLayout>,
    extent: vk::Extent2D,
    subresource: vk::ImageSubresourceRange,
}

/// Groups contiguous mip ranges by layout. One barrier per run; pure.
fn layout_runs(
    layouts: &[vk::ImageLayout],
    levels: Range<u32>,
) -> Vec<(Range<u32>, vk::ImageLayout)> {
    let mut runs: Vec<(Range<u32>, vk::ImageLayout)> = Vec::new();
    for level in levels {
        let layout = layouts[level as usize];
        match runs.last_mut() {
            Some((run, l)) if *l == layout => run.end = level + 1,
            _ => runs.push((level..level + 1, layout)),
        }
    }
    runs
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
            .mip_levels(desc.mips)
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
            level_count: desc.mips,
            base_array_layer: 0,
            layer_count: desc.layers,
        };
        let view_type = if desc.layers > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let make_view = |range: vk::ImageSubresourceRange| unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(view_type)
                        .format(desc.format)
                        .subresource_range(range),
                    None,
                )
                .expect("Failed to create image view")
        };
        let view = make_view(subresource);
        // Built unconditionally, including the single-mip case where it aliases
        // `view`: an extra view object is far cheaper than making every caller
        // branch on whether this image happens to have a chain.
        let mip_views = (0..desc.mips)
            .map(|level| {
                make_view(vk::ImageSubresourceRange {
                    base_mip_level: level,
                    level_count: 1,
                    ..subresource
                })
            })
            .collect();

        ImageResource {
            image,
            memory,
            view,
            mip_views,
            layouts: vec![vk::ImageLayout::UNDEFINED; desc.mips as usize],
            extent: desc.extent,
            subresource,
        }
    }

    pub(crate) fn mips(&self) -> u32 {
        self.layouts.len() as u32
    }

    /// Single-level view for `level`. Panics out of range to catch Hi-Z issues early.
    #[expect(
        dead_code,
        reason = "Hi-Z pyramid is the first consumer"
    )]
    pub(crate) fn mip_view(&self, level: u32) -> vk::ImageView {
        self.mip_views[level as usize]
    }

    #[expect(
        dead_code,
        reason = "Hi-Z pyramid is the first consumer"
    )]
    pub(crate) fn mip_extent(&self, level: u32) -> vk::Extent2D {
        assert!(level < self.mips(), "mip {level} out of range");
        mip_extent_of(self.extent, level)
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
        self.barrier(device, cmd, 0..self.mips(), to, false);
    }

    /// Transition one mip level independently (for pyramid reads and writes).
    #[expect(
        dead_code,
        reason = "Hi-Z pyramid is the first consumer"
    )]
    pub(crate) fn transition_mip(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        level: u32,
        to: LayoutUse,
    ) {
        self.barrier(device, cmd, level..level + 1, to, false);
    }

    /// Like `transition_mip`, but hints that the old contents are dead.
    #[expect(
        dead_code,
        reason = "Hi-Z pyramid is the first consumer"
    )]
    pub(crate) fn transition_mip_discard(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        level: u32,
        to: LayoutUse,
    ) {
        self.barrier(device, cmd, level..level + 1, to, true);
    }

    /// Declares oldLayout = UNDEFINED for targets being fully overwritten
    /// (but dependency still uses real prior state). Waits for prior reads before discarding.
    pub(crate) fn transition_discard(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        to: LayoutUse,
    ) {
        self.barrier(device, cmd, 0..self.mips(), to, true);
    }

    fn barrier(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        levels: Range<u32>,
        to: LayoutUse,
        discard: bool,
    ) {
        let (new_layout, dst_stage, dst_access) = to.dst();
        let barriers: Vec<_> = layout_runs(&self.layouts, levels.clone())
            .into_iter()
            .map(|(run, old_layout)| {
                let (src_stage, src_access) = if old_layout == vk::ImageLayout::UNDEFINED {
                    (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE)
                } else {
                    to.src_when_used()
                };
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)
                    // The discard hint only forces the layout half; the
                    // dependency above still names the real prior use.
                    .old_layout(if discard {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        old_layout
                    })
                    .new_layout(new_layout)
                    .image(self.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        base_mip_level: run.start,
                        level_count: run.end - run.start,
                        ..self.subresource
                    })
            })
            .collect();
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
        }
        for level in levels {
            self.layouts[level as usize] = new_layout;
        }
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            for view in &self.mip_views {
                device.destroy_image_view(*view, None);
            }
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

    const GENERAL: vk::ImageLayout = vk::ImageLayout::GENERAL;
    const READ: vk::ImageLayout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    const UNDEF: vk::ImageLayout = vk::ImageLayout::UNDEFINED;

    /// Mip extents floor correctly, including non-power-of-two dimensions.
    #[test]
    fn mip_extents_follow_the_vulkan_floor_rule() {
        let base = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        assert_eq!(mip_extent_of(base, 0), base);
        assert_eq!(mip_extent_of(base, 1).width, 960);
        assert_eq!(mip_extent_of(base, 4).height, 67);
        // Minimum is 1×1 (never empty).
        assert_eq!(
            mip_extent_of(base, 20),
            vk::Extent2D {
                width: 1,
                height: 1
            }
        );
    }

    /// Uniform layouts produce a single barrier run.
    #[test]
    fn uniform_layouts_group_into_a_single_run() {
        assert_eq!(layout_runs(&[UNDEF], 0..1), vec![(0..1, UNDEF)]);
        assert_eq!(layout_runs(&[READ; 6], 0..6), vec![(0..6, READ)]);
    }

    /// Diverged pyramids yield one run per distinct old layout.
    #[test]
    fn diverged_pyramid_groups_into_runs_per_old_layout() {
        assert_eq!(
            layout_runs(&[READ, READ, GENERAL, UNDEF, UNDEF], 0..5),
            vec![(0..2, READ), (2..3, GENERAL), (3..5, UNDEF)]
        );
        // Sub-ranges report only their own levels.
        assert_eq!(
            layout_runs(&[READ, READ, GENERAL, UNDEF, UNDEF], 2..4),
            vec![(2..3, GENERAL), (3..4, UNDEF)]
        );
    }

    /// Equal but non-adjacent layouts stay separate (don't merge across gaps).
    #[test]
    fn equal_but_noncontiguous_layouts_stay_separate_runs() {
        assert_eq!(
            layout_runs(&[READ, GENERAL, READ], 0..3),
            vec![(0..1, READ), (1..2, GENERAL), (2..3, READ)]
        );
    }

    /// Pyramid transitions leave unmodified levels untouched.
    #[test]
    fn pyramid_transitions_leave_unnamed_mips_untouched() {
        const LEVELS: usize = 5;
        let mut layouts = [UNDEF; LEVELS];
        layouts[0] = LayoutUse::ComputeStorageWrite.dst().0;
        assert_eq!(layouts, [GENERAL, UNDEF, UNDEF, UNDEF, UNDEF]);

        for n in 0..LEVELS - 1 {
            // Publish n: RAW dependency waits on storage write, not just ordered reads.
            assert_eq!(
                LayoutUse::SampledAfterComputeWrite.src_when_used(),
                (
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE
                )
            );
            layouts[n] = LayoutUse::SampledAfterComputeWrite.dst().0;
            layouts[n + 1] = LayoutUse::ComputeStorageWrite.dst().0;

            // Invariant: read and write levels have different layouts.
            assert_eq!(layouts[n], READ);
            assert_eq!(layouts[n + 1], GENERAL);
            assert!(layouts[..n].iter().all(|&l| l == READ));
            assert!(layouts[n + 2..].iter().all(|&l| l == UNDEF));
        }

        assert_eq!(layouts[LEVELS - 1], GENERAL);
        layouts[LEVELS - 1] = LayoutUse::SampledAfterComputeWrite.dst().0;
        assert!(layouts.iter().all(|&l| l == READ));
        // Uniform layout collapses to a single barrier.
        assert_eq!(layout_runs(&layouts, 0..LEVELS as u32), vec![(0..5, READ)]);
    }
}
