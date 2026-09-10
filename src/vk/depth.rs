//! Sampleable depth layout, rest barrier, and previous-frame depth tracking.
//! Split out of `frame_loop` / `mod.rs` (move only).

use ash::vk;

use super::buffers::FRAMES_IN_FLIGHT;
use super::{Renderer, depth_range};

/// Resting layout of the single-sample sampleable depth after the scene pass.
///
/// Contract: when a later pass *this frame* samples that image (see
/// [`sampleable_depth_consumed`]), [`scene_pass::RenderPass::end`] transitions
/// it (the MSAA resolve target when multisampled, else the depth image) from
/// the scene-pass write scope ([`sampleable_depth_attachment_state`]) to this
/// layout in the same `vkCmdPipelineBarrier2` as the offscreen HDR finalize,
/// with dst stage `COMPUTE_SHADER | FRAGMENT_SHADER` and access
/// `SHADER_SAMPLED_READ`. From then on it RESTS here: the quarter-res spill
/// pass (godray sampler) and the VRS classifier sample it in the same submit
/// with no further transition; the present-time fused TAA tonemap samples it
/// in the later copy submit (the render timeline wait covers that fragment
/// shader); the next frame's water-absorption blend samples it as previous
/// depth (same-queue submission order plus this barrier's dst fragment stage
/// is the read-after-write cover).
///
/// When nothing samples it, `end` skips that rest transition and the scene
/// pass stores depth with `DONT_CARE` (and skips the MSAA SAMPLE_ZERO resolve).
/// The next scene pass of this slot begins the image from `UNDEFINED` in
/// either case (contents are cleared every frame, so the discard is free).
/// The multisampled `depth` attachment is unchanged: it still begins from
/// UNDEFINED and is never sampled.
pub(crate) const SAMPLEABLE_DEPTH_REST_LAYOUT: vk::ImageLayout =
    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

/// True when a later pass this frame samples the scene's sampleable depth.
///
/// Consumers, computed once per frame:
/// - VRS classify (same submit), even on unpresented frames;
/// - quarter-res spill/godrays (same submit), on presented frames with bloom
///   or a live godray march;
/// - fused TAA tonemap (later present submit), on presented frames with TAA;
/// - next-frame water absorption (later submit, same queue): this frame stores
///   so the following frame can sample.
///
/// Minimap and screenshot capture do not sample depth.
pub(crate) fn sampleable_depth_consumed(
    will_present: bool,
    taa: bool,
    spill_live: bool,
    classify_vrs: bool,
    absorb_store: bool,
) -> bool {
    absorb_store || classify_vrs || (will_present && (taa || spill_live))
}

const DEPTH_SLOTS: usize = FRAMES_IN_FLIGHT as usize;

/// Per-slot store/sample bits for previous-frame water-absorption depth.
///
/// Frame N samples the depth image of the slot rendered by frame N−1. Validity
/// is exact: false on the first frame, after a swapchain/render-target
/// recreate, when the previous slot did not store, or when the render extent
/// changed. `sampled` covers the write-after-read hazard: the next begin of a
/// slot whose depth was sampled includes `FRAGMENT_SHADER` in the depth
/// transition's source stage.
pub(crate) struct PrevDepthTrack {
    stored: [bool; DEPTH_SLOTS],
    sampled: [bool; DEPTH_SLOTS],
    extent: [Option<(u32, u32)>; DEPTH_SLOTS],
}

impl PrevDepthTrack {
    pub(super) fn new() -> Self {
        Self {
            stored: [false; DEPTH_SLOTS],
            sampled: [false; DEPTH_SLOTS],
            extent: [None; DEPTH_SLOTS],
        }
    }

    pub(super) fn invalidate(&mut self) {
        *self = Self::new();
    }

    pub(super) fn prev_slot(slot: usize) -> usize {
        debug_assert!(slot < DEPTH_SLOTS);
        (slot + DEPTH_SLOTS - 1) % DEPTH_SLOTS
    }

    /// Previous slot stored sampleable depth at this pixel size.
    pub(super) fn valid(&self, slot: usize, extent: vk::Extent2D) -> bool {
        let p = Self::prev_slot(slot);
        self.stored[p] && self.extent[p] == Some((extent.width, extent.height))
    }

    /// Consume the WAR bit for this slot's previous life. True → begin-of-frame
    /// depth transition src must include `FRAGMENT_SHADER`.
    pub(super) fn begin_slot(&mut self, slot: usize) -> bool {
        let war = self.sampled[slot];
        self.sampled[slot] = false;
        war
    }

    pub(super) fn mark_sampled(&mut self, slot: usize) {
        self.sampled[slot] = true;
    }

    pub(super) fn finish(&mut self, slot: usize, stored: bool, extent: vk::Extent2D) {
        self.stored[slot] = stored;
        self.extent[slot] = stored.then_some((extent.width, extent.height));
    }
}

/// Synchronization state of the depth image sampled by post-processing *during
/// the scene pass* (the source scope of the rest-layout barrier).
/// Multisampled rendering writes that image through a resolve operation, whose
/// synchronization scope is COLOR_ATTACHMENT_OUTPUT/COLOR_ATTACHMENT_WRITE.
fn sampleable_depth_attachment_state(
    samples: vk::SampleCountFlags,
    scene_depth_layout: vk::ImageLayout,
) -> (vk::ImageLayout, vk::PipelineStageFlags2, vk::AccessFlags2) {
    if samples == vk::SampleCountFlags::TYPE_1 {
        (
            scene_depth_layout,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
    } else {
        (
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        )
    }
}

impl Renderer {
    /// The layout the scene-pass *attachment* (MS depth, or the single-sample
    /// depth when not multisampled) lives in *during* the pass:
    /// `DEPTH_ATTACHMENT_OPTIMAL`. Water absorption samples the *previous*
    /// slot's stored depth, so this pass never self-depends on its own depth
    /// attachment (that used to force `RENDERING_LOCAL_READ` and defeat Hi-Z).
    ///
    /// After `RenderPass::end`, if a later pass this frame samples it, the
    /// *sampleable* single-sample image leaves this layout and rests in
    /// [`SAMPLEABLE_DEPTH_REST_LAYOUT`]. The next scene pass of this slot
    /// begins from `UNDEFINED` either way.
    pub(super) fn depth_pass_layout() -> vk::ImageLayout {
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
    }

    /// Layout and write scope of the single-sample depth image *during the
    /// scene pass* — the source of the rest-layout barrier in `RenderPass::end`.
    /// With MSAA this is a resolve attachment: Vulkan executes dynamic-rendering
    /// resolves at COLOR_ATTACHMENT_OUTPUT, even for depth. Treating it as an
    /// ordinary depth-test write leaves the resolve unordered on drivers that
    /// implement those stages independently. After that barrier (when issued)
    /// the image rests in [`SAMPLEABLE_DEPTH_REST_LAYOUT`]; see that const for
    /// the contract.
    pub(super) fn sampleable_depth_attachment_state(
        &self,
    ) -> (vk::ImageLayout, vk::PipelineStageFlags2, vk::AccessFlags2) {
        sampleable_depth_attachment_state(self.targets.samples, Self::depth_pass_layout())
    }

    /// Scene-pass → rest: sampleable depth becomes [`SAMPLEABLE_DEPTH_REST_LAYOUT`].
    ///
    /// Issued only when [`sampleable_depth_consumed`] is true. Src is the
    /// attachment-write scope (depth tests, or COLOR_ATTACHMENT_OUTPUT for the
    /// MSAA SAMPLE_ZERO resolve). Dst covers every consumer that samples it
    /// without a further transition: VRS compute, the quarter-res spill compute
    /// (godrays), and the present-time tonemap fragment (fused TAA). The present
    /// copy is a later submit that waits on the render timeline.
    pub(super) fn sampleable_depth_rest_barrier(&self, slot: usize) -> vk::ImageMemoryBarrier2<'_> {
        let (src_layout, src_stage, src_access) = self.sampleable_depth_attachment_state();
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            )
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(src_layout)
            .new_layout(SAMPLEABLE_DEPTH_REST_LAYOUT)
            .image(self.targets.sampleable_depth(slot).image())
            .subresource_range(depth_range())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_depth_resolve_uses_color_output_sync_scope() {
        let scene_layout = vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;
        let (layout, stage, access) =
            sampleable_depth_attachment_state(vk::SampleCountFlags::TYPE_8, scene_layout);
        assert_eq!(layout, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
        assert_eq!(stage, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        assert_eq!(access, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);

        let (layout, stage, access) =
            sampleable_depth_attachment_state(vk::SampleCountFlags::TYPE_1, scene_layout);
        assert_eq!(layout, scene_layout);
        assert!(stage.contains(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS));
        assert!(stage.contains(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS));
        assert!(access.contains(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE));
    }

    #[test]
    fn sampleable_depth_rests_in_shader_read_only() {
        assert_eq!(
            SAMPLEABLE_DEPTH_REST_LAYOUT,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }

    #[test]
    fn sampleable_depth_consumed_gates_on_actual_consumers() {
        // Nothing samples: TAA off, no spill, no VRS, no absorb — even if presenting.
        assert!(!sampleable_depth_consumed(
            false, false, false, false, false
        ));
        assert!(!sampleable_depth_consumed(true, false, false, false, false));
        // VRS classify samples in the same submit, presented or not.
        assert!(sampleable_depth_consumed(false, false, false, true, false));
        assert!(sampleable_depth_consumed(true, false, false, true, false));
        // Fused TAA tonemap samples only on presented frames.
        assert!(sampleable_depth_consumed(true, true, false, false, false));
        assert!(!sampleable_depth_consumed(false, true, false, false, false));
        // Spill/godrays sample only on presented frames.
        assert!(sampleable_depth_consumed(true, false, true, false, false));
        assert!(!sampleable_depth_consumed(false, false, true, false, false));
        // Absorb Blend stores for the next frame, presented or not.
        assert!(sampleable_depth_consumed(false, false, false, false, true));
        assert!(sampleable_depth_consumed(true, false, false, false, true));
    }

    fn extent(w: u32, h: u32) -> vk::Extent2D {
        vk::Extent2D {
            width: w,
            height: h,
        }
    }

    #[test]
    fn prev_depth_invalid_on_first_frame() {
        let t = PrevDepthTrack::new();
        for slot in 0..DEPTH_SLOTS {
            assert!(!t.valid(slot, extent(1920, 1080)), "slot {slot}");
        }
    }

    #[test]
    fn prev_depth_valid_after_previous_slot_stores_same_extent() {
        let mut t = PrevDepthTrack::new();
        let e = extent(1920, 1080);
        t.finish(0, true, e);
        assert!(t.valid(1, e));
        assert!(
            !t.valid(2, e),
            "slot 2's previous is slot 1, not yet stored"
        );
        t.finish(1, true, e);
        assert!(t.valid(2, e));
        t.finish(2, true, e);
        assert!(t.valid(0, e), "ring wrap: slot 0 samples slot 2");
    }

    #[test]
    fn prev_depth_invalid_when_previous_slot_did_not_store() {
        let mut t = PrevDepthTrack::new();
        let e = extent(1920, 1080);
        t.finish(0, false, e);
        assert!(!t.valid(1, e));
        t.finish(0, true, e);
        t.finish(1, false, e);
        assert!(!t.valid(2, e));
        assert!(t.valid(1, e), "slot 0 still stored");
    }

    #[test]
    fn prev_depth_invalid_when_render_extent_changed() {
        let mut t = PrevDepthTrack::new();
        t.finish(0, true, extent(1920, 1080));
        assert!(!t.valid(1, extent(1280, 720)));
        assert!(t.valid(1, extent(1920, 1080)));
    }

    #[test]
    fn prev_depth_invalid_after_recreate() {
        let mut t = PrevDepthTrack::new();
        let e = extent(1920, 1080);
        t.finish(0, true, e);
        t.finish(1, true, e);
        t.mark_sampled(0);
        t.invalidate();
        for slot in 0..DEPTH_SLOTS {
            assert!(!t.valid(slot, e), "slot {slot}");
            assert!(!t.begin_slot(slot), "WAR bit cleared on slot {slot}");
        }
    }

    #[test]
    fn prev_depth_war_bit_set_by_sample_cleared_by_begin() {
        let mut t = PrevDepthTrack::new();
        let e = extent(800, 600);
        t.finish(0, true, e);
        assert!(!t.begin_slot(0), "nothing has sampled slot 0 yet");
        t.mark_sampled(0);
        assert!(t.begin_slot(0));
        assert!(!t.begin_slot(0), "WAR bit is consumed once");
    }

    #[test]
    fn prev_slot_is_the_previous_ring_index() {
        assert_eq!(PrevDepthTrack::prev_slot(0), DEPTH_SLOTS - 1);
        assert_eq!(PrevDepthTrack::prev_slot(1), 0);
        if DEPTH_SLOTS > 2 {
            assert_eq!(PrevDepthTrack::prev_slot(2), 1);
        }
    }
}
