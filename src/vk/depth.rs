//! Sampleable depth layout, rest barrier, and previous-frame depth tracking.
//! Split out of `frame_loop` / `mod.rs` (move only).

use ash::vk;

use super::buffers::FRAMES_IN_FLIGHT;
use super::{Renderer, depth_range};

/// Resting layout of the single-sample sampleable depth after the scene pass.
///
/// Contract: when a later pass *this frame* samples that image
/// ([`SceneDepthUse::Sampleable`]), [`scene_pass::RenderPass::end`] transitions
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
/// The multisampled `depth` attachment also begins from UNDEFINED. It is
/// sampled only on classify-only MSAA frames ([`SceneDepthUse::ClassifyMs`]):
/// stored instead of resolved, it rests in this same layout for the VRS
/// classifier (see [`Renderer::ms_depth_classify_barrier`]).
pub(crate) const SAMPLEABLE_DEPTH_REST_LAYOUT: vk::ImageLayout =
    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

/// What the scene pass leaves of its depth for later passes this frame.
/// Chosen once per frame by [`scene_depth_use`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SceneDepthUse {
    /// Nothing samples depth: `DONT_CARE` store, no MSAA resolve, no rest
    /// barrier.
    Discard,
    /// The single-sample sampleable depth (the MSAA SAMPLE_ZERO resolve, else
    /// the depth attachment itself) is stored and rests in
    /// [`SAMPLEABLE_DEPTH_REST_LAYOUT`]. Every consumer, the VRS classifier
    /// included, samples that image.
    Sampleable,
    /// Multisampled, and the VRS classifier is the only consumer: no resolve.
    /// The MS depth attachment is stored and rests in
    /// [`SAMPLEABLE_DEPTH_REST_LAYOUT`]; `vrs_ms.comp.spv` loads sample 0.
    ClassifyMs,
}

impl SceneDepthUse {
    /// The scene-pass depth attachment (the MS depth when multisampled) is
    /// stored rather than discarded.
    pub(crate) fn stores_attachment(self, multisampled: bool) -> bool {
        match self {
            Self::Discard => false,
            Self::Sampleable => !multisampled,
            Self::ClassifyMs => true,
        }
    }

    /// The single-sample sampleable depth holds this frame's depth after the
    /// pass: what [`PrevDepthTrack::finish`] records as stored.
    pub(crate) fn sampleable_stored(self) -> bool {
        self == Self::Sampleable
    }
}

/// Picks this frame's [`SceneDepthUse`].
///
/// The single-sample sampleable depth is needed by:
/// - quarter-res spill/godrays (same submit), on presented frames with bloom
///   or a live godray march;
/// - fused TAA tonemap (later present submit), on presented frames with TAA;
/// - next-frame water absorption (later submit, same queue): this frame stores
///   so the following frame can sample.
///
/// The VRS classifier (same submit, presented or not) samples it too whenever
/// it exists. When the classifier is the only consumer and `classify_ms`
/// (MSAA with the `vrs_ms` pipeline), it loads sample 0 of the stored MS depth
/// instead, so most unpresented frames skip the full-resolution resolve.
/// Single-sampled, the depth attachment is the sampleable image, so classify
/// alone is still [`SceneDepthUse::Sampleable`].
///
/// Minimap and screenshot capture do not sample depth.
pub(crate) fn scene_depth_use(
    classify_ms: bool,
    will_present: bool,
    taa: bool,
    spill_live: bool,
    classify_vrs: bool,
    absorb_store: bool,
) -> SceneDepthUse {
    let sampleable = absorb_store || (will_present && (taa || spill_live));
    if sampleable || (classify_vrs && !classify_ms) {
        SceneDepthUse::Sampleable
    } else if classify_vrs {
        SceneDepthUse::ClassifyMs
    } else {
        SceneDepthUse::Discard
    }
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
    /// Issued only for [`SceneDepthUse::Sampleable`]. Src is the
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

    /// Scene-pass → classifier read of the multisampled depth attachment
    /// ([`SceneDepthUse::ClassifyMs`]; no resolve ran).
    ///
    /// Src is the attachment's own depth tests and its `STORE` (depth store
    /// ops run in `LATE_FRAGMENT_TESTS` / `DEPTH_STENCIL_ATTACHMENT_WRITE`).
    /// Dst is the classifier alone (`COMPUTE_SHADER` / `SHADER_SAMPLED_READ`),
    /// later in this submit. The image rests in [`SAMPLEABLE_DEPTH_REST_LAYOUT`]
    /// until the next scene pass of this slot discards it from `UNDEFINED`;
    /// that begin adds `COMPUTE_SHADER` to its source scope
    /// (`SlotState::vrs_ms_depth_read`).
    pub(super) fn ms_depth_classify_barrier(&self, slot: usize) -> vk::ImageMemoryBarrier2<'_> {
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(Self::depth_pass_layout())
            .new_layout(SAMPLEABLE_DEPTH_REST_LAYOUT)
            .image(self.targets.depth[slot].image())
            .subresource_range(depth_range())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use SceneDepthUse::{ClassifyMs, Discard, Sampleable};

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

    /// `(will_present, taa, spill_live, classify_vrs, absorb_store)` for all 32
    /// combinations.
    fn all_inputs() -> impl Iterator<Item = (bool, bool, bool, bool, bool)> {
        (0u32..32).map(|m| (m & 1 != 0, m & 2 != 0, m & 4 != 0, m & 8 != 0, m & 16 != 0))
    }

    #[test]
    fn scene_depth_use_gates_on_actual_consumers() {
        // (will_present, taa, spill_live, classify_vrs, absorb_store), then the
        // use without and with the sample-0 MS classifier.
        let cases = [
            // Nothing samples: TAA off, no spill, no VRS, no absorb — even if presenting.
            ((false, false, false, false, false), Discard, Discard),
            ((true, false, false, false, false), Discard, Discard),
            // Fused TAA tonemap samples only on presented frames.
            ((true, true, false, false, false), Sampleable, Sampleable),
            ((false, true, false, false, false), Discard, Discard),
            // Spill/godrays sample only on presented frames.
            ((true, false, true, false, false), Sampleable, Sampleable),
            ((false, false, true, false, false), Discard, Discard),
            // Absorb Blend stores for the next frame, presented or not.
            ((false, false, false, false, true), Sampleable, Sampleable),
            ((true, false, false, false, true), Sampleable, Sampleable),
            // VRS classify reads in the same submit, presented or not. MSAA
            // loads sample 0 of the MS depth unless another consumer needs
            // the resolve anyway.
            ((false, false, false, true, false), Sampleable, ClassifyMs),
            ((true, false, false, true, false), Sampleable, ClassifyMs),
            ((false, true, true, true, false), Sampleable, ClassifyMs),
            ((true, true, false, true, false), Sampleable, Sampleable),
            ((true, false, true, true, false), Sampleable, Sampleable),
            ((false, false, false, true, true), Sampleable, Sampleable),
        ];
        for ((present, taa, spill, classify, absorb), single, ms) in cases {
            let without = scene_depth_use(false, present, taa, spill, classify, absorb);
            let with = scene_depth_use(true, present, taa, spill, classify, absorb);
            assert_eq!((without, with), (single, ms));
        }
    }

    #[test]
    fn scene_depth_use_without_ms_classify_is_the_old_consumed_gate() {
        // Single-sampled (or no `vrs_ms` pipeline): the sampleable depth is
        // stored exactly when any consumer exists, the classifier included.
        for (present, taa, spill, classify, absorb) in all_inputs() {
            let consumed = absorb || classify || (present && (taa || spill));
            let use_ = scene_depth_use(false, present, taa, spill, classify, absorb);
            assert_ne!(use_, ClassifyMs);
            assert_eq!(use_ == Sampleable, consumed, "{use_:?}");
        }
    }

    #[test]
    fn scene_depth_use_ms_classify_resolves_only_for_other_consumers() {
        for (present, taa, spill, classify, absorb) in all_inputs() {
            let resolved = absorb || (present && (taa || spill));
            let use_ = scene_depth_use(true, present, taa, spill, classify, absorb);
            assert_eq!(use_.sampleable_stored(), resolved, "{use_:?}");
            assert_eq!(use_ == ClassifyMs, classify && !resolved, "{use_:?}");
            assert_eq!(use_ == Discard, !classify && !resolved, "{use_:?}");
        }
    }

    #[test]
    fn scene_depth_use_attachment_store() {
        // Single-sampled: the attachment is the sampleable depth.
        assert!(!Discard.stores_attachment(false));
        assert!(Sampleable.stores_attachment(false));
        // MSAA: the resolve feeds consumers, so the MS attachment is stored
        // only when the classifier reads it directly.
        assert!(!Discard.stores_attachment(true));
        assert!(!Sampleable.stores_attachment(true));
        assert!(ClassifyMs.stores_attachment(true));
        // Water absorption sees a stored sampleable depth only from the
        // resolved / single-sample path, never from a classify-only frame.
        assert!(Sampleable.sampleable_stored());
        assert!(!ClassifyMs.sampleable_stored());
        assert!(!Discard.sampleable_stored());
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
