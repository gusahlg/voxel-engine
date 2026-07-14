//! Engine-side shared types and signatures for the renderer fan-out.
//!
//! RULES FOR IMPLEMENTING AGENTS:
//! - Bodies marked `todo!()` are yours (per your work package). Signatures,
//!   layouts, and layout assertions are NOT — signature changes go back to the
//!   orchestrator.
//! - Items whose final home is another module (noted per item) MOVE there when
//!   implemented; leave a `pub use` re-export here until the last consumer
//!   migrates, so parallel packages keep compiling against this path.
//! - The `vk` module is private to this crate: types here hold raw
//!   `ash::vk` handles or are opaque; real construction happens inside `vk/`
//!   with `pub(crate)` constructors added by the implementing package.
#![allow(dead_code)]

use glam::{DVec3, Mat4, Vec2};

// ============================================================================
// ============================================================================

/// Which of the two frames-in-flight this is. Minted ONLY by the renderer's
/// frame pacer (no public constructor); the app obtains it via
/// [`current_slot`]. Type-safe: prevents raw-usize indexing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameSlot(u8);

impl FrameSlot {
    /// The other frame-in-flight (2FIF is a type-level fact: exactly two).
    #[must_use]
    pub fn other(self) -> FrameSlot {
        FrameSlot(1 - self.0)
    }
    /// Crate-internal mint for the pacer and for `PerSlot` iteration.
    pub(crate) fn new(index: usize) -> FrameSlot {
        debug_assert!(index < 2);
        FrameSlot(index as u8)
    }
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// A pair of per-frame-in-flight resources, indexable ONLY by [`FrameSlot`].
/// There is deliberately no `Index<usize>` impl and never will be.
pub struct PerSlot<T>([T; 2]);

impl<T> PerSlot<T> {
    pub fn new(pair: [T; 2]) -> Self {
        PerSlot(pair)
    }

    /// Borrow both slots for lifecycle passes that touch every frame's copy.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
}

impl<T> std::ops::Index<FrameSlot> for PerSlot<T> {
    type Output = T;
    fn index(&self, s: FrameSlot) -> &T {
        &self.0[s.index()]
    }
}

impl<T> std::ops::IndexMut<FrameSlot> for PerSlot<T> {
    fn index_mut(&mut self, s: FrameSlot) -> &mut T {
        &mut self.0[s.index()]
    }
}

// The current recording slot is minted by the pacer: `vk::Renderer::current_slot`
// is the authoritative inherent accessor (the free `current_slot(&Engine)` stub
// was superseded at merge — see Renderer::current_slot.

// ============================================================================
// ============================================================================

/// The view-projection as built by `Frame::begin_3d` — UN-jittered, the only
/// matrix that may be stored or used for culling and TAA reprojection.
///
/// The jittered matrix is produced by a *private* function inside `vk/` at
/// push-constant packing time and must never escape that scope. Do NOT add a
/// public "jittered" type or accessor.
/// If a future pass believes it needs one, that is an orchestrator decision.
#[derive(Clone, Copy, Debug)]
pub struct CleanViewProj(pub Mat4);

/// Sub-pixel camera jitter in PIXELS (±0.5), zero until Phase E. Converted to
/// NDC only inside the private jitter application (it needs the target extent,
/// which is exactly why the conversion cannot live anywhere else). Rides
/// `DrawLists` from `Frame::begin_3d` (the sole injection point).
#[derive(Clone, Copy, Debug, Default)]
pub struct JitterOffset(pub Vec2);

impl JitterOffset {
    pub const ZERO: JitterOffset = JitterOffset(Vec2::ZERO);
}

/// Length of the jitter/dither sequences (shared so TAA sees decorrelated,
/// periodic noise it can integrate).
pub const TEMPORAL_SEQ_LEN: u64 = 16;

/// Halton(2,3) − 0.5, entry `frame % TEMPORAL_SEQ_LEN`. Values come from the
/// generated constants so CPU and any shader consumer agree.
pub fn jitter_at(frame_index: u64) -> JitterOffset {
    // The generated table is already Halton(2,3) − 0.5 in [-0.5, 0.5) per axis,
    // so CPU jitter and any shader consumer read one source.
    let e = crate::genconst::HALTON_23[(frame_index % TEMPORAL_SEQ_LEN) as usize];
    JitterOffset(Vec2::new(e[0], e[1]))
}

// ============================================================================
// ============================================================================

pub const FRAME_UNIFORMS_SET: u32 = 0;
pub const FRAME_UNIFORMS_BINDING: u32 = 2;

// `FrameUniformsGpu` (struct + std140 offset static asserts) is GENERATED from
// the lane_table in build.rs — the twin of `common.slang`'s `FrameUniforms`, so
// a lane can never drift between CPU and GPU. Edit the table there, not here.
include!(concat!(env!("OUT_DIR"), "/gen_frame_uniforms.rs"));

impl FrameUniformsGpu {
    /// Fixed neutral "render lit, not black" default: full ambient, valid sun
    /// direction (avoids NaN in the shader), no fog, exposure 1.0. Used by
    /// [`crate::Lighting::FullBright`] and as filler for pure-2D frames.
    pub fn full_bright() -> Self {
        let mut n = <Self as bytemuck::Zeroable>::zeroed();
        n.sun_dir_elev = [0.0, 1.0, 0.0, std::f32::consts::FRAC_PI_2];
        n.candle[3] = 1.0; // ambient floor luma → diffuse = 1.0
        n.exposure_dither[0] = 1.0; // neutral exposure
        n.extras[0] = 1.0; // stars allowed (invisible anyway at full daylight)
        n
    }
}

/// Bumped when the lane_table layout changes (v2 added the `anim` lane;
/// v3 repurposed `reserved` as `extras` with x = stars gain).
pub const FRAME_UNIFORMS_VERSION: u32 = 3;

// The host-visible per-frame UBO ring's real home is `vk::uniforms::UboRing`
// (`write`/`buffer`/`current_slot` implemented there against a persistently
// mapped `HostBuffer` pair); the frozen `skeleton::UboRing` stub was superseded
// at merge, so only the wire types (`FrameUniformsGpu`) remain here.

// ============================================================================
// ============================================================================

/// Scene exposure: a positive linear multiplier applied before tonemapping.
#[derive(Clone, Copy, Debug)]
pub struct Exposure(pub f32);

impl Exposure {
    /// Pre-metering default — matches today's look at full daylight.
    pub const DEFAULT: Exposure = Exposure(1.0);
}

/// Read view of last-computed exposure (the slot the GPU is NOT writing).
pub struct ExposureRead<'a> {
    pub(crate) buf: &'a ash::vk::Buffer,
}
/// Write view for this frame's compute reduction.
pub struct ExposureWrite<'a> {
    pub(crate) buf: &'a ash::vk::Buffer,
}

// The double-buffered exposure ring's real home is `vk::exposure::ExposureRing`
// (parity baked into its `views`; exposure curve + temporal smoothing in
// `metered`). The `ExposureRead`/`ExposureWrite` view newtypes above stay here
// (the real ring borrows them); the frozen opaque `ExposureRing` stub was
// superseded at merge.

// ============================================================================
// ============================================================================

/// Exactly two cascades, by type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cascade {
    Near,
    Far,
}

/// Per-cascade resources/values, indexable only by [`Cascade`].
pub struct PerCascade<T>([T; 2]);

impl<T> PerCascade<T> {
    pub fn new(pair: [T; 2]) -> Self {
        PerCascade(pair)
    }
}

impl<T> std::ops::Index<Cascade> for PerCascade<T> {
    type Output = T;
    fn index(&self, c: Cascade) -> &T {
        &self.0[c as usize]
    }
}

/// One cascade's fit for this frame: light-space view-proj (CLEAN — shadows
/// never jitter), its far split distance, and world metres per shadow texel
/// (for stable-snap and for the receiver's bias formula).
#[derive(Clone, Copy, Debug)]
pub struct CascadeFit {
    pub view_proj: CleanViewProj,
    pub split: f32,
    pub texel_world: f32,
}

/// Shadow configuration. Every starting value is a re-based formula and
/// provisional: Phase D's checkpoint finalizes them.
pub struct ShadowCfg {
    /// Per-cascade map resolution (one D32 image, 2 layers).
    pub resolution: u32,
    /// Rotated 4-tap PCF radius in texels. Tentative, pending tuning.
    pub blur_texels: f32,
    /// Slope-scaled depth bias. Tentative, pending tuning.
    pub slope_bias: f32,
    /// Per-metre distance bias. Tentative, pending tuning.
    pub dist_bias: f32,
    /// Width (metres) of the map→fallback smoothstep at SHADOW_LIMIT.
    pub fade_band: f32,
    /// Cascade far distances. Tentative, pending tuning.
    pub splits: [f32; 2],
}

// `fit` (cascade fitting around the frustum slice) is implemented in
// `vk::shadow::fit`; the frozen `skeleton::fit` stub was superseded at merge.

/// Sampling-pass uniforms (set 0, binding 3). Separate from FrameUniformsGpu
/// because its lifecycle differs (per-cascade-fit vs per-frame).
/// The shadow PASS itself does NOT use this — a depth-only
/// pipeline pushes CascadeFit's matrix in its own 128 B push budget.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CascadeUniformsGpu {
    pub view_proj: [[[f32; 4]; 4]; 2],
    /// x,y = split distances; z = fade_band; w = SHADOW_LIMIT.
    pub splits_fade: [f32; 4],
    /// x = blur_texels, y = slope_bias, z = dist_bias, w = texel_world(near).
    pub bias: [f32; 4],
}

pub const CASCADE_UNIFORMS_BINDING: u32 = 3;

const _: () = assert!(size_of::<CascadeUniformsGpu>() == 160);
const _: () = assert!(std::mem::offset_of!(CascadeUniformsGpu, splits_fade) == 128);

// ============================================================================
// ============================================================================

/// FROZEN: the history integrator's texel format = the HDR
/// target's, so reprojected blends stay in unclamped linear. TAA resolves
/// BEFORE tonemap; history is never an 8-bit sRGB image.
pub const TAA_HISTORY_FORMAT: ash::vk::Format = ash::vk::Format::R16G16B16A16_SFLOAT;

/// FROZEN pass order — the ONE cross-baseline decision the
/// type system cannot police (wrong order = flicker/ghosting, a SILENT bug, not
/// a compile error). The TAA resolve consumes the resolved HDR and writes the
/// stabilized HDR that exposure meters and tonemap reads:
///
/// ```text
/// shadow depth → main color (mesh, MSAA) → HDR resolve
///   → TAA resolve (history blend, HDR)      ← inserted HERE, vk/mod.rs
///   → exposure compute (meters STABILIZED image)
///   → tonemap (+ post dither)
/// ```
///
/// Validation test (golden): `taa_static_hold` — camera held fixed N
/// frames with TAA on; inter-frame `DiffStats.pct_changed` must converge toward
/// ~0 (temporal stability); plus an orbit shot with no ghost trail (reprojection
/// check). These catch a mis-ordered resolve that
/// no assert can.
///
/// FROZEN TAA-resolve descriptor set bindings (its OWN set, not set 0): the
/// current resolved HDR at 0, the history target at 1, the reprojection UBO at 2.
pub const TAA_RESOLVE_CURRENT_BINDING: u32 = 0;
pub const TAA_RESOLVE_HISTORY_BINDING: u32 = 1;
pub const TAA_RESOLVE_REPROJ_BINDING: u32 = 2;

/// VRS invariant preserved under jitter: the scene fingerprint is computed
/// from the CLEAN view-proj; the jittered matrix is applied only at
/// push-constant packing and must never reach
/// `scene_fingerprint`, so sub-pixel jitter cannot thrash VRS slot reuse.
pub const _TAA_FINGERPRINT_USES_CLEAN: () = ();

/// TAA history color target: persistent (NOT per-slot — history is a single
/// integrator), format = the HDR target's, recreated on resize with
/// contents discarded (history reconverges). Opaque here; real image lives in
/// `vk/`.
pub struct TaaHistory {
    pub(crate) _private: (),
}

/// Reprojection inputs, CPU-side. The `prev` field is always a clean
/// (un-jittered) matrix — reprojecting a jittered matrix causes ghosting.
/// `camera_delta` = prev_origin − current_origin in f64 world space
/// (camera-at-origin: the delta IS the reprojection translation), narrowed
/// to f32 only at [`Reprojection::pack`].
pub struct Reprojection {
    pub prev: CleanViewProj,
    pub camera_delta: DVec3,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReprojectionGpu {
    pub prev_view_proj: [[f32; 4]; 4],
    /// xyz = camera delta (f64-subtract-then-narrow), w unused.
    pub camera_delta: [f32; 4],
}

const _: () = assert!(size_of::<ReprojectionGpu>() == 80);

impl Reprojection {
    /// The ONE narrowing site: compose prev clip transform with the
    /// f64 camera delta, then narrow.
    pub fn pack(&self) -> ReprojectionGpu {
        // The ONE f64→f32 narrowing site. `camera_delta = prev_origin −
        // current_origin` (camera-at-origin ⇒ the delta IS the reprojection
        // translation). A current camera-relative point P maps to its previous
        // camera-relative position `P − camera_delta`, so composing the prev
        // clip transform with that translation gives a single matrix that
        // reprojects current-frame world points straight into previous clip.
        let d = self.camera_delta.as_vec3();
        let m = self.prev.0 * Mat4::from_translation(-d);
        ReprojectionGpu {
            prev_view_proj: m.to_cols_array_2d(),
            camera_delta: [d.x, d.y, d.z, 0.0],
        }
    }
}

// ============================================================================
// ============================================================================

// The real bodies live in `crate::capture` and are re-exported at
// the crate root. This skeleton path is kept as a compatibility re-export for
// existing `voxel_engine::skeleton::{Screenshot, screenshot_to, load_png}`
// consumers (the harness); `screenshot_to` returns `io::Result<()>` (writes to
// exactly `path`), not the draft `PathBuf`.
pub use crate::{Screenshot, load_png, screenshot_to};

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity resolver is its own inverse and never aliases.
    #[test]
    fn frame_slot_other_is_involution() {
        let a = FrameSlot::new(0);
        assert_eq!(a.other().other(), a);
        assert_ne!(a.other(), a);
    }

    /// Verify jitter sequence is centered (mean → 0) and bounded by ±0.5 px.
    #[test]
    fn jitter_sequence_centered_and_bounded() {
        let mut sum = Vec2::ZERO;
        for f in 0..TEMPORAL_SEQ_LEN {
            let j = jitter_at(f).0;
            assert!(j.x.abs() <= 0.5 && j.y.abs() <= 0.5);
            sum += j;
        }
        assert!(sum.length() < 0.1 * TEMPORAL_SEQ_LEN as f32);
    }
}
