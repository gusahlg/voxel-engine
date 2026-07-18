//! Camera, projection, and view-frustum utilities.
//!
//! The engine renders with Vulkan reversed-Z (depth cleared to 0.0, compare
//! GREATER_OR_EQUAL) using `Mat4::perspective_infinite_reverse_rh`, and flips
//! Y via a negative viewport height so NDC is GL-style y-up. `Camera3D`
//! mirrors raylib's camera (fovy in degrees), `world_to_screen` projects with
//! the matrices used for rendering, and `Frustum` extracts frustum planes for
//! AABB culling.

use glam::{Mat4, Vec2, Vec3, Vec4};

/// Near plane distance (single source with shader's WATER_Z_NEAR from genconst).
pub use crate::genconst::Z_NEAR;

/// Validated cylindrical warp strength for the wide-FOV lens: finite and
/// `0 < s <= MAX`. The *absence* of warp is [`Lens::Rectilinear`], not a zero
/// strength — so a `WarpStrength` always denotes an active warp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarpStrength(f32);

impl WarpStrength {
    /// Upper clamp: beyond this the periphery compression is unusable.
    pub const MAX: f32 = 2.0;

    /// Returns `None` for non-finite or `<= 0.0` (that is `Rectilinear`, not a
    /// weak warp); otherwise clamps to `MAX`.
    #[inline]
    pub fn new(s: f32) -> Option<Self> {
        (s.is_finite() && s > 0.0).then(|| Self(s.min(Self::MAX)))
    }

    #[inline]
    pub fn get(self) -> f32 {
        self.0
    }
}

/// How a [`Camera3D`] projects. `Rectilinear` is the zero-cost fast path and
/// carries no warp machinery, so a rectilinear camera is unrepresentable-with-warp
/// by construction. `WideFov` renders a wider rectilinear source (see [`Aspect`])
/// and compresses the periphery in the tonemap resample.
#[derive(Clone, Copy, Debug, Default)]
pub enum Lens {
    #[default]
    Rectilinear,
    WideFov {
        strength: WarpStrength,
    },
}

/// Aspect ratio (framebuffer width / height). Newtype so the *source* (wide)
/// aspect used for the offscreen render can never be confused with the presented
/// window aspect.
#[derive(Clone, Copy, Debug)]
pub struct Aspect(pub f32);

impl Aspect {
    #[inline]
    pub fn get(self) -> f32 {
        self.0
    }

    /// Widens horizontally for the wide-FOV source render (vertical fov unchanged):
    /// `source_aspect = aspect * fov_scale`. Identity leaves the aspect untouched.
    #[inline]
    pub fn source(self, warp: &WarpMap) -> Aspect {
        Aspect(self.0 * warp.fov_scale())
    }
}

/// The single source of truth for the warp maps and the GPU bytes, built once per
/// frame from a [`Lens`]. Both the CPU picking transform ([`WarpMap::warp_ndc`],
/// used by [`world_to_screen`]) and the tonemap push ([`WarpMap::push`]) read the
/// same coefficients, so they cannot desync. Horizontal-only (De Carpentier
/// cylinder): vertical NDC is rectilinear and untouched.
#[derive(Clone, Copy, Debug)]
pub enum WarpMap {
    Identity,
    Active { s: f32, atan_s: f32 },
}

impl WarpMap {
    #[inline]
    pub fn from_lens(lens: Lens) -> Self {
        match lens {
            Lens::Rectilinear => WarpMap::Identity,
            Lens::WideFov { strength } => {
                let s = strength.get();
                WarpMap::Active {
                    s,
                    atan_s: s.atan(),
                }
            }
        }
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        matches!(self, WarpMap::Identity)
    }

    /// Horizontal source-fov widening: a pure function of strength. `1.0` for identity.
    #[inline]
    pub fn fov_scale(&self) -> f32 {
        match self {
            WarpMap::Identity => 1.0,
            WarpMap::Active { s, atan_s } => s / atan_s,
        }
    }

    /// Forward map: source-NDC -> presented-NDC. Mirrors the tonemap frag and is
    /// the picking transform. Horizontal only; vertical passes through.
    #[inline]
    pub fn warp_ndc(&self, ndc: Vec2) -> Vec2 {
        match self {
            WarpMap::Identity => ndc,
            WarpMap::Active { s, atan_s } => Vec2::new((ndc.x * s).atan() / atan_s, ndc.y),
        }
    }

    /// Inverse map: presented-NDC -> source-NDC. The tonemap resample samples the
    /// source here; round-trips [`WarpMap::warp_ndc`] to float precision.
    #[inline]
    pub fn unwarp_ndc(&self, ndc: Vec2) -> Vec2 {
        match self {
            WarpMap::Identity => ndc,
            WarpMap::Active { s, atan_s } => Vec2::new((ndc.x * atan_s).tan() / s, ndc.y),
        }
    }

    /// GPU push bytes for tonemap remap. Identity yields `s = 0` so the frag skips
    /// remapping. Godray carries sun's screen position; all-zero disables march.
    #[inline]
    pub fn push(
        &self,
        exposure: f32,
        godray: Godray,
        vignette: f32,
    ) -> WarpPush {
        let (s, atan_s) = match self {
            WarpMap::Identity => (0.0, 0.0),
            WarpMap::Active { s, atan_s } => (*s, *atan_s),
        };
        WarpPush {
            exposure,
            s,
            atan_s,
            _pad0: 0.0,
            // Godray jitter correction in .w lanes.
            godray0: [godray.sun_uv[0], godray.sun_uv[1], godray.strength, godray.jitter_uv[0]],
            godray1: [godray.tint[0], godray.tint[1], godray.tint[2], godray.jitter_uv[1]],
            vignette,
        }
    }
}

/// Per-frame godray inputs: sun's screen position, strength gate (1 = draw, 0 = off),
/// and linear sun tint. Disables when sun is behind camera or strength = 0.
#[derive(Clone, Copy, Debug)]
pub struct Godray {
    pub sun_uv: [f32; 2],
    pub strength: f32,
    pub tint: [f32; 3],
    /// Raster jitter in UV; corrects godray depth samples to prevent TAA shimmer.
    pub jitter_uv: [f32; 2],
}

impl Godray {
    /// The disabled/no-op value: `strength = 0` skips the march entirely.
    pub const OFF: Self = Self {
        sun_uv: [0.0, 0.0],
        strength: 0.0,
        tint: [0.0, 0.0, 0.0],
        jitter_uv: [0.0, 0.0],
    };

    /// Projects sun to screen position and gates the veil. Returns
    /// [`Godray::OFF`] when disabled or sun is behind camera (w <= 0).
    /// `sun_dir` is world-space direction to the sun; `tint` is its colour.
    pub fn project(
        enabled: bool,
        sun_dir: Vec3,
        tint: [f32; 3],
        cam: &Camera3D,
        screen_w: f32,
        screen_h: f32,
        jitter_uv: [f32; 2],
    ) -> Self {
        if !enabled {
            return Self::OFF;
        }
        // A far point along the sun direction stands in for the sun at infinity.
        let sun_point = cam.position + sun_dir.normalize_or_zero() * 1.0e6;
        let warp = WarpMap::from_lens(cam.lens);
        let source_aspect = Aspect(screen_w / screen_h.max(1.0)).source(&warp);
        let clip = cam.view_proj(source_aspect.get()) * sun_point.extend(1.0);
        // Reversed-Z: w <= 0 means behind camera.
        if clip.w <= 0.0 {
            return Self::OFF;
        }
        let px = world_to_screen(sun_point, cam, screen_w, screen_h);
        Self {
            sun_uv: [px.x / screen_w, px.y / screen_h],
            strength: 1.0,
            tint,
            jitter_uv,
        }
    }
}

/// GPU push constant for the tonemap resample: exposure plus the warp coefficients
/// (`s <= 0` = rectilinear no-op). `exposure` is first for ABI stability with the
/// current single-`f32` tonemap push. Layout mirrored one-to-one in Slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WarpPush {
    pub exposure: f32,
    pub s: f32,
    pub atan_s: f32,
    /// Pad aligning the following `godray0` float4 to 16 bytes.
    pub _pad0: f32,
    /// Godray march: sun screen uv (xy), strength gate (z, 0 = no rays), jitter.x (w).
    pub godray0: [f32; 4],
    /// Godray sun tint: veil colour (rgb), jitter.y (w).
    pub godray1: [f32; 4],
    /// Vignette strength (0 = off). Trailing lane (ABI-appended).
    pub vignette: f32,
}

/// Perspective camera, raylib-parity: `fovy` is the vertical field of view in
/// DEGREES. `lens` selects the plain rectilinear fast path or the wide-FOV mode;
/// [`Lens::Rectilinear`] leaves the image identical to a plain projection.
#[derive(Clone, Copy, Debug)]
pub struct Camera3D {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fovy: f32,
    pub lens: Lens,
}

impl Camera3D {
    /// Right-handed view matrix looking from `position` toward `target`.
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Infinite reversed-Z projection: depth 1.0 at `Z_NEAR`, tending to 0.0
    /// at infinity. `aspect` = framebuffer width / height.
    pub fn proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_infinite_reverse_rh(self.fovy.to_radians(), aspect, Z_NEAR)
    }

    /// Combined projection * view, as consumed by shaders and `Frustum`.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view()
    }
}

/// Projects a world point to framebuffer pixel coordinates (origin top-left).
/// Uses the same matrices as rendering. Points behind the camera produce
/// unusable results (raylib parity — callers pre-filter).
pub fn world_to_screen(p: Vec3, cam: &Camera3D, screen_w: f32, screen_h: f32) -> Vec2 {
    // Project through the SAME widened source frustum the scene renders into
    // (`begin_3d`), then apply the forward warp to land in presented-NDC — the
    // exact inverse of the tonemap resample. Rectilinear leaves both untouched.
    let warp = WarpMap::from_lens(cam.lens);
    let source_aspect = Aspect(screen_w / screen_h.max(1.0)).source(&warp);
    let clip = cam.view_proj(source_aspect.get()) * p.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let ndc = warp.warp_ndc(ndc.truncate());
    // Negative viewport flips y: NDC y-up maps to pixel y downward.
    Vec2::new(
        (ndc.x * 0.5 + 0.5) * screen_w,
        (0.5 - ndc.y * 0.5) * screen_h,
    )
}

/// View frustum for AABB culling: 5 planes (left, right, bottom, top, near)
/// extracted from a view_proj matrix; far plane omitted (infinite reversed-Z).
/// Planes are stored as `Vec4(n.x, n.y, n.z, d)` with inward normals.
pub struct Frustum {
    planes: [Vec4; 5],
}

impl Frustum {
    /// Extracts the 5 frustum planes from a view_proj matrix.
    pub fn from_view_proj(m: &Mat4) -> Self {
        let t = m.transpose();
        let (r0, r1, r2, r3) = (t.x_axis, t.y_axis, t.z_axis, t.w_axis);
        let mut planes = [
            r3 + r0, // left:   -w <= x
            r3 - r0, // right:   x <= w
            r3 + r1, // bottom: -w <= y
            r3 - r1, // top:     y <= w
            r3 - r2, // near:    z <= w (reversed-Z)
        ];
        for p in &mut planes {
            // Normalize planes to world units for distance queries.
            *p /= p.truncate().length();
        }
        Self { planes }
    }

    /// Conservative test: checks the corner farthest along each plane normal.
    /// If any corner is outside a plane, the whole box is culled.
    /// The raw planes (normal.xyz, d), for the GPU cull's identical p-vertex test.
    pub(crate) fn planes(&self) -> [Vec4; 5] {
        self.planes
    }

    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let n = plane.truncate();
            let corner = Vec3::new(
                if n.x >= 0.0 { max.x } else { min.x },
                if n.y >= 0.0 { max.y } else { min.y },
                if n.z >= 0.0 { max.z } else { min.z },
            );
            if n.dot(corner) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f32 = 1280.0;
    const H: f32 = 720.0;
    const ASPECT: f32 = W / H;

    /// Camera at the origin looking down -Z, y-up, 60 degree vertical FOV.
    fn origin_cam() -> Camera3D {
        Camera3D {
            position: Vec3::ZERO,
            target: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::Y,
            fovy: 60.0,
            lens: Lens::Rectilinear,
        }
    }

    fn aabb_around(center: Vec3, half: f32) -> (Vec3, Vec3) {
        (center - Vec3::splat(half), center + Vec3::splat(half))
    }

    #[test]
    fn straight_ahead_maps_to_screen_center() {
        // Non-trivial camera so the view matrix actually does something.
        let cam = Camera3D {
            position: Vec3::new(1.0, 2.0, 3.0),
            target: Vec3::new(4.0, 5.0, 6.0),
            up: Vec3::Y,
            fovy: 60.0,
            lens: Lens::Rectilinear,
        };
        let dir = (cam.target - cam.position).normalize();
        let p = cam.position + dir * 7.0;
        let s = world_to_screen(p, &cam, W, H);
        assert!((s.x - W / 2.0).abs() < 1e-2, "sx = {}", s.x);
        assert!((s.y - H / 2.0).abs() < 1e-2, "sy = {}", s.y);
    }

    #[test]
    fn screen_axes_match_pixel_orientation() {
        let cam = origin_cam();
        // World-up point lands above center: smaller pixel y.
        let s = world_to_screen(Vec3::new(0.0, 1.0, -10.0), &cam, W, H);
        assert!(s.y < H / 2.0, "sy = {}", s.y);
        // World-right point lands right of center: larger pixel x.
        let s = world_to_screen(Vec3::new(1.0, 0.0, -10.0), &cam, W, H);
        assert!(s.x > W / 2.0, "sx = {}", s.x);
    }

    #[test]
    fn frustum_culls_behind_and_outside_fov() {
        let cam = origin_cam();
        let fr = Frustum::from_view_proj(&cam.view_proj(ASPECT));

        // 10 units in front: visible.
        let (mn, mx) = aabb_around(Vec3::new(0.0, 0.0, -10.0), 0.5);
        assert!(fr.intersects_aabb(mn, mx));

        // 10 units behind: culled.
        let (mn, mx) = aabb_around(Vec3::new(0.0, 0.0, 10.0), 0.5);
        assert!(!fr.intersects_aabb(mn, mx));

        // Far outside the horizontal FOV (half-width at z=-10 is ~10.26): culled.
        let (mn, mx) = aabb_around(Vec3::new(100.0, 0.0, -10.0), 0.5);
        assert!(!fr.intersects_aabb(mn, mx));
    }

    #[test]
    fn frustum_keeps_box_straddling_side_plane() {
        let cam = origin_cam();
        let fr = Frustum::from_view_proj(&cam.view_proj(ASPECT));
        // This box straddles a frustum edge; should not be culled.
        let mn = Vec3::new(9.0, -0.5, -10.5);
        let mx = Vec3::new(12.0, 0.5, -9.5);
        assert!(fr.intersects_aabb(mn, mx));
    }

    #[test]
    fn reversed_z_depth_range() {
        let cam = origin_cam();
        let vp = cam.view_proj(ASPECT);

        // Point exactly at the near plane projects to ndc z ~= 1.0.
        let clip = vp * Vec3::new(0.0, 0.0, -Z_NEAR).extend(1.0);
        let ndc_z = clip.z / clip.w;
        assert!((ndc_z - 1.0).abs() < 1e-4, "ndc z at near = {ndc_z}");

        // Very distant point projects to ndc z ~= 0.0.
        let clip = vp * Vec3::new(0.0, 0.0, -10000.0).extend(1.0);
        let ndc_z = clip.z / clip.w;
        assert!(ndc_z.abs() < 1e-4, "ndc z at 10000 = {ndc_z}");
    }

    // ---- warp (high-FOV nonlinear projection) ----

    #[test]
    fn identity_warp_matches_linear_projection() {
        // The regression anchor: ratio 0 must reproduce the pure-perspective
        // pixel exactly, so enabling the feature at rest changes nothing.
        let mut cam = origin_cam();
        cam.lens = Lens::Rectilinear;
        let linear = world_to_screen(Vec3::new(0.7, -0.3, -5.0), &cam, W, H);

        // Build the reference without any warp code path.
        let clip = cam.view_proj(ASPECT) * Vec3::new(0.7, -0.3, -5.0).extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let expect = Vec2::new((ndc.x * 0.5 + 0.5) * W, (0.5 - ndc.y * 0.5) * H);
        assert!((linear - expect).length() < 1e-4, "{linear} vs {expect}");
    }

    fn active_map(s: f32) -> WarpMap {
        WarpMap::from_lens(Lens::WideFov {
            strength: WarpStrength::new(s).unwrap(),
        })
    }

    #[test]
    fn warp_fixes_center_and_edges() {
        let wp = active_map(0.8);
        // View center is a fixed point.
        assert!(wp.warp_ndc(Vec2::ZERO).abs_diff_eq(Vec2::ZERO, 1e-6));
        // Horizontal edges are pinned (presented ±1 maps to source ±1).
        assert!((wp.warp_ndc(Vec2::new(1.0, 0.4)).x - 1.0).abs() < 1e-5);
        assert!((wp.warp_ndc(Vec2::new(-1.0, 0.4)).x + 1.0).abs() < 1e-5);
        // Vertical is left untouched (cylinder about the vertical axis).
        assert!((wp.warp_ndc(Vec2::new(0.5, 0.4)).y - 0.4).abs() < 1e-6);
    }

    #[test]
    fn warp_compresses_periphery() {
        // The forward map magnifies the center and shrinks equal-width columns as
        // they approach the edge, so a fixed interval near the periphery spans
        // fewer presented-NDC units than the same interval near center.
        let wp = active_map(1.0);
        let w =
            |a: f32, b: f32| wp.warp_ndc(Vec2::new(b, 0.0)).x - wp.warp_ndc(Vec2::new(a, 0.0)).x;
        let central = w(0.0, 0.1);
        let peripheral = w(0.85, 0.95);
        assert!(
            peripheral < central,
            "peripheral {peripheral} !< central {central}"
        );
    }

    #[test]
    fn warp_round_trips_through_inverse() {
        // Picking (forward) and the tonemap resample (inverse) must round-trip.
        let wp = active_map(1.2);
        for &p in &[
            Vec2::new(0.0, 0.0),
            Vec2::new(0.9, -0.7),
            Vec2::new(-0.6, 0.5),
            Vec2::new(0.999, 0.999),
        ] {
            let back = wp.unwarp_ndc(wp.warp_ndc(p));
            assert!(back.abs_diff_eq(p, 1e-4), "{p} -> {back}");
        }
    }

    #[test]
    fn widefov_picking_mirrors_the_tonemap_remap() {
        // world_to_screen must land on the pixel the GPU shows: projecting through
        // the widened source frustum then forward-warping is the exact inverse of
        // the tonemap resample (which unwarps presented-NDC to sample source-NDC).
        // So unwarp(presented_ndc) must equal the raw wide-projection source-NDC.
        let mut cam = origin_cam();
        cam.lens = Lens::WideFov {
            strength: WarpStrength::new(1.0).unwrap(),
        };
        let warp = WarpMap::from_lens(cam.lens);
        let source_aspect = Aspect(ASPECT).source(&warp);

        // A peripheral world point (well off-axis, so the warp is non-trivial).
        let p = Vec3::new(9.0, 2.0, -12.0);
        let pixel = world_to_screen(p, &cam, W, H);
        // Back to presented-NDC from the pixel.
        let presented = Vec2::new(pixel.x / W * 2.0 - 1.0, 1.0 - pixel.y / H * 2.0);
        // The source-NDC the frag would sample.
        let sampled_source = warp.unwarp_ndc(presented);
        // The source-NDC the wide projection actually produced.
        let clip = cam.view_proj(source_aspect.get()) * p.extend(1.0);
        let raw_source = (clip.truncate() / clip.w).truncate();
        assert!(
            sampled_source.abs_diff_eq(raw_source, 1e-4),
            "picking desync: sampled {sampled_source} vs projected {raw_source}"
        );
    }

    #[test]
    fn widefov_source_aspect_widens_frustum_to_keep_periphery() {
        // Step 2: the WideFov source frustum (widened aspect) must keep geometry
        // just outside the plain-rectilinear horizontal fov, or the extra periphery
        // the tonemap resample shows would be culled before it is drawn.
        let cam = origin_cam(); // vfov 60
        let window = Aspect(ASPECT);
        let base = Frustum::from_view_proj(&cam.view_proj(window.get()));
        let wide_map = WarpMap::from_lens(Lens::WideFov {
            strength: WarpStrength::new(1.0).unwrap(),
        });
        let source = window.source(&wide_map);
        assert!(source.get() > window.get());
        let wide = Frustum::from_view_proj(&cam.view_proj(source.get()));

        // A small box midway between the base and widened horizontal half-widths
        // at z=-10: outside the base fov, inside the widened source fov.
        let base_half_w = (ASPECT * (30.0_f32.to_radians()).tan()) * 10.0; // ~10.26
        let wide_half_w = base_half_w * wide_map.fov_scale(); // ~13.06
        let mid_x = 0.5 * (base_half_w + wide_half_w);
        let (mn, mx) = aabb_around(Vec3::new(mid_x, 0.0, -10.0), 0.2);
        assert!(
            !base.intersects_aabb(mn, mx),
            "base should cull the periphery"
        );
        assert!(
            wide.intersects_aabb(mn, mx),
            "wide source must keep the periphery"
        );
    }

    #[test]
    fn warp_strength_rejects_nonpositive_and_clamps() {
        assert!(WarpStrength::new(0.0).is_none());
        assert!(WarpStrength::new(-1.0).is_none());
        assert!(WarpStrength::new(f32::NAN).is_none());
        assert_eq!(WarpStrength::new(10.0).unwrap().get(), WarpStrength::MAX);
    }

    #[test]
    fn frustum_near_plane_culls_closer_than_z_near() {
        let cam = origin_cam();
        let fr = Frustum::from_view_proj(&cam.view_proj(ASPECT));
        // Tiny box 0.01 in front of the eye: entirely closer than Z_NEAR,
        // so the near plane must reject it.
        let (mn, mx) = aabb_around(Vec3::new(0.0, 0.0, -0.01), 0.001);
        assert!(!fr.intersects_aabb(mn, mx));
    }
}

// Derivation harness for the wide-FOV remap: validates the atan/tan formulas
// and their properties before embedding in WarpMap and the tonemap frag.
#[cfg(test)]
mod warp_derive {
    #[inline]
    fn warp_x(qx: f32, s: f32) -> f32 {
        (qx * s).atan() / s.atan()
    }
    #[inline]
    fn unwarp_x(px: f32, s: f32) -> f32 {
        (px * s.atan()).tan() / s
    }
    #[inline]
    fn fov_scale(s: f32) -> f32 {
        s / s.atan()
    }

    const STRENGTHS: [f32; 6] = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0];

    #[test]
    fn edges_and_center_are_fixed_points() {
        for &s in &STRENGTHS {
            assert!(warp_x(0.0, s).abs() < 1e-6, "center moved at s={s}");
            assert!((warp_x(1.0, s) - 1.0).abs() < 1e-5, "right edge at s={s}");
            assert!((warp_x(-1.0, s) + 1.0).abs() < 1e-5, "left edge at s={s}");
        }
    }

    #[test]
    fn inverse_sampling_stays_in_source_bounds_and_monotonic() {
        // The load-bearing claim for "no separate wider margin than the source":
        // every presented pixel samples a source qx inside [-1,1] (no starvation),
        // and the map is monotonic (no folding).
        for &s in &STRENGTHS {
            let mut prev = unwarp_x(-1.0, s);
            for i in -100..=100 {
                let px = i as f32 / 100.0;
                let qx = unwarp_x(px, s);
                assert!(qx.abs() <= 1.0 + 1e-6, "s={s} px={px} escaped: {qx}");
                assert!(qx > prev - 1e-6, "s={s} not monotonic at px={px}");
                prev = qx;
            }
        }
    }

    #[test]
    fn maps_round_trip() {
        for &s in &STRENGTHS {
            for i in -95..=95 {
                let px = i as f32 / 100.0;
                let back = warp_x(unwarp_x(px, s), s);
                assert!((back - px).abs() < 1e-4, "s={s} px={px} -> {back}");
            }
        }
    }

    #[test]
    fn center_magnification_equals_fov_scale() {
        // Center magnification (finite difference) must equal fov_scale.
        for &s in &STRENGTHS {
            let h = 1e-4;
            let deriv = (warp_x(h, s) - warp_x(-h, s)) / (2.0 * h);
            assert!(
                (deriv - fov_scale(s)).abs() < 1e-2,
                "s={s}: center slope {deriv} != fov_scale {}",
                fov_scale(s)
            );
        }
    }

    #[test]
    fn print_widening_and_compression_curves() {
        // Not an assertion — records the numbers that size the FOV->strength ramp
        // and the periphery supersample. Run with `cargo test -- --nocapture`.
        let vfov_deg = 70.0_f32;
        let aspect = 16.0 / 9.0;
        let vhalf = (vfov_deg * 0.5).to_radians();
        let base_hhalf = (aspect * vhalf.tan()).atan();
        eprintln!(
            "vfov={vfov_deg}  aspect={aspect}  base hfov={:.1}",
            2.0 * base_hhalf.to_degrees()
        );
        for &s in &STRENGTHS {
            let fs = fov_scale(s);
            let src_hhalf = (fs * base_hhalf.tan()).atan();
            // peripheral compression: source-NDC span mapped into the outer 5% of
            // the presented edge -> how many source texels pack into one edge pixel.
            let outer = unwarp_x(1.0, s) - unwarp_x(0.95, s);
            let inner = unwarp_x(0.05, s) - unwarp_x(0.0, s);
            eprintln!(
                "s={s:>3}  fov_scale={fs:.3}  src_hfov={:.1}  edge/center source-density={:.2}x",
                2.0 * src_hhalf.to_degrees(),
                outer / inner
            );
        }
    }
}
