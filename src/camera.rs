//! Camera, projection, and view-frustum utilities.
//!
//! The engine renders with Vulkan reversed-Z (depth cleared to 0.0, compare
//! GREATER_OR_EQUAL) using `Mat4::perspective_infinite_reverse_rh`, and flips
//! Y via a negative viewport height so NDC is GL-style y-up. `Camera3D`
//! mirrors raylib's camera (fovy in degrees), `world_to_screen` projects with
//! the matrices used for rendering, and `Frustum` extracts frustum planes for
//! AABB culling.

use glam::{Mat4, Vec2, Vec3, Vec4};

/// Near plane distance shared by rendering and culling.
pub const Z_NEAR: f32 = 0.05;

/// Nonlinear projection warp (De Carpentier-family adaptive-cylindrical).
/// Applied per-vertex to NDC xy *after* the linear `view_proj` projection: it
/// pins the horizontal screen edges (nx = ±1) and compresses interior columns
/// toward center, so peripheral geometry occupies fewer pixels — cancelling the
/// wide-FOV corner/edge stretch of a plain perspective projection.
///
/// Forward map (horizontal only): `nx' = atan(nx * s) / atan(s)`, `s =
/// cylindrical_ratio`.
/// - At `s = 0` the warp is the identity, reproducing the pure-perspective image
///   bit-for-bit ([`WarpParams::IDENTITY`], the regression anchor).
/// - At `nx = 0` the output is `nx' = 0` — the view center is a fixed point.
/// - At `nx = ±1` the output is `nx' = ±1` — screen edges are pinned. The
///   silhouette is therefore unchanged and the warp only ever pulls points
///   *inward*, so the existing linear `view_proj` frustum already conservatively
///   bounds the warped image: culling needs no separate matrix (see [`Frustum`]).
///
/// Inverse map (used by the sky pass, which reconstructs world rays from NDC):
/// `nx = tan(nx' * atan(s)) / s`.
///
/// Vertical (`ny`) is left rectilinear — a cylinder about the vertical axis.
/// This is the De Carpentier compromise (horizontal lines still bow slightly
/// off-center) chosen deliberately for flat-monitor comfort over architectural
/// vertical-line straightness.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WarpParams {
    /// Cylindrical blend strength `s`. 0.0 = pure perspective (identity).
    /// Comfortable wide-FOV range is roughly 0.3..1.5; higher compresses the
    /// periphery more.
    pub cylindrical_ratio: f32,
    /// Precomputed `atan(cylindrical_ratio)` so the shader (and `warp_ndc`)
    /// avoid a per-vertex `atan` on this frame-constant. 0.0 in the identity
    /// case, where both maps short-circuit to the identity.
    pub atan_ratio: f32,
}

impl WarpParams {
    /// Pure perspective: the warp is the identity, reproducing today's image.
    pub const IDENTITY: Self = Self {
        cylindrical_ratio: 0.0,
        atan_ratio: 0.0,
    };

    /// Builds params from a cylindrical strength (clamped non-negative),
    /// precomputing `atan_ratio`. `0.0` yields [`WarpParams::IDENTITY`].
    pub fn new(cylindrical_ratio: f32) -> Self {
        let s = cylindrical_ratio.max(0.0);
        if s == 0.0 {
            return Self::IDENTITY;
        }
        Self {
            cylindrical_ratio: s,
            atan_ratio: s.atan(),
        }
    }

    /// True when the warp is the identity (pure perspective).
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.cylindrical_ratio == 0.0
    }

    /// Forward warp of an NDC point. Mirrors the Slang `applyWarp`; any change
    /// here must be echoed there or CPU picking desyncs from the GPU image.
    #[inline]
    pub fn warp_ndc(&self, ndc: Vec2) -> Vec2 {
        if self.is_identity() {
            return ndc;
        }
        Vec2::new((ndc.x * self.cylindrical_ratio).atan() / self.atan_ratio, ndc.y)
    }

    /// Inverse warp of an NDC point. Mirrors the Slang `inverseWarpNdc` and
    /// round-trips [`WarpParams::warp_ndc`] to float precision; the sky pass
    /// uses it to reconstruct the same ray the warped terrain shows per pixel.
    #[inline]
    pub fn unwarp_ndc(&self, ndc: Vec2) -> Vec2 {
        if self.is_identity() {
            return ndc;
        }
        Vec2::new((ndc.x * self.atan_ratio).tan() / self.cylindrical_ratio, ndc.y)
    }
}

/// Perspective camera, raylib-parity: `fovy` is the vertical field of view
/// in DEGREES. `warp` adds an optional nonlinear high-FOV projection on top of
/// the linear perspective; [`WarpParams::IDENTITY`] leaves the image unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Camera3D {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fovy: f32,
    pub warp: WarpParams,
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
    // Clamp aspect denominator to avoid zero height.
    let clip = cam.view_proj(screen_w / screen_h.max(1.0)) * p.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    // Same nonlinear warp the vertex shaders apply, so overlays/picking track
    // the warped geometry. Identity warp leaves `ndc.xy` untouched.
    let ndc = cam.warp.warp_ndc(ndc.truncate());
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
            warp: WarpParams::IDENTITY,
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
            warp: WarpParams::IDENTITY,
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
        cam.warp = WarpParams::IDENTITY;
        let linear = world_to_screen(Vec3::new(0.7, -0.3, -5.0), &cam, W, H);

        // Build the reference without any warp code path.
        let clip = cam.view_proj(ASPECT) * Vec3::new(0.7, -0.3, -5.0).extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let expect = Vec2::new((ndc.x * 0.5 + 0.5) * W, (0.5 - ndc.y * 0.5) * H);
        assert!((linear - expect).length() < 1e-4, "{linear} vs {expect}");
    }

    #[test]
    fn warp_fixes_center_and_edges() {
        let wp = WarpParams::new(0.8);
        // View center is a fixed point.
        assert!(wp.warp_ndc(Vec2::ZERO).abs_diff_eq(Vec2::ZERO, 1e-6));
        // Horizontal edges are pinned (silhouette unchanged → cull stays valid).
        assert!((wp.warp_ndc(Vec2::new(1.0, 0.4)).x - 1.0).abs() < 1e-5);
        assert!((wp.warp_ndc(Vec2::new(-1.0, 0.4)).x + 1.0).abs() < 1e-5);
        // Vertical is left untouched (cylinder about the vertical axis).
        assert!((wp.warp_ndc(Vec2::new(0.5, 0.4)).y - 0.4).abs() < 1e-6);
    }

    #[test]
    fn warp_compresses_periphery_and_stays_in_silhouette() {
        // The de-stretch is a *relative* effect: the map magnifies the center
        // and shrinks equal-width columns as they approach the edge. So a fixed
        // interval near the periphery must project to fewer NDC units than the
        // same interval near center.
        let wp = WarpParams::new(1.0);
        let w = |a: f32, b: f32| wp.warp_ndc(Vec2::new(b, 0.0)).x - wp.warp_ndc(Vec2::new(a, 0.0)).x;
        let central = w(0.0, 0.1);
        let peripheral = w(0.85, 0.95);
        assert!(peripheral < central, "peripheral {peripheral} !< central {central}");

        // No point escapes the linear silhouette [-1, 1]: the map is monotonic
        // and pins ±1, so nothing off-screen is pulled on-screen. This is why
        // `Frustum::from_view_proj` on the linear matrix stays conservative and
        // needs no widening.
        let mut prev = wp.warp_ndc(Vec2::new(-1.0, 0.0)).x;
        for i in -99..=100 {
            let nx = i as f32 / 100.0;
            let out = wp.warp_ndc(Vec2::new(nx, 0.0)).x;
            assert!(out.abs() <= 1.0 + 1e-6, "nx={nx} escaped: {out}");
            assert!(out > prev - 1e-6, "not monotonic at nx={nx}");
            prev = out;
        }
    }

    #[test]
    fn warp_round_trips_through_inverse() {
        // The sky pass relies on unwarp(warp(x)) = x to reconstruct rays.
        let wp = WarpParams::new(1.2);
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
    fn frustum_near_plane_culls_closer_than_z_near() {
        let cam = origin_cam();
        let fr = Frustum::from_view_proj(&cam.view_proj(ASPECT));
        // Tiny box 0.01 in front of the eye: entirely closer than Z_NEAR,
        // so the near plane must reject it.
        let (mn, mx) = aabb_around(Vec3::new(0.0, 0.0, -0.01), 0.001);
        assert!(!fr.intersects_aabb(mn, mx));
    }
}
