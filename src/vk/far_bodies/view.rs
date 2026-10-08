//! The sky pass's view of the far bodies: [`FarView`], the orthonormal view
//! basis, and the frustum tests that drop bodies whose cone cannot meet it.

use super::cones::mapped_horizon_half;
use crate::far_body::{FarBody, FarShape};

/// What the sky pass needs to drop bodies that cannot meet this frame's view.
/// `view_proj` is the clean (unjittered) camera-relative matrix. `px_max` is an
/// upper bound on the pixel angle, with a factor of two over the projection.
/// `width` and `height` are the render extent the sky pass draws, the same one
/// `px_max` was measured against.
#[derive(Clone, Copy)]
pub(crate) struct FarView {
    pub view_proj: glam::Mat4,
    pub px_max: f32,
    pub width: u32,
    pub height: u32,
}

/// Vertical pixel angle `2 * tan(fovy/2) / height`, the horizontal equivalent
/// from the projection, the max of those, times two.
pub(super) fn px_max(fovy_tan_half: f32, view_proj: glam::Mat4, width: u32, height: u32) -> f32 {
    let h = height.max(1) as f32;
    let w = width.max(1) as f32;
    let vert = 2.0 * fovy_tan_half / h;
    // Row 0 of `proj * view` has length `1 / tan(fovx/2)` (the view is rigid).
    let row0 = view_proj.transpose().x_axis.truncate().length().max(1e-20);
    let horiz = 2.0 / (row0 * w);
    vert.max(horiz) * 2.0
}

pub(crate) fn far_view(
    fovy_tan_half: f32,
    view_proj: glam::Mat4,
    width: u32,
    height: u32,
) -> FarView {
    let width = width.max(1);
    let height = height.max(1);
    FarView {
        view_proj,
        px_max: px_max(fovy_tan_half, view_proj, width, height),
        width,
        height,
    }
}

/// Inward normals of the four side planes, for directions through the eye.
/// Gribb-Hartmann on the rows of `view_proj` (row3 ± row0, row3 ± row1); the
/// translation drops out because each side plane contains the camera.
pub(super) fn side_normals(view_proj: glam::Mat4) -> [glam::Vec3; 4] {
    let rows = view_proj.transpose();
    let (r0, r1, r3) = (rows.x_axis, rows.y_axis, rows.w_axis);
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1].map(|plane| {
        let n = plane.truncate();
        let len = n.length();
        if len > 1e-20 {
            n / len
        } else {
            glam::Vec3::ZERO
        }
    })
}

/// `bound` is the cone's sine, or `-1` to always keep the body. A cone of sine
/// `a = min(bound + 3 px, 1)` lies outside an inward plane `n` when
/// `dot(n, dir) < -a`. `a >= 1` is a hemisphere (`f > 0` in the shader, since
/// `|ray × dir|` cannot exceed 1), not the whole sky: keep it only when that
/// hemisphere meets the frustum.
fn cone_in_view(dir: glam::Vec3, bound: f32, view: &FarView) -> bool {
    if !(bound >= 0.0) || !bound.is_finite() {
        return true;
    }
    let len2 = dir.length_squared();
    if !(len2 > 0.0) || !len2.is_finite() {
        return true;
    }
    let dir = dir / len2.sqrt();
    let a_sin = (bound + 3.0 * view.px_max).min(1.0);
    if !a_sin.is_finite() {
        return true;
    }
    if a_sin >= 1.0 {
        return hemisphere_meets_frustum(dir, view.view_proj);
    }
    for n in side_normals(view.view_proj) {
        if n.length_squared() == 0.0 {
            continue;
        }
        if n.dot(dir) < -a_sin {
            return false;
        }
    }
    true
}

/// The open hemisphere `dot(ray, dir) > 0` meets the view if any frustum corner
/// does. The four corners are the extreme rays of the side-plane cone, and
/// `dot` is linear, so the max over the frustum is a corner. A hair of slack
/// keeps a graze the pixel test might still accept.
fn hemisphere_meets_frustum(dir: glam::Vec3, view_proj: glam::Mat4) -> bool {
    let Some(inv) = view_proj.try_inverse() else {
        return true;
    };
    for y in [-1.0f32, 1.0] {
        for x in [-1.0f32, 1.0] {
            let ray = (inv * glam::Vec4::new(x, y, 0.0, 1.0)).truncate();
            let len2 = ray.length_squared();
            if !(len2 > 0.0) || !ray.is_finite() {
                return true;
            }
            if dir.dot(ray / len2.sqrt()) > -1.0e-4 {
                return true;
            }
        }
    }
    false
}

/// Orthonormal view axes and the projection scales, read off `view_proj`.
/// `right`/`up`/`back` are world-space. A view-space direction is
/// `(right · d, up · d, back · d)`, with `back` the camera's +Z (the camera
/// looks down −Z). `focal_x` is `1 / tan(fovx/2)`, `focal_y` is `1 / tan(fovy/2)`.
pub(super) struct ViewBasis {
    right: glam::Vec3,
    up: glam::Vec3,
    pub(super) back: glam::Vec3,
    pub(super) focal_x: f32,
    pub(super) focal_y: f32,
}

impl ViewBasis {
    pub(super) fn from_view_proj(view_proj: glam::Mat4) -> Option<Self> {
        let rows = view_proj.transpose();
        let right = rows.x_axis.truncate();
        let up = rows.y_axis.truncate();
        // Row 3's xyz is the forward direction (camera looks down −Z).
        let forward = rows.w_axis.truncate();
        let focal_x = right.length();
        let focal_y = up.length();
        let forward_len = forward.length();
        if !(focal_x.is_finite() && focal_y.is_finite() && forward_len.is_finite())
            || focal_x <= 1e-12
            || focal_y <= 1e-12
            || forward_len <= 1e-12
        {
            return None;
        }
        Some(Self {
            right: right / focal_x,
            up: up / focal_y,
            back: -forward / forward_len,
            focal_x,
            focal_y,
        })
    }

    pub(super) fn to_view(&self, world: glam::Vec3) -> glam::Vec3 {
        glam::Vec3::new(
            self.right.dot(world),
            self.up.dot(world),
            self.back.dot(world),
        )
    }

    /// View-space ray through NDC `(x, y)`. y is up, matching the sky unproject
    /// (`inv_view_proj * (ndc, 0, 1)`).
    pub(super) fn ray_ndc(&self, ndc_x: f32, ndc_y: f32) -> glam::Vec3 {
        glam::Vec3::new(ndc_x / self.focal_x, ndc_y / self.focal_y, -1.0).normalize()
    }
}

/// Maximum of `dot(unit ray, dir)` on the view's direction cone.
///
/// `dir` is unit. The maximum on a convex spherical polygon is 1 when `dir`
/// lies inside, otherwise on an edge. An edge's maximum is a corner or the
/// point where the great circle passes closest to `dir`, so the four corners
/// alone miss a graze through the middle of a side. `None` keeps the body.
fn max_dir_dot_on_frustum(dir: glam::Vec3, view_proj: glam::Mat4) -> Option<f32> {
    let inv = view_proj.try_inverse()?;
    let mut corners = [glam::Vec3::ZERO; 4];
    let mut i = 0;
    for y in [-1.0f32, 1.0] {
        for x in [-1.0f32, 1.0] {
            let ray = (inv * glam::Vec4::new(x, y, 0.0, 1.0)).truncate();
            let len2 = ray.length_squared();
            if !(len2 > 0.0) || !ray.is_finite() {
                return None;
            }
            corners[i] = ray / len2.sqrt();
            i += 1;
        }
    }
    let clip = view_proj * glam::Vec4::new(dir.x, dir.y, dir.z, 0.0);
    if clip.is_finite() && clip.w > 0.0 {
        let slack = 1.0e-4 * clip.w.abs();
        if clip.x.abs() <= clip.w + slack && clip.y.abs() <= clip.w + slack {
            return Some(1.0);
        }
    }
    let mut best = f32::NEG_INFINITY;
    for corner in corners {
        best = best.max(dir.dot(corner));
    }
    // NDC order from the loops above: (-1,-1), (1,-1), (-1,1), (1,1).
    for (ia, ib) in [(0usize, 1usize), (2, 3), (0, 2), (1, 3)] {
        let (a, b) = (corners[ia], corners[ib]);
        let n = a.cross(b);
        let nlen2 = n.length_squared();
        if !(nlen2 > 1.0e-20) {
            continue;
        }
        let n = n / nlen2.sqrt();
        let proj = dir - n * dir.dot(n);
        let plen2 = proj.length_squared();
        if !(plen2 > 1.0e-20) {
            continue;
        }
        let u = proj / plen2.sqrt();
        let ab = a.cross(b);
        let on_arc = a.cross(u).dot(ab) >= -1.0e-5 && u.cross(b).dot(ab) >= -1.0e-5;
        if on_arc {
            best = best.max(dir.dot(u));
        }
    }
    best.is_finite().then_some(best)
}

/// The widened horizon cone meets the view. A missing inverse keeps the body.
fn horizon_cone_meets_frustum(dir: glam::Vec3, cos_alpha: f32, view_proj: glam::Mat4) -> bool {
    let len2 = dir.length_squared();
    if !(len2 > 0.0) || !dir.is_finite() || !cos_alpha.is_finite() {
        return true;
    }
    let dir = dir / len2.sqrt();
    match max_dir_dot_on_frustum(dir, view_proj) {
        Some(max_dot) => max_dot >= cos_alpha - 1.0e-4,
        None => true,
    }
}

/// Frustum keep. A mapped body with `horizon < 1` uses the widened horizon
/// cone, including when [`cone_bound`] is the per-pixel sentinel. `horizon >= 1`
/// and every other shape keep the pixel-cone test.
///
/// [`cone_bound`]: super::cones::cone_bound
pub(super) fn body_meets_view(body: &FarBody, bound: f32, view: &FarView) -> bool {
    if let FarShape::Mapped { horizon, air, .. } = body.shape {
        if horizon < 1.0 {
            return match mapped_horizon_half(horizon, air, body.distance, view.px_max) {
                None => true,
                Some((_, cos_alpha)) => {
                    horizon_cone_meets_frustum(body.dir, cos_alpha, view.view_proj)
                }
            };
        }
    }
    cone_in_view(body.dir, bound, view)
}
