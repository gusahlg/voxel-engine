//! The screen tile grid: its layout, the view-space tile cones, the per-tile
//! body masks, and the three tile-index runs the tile-quad draws read.

use super::cones::{add_angles, mapped_horizon_half};
use super::table::{FarBodyGpu, FarTableGpu, MAX_FAR_TILES};
use super::view::{FarView, ViewBasis};
use crate::far_body::MAX_FAR_BODIES;

/// Preferred tile, in pixels. Coarsens when the grid would exceed [`MAX_FAR_TILES`].
const TILE_FINE_PX: u32 = 64;
const TILE_COARSE_PX: u32 = 128;

/// `(tile size, tiles_x, tiles_y)`. 64 px unless that grid exceeds
/// [`MAX_FAR_TILES`], then 128, then coarser powers of two. The chosen size is
/// what the shader reads from the header, so the mask never overruns the array.
pub(super) fn tile_layout(width: u32, height: u32) -> (u32, u32, u32) {
    let width = width.max(1);
    let height = height.max(1);
    let mut tile = TILE_FINE_PX;
    loop {
        let tiles_x = width.div_ceil(tile);
        let tiles_y = height.div_ceil(tile);
        if tiles_x as u64 * tiles_y as u64 <= MAX_FAR_TILES as u64 {
            return (tile, tiles_x, tiles_y);
        }
        if tile >= width.max(height) {
            return (width.max(height), 1, 1);
        }
        let next = if tile < TILE_COARSE_PX {
            TILE_COARSE_PX
        } else {
            tile.saturating_mul(2)
        };
        tile = next.min(width.max(height)).max(tile + 1);
    }
}

/// 1.25 on the angular pixel bound, then two pixels of pad.
const RECT_SAFETY: f32 = 1.25;
const RECT_PAD_PX: f32 = 2.0;
/// A screen rect is not trustworthy once the cone reaches within 10° of the
/// view plane (80° from the view axis). Those cones take the angular test.
const VIEW_PLANE_LIMIT_RAD: f32 = 80.0 * std::f32::consts::PI / 180.0;
/// Sine past which the disc-on-screen rect is a poor fit (`asin(0.5) = 30°`).
/// Wider cones, including a hemisphere at sine 1, take the angular test.
const LARGE_CONE_SINE: f32 = 0.5;
/// Slack on `dot >= cos(a_body + a_tile)` so a tile on the boundary is kept.
const ANGULAR_COS_EPS: f32 = 1.0e-5;
/// Focal lengths are rotation-invariant in exact arithmetic. A relative slop
/// absorbs the ulp wobble of `|row|` as the camera turns, and is far inside
/// one pixel, so a reused tile frame stays conservative.
const FOCAL_REL_EPS: f32 = 2.0e-5;

enum TileCover {
    /// Bit set in every tile.
    All,
    /// Fully off the render extent. The frustum cull should already have dropped it.
    None,
    /// Inclusive tile coordinates. Small cones only.
    Rect {
        tx0: u32,
        ty0: u32,
        tx1: u32,
        ty1: u32,
    },
    /// Cone too wide, or too close to the view plane, for the rect.
    Angular,
}

/// Framebuffer pixel (origin top-left, y down) to GL NDC (y up).
pub(super) fn pixel_ndc(x: f32, y: f32, width: f32, height: f32) -> (f32, f32) {
    ((x / width) * 2.0 - 1.0, 1.0 - (y / height) * 2.0)
}

fn focals_match(a: f32, b: f32) -> bool {
    if !(a.is_finite() && b.is_finite()) || a <= 0.0 || b <= 0.0 {
        return false;
    }
    (a - b).abs() <= a.max(b) * FOCAL_REL_EPS
}

/// One tile's view-space centre ray and the sine/cosine of its angular radius
/// (max angle from the centre to the four corner rays, plus one pixel).
pub(super) struct TileSample {
    pub(super) centre: glam::Vec3,
    pub(super) sin_r: f32,
    pub(super) cos_r: f32,
}

/// View-space tile cones for one projection. The directions do not depend on
/// the camera orientation, so the ring reuses them until the projection, the
/// render extent, or the tile size changes.
pub(super) struct TileFrames {
    focal_x: f32,
    focal_y: f32,
    width: u32,
    height: u32,
    tile_px: u32,
    pub(super) samples: Vec<TileSample>,
}

impl TileFrames {
    pub(super) fn empty() -> Self {
        Self {
            focal_x: 0.0,
            focal_y: 0.0,
            width: 0,
            height: 0,
            tile_px: 0,
            samples: Vec::new(),
        }
    }

    /// `view`'s basis when these frames were built for its projection,
    /// render extent and tile size.
    fn matched_basis(&self, view: &FarView) -> Option<ViewBasis> {
        let (tile_px, tiles_x, tiles_y) = tile_layout(view.width, view.height);
        let expect = tiles_x as usize * tiles_y as usize;
        if self.width != view.width
            || self.height != view.height
            || self.tile_px != tile_px
            || self.samples.len() != expect
        {
            return None;
        }
        let basis = ViewBasis::from_view_proj(view.view_proj)?;
        (focals_match(self.focal_x, basis.focal_x) && focals_match(self.focal_y, basis.focal_y))
            .then_some(basis)
    }

    #[cfg(test)]
    pub(super) fn matches(&self, view: &FarView) -> bool {
        self.matched_basis(view).is_some()
    }

    /// These frames paired with `view`, when they match it.
    #[cfg(test)]
    pub(super) fn view<'a>(&'a self, view: &'a FarView) -> Option<TileView<'a>> {
        let basis = self.matched_basis(view)?;
        Some(TileView {
            frames: self,
            view,
            basis,
        })
    }

    /// Rebuild unless the frames match `view`, then pair them with it. While
    /// the projection holds, this is the frame's one match test. `None`
    /// when `view` has no basis (or the rebuild still does not match).
    pub(super) fn sync<'a>(&'a mut self, view: &'a FarView) -> Option<TileView<'a>> {
        let basis = match self.matched_basis(view) {
            Some(basis) => basis,
            None => {
                *self = Self::build(view).unwrap_or_else(Self::empty);
                self.matched_basis(view)?
            }
        };
        Some(TileView {
            frames: self,
            view,
            basis,
        })
    }

    /// `true` when the samples were derived again.
    #[cfg(test)]
    pub(super) fn rebuild_if_changed(&mut self, view: &FarView) -> bool {
        if self.matches(view) {
            return false;
        }
        *self = Self::build(view).unwrap_or_else(Self::empty);
        true
    }

    pub(super) fn build(view: &FarView) -> Option<Self> {
        let basis = ViewBasis::from_view_proj(view.view_proj)?;
        let (tile_px, tiles_x, tiles_y) = tile_layout(view.width, view.height);
        let width = view.width.max(1);
        let height = view.height.max(1);
        let wf = width as f32;
        let hf = height as f32;
        let ray_at = |x: f32, y: f32| {
            let (nx, ny) = pixel_ndc(x, y, wf, hf);
            basis.ray_ndc(nx, ny)
        };
        let mut samples = Vec::with_capacity(tiles_x as usize * tiles_y as usize);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let x0 = (tx * tile_px) as f32;
                let y0 = (ty * tile_px) as f32;
                let x1 = ((tx + 1) * tile_px).min(width) as f32;
                let y1 = ((ty + 1) * tile_px).min(height) as f32;
                let centre = ray_at(0.5 * (x0 + x1), 0.5 * (y0 + y1));
                let corners = [
                    ray_at(x0, y0),
                    ray_at(x1, y0),
                    ray_at(x0, y1),
                    ray_at(x1, y1),
                ];
                let mut cos_corner = 1.0f32;
                for corner in corners {
                    cos_corner = cos_corner.min(centre.dot(corner));
                }
                // One pixel past each corner, so a ray at the pixel edge and the
                // shader's derivative step stay inside the radius.
                let mut cos_px = 1.0f32;
                let steps = [
                    (x0, y0, -1.0, 0.0),
                    (x0, y0, 0.0, -1.0),
                    (x1, y0, 1.0, 0.0),
                    (x1, y0, 0.0, -1.0),
                    (x0, y1, -1.0, 0.0),
                    (x0, y1, 0.0, 1.0),
                    (x1, y1, 1.0, 0.0),
                    (x1, y1, 0.0, 1.0),
                ];
                for (x, y, dx, dy) in steps {
                    let a = ray_at(x, y);
                    let b = ray_at(x + dx, y + dy);
                    cos_px = cos_px.min(a.dot(b));
                }
                let (sin_r, cos_r) =
                    add_angles(cos_corner.clamp(-1.0, 1.0), cos_px.clamp(-1.0, 1.0));
                samples.push(TileSample {
                    centre,
                    sin_r,
                    cos_r,
                });
            }
        }
        Some(Self {
            focal_x: basis.focal_x,
            focal_y: basis.focal_y,
            width,
            height,
            tile_px,
            samples,
        })
    }
}

/// Tile frames that match one view, with that view's basis. Built once per
/// frame by [`TileFrames::sync`], so the passes after the pack neither test
/// the match again nor rebuild the basis.
pub(super) struct TileView<'a> {
    pub(super) frames: &'a TileFrames,
    pub(super) view: &'a FarView,
    pub(super) basis: ViewBasis,
}

/// `angle(tile centre, dir) <= a_body + a_tile`, compared in cosines.
/// `sin_body` / `cos_body` are the body's drawable half-angle. `cos_body` is
/// negative when the half-angle is wider than a hemisphere.
pub(super) fn tile_within(
    tile: &TileSample,
    dir_view: glam::Vec3,
    sin_body: f32,
    cos_body: f32,
) -> bool {
    if tile.sin_r > 1.0 {
        return true;
    }
    let sin_sum = sin_body * tile.cos_r + cos_body * tile.sin_r;
    let cos_sum = cos_body * tile.cos_r - sin_body * tile.sin_r;
    // Sum past π: every direction is inside the cone.
    if sin_sum < 0.0 {
        return true;
    }
    dir_view.dot(tile.centre) + ANGULAR_COS_EPS >= cos_sum
}

/// The whole tile cone lies strictly inside the body's half-angle.
///
/// `angle(centre, dir) + a_tile < a_body`. A tile that only touches the
/// boundary stays out, so the limb and the silhouette stay at 1×1.
pub(super) fn tile_strictly_inside(
    tile: &TileSample,
    dir_view: glam::Vec3,
    sin_body: f32,
    cos_body: f32,
) -> bool {
    if tile.sin_r > 1.0 || !(tile.sin_r.is_finite() && tile.cos_r.is_finite()) {
        return false;
    }
    let sin_g = sin_body * tile.cos_r - cos_body * tile.sin_r;
    let cos_g = cos_body * tile.cos_r + sin_body * tile.sin_r;
    if !(sin_g > 0.0) || !sin_g.is_finite() || !cos_g.is_finite() {
        return false;
    }
    dir_view.dot(tile.centre) > cos_g + ANGULAR_COS_EPS
}

/// Screen tiles touched by kept body `dir`/`bound`. `bound` is the cone sine,
/// or `-1` to keep the body everywhere. The drawable angular radius is
/// `asin(min(bound + 3 px_max, 1))` on the unit centre direction. A wide cone,
/// or one that reaches within 10° of the view plane, is [`TileCover::Angular`]:
/// the screen rect is not a bound there.
fn tile_cover(
    dir: glam::Vec3,
    bound: f32,
    view: &FarView,
    tile_px: u32,
    tiles_x: u32,
    tiles_y: u32,
) -> TileCover {
    if !(bound >= 0.0) || !bound.is_finite() {
        return TileCover::All;
    }
    let len2 = dir.length_squared();
    if !(len2 > 0.0) || !len2.is_finite() {
        return TileCover::All;
    }
    let dir = dir / len2.sqrt();
    let lim = bound + 3.0 * view.px_max;
    if !lim.is_finite() {
        return TileCover::All;
    }
    let a_sin = lim.min(1.0);
    let a = a_sin.asin();
    // Direction through the eye: translation in `view_proj` drops out at w = 0.
    // `clip.w` is cos(theta) for a unit direction (camera looks down −Z).
    let clip = view.view_proj * glam::Vec4::new(dir.x, dir.y, dir.z, 0.0);
    let theta = if clip.is_finite() {
        clip.w.clamp(-1.0, 1.0).acos()
    } else {
        f32::NAN
    };
    let cos_outer = (theta + a).cos();
    let wide = bound >= LARGE_CONE_SINE || a_sin >= 1.0;
    let near_plane = !theta.is_finite()
        || theta + a >= VIEW_PLANE_LIMIT_RAD
        || !(cos_outer > 0.0)
        || !(clip.w > 0.0);
    if wide || near_plane {
        return TileCover::Angular;
    }
    let rows = view.view_proj.transpose();
    let f_px_x = 0.5 * view.width as f32 * rows.x_axis.truncate().length();
    let f_px_y = 0.5 * view.height as f32 * rows.y_axis.truncate().length();
    if !(f_px_x.is_finite() && f_px_y.is_finite()) || f_px_x <= 0.0 || f_px_y <= 0.0 {
        return TileCover::All;
    }
    // Pixel radius of an angular disc of radius `a` at angle `theta` from the
    // view axis. d(tan)/dθ = sec², largest at the outer edge. Per-axis focal
    // length, then 1.25 and 2 px.
    let stretch = a / (cos_outer * cos_outer);
    let rx = RECT_SAFETY * stretch * f_px_x + RECT_PAD_PX;
    let ry = RECT_SAFETY * stretch * f_px_y + RECT_PAD_PX;
    if !(rx.is_finite() && ry.is_finite()) {
        return TileCover::All;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    if !(ndc_x.is_finite() && ndc_y.is_finite()) {
        return TileCover::All;
    }
    let w = view.width as f32;
    let h = view.height as f32;
    // Negative viewport: NDC y-up maps to framebuffer y-down, matching SV_Position.
    let cx = (ndc_x * 0.5 + 0.5) * w;
    let cy = (0.5 - ndc_y * 0.5) * h;
    if !(cx.is_finite() && cy.is_finite()) {
        return TileCover::All;
    }
    let left = cx - rx;
    let right = cx + rx;
    let top = cy - ry;
    let bot = cy + ry;
    if right < 0.0 || bot < 0.0 || left >= w || top >= h {
        return TileCover::None;
    }
    let x0 = left.max(0.0).floor() as u32;
    let y0 = top.max(0.0).floor() as u32;
    let x1 = (right.floor().max(0.0) as u32).min(view.width - 1);
    let y1 = (bot.floor().max(0.0) as u32).min(view.height - 1);
    if x0 > x1 || y0 > y1 {
        return TileCover::None;
    }
    let tx0 = (x0 / tile_px).min(tiles_x - 1);
    let tx1 = (x1 / tile_px).min(tiles_x - 1);
    let ty0 = (y0 / tile_px).min(tiles_y - 1);
    let ty1 = (y1 / tile_px).min(tiles_y - 1);
    TileCover::Rect { tx0, ty0, tx1, ty1 }
}

/// Fill `header.yzw`, the live mask prefix and the tile lists, and return
/// the runs. No view leaves the tile header at zero; the shader then walks
/// every kept body. `tiles` are the ring's view-space tile frames when they
/// match this projection. `cull` off (`VOXEL_FAR_CULL=0`) sets every kept
/// body in every live tile, the mapped horizon cone included.
pub(super) fn stamp_tiles(
    table: &mut FarTableGpu,
    view: Option<&FarView>,
    tiles: Option<&TileView>,
    cull: bool,
) -> TileRuns {
    let Some(view) = view else {
        return TileRuns::default();
    };
    let (tile_px, tiles_x, tiles_y) = tile_layout(view.width, view.height);
    table.header[1] = tile_px;
    table.header[2] = tiles_x;
    table.header[3] = tiles_y;
    let tile_count = tiles_x as usize * tiles_y as usize;
    let kept = table.kept();
    let mut blanket = 0u32;
    // `(bit, dir, sin, cos)` of the drawable half-angle. `cos` may be negative.
    let mut angular = [(0u32, glam::Vec3::ZERO, 0.0f32, 0.0f32); MAX_FAR_BODIES];
    let mut n_angular = 0usize;
    for k in 0..kept {
        let bit = 1u32 << k;
        if !cull {
            blanket |= bit;
            continue;
        }
        let bound = table.cone[k][3];
        let dir = table.cone_dir(k);
        let gpu = &table.body[k];
        let horizon = gpu.horizon();
        // Mapped + a real horizon: the tile cone is the horizon gate, even
        // when `cone.w` is the per-pixel sentinel. `horizon >= 1` falls
        // through to that sentinel and paints every tile.
        if gpu.is_mapped() && horizon < 1.0 {
            match mapped_horizon_half(horizon, gpu.air(), gpu.distance(), view.px_max) {
                None => blanket |= bit,
                Some((sin_b, cos_b)) => {
                    angular[n_angular] = (bit, dir, sin_b, cos_b);
                    n_angular += 1;
                }
            }
            continue;
        }
        match tile_cover(dir, bound, view, tile_px, tiles_x, tiles_y) {
            TileCover::All => blanket |= bit,
            TileCover::None => {}
            TileCover::Rect { tx0, ty0, tx1, ty1 } => {
                for ty in ty0..=ty1 {
                    let row = ty as usize * tiles_x as usize;
                    for tx in tx0..=tx1 {
                        table.tile_mask[row + tx as usize] |= bit;
                    }
                }
            }
            TileCover::Angular => {
                let sin_body = (bound + 3.0 * view.px_max).clamp(0.0, 1.0);
                if !sin_body.is_finite() {
                    blanket |= bit;
                } else {
                    let cos_body = (1.0 - sin_body * sin_body).max(0.0).sqrt();
                    angular[n_angular] = (bit, dir, sin_body, cos_body);
                    n_angular += 1;
                }
            }
        }
    }
    if n_angular > 0 {
        paint_angular(table, view, tiles, &angular[..n_angular], &mut blanket);
    }
    if blanket != 0 {
        for mask in &mut table.tile_mask[..tile_count] {
            *mask |= blanket;
        }
    }
    fill_tile_lists(table, view.width, view.height)
}

/// Set bit `k` on each tile whose view-space cone meets the body's half-angle
/// `(sin, cos)`. `cos` is negative past 90°. Without matching `tiles` the
/// frames are built for this view; a projection that yields no basis paints
/// the body into every tile.
fn paint_angular(
    table: &mut FarTableGpu,
    view: &FarView,
    tiles: Option<&TileView>,
    angular: &[(u32, glam::Vec3, f32, f32)],
    blanket: &mut u32,
) {
    let owned;
    let (frames, basis) = match tiles {
        Some(tiles) => (tiles.frames, &tiles.basis),
        None => {
            owned = TileFrames::build(view).zip(ViewBasis::from_view_proj(view.view_proj));
            let Some((frames, basis)) = owned.as_ref() else {
                for (bit, _, _, _) in angular {
                    *blanket |= *bit;
                }
                return;
            };
            (frames, basis)
        }
    };
    let n = frames.samples.len().min(table.tile_mask.len());
    for &(bit, dir, sin_body, cos_body) in angular {
        let len2 = dir.length_squared();
        if !(len2 > 0.0) || !len2.is_finite() {
            *blanket |= bit;
            continue;
        }
        let dir_view = basis.to_view(dir / len2.sqrt());
        let dir_len2 = dir_view.length_squared();
        if !(dir_len2 > 0.0) || !dir_view.is_finite() {
            *blanket |= bit;
            continue;
        }
        let dir_view = dir_view / dir_len2.sqrt();
        if !(sin_body.is_finite() && cos_body.is_finite()) {
            *blanket |= bit;
            continue;
        }
        for (i, tile) in frames.samples.iter().enumerate().take(n) {
            if tile_within(tile, dir_view, sin_body, cos_body) {
                table.tile_mask[i] |= bit;
            }
        }
    }
}

/// Live mask words (`tiles_x * tiles_y`), capped at the array.
pub(super) fn used_tiles(table: &FarTableGpu) -> usize {
    let n = (table.header[2] as u64).saturating_mul(table.header[3] as u64);
    n.min(MAX_FAR_TILES as u64) as usize
}

/// Screen rect of tile `index` (`ty * tiles_x + tx`), in pixels, clamped to
/// the render extent. The right and bottom tiles are shorter when `width` or
/// `height` is not a multiple of `tile_px`. Matches `sky_tile.vert`.
#[cfg(test)]
pub(super) fn tile_rect(
    index: u32,
    tile_px: u32,
    tiles_x: u32,
    width: u32,
    height: u32,
) -> [u32; 4] {
    debug_assert!(tiles_x > 0);
    let tx = index % tiles_x;
    let ty = index / tiles_x;
    let x0 = tx.saturating_mul(tile_px);
    let y0 = ty.saturating_mul(tile_px);
    let x1 = x0.saturating_add(tile_px).min(width);
    let y1 = y0.saturating_add(tile_px).min(height);
    [x0, y0, x1, y1]
}

/// Inverse of the scene pass's negative-height viewport. NDC y is up,
/// framebuffer y is down. Matches `sky_tile.vert` and the fullscreen
/// triangle's clip xy at the same sample.
#[cfg(test)]
pub(super) fn framebuffer_ndc(xf: f32, yf: f32, width: u32, height: u32) -> [f32; 2] {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    [(xf / w) * 2.0 - 1.0, 1.0 - (yf / h) * 2.0]
}

/// Kept bodies whose shader is not the sphere/inner pair. Bit i is kept index i.
/// Shape 0 is the cube. An unrecognised shape stays on the full march.
pub(super) fn heavy_mask(table: &FarTableGpu) -> u32 {
    table.kept_bits(|body| !body.is_light())
}

/// Kept Mapped bodies. Bit i is kept index i.
pub(super) fn mapped_mask(table: &FarTableGpu) -> u32 {
    table.kept_bits(FarBodyGpu::is_mapped)
}

/// Lengths of the three tile-index runs [`fill_tile_lists`] writes, in
/// upload order: mask == 0, then sphere-only (a non-zero mask that misses
/// every heavy body), then heavy. The pack returns them, so one count serves
/// the draw plan and every split of the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TileRuns {
    pub(super) base: u32,
    pub(super) sphere: u32,
    pub(super) heavy: u32,
}

impl TileRuns {
    /// Count the live masks.
    #[cfg(test)]
    pub(super) fn count(table: &FarTableGpu) -> Self {
        Self::count_with(table, heavy_mask(table))
    }

    fn count_with(table: &FarTableGpu, heavy: u32) -> Self {
        let mut runs = Self::default();
        for mask in &table.tile_mask[..used_tiles(table)] {
            if *mask == 0 {
                runs.base += 1;
            } else if mask & heavy == 0 {
                runs.sphere += 1;
            } else {
                runs.heavy += 1;
            }
        }
        runs
    }

    /// The heavy run's slots in `tile_index`.
    pub(super) fn heavy_slots(&self) -> std::ops::Range<usize> {
        let start = (self.base + self.sphere) as usize;
        start..start + self.heavy as usize
    }
}

/// Partition the live tiles into mask == 0, then sphere-only, then heavy.
/// Each run is ascending row-major. `list_header.y` is `n_sphere + n_heavy`.
pub(super) fn fill_tile_lists(table: &mut FarTableGpu, width: u32, height: u32) -> TileRuns {
    let heavy = heavy_mask(table);
    let runs = TileRuns::count_with(table, heavy);
    let n = used_tiles(table);
    let mut base_i = 0u32;
    let mut sphere_i = runs.base;
    let mut heavy_i = runs.base + runs.sphere;
    for (i, mask) in table.tile_mask[..n].iter().copied().enumerate() {
        if mask == 0 {
            table.tile_index[base_i as usize] = i as u32;
            base_i += 1;
        } else if mask & heavy == 0 {
            table.tile_index[sphere_i as usize] = i as u32;
            sphere_i += 1;
        } else {
            table.tile_index[heavy_i as usize] = i as u32;
            heavy_i += 1;
        }
    }
    table.list_header = [runs.base, runs.sphere + runs.heavy, width, height];
    runs
}
