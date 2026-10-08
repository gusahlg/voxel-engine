//! Building one azimuthal horizon table from a Mapped body's datum: the
//! chart-cell geometry cache, the cap bounds and the adaptive 4×4 splits.

use super::cones::unit_dir;
use super::horizon::horizon_axes;
use super::table::HORIZON_BINS;

/// Added to every stored sine. A sample sitting on the bound stays inside.
const HORIZON_SIN_PAD: f32 = 1.0e-4;
/// Bilinear 4×4 splits of a datum cell whose cap reaches the up axis.
/// Six levels take a ~1.5e6-block cell (g = 33) down to ~370 m.
const HORIZON_SPLIT_DEPTH: u32 = 6;
/// Sub-cells along one chart edge.
const HORIZON_SPLIT_N: u32 = 4;
/// Fine-chart stride of one datum cell, `SPLIT_N ^ SPLIT_DEPTH`. Every
/// split corner lands on this grid, so the per-frame walk does not call `tan`.
const HORIZON_FINE_STRIDE: u32 = 4096;
const _: () = assert!(HORIZON_FINE_STRIDE == 4u32.pow(HORIZON_SPLIT_DEPTH));
/// Extra azimuth half-width. Covers the fast `atan` (under 1e-4 rad) twice,
/// once for the bin centre and once for the half-angle.
pub(super) const HORIZON_AZ_PAD: f32 = 2.0e-4;
/// Patch visits before further splits fall back to the corner-max bound.
/// The axis chain and a one-level straddle stay far under this. A 4×4 of
/// every cell would not.
const HORIZON_PATCH_BUDGET: u32 = 8192;

/// Sine of the elevation of a point at normalised radius `rho` whose
/// centre-direction has `dot(s, up) = mu`. The eye is the origin and the
/// centre is `-up`. Highest radius is the higher elevation: the derivative
/// in `rho` is non-negative. The derivative in `mu` peaks at `mu = rho`
/// when `rho < 1`, and keeps rising up to the pole when `rho >= 1`.
fn horizon_elev_sin(rho: f32, mu: f32) -> f32 {
    let mu = mu.clamp(-1.0, 1.0);
    if !(rho > 0.0) || !rho.is_finite() {
        return -1.0;
    }
    let disc = (1.0 - rho) * (1.0 - rho) + 2.0 * rho * (1.0 - mu);
    if !(disc > 1.0e-20) {
        return 1.0;
    }
    let s = (rho * mu - 1.0) / disc.sqrt();
    if s.is_finite() {
        s.clamp(-1.0, 1.0)
    } else {
        1.0
    }
}

/// Upper bound on elevation over the cap. `mu*` is `rho` when the cap
/// straddles the limb (`mu = rho`), otherwise the endpoint closer to it.
fn cap_elev(rho: f32, mu_far: f32, mu_near: f32) -> f32 {
    let mu = if rho >= 1.0 {
        mu_near
    } else {
        rho.clamp(mu_far, mu_near)
    };
    horizon_elev_sin(rho, mu)
}

fn chart_dir(face: usize, xi: f32, eta: f32) -> Option<glam::Vec3> {
    let (tu, n, tv) = crate::far_body::far_map_basis(face);
    let a = (xi * std::f32::consts::FRAC_PI_4).tan();
    let b = (eta * std::f32::consts::FRAC_PI_4).tan();
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let d = n + tu * a + tv * b;
    unit_dir(d)
}

fn raise_bins(bins: &mut [f32; HORIZON_BINS], az: f32, half: Option<f32>, elev: f32) {
    if !elev.is_finite() {
        return;
    }
    let elev = (elev + HORIZON_SIN_PAD).clamp(-1.0, 1.0);
    let Some(half) = half else {
        for bin in bins.iter_mut() {
            *bin = bin.max(elev);
        }
        return;
    };
    let width = std::f32::consts::TAU / HORIZON_BINS as f32;
    let mut i0 = ((az - half) / width).floor() as i32;
    let i1 = ((az + half) / width).floor() as i32;
    if i1 - i0 >= HORIZON_BINS as i32 {
        for bin in bins.iter_mut() {
            *bin = bin.max(elev);
        }
        return;
    }
    while i0 <= i1 {
        let idx = i0.rem_euclid(HORIZON_BINS as i32) as usize;
        bins[idx] = bins[idx].max(elev);
        i0 += 1;
    }
}

fn horizon_bilerp(h00: f32, h10: f32, h01: f32, h11: f32, u: f32, v: f32) -> f32 {
    let a = h00 + (h10 - h00) * u;
    let b = h01 + (h11 - h01) * u;
    a + (b - a) * v
}

/// `atan` on `[0, 1]`. Degree-9 odd polynomial, max abs error under 3e-5 rad.
#[inline(always)]
pub(super) fn fast_atan(z: f32) -> f32 {
    let z = z.abs();
    let u = z * z;
    let mut p = 0.024597976f32;
    p = p * u + -0.09351313;
    p = p * u + 0.18633223;
    p = p * u + -0.33197755;
    p = p * u + 0.99998576;
    z * p
}

/// `atan2` via one [`fast_atan`] on a reduced `[0, 1]` argument.
#[inline(always)]
pub(super) fn fast_atan2(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    if !(ax + ay > 0.0) {
        return 0.0;
    }
    let a = if ax >= ay {
        fast_atan(ay / ax)
    } else {
        std::f32::consts::FRAC_PI_2 - fast_atan(ax / ay)
    };
    let a = if x < 0.0 { std::f32::consts::PI - a } else { a };
    if y < 0.0 { -a } else { a }
}

/// `asin` on `[0, 1]` via `atan2(s, sqrt(1-s²))`.
#[inline(always)]
pub(super) fn fast_asin(s: f32) -> f32 {
    let s = s.clamp(0.0, 1.0);
    let c = (1.0 - s * s).max(0.0).sqrt();
    let a = fast_atan2(s, c);
    if a < 0.0 { -a } else { a }
}

/// `(cos, sin)` of `beta + 1e-5`, where `beta = acos(min_cos)`.
///
/// The extra tenth of a milliradian keeps the f32 cos/sin identity from
/// shrinking the cap inside the true corner angle.
#[inline]
fn horizon_cap_axes(min_cos: f32) -> (f32, f32) {
    let cos_b = min_cos.clamp(-1.0, 1.0);
    let sin_b = (1.0 - cos_b * cos_b).max(0.0).sqrt();
    let d = 1.0e-5f32;
    let c = cos_b - sin_b * d;
    let s = (sin_b + cos_b * d).max(0.0);
    let len = (c * c + s * s).sqrt();
    if !(len > 0.0) {
        return (1.0, 0.0);
    }
    ((c / len).clamp(-1.0, 1.0), (s / len).clamp(0.0, 1.0))
}

/// `(mu_far, mu_near)` without `acos`. `cos_b` / `sin_b` are a cap half-angle.
#[inline(always)]
fn cap_mu_fast(mu: f32, cos_b: f32, sin_b: f32, sin_g: f32) -> (f32, f32) {
    let near_c = (mu * cos_b + sin_g * sin_b).clamp(-1.0, 1.0);
    let far_c = (mu * cos_b - sin_g * sin_b).clamp(-1.0, 1.0);
    let mu_near = if mu >= cos_b { 1.0 } else { near_c };
    let mu_far = if mu <= -cos_b { -1.0 } else { far_c };
    (mu_far.min(mu_near), mu_near)
}

/// One datum cell on the equiangular chart. Built once per `g`.
struct HorizonCellGeom {
    centre: glam::Vec3,
    cos_beta: f32,
    sin_beta: f32,
}

struct HorizonGeom {
    cells: u32,
    stride: u32,
    /// `tan(ξ π/4)` at `ξ = 2k/fine - 1`, `k = 0..=cells*stride`.
    tan_q: Vec<f32>,
    /// Face-major, then row `j`, then column `i`. `sin_beta < 0` is unusable.
    cell: Vec<HorizonCellGeom>,
}

fn build_horizon_geom(g: u32) -> HorizonGeom {
    let cells = g.max(2) - 1;
    let stride = HORIZON_FINE_STRIDE;
    let fine = cells * stride;
    let mut tan_q = Vec::with_capacity(fine as usize + 1);
    let denom = fine as f32;
    for k in 0..=fine {
        let xi = 2.0 * k as f32 / denom - 1.0;
        tan_q.push((xi * std::f32::consts::FRAC_PI_4).tan());
    }
    let mut cell = Vec::with_capacity(6 * cells as usize * cells as usize);
    for face in 0..6usize {
        let (tu, n, tv) = crate::far_body::far_map_basis(face);
        for j in 0..cells {
            for i in 0..cells {
                let ku0 = i * stride;
                let kv0 = j * stride;
                let corner = |ku: u32, kv: u32| {
                    let a = tan_q[ku as usize];
                    let b = tan_q[kv as usize];
                    unit_dir(n + tu * a + tv * b)
                };
                let c00 = corner(ku0, kv0);
                let c10 = corner(ku0 + stride, kv0);
                let c01 = corner(ku0, kv0 + stride);
                let c11 = corner(ku0 + stride, kv0 + stride);
                let centre = corner(ku0 + stride / 2, kv0 + stride / 2);
                let geom = match (centre, c00, c10, c01, c11) {
                    (Some(centre), Some(a), Some(b), Some(c), Some(d)) => {
                        let min_cos = centre
                            .dot(a)
                            .min(centre.dot(b))
                            .min(centre.dot(c))
                            .min(centre.dot(d));
                        let (cos_beta, sin_beta) = horizon_cap_axes(min_cos);
                        HorizonCellGeom {
                            centre,
                            cos_beta,
                            sin_beta,
                        }
                    }
                    _ => HorizonCellGeom {
                        centre: glam::Vec3::ZERO,
                        cos_beta: 0.0,
                        sin_beta: -1.0,
                    },
                };
                cell.push(geom);
            }
        }
    }
    HorizonGeom {
        cells,
        stride,
        tan_q,
        cell,
    }
}

fn horizon_geom(g: u32) -> Option<&'static HorizonGeom> {
    if !(2..=65).contains(&g) {
        return None;
    }
    use std::sync::OnceLock;
    static SLOTS: OnceLock<Box<[OnceLock<HorizonGeom>]>> = OnceLock::new();
    let slots = SLOTS.get_or_init(|| {
        let mut v = Vec::with_capacity(66);
        for _ in 0..66 {
            v.push(OnceLock::new());
        }
        v.into_boxed_slice()
    });
    Some(slots[g as usize].get_or_init(|| build_horizon_geom(g)))
}

/// Split a patch whose cap contains the up axis, or sits within one cap-radius
/// of it, and a patch whose terrain corners straddle the eye radius.
///
/// `cos_b` / `sin_b` are the cap half-angle. The axis cap paints every bin, so
/// it recurses to [`HORIZON_SPLIT_DEPTH`]. A cap that only comes close is split
/// twice: past that, each level multiplies the patch count by four and the
/// walk covers the globe. A far straddle is split once. Uniform terrain already
/// has its max at a corner. Splitting every cell whose shell contains the
/// camera would be the whole planet (the pixel shell is thicker than 50 km)
/// and does not fit in the frame.
fn horizon_should_split(
    depth: u32,
    mu: f32,
    cos_b: f32,
    sin_b: f32,
    radius: f32,
    min_off: f32,
    max_off: f32,
    distance: f32,
) -> bool {
    if depth >= HORIZON_SPLIT_DEPTH || !(cos_b.abs() <= 1.0) || !mu.is_finite() || sin_b < 0.0 {
        return false;
    }
    let mu = mu.clamp(-1.0, 1.0);
    // gamma <= beta + 1e-6, with cos decreasing on [0, π].
    horizon_reaches_axis(depth, mu, cos_b, sin_b)
        || horizon_straddle(depth, radius, min_off, max_off, distance)
}

/// Cap contains the up axis, or sits within one cap-radius of it.
///
/// The axis cap paints every bin, so it recurses to [`HORIZON_SPLIT_DEPTH`].
/// A cap that only comes close is split twice: past that each level multiplies
/// the patch count by four.
#[inline]
fn horizon_reaches_axis(depth: u32, mu: f32, cos_b: f32, sin_b: f32) -> bool {
    if depth >= HORIZON_SPLIT_DEPTH || !(cos_b.abs() <= 1.0) || !mu.is_finite() || sin_b < 0.0 {
        return false;
    }
    let mu = mu.clamp(-1.0, 1.0);
    // gamma <= beta + 1e-6, with cos decreasing on [0, π].
    let cos_contains = cos_b - sin_b * 1.0e-6;
    if mu >= cos_contains {
        return true;
    }
    if depth >= 2 {
        return false;
    }
    if cos_b <= 0.0 {
        return true;
    }
    let cos_2b = 2.0 * cos_b * cos_b - 1.0;
    let sin_2b = 2.0 * sin_b * cos_b;
    mu >= cos_2b - sin_2b * 1.0e-6
}

/// Terrain corners on opposite sides of the eye radius. One 4×4 split; the
/// sub-cell max stays at a corner. The shell is not part of this test: once
/// it contains the camera every cell would split.
#[inline]
fn horizon_straddle(depth: u32, radius: f32, min_off: f32, max_off: f32, distance: f32) -> bool {
    if depth >= 1 {
        return false;
    }
    let r_max = radius + max_off;
    let r_min = radius + min_off;
    r_max >= distance && r_min < distance
}

/// Elevation sine of one datum patch.
///
/// A patch that contains the up axis and whose terrain is below the eye does
/// not take `theta_min = 0` at the shell radius. That sample is the zenith
/// spike — the shell above the camera — and it is not a shaded ray
/// (`facing > 0` stays below the horizontal). Smearing it paints every bin
/// with sine 1, which is what a coarse cell under the eye did. The patch
/// contributes the smooth-sphere limb of its own radius instead: negative
/// while the eye is outside that sphere, and 0 once the shell contains the
/// eye. `theta_min = 0` is used only when the terrain itself reaches the eye.
///
/// A patch that does not contain the axis still drops a positive shell spike
/// when the terrain is below the eye. The drawable limb is the horizontal.
fn horizon_patch_elev(
    rho_outer: f32,
    rho_terrain: f32,
    mu_far: f32,
    mu_near: f32,
    contains_axis: bool,
) -> f32 {
    if contains_axis && rho_terrain < 1.0 {
        if rho_outer < 1.0 {
            return -(1.0 - rho_outer * rho_outer).max(0.0).sqrt();
        }
        return 0.0;
    }
    if rho_outer >= 1.0 && rho_terrain < 1.0 {
        let spiked = cap_elev(rho_outer, mu_far, mu_near);
        if spiked > 0.0 {
            return cap_elev(rho_terrain, mu_far, mu_near).max(0.0);
        }
        return spiked;
    }
    cap_elev(rho_outer, mu_far, mu_near)
}

/// Cheap upper bound on [`horizon_patch_elev`] for a patch of this `max_off`.
/// Depends only on the radius, so a cell at or under the bound cannot raise a
/// bin the axis cap has already filled.
#[inline(always)]
fn horizon_elev_upper(radius: f32, max_off: f32, shell: f32, distance: f32) -> f32 {
    let rho_t = (radius + max_off) / distance;
    if !rho_t.is_finite() || rho_t >= 1.0 {
        return 1.0;
    }
    let rho_o = (radius + max_off + shell) / distance;
    if !(rho_o > 0.0) || !rho_o.is_finite() {
        return -1.0;
    }
    if rho_o >= 1.0 {
        return 0.0;
    }
    -(1.0 - rho_o * rho_o).max(0.0).sqrt()
}

struct HorizonAccum<'a> {
    bins: &'a mut [f32; HORIZON_BINS],
    geom: &'a HorizonGeom,
    /// Eye axes in body space (`R⁻¹` of the world axes). Azimuth is
    /// `atan2(centre·east, centre·north)` and matches [`horizon_azimuth`].
    ///
    /// [`horizon_azimuth`]: super::horizon::horizon_azimuth
    up_b: glam::Vec3,
    east_b: glam::Vec3,
    north_b: glam::Vec3,
    radius: f32,
    distance: f32,
    shell: f32,
    bin_w: f32,
    /// Bins already at or above this sine cannot be raised by a patch whose
    /// upper bound is lower. `-∞` until the axis cap has been written.
    floor: f32,
    visited: u32,
    truncated: bool,
}

impl HorizonAccum<'_> {
    fn dir_k(
        &self,
        basis: (glam::Vec3, glam::Vec3, glam::Vec3),
        ku: u32,
        kv: u32,
    ) -> Option<glam::Vec3> {
        let a = self.geom.tan_q[ku as usize];
        let b = self.geom.tan_q[kv as usize];
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let (tu, n, tv) = basis;
        unit_dir(n + tu * a + tv * b)
    }

    /// Chart midpoint. Spans coarser than one fine step land on the tan grid.
    fn dir_mid(
        &self,
        face: usize,
        basis: (glam::Vec3, glam::Vec3, glam::Vec3),
        ku0: u32,
        kv0: u32,
        ku1: u32,
        kv1: u32,
    ) -> Option<glam::Vec3> {
        if ku1 > ku0 + 1 && kv1 > kv0 + 1 {
            return self.dir_k(basis, (ku0 + ku1) / 2, (kv0 + kv1) / 2);
        }
        let fine = self.geom.cells * self.geom.stride;
        let mid = |k0: u32, k1: u32| {
            let t = 0.5 * (k0 as f32 + k1 as f32);
            2.0 * t / fine as f32 - 1.0
        };
        chart_dir(face, mid(ku0, ku1), mid(kv0, kv1))
    }

    #[inline(always)]
    fn commit(&mut self, centre: glam::Vec3, cos_b: f32, sin_b: f32, max_off: f32) {
        let mu = centre.dot(self.up_b).clamp(-1.0, 1.0);
        let sin_g = (1.0 - mu * mu).max(0.0).sqrt();
        let rho_outer = (self.radius + max_off + self.shell) / self.distance;
        let rho_terrain = (self.radius + max_off) / self.distance;
        if !(rho_outer > 0.0) || !rho_outer.is_finite() {
            return;
        }
        let (mu_far, mu_near) = cap_mu_fast(mu, cos_b, sin_b, sin_g);
        let cos_contains = cos_b - sin_b * 1.0e-6;
        let contains = mu >= cos_contains;
        let south = mu <= -cos_contains;
        let elev = horizon_patch_elev(rho_outer, rho_terrain, mu_far, mu_near, contains);
        let half = if contains || south || !(sin_g > 1.0e-8) {
            None
        } else {
            let ratio = (sin_b / sin_g).clamp(0.0, 1.0);
            Some(fast_asin(ratio) + self.bin_w + HORIZON_AZ_PAD)
        };
        let az = fast_atan2(centre.dot(self.east_b), centre.dot(self.north_b));
        raise_bins(self.bins, az, half, elev);
    }

    fn patch(
        &mut self,
        face: usize,
        ku0: u32,
        kv0: u32,
        ku1: u32,
        kv1: u32,
        h00: f32,
        h10: f32,
        h01: f32,
        h11: f32,
        depth: u32,
    ) {
        self.visited += 1;
        if !h00.is_finite() || !h10.is_finite() || !h01.is_finite() || !h11.is_finite() {
            return;
        }
        let max_off = h00.max(h10).max(h01).max(h11);
        let min_off = h00.min(h10).min(h01).min(h11);
        if self.floor.is_finite()
            && horizon_elev_upper(self.radius, max_off, self.shell, self.distance) + HORIZON_SIN_PAD
                <= self.floor
        {
            return;
        }
        let basis = crate::far_body::far_map_basis(face);
        let (Some(c00), Some(c10), Some(c01), Some(c11), Some(centre)) = (
            self.dir_k(basis, ku0, kv0),
            self.dir_k(basis, ku1, kv0),
            self.dir_k(basis, ku0, kv1),
            self.dir_k(basis, ku1, kv1),
            self.dir_mid(face, basis, ku0, kv0, ku1, kv1),
        ) else {
            return;
        };
        let min_cos = centre
            .dot(c00)
            .min(centre.dot(c10))
            .min(centre.dot(c01))
            .min(centre.dot(c11));
        let (cos_b, sin_b) = horizon_cap_axes(min_cos);
        // Corners closer than an f32 ulp share one direction. Deeper 4×4
        // splits stay on that point and only multiply the patch count.
        if min_cos >= 1.0 - 1.0e-7 {
            self.commit(centre, cos_b, sin_b, max_off);
            return;
        }
        let mu = centre.dot(self.up_b);
        let want = horizon_should_split(
            depth,
            mu,
            cos_b,
            sin_b,
            self.radius,
            min_off,
            max_off,
            self.distance,
        );
        let split = want && self.visited < HORIZON_PATCH_BUDGET;
        if want && !split {
            self.truncated = true;
        }
        if split {
            let step_u = (ku1 - ku0) / HORIZON_SPLIT_N;
            let step_v = (kv1 - kv0) / HORIZON_SPLIT_N;
            if step_u == 0 || step_v == 0 {
                self.commit(centre, cos_b, sin_b, max_off);
                return;
            }
            let nf = HORIZON_SPLIT_N as f32;
            for sv in 0..HORIZON_SPLIT_N {
                let v0 = sv as f32 / nf;
                let v1 = (sv + 1) as f32 / nf;
                let kv_a = kv0 + step_v * sv;
                let kv_b = kv_a + step_v;
                for su in 0..HORIZON_SPLIT_N {
                    let u0 = su as f32 / nf;
                    let u1 = (su + 1) as f32 / nf;
                    let ku_a = ku0 + step_u * su;
                    let ku_b = ku_a + step_u;
                    self.patch(
                        face,
                        ku_a,
                        kv_a,
                        ku_b,
                        kv_b,
                        horizon_bilerp(h00, h10, h01, h11, u0, v0),
                        horizon_bilerp(h00, h10, h01, h11, u1, v0),
                        horizon_bilerp(h00, h10, h01, h11, u0, v1),
                        horizon_bilerp(h00, h10, h01, h11, u1, v1),
                        depth + 1,
                    );
                }
            }
            return;
        }
        self.commit(centre, cos_b, sin_b, max_off);
    }
}

/// 256-bin horizon table for one Mapped body.
///
/// `up` is the unit centre→eye direction. A cell's radius is the max of its
/// four corner offsets (exact for bilinear) plus `radius` plus the shell
/// `max(air, px * distance)`. `px` is an upper bound on the shader's pixel
/// angle. The drawn limb is the air the ray crosses and is not widened by
/// `px`; the table's extra pixel keeps that limb inside the bins. Directions
/// of the cell lie in a cap of the half-diagonal angle around the
/// chart-midpoint direction. The cap's highest elevation at that radius is
/// written into every azimuth bin the cap overlaps; a cap that contains the
/// up axis covers every bin. An unwritten bin becomes 1 so it cannot reject.
///
/// A datum cell at g = 33 is ~1.5e6 blocks wide. The cell under the eye can
/// hold a corner above the camera while the ground underfoot is far below, and
/// the air shell can contain the eye on its own. That cap used to write sine 1
/// into every bin. Those cells are split bilinearly into 4×4 sub-cells
/// ([`horizon_should_split`]); a sub-cell's max stays at a corner. Chart
/// directions for a given `g` are cached. Cells that cannot rise above the
/// sine the axis cap wrote are skipped.
pub(super) fn build_horizon_bins(
    g: u32,
    datum: &[f32],
    rotation: glam::Quat,
    up: glam::Vec3,
    radius: f32,
    distance: f32,
    air: f32,
    px: f32,
) -> [f32; HORIZON_BINS] {
    let full = [1.0; HORIZON_BINS];
    if g < 2 || !(distance > 0.0) || !distance.is_finite() || !radius.is_finite() {
        return full;
    }
    let gg = g as usize;
    let n = 6 * gg * gg;
    if datum.len() < n || datum.iter().take(n).any(|v| !v.is_finite()) {
        return full;
    }
    let Some(up) = unit_dir(up) else {
        return full;
    };
    let Some((east, north)) = horizon_axes(up) else {
        return full;
    };
    let rotation = if rotation.is_finite() && rotation.length_squared() > 0.0 {
        rotation.normalize()
    } else {
        glam::Quat::IDENTITY
    };
    let shell = air.max(0.0).max(px.max(0.0) * distance);
    let mut bins = [-2.0f32; HORIZON_BINS];
    let bin_w = std::f32::consts::TAU / HORIZON_BINS as f32;
    let cells = g - 1;
    let owned;
    let geom: &HorizonGeom = match horizon_geom(g) {
        Some(cached) => cached,
        None => {
            owned = build_horizon_geom(g);
            &owned
        }
    };
    let inv = rotation.conjugate();
    let mut accum = HorizonAccum {
        bins: &mut bins,
        geom,
        up_b: inv * up,
        east_b: inv * east,
        north_b: inv * north,
        radius,
        distance,
        shell,
        bin_w,
        floor: f32::NEG_INFINITY,
        visited: 0,
        truncated: false,
    };
    let stride = geom.stride;
    // Axis caps first. One of them paints every bin, and that sine is the
    // floor the other cells have to beat.
    for face in 0..6usize {
        let face_base = face * gg * gg;
        for j in 0..cells {
            let row = face_base + j as usize * gg;
            for i in 0..cells {
                let geom_i = (face * cells as usize + j as usize) * cells as usize + i as usize;
                let sample = &geom.cell[geom_i];
                if sample.sin_beta >= 0.0
                    && !horizon_reaches_axis(
                        0,
                        sample.centre.dot(accum.up_b),
                        sample.cos_beta,
                        sample.sin_beta,
                    )
                {
                    continue;
                }
                let h00 = datum[row + i as usize];
                let h10 = datum[row + i as usize + 1];
                let h01 = datum[row + gg + i as usize];
                let h11 = datum[row + gg + i as usize + 1];
                let ku0 = i * stride;
                let kv0 = j * stride;
                accum.patch(
                    face,
                    ku0,
                    kv0,
                    ku0 + stride,
                    kv0 + stride,
                    h00,
                    h10,
                    h01,
                    h11,
                    0,
                );
            }
        }
    }
    let mut floor = f32::INFINITY;
    for bin in accum.bins.iter() {
        if *bin < -1.5 {
            floor = f32::NEG_INFINITY;
            break;
        }
        floor = floor.min(*bin);
    }
    accum.floor = floor;
    for face in 0..6usize {
        let face_base = face * gg * gg;
        for j in 0..cells {
            let row = face_base + j as usize * gg;
            for i in 0..cells {
                let geom_i = (face * cells as usize + j as usize) * cells as usize + i as usize;
                let sample = &geom.cell[geom_i];
                let h00 = datum[row + i as usize];
                let h10 = datum[row + i as usize + 1];
                let h01 = datum[row + gg + i as usize];
                let h11 = datum[row + gg + i as usize + 1];
                let max_off = h00.max(h10).max(h01).max(h11);
                if sample.sin_beta >= 0.0
                    && horizon_elev_upper(radius, max_off, shell, distance) + HORIZON_SIN_PAD
                        <= floor
                {
                    continue;
                }
                let mu = sample.centre.dot(accum.up_b);
                if sample.sin_beta < 0.0
                    || horizon_reaches_axis(0, mu, sample.cos_beta, sample.sin_beta)
                {
                    continue;
                }
                let min_off = h00.min(h10).min(h01).min(h11);
                if horizon_straddle(0, radius, min_off, max_off, distance) {
                    let ku0 = i * stride;
                    let kv0 = j * stride;
                    accum.patch(
                        face,
                        ku0,
                        kv0,
                        ku0 + stride,
                        kv0 + stride,
                        h00,
                        h10,
                        h01,
                        h11,
                        0,
                    );
                } else {
                    accum.commit(sample.centre, sample.cos_beta, sample.sin_beta, max_off);
                }
            }
        }
    }
    debug_assert!(
        !accum.truncated,
        "horizon split stopped after {} patches",
        accum.visited
    );
    for bin in &mut bins {
        if *bin < -1.5 {
            *bin = 1.0;
        }
    }
    bins
}
