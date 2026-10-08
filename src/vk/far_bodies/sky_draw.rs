//! Which tiles draw with which sky pipeline: the coarse and single-Mapped
//! splits of the tile-index runs, and [`SkyDraw`], the plan `record_sky`
//! draws from.

use super::cones::{add_angles, lo_disc_interior, mapped_interior_half, mapped_rho_lo, unit_dir};
use super::sky_mapsolo_enabled;
use super::table::FarTableGpu;
use super::tiles::{
    TileFrames, has_mapped, heavy_mask, tile_split, tile_strictly_inside, tile_within, used_tiles,
};
use super::view::{FarView, ViewBasis};
use crate::far_body::{MAX_FAR_BODIES, MAX_FAR_MAPS};
use crate::vk::pipeline::{SkyFrag, SkyRate};

/// Disc and star inputs for [`split_coarse_base`]. The rims are the push
/// constant's outer cosines (`SkyParams::disc_rims`), not the solid cores.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SkyCoarseQuery {
    /// World-space sun direction, the same vector the push constant normalises.
    /// A zero or non-finite direction shades every base tile at 1×1.
    pub sun_dir: glam::Vec3,
    pub sun_cos_rim: f32,
    pub moon_cos_rim: f32,
    /// `max(night, star_floor) * stars_gain > 0` in `sky.frag`.
    pub stars: bool,
}

/// Same gate as `sky.frag`: stars draw when `max(night, star_floor) * gain`
/// is positive. A non-finite factor counts as stars on, so no tile goes coarse.
pub(crate) fn stars_drawn(night: f32, star_floor: f32, stars_gain: f32) -> bool {
    if !(night.is_finite() && star_floor.is_finite() && stars_gain.is_finite()) {
        return true;
    }
    night.max(star_floor) * stars_gain > 0.0
}

/// Widen a disc-rim cosine by `margin_rad`. `None` when the inputs are not a
/// finite cone, or the widened angle covers the sphere (every tile is touched).
fn widened_disc(cos_rim: f32, margin_rad: f32) -> Option<(f32, f32)> {
    if !(cos_rim.is_finite() && margin_rad.is_finite()) || !(-1.0..=1.0).contains(&cos_rim) {
        return None;
    }
    if margin_rad < 0.0 {
        return None;
    }
    let (sin_w, cos_w) = add_angles(cos_rim, margin_rad.cos());
    if !(sin_w.is_finite() && cos_w.is_finite()) || sin_w < 0.0 || sin_w > 1.0 {
        return None;
    }
    Some((sin_w, cos_w))
}

/// Stably partition the base prefix into tiles neither disc can touch, then
/// the rest. Sphere and heavy runs are left where [`fill_tile_lists`] put
/// them. Returns the coarse count. Stars, a missing frame, or an unusable
/// disc leave the prefix unchanged and return 0.
///
/// `px_max` is twice the larger-axis pixel angle, so the margin is 2 px on
/// top of the tile cone (which already reaches 1 px past its corners).
///
/// [`fill_tile_lists`]: super::tiles::fill_tile_lists
pub(super) fn split_coarse_base(
    table: &mut FarTableGpu,
    frames: &TileFrames,
    view: &FarView,
    query: &SkyCoarseQuery,
) -> u32 {
    if query.stars {
        return 0;
    }
    let n_base = table.list_header[0] as usize;
    if n_base == 0 || !frames.matches(view) {
        return 0;
    }
    let Some(basis) = ViewBasis::from_view_proj(view.view_proj) else {
        return 0;
    };
    let Some(sun) = unit_dir(query.sun_dir) else {
        return 0;
    };
    let Some(sun_view) = unit_dir(basis.to_view(sun)) else {
        return 0;
    };
    let Some(moon_view) = unit_dir(basis.to_view(-sun)) else {
        return 0;
    };
    let Some((sun_sin, sun_cos)) = widened_disc(query.sun_cos_rim, view.px_max) else {
        return 0;
    };
    let Some((moon_sin, moon_cos)) = widened_disc(query.moon_cos_rim, view.px_max) else {
        return 0;
    };
    // Classify before writing, so a bad index leaves the run untouched.
    let mut touch = Vec::with_capacity(n_base);
    for slot in 0..n_base {
        let index = table.tile_index[slot];
        let Some(tile) = frames.samples.get(index as usize) else {
            return 0;
        };
        touch.push(
            tile_within(tile, sun_view, sun_sin, sun_cos)
                || tile_within(tile, moon_view, moon_sin, moon_cos),
        );
    }
    let mut coarse = Vec::with_capacity(n_base);
    let mut fine = Vec::with_capacity(n_base);
    for (slot, hit) in touch.into_iter().enumerate() {
        let index = table.tile_index[slot];
        if hit {
            fine.push(index);
        } else {
            coarse.push(index);
        }
    }
    let n_coarse = coarse.len();
    for (slot, index) in coarse.into_iter().chain(fine).enumerate() {
        table.tile_index[slot] = index;
    }
    n_coarse as u32
}

/// Stably partition the heavy run into mapped-interior tiles, then the rest.
/// Returns the coarse count. Both runs use the fixed-point march. The prefix
/// is drawn at 2×2; the suffix, where the silhouette and the limb live, stays
/// at full rate. A tile qualifies when its mask is exactly one
/// mapped body (no other body, and so no nearer body, can cover it) and its
/// cone lies strictly inside one of:
///
/// * the horizon disc inset by the air limb and 3 px, when `horizon < 1`;
/// * the lo-sphere disc inset by 2 px, when the eye is outside that sphere.
///   The tile cone already carries the tile's angular radius, so the inset
///   is the rest of the margin.
///
/// Every pixel of that tile is a surface hit, and the hit replaces the sky
/// colour. Stars and the sun/moon discs are composited before far bodies, so
/// they do not keep the tile at 1×1 — that gate is only for empty base-sky
/// tiles ([`split_coarse_base`]). A missing frame leaves the run unchanged
/// and returns 0. `map_min[i]` is map `i`'s minimum datum offset.
pub(super) fn split_coarse_far(
    table: &mut FarTableGpu,
    frames: &TileFrames,
    view: &FarView,
    map_min: &[f32; MAX_FAR_MAPS],
) -> u32 {
    let (n_base, n_sphere, n_heavy) = tile_split(table);
    if n_heavy == 0 || !frames.matches(view) {
        return 0;
    }
    let Some(basis) = ViewBasis::from_view_proj(view.view_proj) else {
        return 0;
    };
    let kept = (table.header[0] as usize).min(MAX_FAR_BODIES);
    // Horizon-disc cone, then lo-sphere disc. Either one admits the tile.
    let mut interior: [Option<(glam::Vec3, f32, f32)>; MAX_FAR_BODIES] = [None; MAX_FAR_BODIES];
    let mut lo_interior: [Option<(glam::Vec3, f32, f32)>; MAX_FAR_BODIES] = [None; MAX_FAR_BODIES];
    for k in 0..kept {
        let gpu = &table.body[k];
        let shape = gpu.atmosphere[3];
        if !(3.5..4.5).contains(&shape) {
            continue;
        }
        let dir = glam::Vec3::new(table.cone[k][0], table.cone[k][1], table.cone[k][2]);
        let Some(dir) = unit_dir(dir) else {
            continue;
        };
        let Some(dir_view) = unit_dir(basis.to_view(dir)) else {
            continue;
        };
        let horizon = gpu.albedo0[3];
        if horizon < 1.0
            && let Some((sin_b, cos_b)) =
                mapped_interior_half(horizon, gpu.albedo1[3], gpu.albedo2[3], view.px_max)
        {
            interior[k] = Some((dir_view, sin_b, cos_b));
        }
        // `px_max` is already two pixel-angles, the 2 px inset.
        if let Some(rho_lo) = mapped_rho_lo(gpu, map_min)
            && let Some((sin_b, cos_b)) = lo_disc_interior(rho_lo, view.px_max)
        {
            lo_interior[k] = Some((dir_view, sin_b, cos_b));
        }
    }
    let start = (n_base + n_sphere) as usize;
    let end = start + n_heavy as usize;
    if end > table.tile_index.len() {
        return 0;
    }
    // Classify before writing, so a bad index leaves the run untouched.
    let mut take = Vec::with_capacity(n_heavy as usize);
    for slot in start..end {
        let index = table.tile_index[slot];
        let Some(tile) = frames.samples.get(index as usize) else {
            return 0;
        };
        let mask = table.tile_mask[index as usize];
        let inside = mask.count_ones() == 1 && {
            let k = mask.trailing_zeros() as usize;
            let in_horizon = match interior[k] {
                Some((dir_view, sin_b, cos_b)) => {
                    tile_strictly_inside(tile, dir_view, sin_b, cos_b)
                }
                None => false,
            };
            let in_lo = match lo_interior[k] {
                Some((dir_view, sin_b, cos_b)) => {
                    tile_strictly_inside(tile, dir_view, sin_b, cos_b)
                }
                None => false,
            };
            in_horizon || in_lo
        };
        take.push(inside);
    }
    let mut coarse = Vec::with_capacity(n_heavy as usize);
    let mut fine = Vec::with_capacity(n_heavy as usize);
    for (offset, inside) in take.into_iter().enumerate() {
        let index = table.tile_index[start + offset];
        if inside {
            coarse.push(index);
        } else {
            fine.push(index);
        }
    }
    let n_coarse = coarse.len();
    for (offset, index) in coarse.into_iter().chain(fine).enumerate() {
        table.tile_index[start + offset] = index;
    }
    n_coarse as u32
}

/// One live mask bit, and that kept body is Mapped (shape lane in `(3.5, 4.5)`).
pub(super) fn mask_is_mapsolo(table: &FarTableGpu, mask: u32) -> bool {
    if mask.count_ones() != 1 {
        return false;
    }
    let k = mask.trailing_zeros() as usize;
    let n = (table.header[0] as usize).min(MAX_FAR_BODIES);
    if k >= n {
        return false;
    }
    let shape = table.body[k].atmosphere[3];
    (3.5..4.5).contains(&shape)
}

/// Live tiles whose mask is exactly one Mapped body. Zero when mapsolo is off.
/// Counts masks, not the index list: every such tile sits in the heavy run.
fn count_mapsolo(table: &FarTableGpu) -> u32 {
    if !sky_mapsolo_enabled() {
        return 0;
    }
    table.tile_mask[..used_tiles(table)]
        .iter()
        .filter(|mask| mask_is_mapsolo(table, **mask))
        .count() as u32
}

/// Stably partition the heavy run into single-Mapped tiles, then the rest.
/// Returns that count. Base and sphere runs stay where [`fill_tile_lists`]
/// put them. Called only while quads are on and mapsolo is enabled, and only
/// before [`split_coarse_far`], so the interior prefix of this run is the
/// coarse mapsolo tiles.
///
/// [`fill_tile_lists`]: super::tiles::fill_tile_lists
pub(super) fn partition_mapsolo(table: &mut FarTableGpu) -> u32 {
    let (n_base, n_sphere, n_heavy) = tile_split(table);
    if n_heavy == 0 {
        return 0;
    }
    let start = (n_base + n_sphere) as usize;
    let end = start + n_heavy as usize;
    if end > table.tile_index.len() {
        return 0;
    }
    let mut solo = Vec::with_capacity(n_heavy as usize);
    let mut rest = Vec::with_capacity(n_heavy as usize);
    for slot in start..end {
        let index = table.tile_index[slot];
        let mask = table.tile_mask.get(index as usize).copied().unwrap_or(0);
        if mask_is_mapsolo(table, mask) {
            solo.push(index);
        } else {
            rest.push(index);
        }
    }
    let n = solo.len();
    for (offset, index) in solo.into_iter().chain(rest).enumerate() {
        table.tile_index[start + offset] = index;
    }
    n as u32
}

/// Which body fragment a draw needs. `Full` has every shape. `NoMap` drops the
/// mapped march. `Sphere` keeps spheres and inner spheres.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SkyBodyPipe {
    #[default]
    Full,
    NoMap,
    Sphere,
}

impl SkyBodyPipe {
    pub(crate) fn frag(self) -> SkyFrag {
        match self {
            Self::Full => SkyFrag::Full,
            Self::NoMap => SkyFrag::NoMap,
            Self::Sphere => SkyFrag::Sphere,
        }
    }
}

/// One instanced tile-quad draw: `count` tiles from tile-list slot `first`
/// (the draw's firstInstance) on the `(frag, Tile, rate)` sky pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SkyRun {
    pub frag: SkyFrag,
    pub rate: SkyRate,
    pub first: u32,
    pub count: u32,
}

/// How `record_sky` draws this frame. Tile quads only when the mask is a real
/// per-tile classification: a view, culling on, a non-zero tile size, and at
/// least one kept body. Otherwise one fullscreen triangle. No kept bodies use
/// the body-free pipeline. A frame with no mapped body uses `NoMap`, and a
/// frame of only spheres and inner spheres uses `Sphere`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SkyDraw {
    pub quads: bool,
    /// Fullscreen triangle uses the body-free pipeline.
    pub base: bool,
    /// Fullscreen body pipeline, or the pipeline for heavy tiles.
    pub body: SkyBodyPipe,
    pub n_base: u32,
    /// Tiles whose mask is only spheres and inner spheres.
    pub n_sphere: u32,
    /// Tiles whose mask meets a cube, rounded or mapped body.
    pub n_heavy: u32,
    /// Coarse-eligible prefix of the base run. `sky.coarse`. Zero unless a
    /// query actually split the prefix.
    pub n_coarse: u32,
    /// Coarse-eligible prefix of the heavy run. `sky.coarse_far`. Zero unless
    /// a query actually split that run. With mapsolo on this prefix is a
    /// prefix of [`Self::n_mapsolo`].
    pub n_coarse_far: u32,
    /// Heavy tiles drawn with the loop-free single-Mapped fragment, including
    /// the 2×2 interior prefix. `sky.mapsolo`. Zero when mapsolo is off or the
    /// sky is one fullscreen triangle.
    pub n_mapsolo: u32,
}

impl SkyDraw {
    /// Tile-quad draws in upload order: base coarse, base fine, sphere,
    /// mapsolo coarse, mapsolo fine, heavy coarse, heavy fine. Each `first`
    /// is the sum of the counts before it, which is where [`fill_tile_lists`],
    /// [`partition_mapsolo`] and the `split_coarse_*` pair put those tiles.
    /// Empty runs are `None`; all are `None` when the sky is one fullscreen
    /// triangle.
    ///
    /// `coarse` is [`SkyPipelines::has_coarse`], the same switch the CPU
    /// split read. Without the 2×2 pipelines a coarse prefix stays at the
    /// head of its fine run. With mapsolo on, the heavy interior is a prefix
    /// of the mapsolo run, so the heavy coarse run is empty. With it off,
    /// mapped-interior tiles are a prefix of the whole heavy run and draw on
    /// the full fragment at 2×2.
    ///
    /// [`fill_tile_lists`]: super::tiles::fill_tile_lists
    /// [`SkyPipelines::has_coarse`]: crate::vk::pipeline::SkyPipelines::has_coarse
    pub(crate) fn runs(&self, coarse: bool) -> [Option<SkyRun>; 7] {
        if !self.quads {
            return [None; 7];
        }
        let if_coarse = |n: u32| if coarse { n } else { 0 };
        let base_coarse = if_coarse(self.n_coarse.min(self.n_base));
        let mapsolo = self.n_mapsolo.min(self.n_heavy);
        let mapsolo_coarse = if_coarse(self.n_coarse_far.min(mapsolo));
        let heavy_coarse = if_coarse(
            self.n_coarse_far
                .saturating_sub(mapsolo)
                .min(self.n_heavy - mapsolo),
        );
        let mut first = 0u32;
        [
            (SkyFrag::Base, SkyRate::Coarse, base_coarse),
            (SkyFrag::Base, SkyRate::Fine, self.n_base - base_coarse),
            (SkyFrag::Sphere, SkyRate::Fine, self.n_sphere),
            (SkyFrag::MapSolo, SkyRate::Coarse, mapsolo_coarse),
            (SkyFrag::MapSolo, SkyRate::Fine, mapsolo - mapsolo_coarse),
            (SkyFrag::Full, SkyRate::Coarse, heavy_coarse),
            (
                self.body.frag(),
                SkyRate::Fine,
                self.n_heavy - mapsolo - heavy_coarse,
            ),
        ]
        .map(|(frag, rate, count)| {
            let run = (count > 0).then_some(SkyRun {
                frag,
                rate,
                first,
                count,
            });
            first += count;
            run
        })
    }

    /// The fullscreen triangle's fragment: body-free with no kept body, else
    /// the frame's body fragment.
    pub(crate) fn fullscreen_frag(&self) -> SkyFrag {
        if self.base {
            SkyFrag::Base
        } else {
            self.body.frag()
        }
    }

    /// `sky.coarse`, `sky.coarse_far` and `sky.mapsolo`.
    pub(super) fn publish_gauges(&self) {
        crate::profile::gauge(crate::profile::Gauge::SkyCoarse, u64::from(self.n_coarse));
        crate::profile::gauge(
            crate::profile::Gauge::SkyCoarseFar,
            u64::from(self.n_coarse_far),
        );
        crate::profile::gauge(crate::profile::Gauge::SkyMapsolo, u64::from(self.n_mapsolo));
    }

    /// Zeroes every gauge [`FarBodyRing::write`] publishes: a frame with no sky.
    ///
    /// [`FarBodyRing::write`]: super::FarBodyRing::write
    pub(crate) fn zero_gauges() {
        publish_far_gauges(0, 0, 0);
        Self::default().publish_gauges();
    }

    fn body_pipe(table: &FarTableGpu) -> SkyBodyPipe {
        if has_mapped(table) {
            SkyBodyPipe::Full
        } else if heavy_mask(table) != 0 {
            SkyBodyPipe::NoMap
        } else {
            SkyBodyPipe::Sphere
        }
    }

    pub(super) fn from_table(table: &FarTableGpu, cull: bool) -> Self {
        let bodies = table.header[0];
        let tile_px = table.header[1];
        let tiles_x = table.header[2];
        let tiles_y = table.header[3];
        let body = Self::body_pipe(table);
        let tiled = cull && tile_px != 0 && tiles_x != 0 && tiles_y != 0 && bodies != 0;
        if !tiled {
            return Self {
                base: bodies == 0,
                body,
                ..Self::default()
            };
        }
        let (n_base, n_sphere, n_heavy) = tile_split(table);
        Self {
            quads: true,
            body,
            n_base,
            n_sphere,
            n_heavy,
            n_mapsolo: count_mapsolo(table),
            ..Self::default()
        }
    }
}

/// `far.bodies` (offered), `far.drawn` (kept) and `far.tiles` (mask != 0).
pub(super) fn publish_far_gauges(offered: u64, drawn: u64, tiles: u64) {
    crate::profile::gauge(crate::profile::Gauge::FarBodies, offered);
    crate::profile::gauge(crate::profile::Gauge::FarDrawn, drawn);
    crate::profile::gauge(crate::profile::Gauge::FarTiles, tiles);
}
