//! Which tiles draw with which sky pipeline: the coarse, single-Mapped and
//! single-Rounded splits of the tile-index runs, and [`SkyDraw`], the plan
//! `record_sky` draws from.

use super::FarSwitches;
use super::cones::{add_angles, lo_disc_interior, mapped_interior_half, mapped_rho_lo, unit_dir};
use super::horizon::clear_tiles_above_horizon;
use super::table::FarTableGpu;
use super::tiles::{
    TileRuns, TileView, heavy_mask, mapped_mask, rounded_mask, tile_strictly_inside, tile_within,
    used_tiles,
};
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

/// Stably move the tile indices `take` accepts to the front of `run` and the
/// rest after them, each group in its old order. Returns the accepted count.
/// Every index is classified before `run` is written, so a `None` from `take`
/// (an index with no tile) leaves `run` untouched and returns `None`.
/// `scratch` is the ring's: no allocation once it holds `run.len()` words.
pub(super) fn stable_partition(
    run: &mut [u32],
    scratch: &mut Vec<u32>,
    mut take: impl FnMut(u32) -> Option<bool>,
) -> Option<u32> {
    let n = run.len();
    scratch.clear();
    scratch.resize(n, 0);
    // Accepted indices fill from the front, the rest from the back (so
    // reversed), and the two meet.
    let mut front = 0usize;
    let mut back = n;
    for &index in run.iter() {
        if take(index)? {
            scratch[front] = index;
            front += 1;
        } else {
            back -= 1;
            scratch[back] = index;
        }
    }
    let (taken, rest) = scratch.split_at(front);
    run[..front].copy_from_slice(taken);
    for (slot, &index) in run[front..].iter_mut().zip(rest.iter().rev()) {
        *slot = index;
    }
    Some(front as u32)
}

/// Stably partition the base run into tiles neither disc can touch, then
/// the rest. Sphere and heavy runs are left where [`fill_tile_lists`] put
/// them. Returns the coarse count. Stars or an unusable disc leave the run
/// unchanged and return 0.
///
/// `px_max` is twice the larger-axis pixel angle, so the margin is 2 px on
/// top of the tile cone (which already reaches 1 px past its corners).
///
/// [`fill_tile_lists`]: super::tiles::fill_tile_lists
pub(super) fn split_coarse_base(
    table: &mut FarTableGpu,
    runs: TileRuns,
    tiles: &TileView,
    query: &SkyCoarseQuery,
    scratch: &mut Vec<u32>,
) -> u32 {
    if query.stars {
        return 0;
    }
    let n_base = runs.base as usize;
    if n_base == 0 {
        return 0;
    }
    let Some(sun) = unit_dir(query.sun_dir) else {
        return 0;
    };
    let Some(sun_view) = unit_dir(tiles.basis.to_view(sun)) else {
        return 0;
    };
    let Some(moon_view) = unit_dir(tiles.basis.to_view(-sun)) else {
        return 0;
    };
    let px_max = tiles.view.px_max;
    let Some((sun_sin, sun_cos)) = widened_disc(query.sun_cos_rim, px_max) else {
        return 0;
    };
    let Some((moon_sin, moon_cos)) = widened_disc(query.moon_cos_rim, px_max) else {
        return 0;
    };
    let samples = &tiles.frames.samples;
    stable_partition(&mut table.tile_index[..n_base], scratch, |index| {
        let tile = samples.get(index as usize)?;
        let touch = tile_within(tile, sun_view, sun_sin, sun_cos)
            || tile_within(tile, moon_view, moon_sin, moon_cos);
        Some(!touch)
    })
    .unwrap_or(0)
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
/// tiles ([`split_coarse_base`]). `map_min[i]` is map `i`'s minimum datum
/// offset.
pub(super) fn split_coarse_far(
    table: &mut FarTableGpu,
    runs: TileRuns,
    tiles: &TileView,
    map_min: &[f32; MAX_FAR_MAPS],
    scratch: &mut Vec<u32>,
) -> u32 {
    if runs.heavy == 0 {
        return 0;
    }
    let px_max = tiles.view.px_max;
    // Horizon-disc cone, then lo-sphere disc. Either one admits the tile.
    let mut interior: [Option<(glam::Vec3, f32, f32)>; MAX_FAR_BODIES] = [None; MAX_FAR_BODIES];
    let mut lo_interior: [Option<(glam::Vec3, f32, f32)>; MAX_FAR_BODIES] = [None; MAX_FAR_BODIES];
    for k in 0..table.kept() {
        let gpu = &table.body[k];
        if !gpu.is_mapped() {
            continue;
        }
        let Some(dir) = unit_dir(table.cone_dir(k)) else {
            continue;
        };
        let Some(dir_view) = unit_dir(tiles.basis.to_view(dir)) else {
            continue;
        };
        let horizon = gpu.horizon();
        if horizon < 1.0
            && let Some((sin_b, cos_b)) =
                mapped_interior_half(horizon, gpu.air(), gpu.distance(), px_max)
        {
            interior[k] = Some((dir_view, sin_b, cos_b));
        }
        // `px_max` is already two pixel-angles, the 2 px inset.
        if let Some(rho_lo) = mapped_rho_lo(gpu, map_min)
            && let Some((sin_b, cos_b)) = lo_disc_interior(rho_lo, px_max)
        {
            lo_interior[k] = Some((dir_view, sin_b, cos_b));
        }
    }
    let slots = runs.heavy_slots();
    if slots.end > table.tile_index.len() {
        return 0;
    }
    let samples = &tiles.frames.samples;
    let masks = &table.tile_mask;
    let inside = |cone: Option<(glam::Vec3, f32, f32)>, tile| match cone {
        Some((dir_view, sin_b, cos_b)) => tile_strictly_inside(tile, dir_view, sin_b, cos_b),
        None => false,
    };
    stable_partition(&mut table.tile_index[slots], scratch, |index| {
        let tile = samples.get(index as usize)?;
        let mask = masks[index as usize];
        Some(
            mask.count_ones() == 1 && {
                let k = mask.trailing_zeros() as usize;
                inside(interior[k], tile) || inside(lo_interior[k], tile)
            },
        )
    })
    .unwrap_or(0)
}

/// One live mask bit, and that kept body is Mapped. `mapped` is
/// [`mapped_mask`]: bits past the kept count are clear.
fn is_mapsolo(mask: u32, mapped: u32) -> bool {
    mask.count_ones() == 1 && mask & mapped != 0
}

/// One live mask bit, and that kept body is Mapped.
#[cfg(test)]
pub(super) fn mask_is_mapsolo(table: &FarTableGpu, mask: u32) -> bool {
    is_mapsolo(mask, mapped_mask(table))
}

/// Live tiles whose mask is exactly one Mapped body. Counts masks, not the
/// index list: every such tile sits in the heavy run.
fn count_mapsolo(table: &FarTableGpu, mapped: u32) -> u32 {
    table.tile_mask[..used_tiles(table)]
        .iter()
        .filter(|mask| is_mapsolo(**mask, mapped))
        .count() as u32
}

/// One live mask bit, and that kept body is Rounded. `rounded` is
/// [`rounded_mask`]: bits past the kept count are clear.
fn is_roundsolo(mask: u32, rounded: u32) -> bool {
    mask.count_ones() == 1 && mask & rounded != 0
}

/// One live mask bit, and that kept body is Rounded.
#[cfg(test)]
pub(super) fn mask_is_roundsolo(table: &FarTableGpu, mask: u32) -> bool {
    is_roundsolo(mask, rounded_mask(table))
}

/// Live tiles whose mask is exactly one Rounded body. Like
/// [`count_mapsolo`], every such tile sits in the heavy run.
fn count_roundsolo(table: &FarTableGpu, rounded: u32) -> u32 {
    table.tile_mask[..used_tiles(table)]
        .iter()
        .filter(|mask| is_roundsolo(**mask, rounded))
        .count() as u32
}

/// Stably partition the heavy run into single-Mapped tiles, then the rest.
/// Returns that count. Base and sphere runs stay where [`fill_tile_lists`]
/// put them. Called only while quads are on and mapsolo is enabled, and only
/// before [`split_coarse_far`], so the interior prefix of this run is the
/// coarse mapsolo tiles.
///
/// [`fill_tile_lists`]: super::tiles::fill_tile_lists
pub(super) fn partition_mapsolo(
    table: &mut FarTableGpu,
    runs: TileRuns,
    scratch: &mut Vec<u32>,
) -> u32 {
    if runs.heavy == 0 {
        return 0;
    }
    let slots = runs.heavy_slots();
    if slots.end > table.tile_index.len() {
        return 0;
    }
    let mapped = mapped_mask(table);
    let masks = &table.tile_mask;
    stable_partition(&mut table.tile_index[slots], scratch, |index| {
        let mask = masks.get(index as usize).copied().unwrap_or(0);
        Some(is_mapsolo(mask, mapped))
    })
    .unwrap_or(0)
}

/// Stably partition the heavy run into the rest, then the single-Rounded
/// tiles, and return how many those are. A suffix: the mapsolo prefix stays
/// where [`partition_mapsolo`] put it, and [`split_coarse_far`], which takes
/// only single-Mapped tiles, keeps this a suffix. Base and sphere runs stay.
/// Called only while quads are on and roundsolo is enabled.
pub(super) fn partition_roundsolo(
    table: &mut FarTableGpu,
    runs: TileRuns,
    scratch: &mut Vec<u32>,
) -> u32 {
    if runs.heavy == 0 {
        return 0;
    }
    let slots = runs.heavy_slots();
    if slots.end > table.tile_index.len() {
        return 0;
    }
    let rounded = rounded_mask(table);
    let masks = &table.tile_mask;
    stable_partition(&mut table.tile_index[slots], scratch, |index| {
        let mask = masks.get(index as usize).copied().unwrap_or(0);
        Some(!is_roundsolo(mask, rounded))
    })
    .map_or(0, |rest| runs.heavy - rest)
}

/// The per-frame tile classification after the pack and the horizon tables:
/// with culling on, drop tiles above those tables; plan the draws; then
/// stably reorder the index runs, mapsolo first, so the interior split pulls
/// its coarse prefix out of that single-Mapped run and the full-shader tiles
/// stay after it, and roundsolo last in the heavy run; then the two coarse
/// splits. `runs` are the pack's.
/// `tiles` are the ring's frames when they match the view. The splits only
/// rewrite the index prefix, so this runs before the byte match. Host-only,
/// and no allocation once `scratch` holds the longest run.
pub(super) fn classify_tiles(
    table: &mut FarTableGpu,
    runs: TileRuns,
    tiles: Option<&TileView>,
    coarse: Option<&SkyCoarseQuery>,
    map_min: &[f32; MAX_FAR_MAPS],
    switches: FarSwitches,
    scratch: &mut Vec<u32>,
) -> SkyDraw {
    let runs = match tiles {
        Some(tiles) if switches.cull => clear_tiles_above_horizon(table, tiles).unwrap_or(runs),
        _ => runs,
    };
    let mut draw = SkyDraw::from_runs(table, runs, switches);
    // Fullscreen sky (no tile grid) stays on the 1×1 triangle.
    if draw.quads {
        if switches.mapsolo {
            let n = partition_mapsolo(table, runs, scratch);
            debug_assert_eq!(n, draw.n_mapsolo);
            draw.n_mapsolo = n;
        }
        if switches.roundsolo {
            let n = partition_roundsolo(table, runs, scratch);
            debug_assert_eq!(n, draw.n_roundsolo);
            draw.n_roundsolo = n;
        }
        if let (Some(query), Some(tiles)) = (coarse, tiles) {
            draw.n_coarse = split_coarse_base(table, runs, tiles, query, scratch);
            draw.n_coarse_far = split_coarse_far(table, runs, tiles, map_min, scratch);
        }
    }
    draw
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
    /// Heavy tiles drawn with the loop-free single-Rounded fragment: the
    /// suffix of the heavy run, all at 1×1. `sky.roundsolo`. Zero when
    /// roundsolo is off or the sky is one fullscreen triangle.
    pub n_roundsolo: u32,
}

impl SkyDraw {
    /// Tile-quad draws in upload order: base coarse, base fine, sphere,
    /// mapsolo coarse, mapsolo fine, heavy coarse, heavy fine, roundsolo.
    /// Each `first` is the sum of the counts before it, which is where
    /// [`fill_tile_lists`], [`partition_mapsolo`], [`partition_roundsolo`]
    /// and the `split_coarse_*` pair put those tiles. Empty runs are `None`;
    /// all are `None` when the sky is one fullscreen triangle.
    ///
    /// `coarse` is [`SkyPipelines::has_coarse`], the same switch the CPU
    /// split read. Without the 2×2 pipelines a coarse prefix stays at the
    /// head of its fine run. With mapsolo on, the heavy interior is a prefix
    /// of the mapsolo run, so the heavy coarse run is empty. With it off,
    /// mapped-interior tiles are a prefix of the whole heavy run and draw on
    /// the full fragment at 2×2. The roundsolo suffix is never coarse: the
    /// interior split takes only single-Mapped tiles.
    ///
    /// [`fill_tile_lists`]: super::tiles::fill_tile_lists
    /// [`SkyPipelines::has_coarse`]: crate::vk::pipeline::SkyPipelines::has_coarse
    pub(crate) fn runs(&self, coarse: bool) -> [Option<SkyRun>; 8] {
        if !self.quads {
            return [None; 8];
        }
        let if_coarse = |n: u32| if coarse { n } else { 0 };
        let base_coarse = if_coarse(self.n_coarse.min(self.n_base));
        let mapsolo = self.n_mapsolo.min(self.n_heavy);
        let roundsolo = self.n_roundsolo.min(self.n_heavy - mapsolo);
        let rest = self.n_heavy - mapsolo - roundsolo;
        let mapsolo_coarse = if_coarse(self.n_coarse_far.min(mapsolo));
        let heavy_coarse = if_coarse(self.n_coarse_far.saturating_sub(mapsolo).min(rest));
        let mut first = 0u32;
        [
            (SkyFrag::Base, SkyRate::Coarse, base_coarse),
            (SkyFrag::Base, SkyRate::Fine, self.n_base - base_coarse),
            (SkyFrag::Sphere, SkyRate::Fine, self.n_sphere),
            (SkyFrag::MapSolo, SkyRate::Coarse, mapsolo_coarse),
            (SkyFrag::MapSolo, SkyRate::Fine, mapsolo - mapsolo_coarse),
            (SkyFrag::Full, SkyRate::Coarse, heavy_coarse),
            (self.body.frag(), SkyRate::Fine, rest - heavy_coarse),
            (SkyFrag::RoundSolo, SkyRate::Fine, roundsolo),
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

    /// `sky.coarse`, `sky.coarse_far`, `sky.mapsolo` and `sky.roundsolo`.
    pub(super) fn publish_gauges(&self) {
        crate::profile::gauge(crate::profile::Gauge::SkyCoarse, u64::from(self.n_coarse));
        crate::profile::gauge(
            crate::profile::Gauge::SkyCoarseFar,
            u64::from(self.n_coarse_far),
        );
        crate::profile::gauge(crate::profile::Gauge::SkyMapsolo, u64::from(self.n_mapsolo));
        crate::profile::gauge(
            crate::profile::Gauge::SkyRoundsolo,
            u64::from(self.n_roundsolo),
        );
    }

    /// Zeroes every gauge [`FarBodyRing::write`] publishes: a frame with no sky.
    ///
    /// [`FarBodyRing::write`]: super::FarBodyRing::write
    pub(crate) fn zero_gauges() {
        publish_far_gauges(0, 0, 0);
        Self::default().publish_gauges();
    }

    fn body_pipe(heavy: u32, mapped: u32) -> SkyBodyPipe {
        if mapped != 0 {
            SkyBodyPipe::Full
        } else if heavy != 0 {
            SkyBodyPipe::NoMap
        } else {
            SkyBodyPipe::Sphere
        }
    }

    /// The draw plan for a packed table whose tile lists hold `runs`.
    /// [`FarSwitches`]: culling off draws one fullscreen triangle, mapsolo
    /// off counts no single-Mapped tile, roundsolo off no single-Rounded one.
    pub(super) fn from_runs(table: &FarTableGpu, runs: TileRuns, switches: FarSwitches) -> Self {
        let bodies = table.header[0];
        let tile_px = table.header[1];
        let tiles_x = table.header[2];
        let tiles_y = table.header[3];
        let mapped = mapped_mask(table);
        let body = Self::body_pipe(heavy_mask(table), mapped);
        let tiled = switches.cull && tile_px != 0 && tiles_x != 0 && tiles_y != 0 && bodies != 0;
        if !tiled {
            return Self {
                base: bodies == 0,
                body,
                ..Self::default()
            };
        }
        Self {
            quads: true,
            body,
            n_base: runs.base,
            n_sphere: runs.sphere,
            n_heavy: runs.heavy,
            n_mapsolo: if switches.mapsolo {
                count_mapsolo(table, mapped)
            } else {
                0
            },
            n_roundsolo: if switches.roundsolo {
                count_roundsolo(table, rounded_mask(table))
            } else {
                0
            },
            ..Self::default()
        }
    }

    /// [`Self::from_runs`] with the runs counted from the masks, mapsolo and
    /// roundsolo on.
    #[cfg(test)]
    pub(super) fn from_table(table: &FarTableGpu, cull: bool) -> Self {
        let switches = FarSwitches {
            cull,
            mapsolo: true,
            roundsolo: true,
        };
        Self::from_runs(table, TileRuns::count(table), switches)
    }
}

/// `far.bodies` (offered), `far.drawn` (kept) and `far.tiles` (mask != 0).
pub(super) fn publish_far_gauges(offered: u64, drawn: u64, tiles: u64) {
    crate::profile::gauge(crate::profile::Gauge::FarBodies, offered);
    crate::profile::gauge(crate::profile::Gauge::FarDrawn, drawn);
    crate::profile::gauge(crate::profile::Gauge::FarTiles, tiles);
}
