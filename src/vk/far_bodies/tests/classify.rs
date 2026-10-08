//! The per-frame tile classification against the frozen d2f0e9f copy
//! (`frozen.rs`), the shared stable partition, and `VOXEL_FAR_CULL=0`.

use glam::{Quat, Vec3};

use super::frozen::{old_classify, old_horizon_candidates, old_pack_table_cached};
use super::support::{Rng, mapped_down, placed, view_pitched};
use crate::camera::{Camera3D, Lens};
use crate::color::LinearRgb;
use crate::far_body::{FarBody, FarMapId, FarShape, MAX_FAR_BODIES, MAX_FAR_MAPS};
use crate::vk::far_bodies::FarSwitches;
use crate::vk::far_bodies::horizon::horizon_candidates;
use crate::vk::far_bodies::sky_draw::{
    SkyBodyPipe, SkyCoarseQuery, SkyDraw, classify_tiles, stable_partition,
};
use crate::vk::far_bodies::table::{
    FarTableGpu, HORIZON_BINS, HORIZON_TABLES, list_bytes, pack_table_into, table_bytes,
    zeroed_table,
};
use crate::vk::far_bodies::tiles::{TileFrames, TileRuns, used_tiles};
use crate::vk::far_bodies::view::{FarView, far_view};

fn copy_table(src: &FarTableGpu) -> Box<FarTableGpu> {
    let mut table = zeroed_table();
    bytemuck::bytes_of_mut(&mut *table).copy_from_slice(bytemuck::bytes_of(src));
    table
}

fn bytes(table: &FarTableGpu) -> &[u8] {
    bytemuck::bytes_of(table)
}

#[test]
fn stable_partition_keeps_both_orders_and_aborts_untouched() {
    let mut scratch = Vec::new();
    let mut run = [9u32, 2, 7, 4, 4, 1, 8, 3];
    let taken = stable_partition(&mut run, &mut scratch, |i| Some(i.is_multiple_of(2)));
    assert_eq!(taken, Some(4));
    assert_eq!(run, [2, 4, 4, 8, 9, 7, 1, 3]);
    // All, none and empty.
    let mut all = [5u32, 1, 3];
    assert_eq!(
        stable_partition(&mut all, &mut scratch, |_| Some(true)),
        Some(3)
    );
    assert_eq!(all, [5, 1, 3]);
    assert_eq!(
        stable_partition(&mut all, &mut scratch, |_| Some(false)),
        Some(0)
    );
    assert_eq!(all, [5, 1, 3]);
    assert_eq!(
        stable_partition(&mut [], &mut scratch, |_| Some(true)),
        Some(0)
    );
    // A bad index anywhere, even after some were taken, leaves the run.
    let mut bad = [1u32, 2, 3, 99, 4];
    assert_eq!(
        stable_partition(&mut bad, &mut scratch, |i| (i < 10).then_some(i > 2)),
        None
    );
    assert_eq!(bad, [1, 2, 3, 99, 4]);
    // The scratch keeps its capacity: no allocation once warm.
    let cap = scratch.capacity();
    let mut again = [3u32, 1, 2];
    stable_partition(&mut again, &mut scratch, |i| Some(i < 3));
    assert_eq!(scratch.capacity(), cap);
}

/// A view with any yaw and pitch, so tiles meet bodies from every side.
fn view_yaw_pitch(yaw_deg: f32, pitch_deg: f32, fovy: f32, width: u32, height: u32) -> FarView {
    let (yaw, pitch) = (yaw_deg.to_radians(), pitch_deg.to_radians());
    let forward = Vec3::new(
        pitch.cos() * yaw.sin(),
        pitch.sin(),
        -pitch.cos() * yaw.cos(),
    );
    let position = Vec3::new(3.0, -2.0, 7.0);
    let cam = Camera3D {
        position,
        target: position + forward,
        up: Vec3::Y,
        fovy,
        lens: Lens::Rectilinear,
    };
    let aspect = width as f32 / height as f32;
    let tan_half = (fovy.to_radians() * 0.5).tan();
    far_view(tan_half, cam.view_proj(aspect), width, height)
}

fn random_quat(rng: &mut Rng) -> Quat {
    Quat::from_axis_angle(rng.unit_vec(), rng.range(0.0, std::f32::consts::TAU))
}

/// Every shape. Mapped bodies take every horizon regime (a real cone, the
/// `>= 1` sentinel, below `-1`), map ids past the table, thin and thick air,
/// and some put the eye inside the hi+air ball so they win a horizon table.
fn random_body(rng: &mut Rng, seed: u32) -> FarBody {
    let mut dir = rng.unit_vec();
    if rng.next().is_multiple_of(40) {
        dir = Vec3::ZERO;
    }
    // Mostly small cones: a wide one (or any inner sphere) lands in every
    // tile and leaves no base or single-Mapped tile to split.
    let wide = rng.next().is_multiple_of(8);
    let size = |rng: &mut Rng, max: f32| {
        if wide {
            rng.range(0.3, max)
        } else {
            rng.range(0.002, 0.3)
        }
    };
    let (shape, distance, radius) = match rng.next() % 16 {
        0..=3 => (FarShape::Sphere, 1.0, size(rng, 1.2)),
        4..=6 => (FarShape::Cube, 1.0, size(rng, 0.7)),
        7..=8 => (
            FarShape::Rounded {
                exponent: rng.range(1.0, 40.0),
            },
            1.0,
            size(rng, 0.8),
        ),
        9 => (FarShape::InnerSphere, 2.0, 5.0),
        _ => {
            let distance = [1.0f32, 40.0, 3.0e4][rng.next() as usize % 3];
            let rho = if rng.next().is_multiple_of(2) {
                rng.range(0.9, 1.05)
            } else {
                rng.range(0.01, 0.9)
            };
            let horizon = match rng.next() % 6 {
                0 => 1.0,
                1 => 1.5,
                2 => -1.2,
                _ => rng.range(-0.95, 0.95),
            };
            let air = distance * rng.range(-0.005, 0.06);
            let map = FarMapId((rng.next() % (MAX_FAR_MAPS as u32 + 1)) as u8);
            if rng.next().is_multiple_of(3) {
                // Below the eye, as a planet underfoot.
                dir = (-Vec3::Y + 0.3 * rng.unit_vec()).normalize();
            }
            (
                FarShape::Mapped { map, horizon, air },
                distance,
                rho * distance,
            )
        }
    };
    FarBody {
        dir,
        distance,
        radius,
        shape,
        rotation: random_quat(rng),
        albedo: [LinearRgb([0.3, 0.2, 0.1]); 6],
        atmosphere: LinearRgb([0.05, 0.1, 0.2]),
        seed,
    }
}

fn random_query(rng: &mut Rng) -> SkyCoarseQuery {
    let (sun_cos_rim, moon_cos_rim) =
        crate::vk::pipeline::SkyParams::disc_rims(rng.range(0.004, 0.2));
    SkyCoarseQuery {
        sun_dir: if rng.next().is_multiple_of(25) {
            Vec3::ZERO
        } else {
            rng.unit_vec()
        },
        sun_cos_rim,
        moon_cos_rim,
        stars: rng.next().is_multiple_of(4),
    }
}

/// Give up to two kept Mapped bodies a horizon table, the candidates first,
/// with bins around a random level so some tiles sit above them.
fn fill_horizons(table: &mut FarTableGpu, rng: &mut Rng, map_max: &[f32; MAX_FAR_MAPS]) {
    let mut ids: Vec<u32> = horizon_candidates(table, map_max)
        .iter()
        .flatten()
        .map(|cand| cand.index as u32)
        .collect();
    if ids.is_empty() && rng.next().is_multiple_of(2) {
        let mapped: Vec<u32> = (0..table.kept() as u32)
            .filter(|&k| table.body[k as usize].is_mapped())
            .collect();
        if !mapped.is_empty() {
            ids.push(mapped[rng.next() as usize % mapped.len()]);
        }
    }
    for (slot, id) in ids.into_iter().take(HORIZON_TABLES).enumerate() {
        table.horizon_id[slot] = id;
        let level = rng.range(-0.7, 0.6);
        let start = slot * HORIZON_BINS;
        for bin in &mut table.horizon_sin[start..start + HORIZON_BINS] {
            *bin = (level + rng.range(-0.15, 0.15)).clamp(-1.0, 1.0);
        }
    }
}

/// The ring's host pipeline against the frozen d2f0e9f one, culling on:
/// random scenes (every shape, 0 to 35 bodies, every horizon regime),
/// projections from 100×70 to 8K (coarse tiles past 8192), several turns per
/// projection so both rings reuse their tile frames, an occasional
/// degenerate view and no view, synthetic horizon tables, and each scene with
/// coarse on and off times mapsolo on and off. The new pack writes into one
/// reused, dirtied table. The whole table must be byte-identical (so the
/// uploaded `table_bytes`/`list_bytes`: header, cones, bodies, horizon,
/// masks, `list_header` and the index prefix), and so must the draw plan,
/// the dip, the runs and the horizon pick.
#[test]
fn classification_matches_the_frozen_copy_on_random_scenes() {
    let extents = [
        (100u32, 70u32),
        (320, 180),
        (640, 360),
        (1280, 720),
        (1920, 1080),
        (3440, 1440),
        (7680, 4320),
        (9000, 5000),
    ];
    let mut rng = Rng(0x5EED_0A15);
    let mut old_frames = TileFrames::empty();
    let mut new_frames = TileFrames::empty();
    let mut reused = zeroed_table();
    let mut scratch = Vec::new();
    let mut scenes = 0u32;
    // Cases with tile quads, each split taking tiles, and a horizon clear.
    let mut seen = [0u32; 5];
    for projection in 0..150u32 {
        // Mostly small extents; the 8K pair is slow in a debug build.
        let pick = rng.next() as usize % (extents.len() * 3);
        let (width, height) = if pick < extents.len() {
            extents[pick]
        } else {
            extents[pick % 5]
        };
        let fovy = rng.range(25.0, 120.0);
        for turn in 0..4u32 {
            let view = match (projection + turn) % 23 {
                0 => None,
                11 => Some(FarView {
                    view_proj: glam::Mat4::ZERO,
                    px_max: 0.002,
                    width,
                    height,
                }),
                _ => Some(view_yaw_pitch(
                    rng.range(0.0, 360.0),
                    rng.range(-89.0, 89.0),
                    fovy,
                    width,
                    height,
                )),
            };
            // Mostly a few bodies, sometimes past the 32-body cap.
            let n_bodies = match rng.next() % 10 {
                0..=5 => rng.next() as usize % 5,
                6..=8 => 5 + rng.next() as usize % 8,
                _ => 13 + rng.next() as usize % (MAX_FAR_BODIES - 9),
            };
            let bodies: Vec<FarBody> = (0..n_bodies)
                .map(|i| random_body(&mut rng, i as u32 + 1))
                .collect();
            let map_max: [f32; MAX_FAR_MAPS] = std::array::from_fn(|_| rng.range(0.0, 0.04));
            let map_min: [f32; MAX_FAR_MAPS] = std::array::from_fn(|_| -rng.range(0.0, 0.04));
            let sky_up = rng.unit_vec();
            let query = random_query(&mut rng);

            // Old: `rebuild_if_changed`, then the pack with the ring's frames.
            if let Some(view) = view.as_ref() {
                old_frames.rebuild_if_changed(view);
            }
            let (mut old, old_dip) = old_pack_table_cached(
                &bodies,
                view.as_ref(),
                Some(&old_frames),
                &map_max,
                &map_min,
                sky_up,
                true,
            );
            // New: one `sync`, then the pack into a dirty reused table.
            let tiles = view.as_ref().and_then(|view| new_frames.sync(view));
            bytemuck::bytes_of_mut(&mut *reused).fill(0xA5);
            let (new_dip, runs) = pack_table_into(
                &mut reused,
                &bodies,
                view.as_ref(),
                tiles.as_ref(),
                &map_max,
                &map_min,
                sky_up,
                true,
            );
            let label = format!("projection {projection} turn {turn} {width}x{height}");
            assert_eq!(new_dip.to_bits(), old_dip.to_bits(), "{label}");
            assert!(bytes(&reused) == bytes(&old), "{label}: packed table");
            assert_eq!(runs, TileRuns::count(&reused), "{label}");
            let new_pick: Vec<(usize, f32, usize)> = horizon_candidates(&reused, &map_max)
                .iter()
                .flatten()
                .map(|cand| (cand.index, cand.rho, cand.map))
                .collect();
            assert_eq!(new_pick, old_horizon_candidates(&old, &map_max), "{label}");

            fill_horizons(&mut old, &mut rng, &map_max);
            bytemuck::bytes_of_mut(&mut *reused).copy_from_slice(bytes(&old));
            for (coarse, mapsolo) in [(false, false), (false, true), (true, false), (true, true)] {
                let coarse = coarse.then_some(&query);
                let mut old_case = copy_table(&old);
                let old_draw = old_classify(
                    &mut old_case,
                    view.as_ref(),
                    &old_frames,
                    coarse,
                    &map_min,
                    true,
                    mapsolo,
                );
                let mut new_case = copy_table(&reused);
                let new_draw = classify_tiles(
                    &mut new_case,
                    runs,
                    tiles.as_ref(),
                    coarse,
                    &map_min,
                    FarSwitches {
                        cull: true,
                        mapsolo,
                    },
                    &mut scratch,
                );
                let label = format!("{label} coarse {} mapsolo {mapsolo}", coarse.is_some());
                assert_eq!(new_draw, old_draw, "{label}");
                assert!(table_bytes(&new_case) == table_bytes(&old_case), "{label}");
                assert!(list_bytes(&new_case) == list_bytes(&old_case), "{label}");
                assert!(bytes(&new_case) == bytes(&old_case), "{label}: whole table");
                scenes += 1;
                let n = used_tiles(&old);
                for (count, hit) in seen.iter_mut().zip([
                    new_draw.quads,
                    new_draw.n_coarse > 0,
                    new_draw.n_coarse_far > 0,
                    new_draw.n_mapsolo > 0,
                    new_case.tile_mask[..n] != old.tile_mask[..n],
                ]) {
                    *count += u32::from(hit);
                }
            }
        }
    }
    // The sweep reached every path it claims to.
    assert_eq!(scenes, 150 * 4 * 4);
    let [quads, coarse_base, coarse_far, mapsolo, cleared] = seen;
    eprintln!(
        "{scenes} cases: quads {quads}, coarse base {coarse_base}, coarse far {coarse_far}, \
         mapsolo {mapsolo}, horizon clear {cleared}"
    );
    assert!(quads > scenes / 2, "quads {quads} of {scenes}");
    assert!(coarse_base >= 100, "coarse base split {coarse_base} times");
    assert!(coarse_far >= 100, "coarse far split {coarse_far} times");
    assert!(mapsolo >= 100, "mapsolo split {mapsolo} times");
    assert!(
        cleared >= 100,
        "horizon tables cleared tiles {cleared} times"
    );
}

/// The ring's partitions reuse one scratch: none of them allocates once it
/// holds the longest run. Measured by capacity, so it runs on any allocator.
#[test]
fn classification_reuses_the_ring_scratch() {
    let view = view_pitched(-5.0, 70.0, 1920, 1080);
    let mut frames = TileFrames::empty();
    let tiles = frames.sync(&view).expect("tile frames");
    let bodies = [
        mapped_down(4.0, 1.0, 0.0, 0.0),
        placed(Vec3::new(0.0, -1.0, -1.0), 0.15, FarShape::Sphere, 2),
        placed(Vec3::new(0.4, 0.3, -1.0), 0.05, FarShape::Sphere, 3),
    ];
    let map = [0.0f32; MAX_FAR_MAPS];
    let (sun_cos_rim, moon_cos_rim) = crate::vk::pipeline::SkyParams::disc_rims(0.03);
    let query = SkyCoarseQuery {
        sun_dir: Vec3::new(-0.4, 0.4, -1.0),
        sun_cos_rim,
        moon_cos_rim,
        stars: false,
    };
    let switches = FarSwitches {
        cull: true,
        mapsolo: true,
    };
    let mut table = zeroed_table();
    let mut scratch = Vec::with_capacity(used_tiles_for(&view));
    let cap = scratch.capacity();
    for _ in 0..3 {
        let (_, runs) = pack_table_into(
            &mut table,
            &bodies,
            Some(&view),
            Some(&tiles),
            &map,
            &map,
            Vec3::Y,
            true,
        );
        let draw = classify_tiles(
            &mut table,
            runs,
            Some(&tiles),
            Some(&query),
            &map,
            switches,
            &mut scratch,
        );
        assert!(draw.n_coarse > 0 && draw.n_coarse_far > 0 && draw.n_mapsolo > 0);
        assert_eq!(scratch.capacity(), cap);
    }
}

fn used_tiles_for(view: &FarView) -> usize {
    let (_, tiles_x, tiles_y) = crate::vk::far_bodies::tiles::tile_layout(view.width, view.height);
    (tiles_x * tiles_y) as usize
}

/// `VOXEL_FAR_CULL=0` turns off every per-tile cull: every body is kept with
/// the `-1` cone (no pixel reject), every live tile carries every kept body
/// (the mapped horizon cone included), a horizon table clears nothing, and
/// the sky is one fullscreen triangle with no split. The tile header stays
/// live, and the fragment's `far_tile_mask` reads the mask word whenever the
/// tile size is non-zero, so a full mask is what lets it walk every body.
#[test]
fn far_cull_off_keeps_every_kept_body_in_every_live_tile() {
    let view = view_pitched(-10.0, 70.0, 640, 360);
    let frames = TileFrames::build(&view).expect("tile frames");
    let tiles = frames.view(&view).expect("the frames match the view");
    let map = [0.0f32; MAX_FAR_MAPS];
    // A planet below with a real horizon, a sphere behind the camera, and a
    // cube and a rounded body in view.
    let bodies = [
        mapped_down(4.0, 1.0, 0.0, 0.0),
        placed(Vec3::Z, 0.05, FarShape::Sphere, 2),
        placed(Vec3::new(0.2, -0.1, -1.0), 0.02, FarShape::Cube, 3),
        placed(
            Vec3::new(-0.3, 0.2, -1.0),
            0.02,
            FarShape::Rounded { exponent: 4.0 },
            4,
        ),
    ];
    let pack = |cull: bool| {
        let mut table = zeroed_table();
        let (_, runs) = pack_table_into(
            &mut table,
            &bodies,
            Some(&view),
            Some(&tiles),
            &map,
            &map,
            Vec3::Y,
            cull,
        );
        (table, runs)
    };
    let n = used_tiles(&pack(true).0);
    assert!(n > 1);

    // Culling on: the sphere behind the camera is dropped, and the planet's
    // horizon cone leaves the tiles above the horizon without its bit.
    let (on, _) = pack(true);
    assert_eq!(on.header[0], 3);
    assert!(on.tile_mask[..n].iter().any(|mask| mask & 1 == 0));
    // Before the fix the horizon cone was painted with culling off too.
    let (old_off, _) = old_pack_table_cached(
        &bodies,
        Some(&view),
        Some(&frames),
        &map,
        &map,
        Vec3::Y,
        false,
    );
    assert_eq!(old_off.header[0], 4);
    assert!(old_off.tile_mask[..n].iter().any(|mask| mask & 1 == 0));

    let (mut off, runs) = pack(false);
    let kept = bodies.len();
    assert_eq!(off.header[0], kept as u32);
    assert!(off.cone[..kept].iter().all(|cone| cone[3] == -1.0));
    let all = (1u32 << kept) - 1;
    assert!(
        off.tile_mask[..n].iter().all(|mask| *mask == all),
        "a live tile lost a kept body with culling off"
    );
    assert_eq!(
        runs,
        TileRuns {
            base: 0,
            sphere: 0,
            heavy: n as u32,
        }
    );

    // A table every tile sits above: culling on drops the planet's bit
    // almost everywhere, culling off leaves the mask and the lists alone.
    off.horizon_id[0] = 0;
    off.horizon_sin[..HORIZON_BINS].fill(-1.0);
    let (sun_cos_rim, moon_cos_rim) = crate::vk::pipeline::SkyParams::disc_rims(0.03);
    let query = SkyCoarseQuery {
        sun_dir: Vec3::new(0.1, 0.2, -1.0),
        sun_cos_rim,
        moon_cos_rim,
        stars: false,
    };
    for mapsolo in [false, true] {
        let mut table = copy_table(&off);
        let draw = classify_tiles(
            &mut table,
            runs,
            Some(&tiles),
            Some(&query),
            &map,
            FarSwitches {
                cull: false,
                mapsolo,
            },
            &mut Vec::new(),
        );
        assert_eq!(
            draw,
            SkyDraw {
                body: SkyBodyPipe::Full,
                ..SkyDraw::default()
            }
        );
        assert!(
            bytes(&table) == bytes(&off),
            "culling off touched the table"
        );
    }
    let mut culled = copy_table(&off);
    let draw = classify_tiles(
        &mut culled,
        runs,
        Some(&tiles),
        Some(&query),
        &map,
        FarSwitches {
            cull: true,
            mapsolo: true,
        },
        &mut Vec::new(),
    );
    assert!(draw.quads);
    assert!(culled.tile_mask[..n].iter().any(|mask| mask & 1 == 0));
}
