use bytemuck::Zeroable;
use glam::{Quat, Vec3};

use super::support::{
    datum_from, mapped_down, placed, ray_at, tile_at, view_along_neg_z, view_pitched,
};
use crate::far_body::mirror::{lowland_up, ray_mapped_fast};
use crate::far_body::{FarShape, MAX_FAR_MAPS};
use crate::vk::far_bodies::cones::{lo_disc_interior, mapped_interior_half};
use crate::vk::far_bodies::sky_draw::{
    self as sky, SkyBodyPipe, SkyCoarseQuery, SkyDraw, SkyRun, mask_is_mapsolo, stars_drawn,
};
use crate::vk::far_bodies::table::{FarTableGpu, zeroed_table};
use crate::vk::far_bodies::tiles::{
    TileFrames, TileRuns, fill_tile_lists, heavy_mask, tile_layout, tile_rect,
    tile_strictly_inside, used_tiles,
};
use crate::vk::far_bodies::view::{FarView, ViewBasis};
use crate::vk::pipeline::{SkyFrag, SkyRate};

// The splits as these tests call them: runs counted from the masks, fresh
// scratch, and `frames` matched against `view` here (0 when they do not
// match, as the ring never splits then).

fn partition_mapsolo(table: &mut FarTableGpu) -> u32 {
    sky::partition_mapsolo(table, TileRuns::count(table), &mut Vec::new())
}

fn split_coarse_base(
    table: &mut FarTableGpu,
    frames: &TileFrames,
    view: &FarView,
    query: &SkyCoarseQuery,
) -> u32 {
    let Some(tiles) = frames.view(view) else {
        return 0;
    };
    sky::split_coarse_base(
        table,
        TileRuns::count(table),
        &tiles,
        query,
        &mut Vec::new(),
    )
}

fn split_coarse_far(
    table: &mut FarTableGpu,
    frames: &TileFrames,
    view: &FarView,
    map_min: &[f32; MAX_FAR_MAPS],
) -> u32 {
    let Some(tiles) = frames.view(view) else {
        return 0;
    };
    sky::split_coarse_far(
        table,
        TileRuns::count(table),
        &tiles,
        map_min,
        &mut Vec::new(),
    )
}

#[test]
fn mapsolo_is_the_single_mapped_prefix_of_the_heavy_run() {
    let mut table = FarTableGpu::zeroed();
    // 192×128 is 3×2 tiles of 64 px.
    let width = 192u32;
    let height = 128u32;
    let tile_px = 64u32;
    let tiles_x = width.div_ceil(tile_px);
    let tiles_y = height.div_ceil(tile_px);
    assert_eq!((tiles_x, tiles_y), (3, 2));
    table.header = [3, tile_px, tiles_x, tiles_y];
    // Kept 0 is a sphere, kept 1 is mapped, kept 2 is a cube.
    table.body[0].atmosphere[3] = 1.0;
    table.body[1].atmosphere[3] = 4.0;
    table.body[2].atmosphere[3] = 0.0;
    // 0 base, 1 mapped, 2 sphere, 3 mapped+sphere, 4 cube, 5 mapped.
    table.tile_mask[0] = 0;
    table.tile_mask[1] = 0b010;
    table.tile_mask[2] = 0b001;
    table.tile_mask[3] = 0b011;
    table.tile_mask[4] = 0b100;
    table.tile_mask[5] = 0b010;
    fill_tile_lists(&mut table, width, height);

    let draw = SkyDraw::from_table(&table, true);
    assert!(draw.quads);
    assert_eq!(
        (
            draw.n_base,
            draw.n_sphere,
            draw.n_heavy,
            table.list_header[1],
            draw.n_mapsolo
        ),
        (1, 1, 4, 5, 2)
    );
    assert_eq!(
        draw.n_base + draw.n_sphere + draw.n_heavy,
        tiles_x * tiles_y
    );
    assert!(draw.n_mapsolo <= draw.n_heavy);
    assert_eq!(draw.body, SkyBodyPipe::Full);
    // Base, then the single sphere. Heavy is still row-major here.
    assert_eq!(&table.tile_index[..2], &[0, 2]);
    assert_eq!(&table.tile_index[2..6], &[1, 3, 4, 5]);
    let base_sphere = table.tile_index[..2].to_vec();
    let n = partition_mapsolo(&mut table);
    assert_eq!(n, draw.n_mapsolo);
    assert_eq!(&table.tile_index[..2], base_sphere.as_slice());
    // Mapped tiles stay sorted, then the shared and cube tiles stay sorted.
    assert_eq!(&table.tile_index[2..6], &[1, 5, 3, 4]);
    assert!(mask_is_mapsolo(&table, table.tile_mask[1]));
    assert!(mask_is_mapsolo(&table, table.tile_mask[5]));
    assert!(
        !mask_is_mapsolo(&table, table.tile_mask[3]),
        "mapped+sphere"
    );
    assert!(!mask_is_mapsolo(&table, table.tile_mask[2]), "sphere");
    assert!(!mask_is_mapsolo(&table, table.tile_mask[4]), "cube");
    assert!(!mask_is_mapsolo(&table, 0));
    let heavy = &table.tile_index[2..6];
    assert!(heavy[..n as usize].windows(2).all(|w| w[0] < w[1]));
    assert!(heavy[n as usize..].windows(2).all(|w| w[0] < w[1]));
}

fn coarse_query(sun: Vec3, stars: bool) -> SkyCoarseQuery {
    let (sun_cos_rim, moon_cos_rim) = crate::vk::pipeline::SkyParams::disc_rims(0.03);
    SkyCoarseQuery {
        sun_dir: sun,
        sun_cos_rim,
        moon_cos_rim,
        stars,
    }
}

/// 320×180 at 60° is 5×3 tiles of 64 px. The screen centre sits in tile 7.
fn coarse_grid() -> (Box<FarTableGpu>, FarView, TileFrames) {
    let width = 320u32;
    let height = 180u32;
    let view = view_along_neg_z(60.0, width, height);
    let frames = TileFrames::build(&view).expect("view basis");
    assert!(frames.matches(&view));
    let (tile_px, tiles_x, tiles_y) = tile_layout(width, height);
    assert_eq!((tile_px, tiles_x, tiles_y), (64, 5, 3));
    let mut table = zeroed_table();
    table.header = [0, tile_px, tiles_x, tiles_y];
    fill_tile_lists(&mut table, width, height);
    let n = (tiles_x * tiles_y) as usize;
    assert_eq!(table.list_header[0] as usize, n);
    assert!(table.tile_index[..n].windows(2).all(|w| w[0] < w[1]));
    (table, view, frames)
}

fn assert_disc_split(sun: Vec3, center_coarse: bool) {
    let (mut table, view, frames) = coarse_grid();
    let n_base = table.list_header[0];
    let before = table.tile_index[..n_base as usize].to_vec();
    let header = table.list_header;
    let n_coarse = split_coarse_base(&mut table, &frames, &view, &coarse_query(sun, false));
    assert_eq!(table.list_header, header);
    assert!(n_coarse > 0 && n_coarse < n_base);
    let coarse = &table.tile_index[..n_coarse as usize];
    let fine = &table.tile_index[n_coarse as usize..n_base as usize];
    assert!(coarse.windows(2).all(|w| w[0] < w[1]));
    assert!(fine.windows(2).all(|w| w[0] < w[1]));
    let mut merged = table.tile_index[..n_base as usize].to_vec();
    merged.sort_unstable();
    assert_eq!(merged, before);
    let center = 7u32;
    assert_eq!(tile_at(160.0, 90.0, 64, 5), center as usize);
    let center_in_coarse = coarse.contains(&center);
    assert_eq!(center_in_coarse, center_coarse);
    assert!(coarse.contains(&0), "corner tile must stay coarse");
    assert!(!fine.contains(&0));
}

#[test]
fn coarse_base_excludes_a_tile_a_disc_touches() {
    // Sun on the view axis: the centre tile meets the sun disc, the moon
    // is behind the camera. Corner tile 0 is many degrees off both discs.
    assert_disc_split(-Vec3::Z, false);
    // Sun behind the camera puts the moon on the view axis.
    assert_disc_split(Vec3::Z, false);
}

#[test]
fn coarse_base_takes_every_tile_when_both_discs_miss() {
    // +Y is 90° off a 60° view along −Z. Both discs sit outside the frustum.
    let (mut table, view, frames) = coarse_grid();
    let n_base = table.list_header[0];
    let before = table.tile_index[..n_base as usize].to_vec();
    let n_coarse = split_coarse_base(&mut table, &frames, &view, &coarse_query(Vec3::Y, false));
    assert_eq!(n_coarse, n_base);
    assert_eq!(&table.tile_index[..n_base as usize], before.as_slice());
}

#[test]
fn stars_disable_coarse_and_leave_the_base_run() {
    let (mut table, view, frames) = coarse_grid();
    let n_base = table.list_header[0] as usize;
    let before = table.tile_index[..n_base].to_vec();
    let n_coarse = split_coarse_base(&mut table, &frames, &view, &coarse_query(-Vec3::Z, true));
    assert_eq!(n_coarse, 0);
    assert_eq!(&table.tile_index[..n_base], before.as_slice());
    // Day, no star floor: the product is zero even with the gain left on.
    assert!(!stars_drawn(0.0, 0.0, 1.0));
    assert!(stars_drawn(1.0, 0.0, 1.0));
    assert!(stars_drawn(0.0, 0.4, 1.0));
    assert!(!stars_drawn(1.0, 1.0, 0.0));
    assert!(stars_drawn(f32::NAN, 0.0, 1.0));
    // A non-finite rim shades nothing coarse and does not reorder.
    let (mut table, view, frames) = coarse_grid();
    let mut query = coarse_query(-Vec3::Z, false);
    query.sun_cos_rim = f32::NAN;
    let n_coarse = split_coarse_base(&mut table, &frames, &view, &query);
    assert_eq!(n_coarse, 0);
    assert_eq!(&table.tile_index[..n_base], before.as_slice());
}

#[test]
fn coarse_split_keeps_the_three_way_partition() {
    let (mut table, view, frames) = coarse_grid();
    table.header[0] = 2;
    table.body[0].atmosphere[3] = 1.0;
    table.body[1].atmosphere[3] = 0.0;
    // Tile 1 is sphere-only, tile 3 meets a cube. The centre (tile 7) stays
    // base, so the sun on −Z pulls it into the fine suffix and the base
    // order actually changes.
    table.tile_mask[1] = 0b001;
    table.tile_mask[3] = 0b010;
    fill_tile_lists(&mut table, view.width, view.height);
    let n_base = table.list_header[0] as usize;
    let n_full = table.list_header[1] as usize;
    assert_eq!(n_base + n_full, 15);
    assert_eq!(n_full, 2);
    let base_before = table.tile_index[..n_base].to_vec();
    let full_before = table.tile_index[n_base..n_base + n_full].to_vec();
    let header = table.list_header;
    let n_coarse = split_coarse_base(&mut table, &frames, &view, &coarse_query(-Vec3::Z, false));
    assert!(n_coarse > 0 && (n_coarse as usize) < n_base);
    assert_eq!(table.list_header, header);
    assert_eq!(
        &table.tile_index[n_base..n_base + n_full],
        full_before.as_slice()
    );
    let mut merged = table.tile_index[..n_base].to_vec();
    merged.sort_unstable();
    assert_eq!(merged, base_before);
    let coarse = &table.tile_index[..n_coarse as usize];
    let fine = &table.tile_index[n_coarse as usize..n_base];
    assert!(coarse.windows(2).all(|w| w[0] < w[1]));
    assert!(fine.windows(2).all(|w| w[0] < w[1]));
    assert!(fine.contains(&7), "the sunlit centre tile stays at 1x1");
    for &index in coarse {
        assert_eq!(table.tile_mask[index as usize], 0);
    }
    for &index in fine {
        assert_eq!(table.tile_mask[index as usize], 0);
    }
}

fn heavy_run(table: &FarTableGpu) -> (usize, usize) {
    let slots = TileRuns::count(table).heavy_slots();
    (slots.start, slots.end)
}

#[test]
fn coarse_far_is_the_mapped_interior_only() {
    let view = view_pitched(0.0, 90.0, 1280, 720);
    let frames = TileFrames::build(&view).expect("frames");
    let map_max = [0.0f32; MAX_FAR_MAPS];
    let body = mapped_down(4.0, 1.0, 0.0, 0.0);
    let mut table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
    assert_eq!(table.header[0], 1);
    let (start, end) = heavy_run(&table);
    let n_heavy = (end - start) as u32;
    assert!(n_heavy > 1, "heavy {n_heavy}");
    let before = table.tile_index[start..end].to_vec();
    let header = table.list_header;
    let map_min = [0.0f32; MAX_FAR_MAPS];
    let n = split_coarse_far(&mut table, &frames, &view, &map_min);
    assert!(n > 0 && n < n_heavy, "coarse {n} of {n_heavy}");
    assert_eq!(table.list_header, header);
    let coarse = &table.tile_index[start..start + n as usize];
    let fine = &table.tile_index[start + n as usize..end];
    assert!(coarse.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(fine.windows(2).all(|pair| pair[0] < pair[1]));
    let mut merged = table.tile_index[start..end].to_vec();
    merged.sort_unstable();
    let mut expect = before.clone();
    expect.sort_unstable();
    assert_eq!(merged, expect);

    let (sin_b, cos_b) = mapped_interior_half(0.0, 0.0, 4.0, view.px_max).expect("interior cone");
    let basis = ViewBasis::from_view_proj(view.view_proj).expect("basis");
    let dir_view = basis.to_view(-Vec3::Y).normalize();
    let mut boundary = 0u32;
    for &index in &before {
        let tile = &frames.samples[index as usize];
        let inside = tile_strictly_inside(tile, dir_view, sin_b, cos_b);
        if inside {
            assert!(
                coarse.contains(&index),
                "interior tile {index} stayed at 1x1"
            );
        } else {
            assert!(fine.contains(&index), "limb tile {index} went coarse");
            boundary += 1;
        }
    }
    assert!(boundary > 0, "every heavy tile was interior");

    // horizon >= 1 disables the horizon disc. The lo sphere still admits
    // tiles: min offset 0 makes rho_lo = rho = 0.75, and a downward view
    // puts the centre inside that disc while the corners stay out.
    let disabled = mapped_down(4.0, 3.0, 1.0, 0.0);
    let down = view_pitched(-50.0, 70.0, 640, 360);
    let down_frames = TileFrames::build(&down).expect("down frames");
    let mut off = super::pack_table(std::slice::from_ref(&disabled), Some(down), &map_max);
    let (off_start, off_end) = heavy_run(&off);
    assert!(off_end > off_start, "horizon >= 1 painted no heavy tile");
    let off_before = off.tile_index[off_start..off_end].to_vec();
    let n_lo = split_coarse_far(&mut off, &down_frames, &down, &map_min);
    assert!(n_lo > 0 && (n_lo as usize) < off_end - off_start);
    let (sin_lo, cos_lo) = lo_disc_interior(0.75, down.px_max).expect("lo disc");
    let down_basis = ViewBasis::from_view_proj(down.view_proj).expect("down basis");
    let down_dir = down_basis.to_view(-Vec3::Y).normalize();
    let lo_coarse = &off.tile_index[off_start..off_start + n_lo as usize];
    let lo_fine = &off.tile_index[off_start + n_lo as usize..off_end];
    for &index in &off_before {
        let tile = &down_frames.samples[index as usize];
        let inside = tile_strictly_inside(tile, down_dir, sin_lo, cos_lo);
        if inside {
            assert!(
                lo_coarse.contains(&index),
                "lo-disc tile {index} stayed 1x1"
            );
        } else {
            assert!(lo_fine.contains(&index), "outside tile {index} went coarse");
        }
    }

    // A second body on the same tiles drops those tiles out of the prefix.
    // Bottom-centre of this view, inside the mapped hemisphere and on screen.
    let companion = placed(Vec3::new(0.0, -1.0, -1.0), 0.15, FarShape::Sphere, 2);
    let mut both = super::pack_table(&[body, companion], Some(view), &map_max);
    assert_eq!(both.header[0], 2);
    let n_both = split_coarse_far(&mut both, &frames, &view, &map_min);
    assert!(n_both > 0, "the companion erased every interior tile");
    let (both_start, both_end) = heavy_run(&both);
    let both_coarse = &both.tile_index[both_start..both_start + n_both as usize];
    let mut shared = 0u32;
    for &index in &both.tile_index[both_start..both_end] {
        let mask = both.tile_mask[index as usize];
        if mask.count_ones() != 1 {
            shared += 1;
            assert!(
                !both_coarse.contains(&index),
                "shared tile {index} mask {mask:#x} went coarse"
            );
        }
    }
    assert!(shared > 0, "the companion shared no tile");
    for &index in both_coarse {
        assert_eq!(both.tile_mask[index as usize], 1);
    }
}

#[test]
fn mapsolo_interior_is_coarse_and_a_shared_tile_stays_full() {
    let view = view_pitched(0.0, 90.0, 1280, 720);
    let frames = TileFrames::build(&view).expect("frames");
    let map_max = [0.0f32; MAX_FAR_MAPS];
    let map_min = [0.0f32; MAX_FAR_MAPS];
    let body = mapped_down(4.0, 1.0, 0.0, 0.0);
    let mut table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
    let draw = SkyDraw::from_table(&table, true);
    assert!(draw.quads);
    assert!(draw.n_mapsolo > 1, "mapsolo {}", draw.n_mapsolo);
    assert_eq!(draw.n_mapsolo, draw.n_heavy);
    assert_eq!(draw.body, SkyBodyPipe::Full);
    let (start, end) = heavy_run(&table);
    let head = table.tile_index[..start].to_vec();
    let heavy = table.tile_index[start..end].to_vec();
    let n = partition_mapsolo(&mut table);
    assert_eq!(n, draw.n_mapsolo);
    // One mapped body: the partition is the identity, and the base run stays.
    assert_eq!(&table.tile_index[..start], head.as_slice());
    assert_eq!(&table.tile_index[start..end], heavy.as_slice());
    let n_coarse = split_coarse_far(&mut table, &frames, &view, &map_min);
    assert!(
        n_coarse > 0 && n_coarse <= n,
        "coarse {n_coarse} of mapsolo {n}"
    );
    for &index in &table.tile_index[start..start + n_coarse as usize] {
        assert!(
            mask_is_mapsolo(&table, table.tile_mask[index as usize]),
            "coarse tile {index} is not mapsolo"
        );
    }

    // A sphere sharing tiles with the planet leaves those tiles on the full
    // fragment, after the mapsolo prefix, and out of the coarse run.
    let companion = placed(Vec3::new(0.0, -1.0, -1.0), 0.15, FarShape::Sphere, 2);
    let mut both = super::pack_table(&[body, companion], Some(view), &map_max);
    assert_eq!(both.header[0], 2);
    let both_draw = SkyDraw::from_table(&both, true);
    assert_eq!(both_draw.body, SkyBodyPipe::Full);
    assert!(both_draw.n_mapsolo < both_draw.n_heavy);
    assert_eq!(
        both_draw.n_base + both_draw.n_sphere + both_draw.n_heavy,
        used_tiles(&both) as u32
    );
    let (both_start, both_end) = heavy_run(&both);
    let both_head = both.tile_index[..both_start].to_vec();
    let n_sphere = both_draw.n_sphere as usize;
    let n_base = both_draw.n_base as usize;
    for &index in &both.tile_index[n_base..n_base + n_sphere] {
        let mask = both.tile_mask[index as usize];
        assert_ne!(mask, 0);
        assert!(!mask_is_mapsolo(&both, mask), "sphere tile {index}");
    }
    let n_solo = partition_mapsolo(&mut both);
    assert_eq!(n_solo, both_draw.n_mapsolo);
    assert_eq!(&both.tile_index[..both_start], both_head.as_slice());
    let solo = &both.tile_index[both_start..both_start + n_solo as usize];
    let rest = &both.tile_index[both_start + n_solo as usize..both_end];
    assert!(solo.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(rest.windows(2).all(|pair| pair[0] < pair[1]));
    let mut shared = 0u32;
    for &index in solo {
        assert!(mask_is_mapsolo(&both, both.tile_mask[index as usize]));
    }
    for &index in rest {
        let mask = both.tile_mask[index as usize];
        assert!(!mask_is_mapsolo(&both, mask), "tile {index} mask {mask:#x}");
        if mask.count_ones() != 1 {
            shared += 1;
        }
    }
    assert!(shared > 0, "the companion shared no heavy tile");
    let n_both = split_coarse_far(&mut both, &frames, &view, &map_min);
    assert!(
        n_both > 0 && n_both <= n_solo,
        "coarse {n_both} of mapsolo {n_solo}"
    );
    let both_coarse = &both.tile_index[both_start..both_start + n_both as usize];
    for &index in both_coarse {
        assert!(mask_is_mapsolo(&both, both.tile_mask[index as usize]));
    }
    for &index in &both.tile_index[both_start + n_solo as usize..both_end] {
        assert!(
            !both_coarse.contains(&index),
            "full-run tile {index} went coarse"
        );
    }
}

/// One command of the pre-table `record_sky` tile path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyCmd {
    Bind(SkyFrag, SkyRate),
    Draw { count: u32, first: u32 },
}

/// `record_sky`'s bind+draw sequence as of 09812b7 (`scene_pass.rs`
/// 876-936 and 1013-1091), copied with each pipeline named by its
/// `(frag, rate)` row. The three coarse `Option`s were built together, so
/// they are one `coarse` flag; `sky_tile_heavy` is the body fragment at
/// 1×1. `Err` is the fullscreen triangle's fragment.
fn legacy_sky_cmds(draw: &SkyDraw, coarse: bool) -> Result<Vec<LegacyCmd>, SkyFrag> {
    use LegacyCmd::{Bind, Draw};
    let pipe = coarse.then_some(());
    let tile_base_coarse = pipe.map(|_| (SkyFrag::Base, SkyRate::Coarse));
    let tile_base = (SkyFrag::Base, SkyRate::Fine);
    let tile_sphere = (SkyFrag::Sphere, SkyRate::Fine);
    let tile_mapsolo = (SkyFrag::MapSolo, SkyRate::Fine);
    let tile_mapsolo_coarse = pipe.map(|_| (SkyFrag::MapSolo, SkyRate::Coarse));
    let tile_coarse = pipe.map(|_| (SkyFrag::Full, SkyRate::Coarse));
    let tile_heavy = (draw.body.frag(), SkyRate::Fine);
    let quads = draw.quads && (draw.n_base > 0 || draw.n_sphere > 0 || draw.n_heavy > 0);
    let fullscreen = if draw.base {
        SkyFrag::Base
    } else {
        draw.body.frag()
    };
    let n_coarse = tile_base_coarse.map_or(0, |_| draw.n_coarse.min(draw.n_base));
    let n_fine = draw.n_base - n_coarse;
    let n_mapsolo = draw.n_mapsolo.min(draw.n_heavy);
    let n_mapsolo_coarse = tile_mapsolo_coarse.map_or(0, |_| draw.n_coarse_far.min(n_mapsolo));
    let n_mapsolo_fine = n_mapsolo - n_mapsolo_coarse;
    let n_full_coarse = tile_coarse.map_or(0, |_| {
        draw.n_coarse_far
            .saturating_sub(n_mapsolo)
            .min(draw.n_heavy - n_mapsolo)
    });
    let n_full_fine = draw.n_heavy - n_mapsolo - n_full_coarse;
    if !quads {
        return Err(fullscreen);
    }
    let first = if n_coarse > 0 {
        tile_base_coarse.unwrap_or(tile_base)
    } else if n_fine > 0 {
        tile_base
    } else if draw.n_sphere > 0 {
        tile_sphere
    } else if n_mapsolo_coarse > 0 {
        tile_mapsolo_coarse.unwrap_or(tile_mapsolo)
    } else if n_mapsolo_fine > 0 {
        tile_mapsolo
    } else if n_full_coarse > 0 {
        tile_coarse.unwrap_or(tile_heavy)
    } else {
        tile_heavy
    };
    let bind = |(frag, rate): (SkyFrag, SkyRate)| Bind(frag, rate);
    let mut cmds = vec![bind(first)];
    if n_coarse > 0 {
        cmds.push(Draw {
            count: n_coarse,
            first: 0,
        });
    }
    if n_fine > 0 {
        if n_coarse > 0 {
            cmds.push(bind(tile_base));
        }
        cmds.push(Draw {
            count: n_fine,
            first: n_coarse,
        });
    }
    if draw.n_sphere > 0 {
        if draw.n_base > 0 {
            cmds.push(bind(tile_sphere));
        }
        cmds.push(Draw {
            count: draw.n_sphere,
            first: draw.n_base,
        });
    }
    let heavy_start = draw.n_base + draw.n_sphere;
    if n_mapsolo_coarse > 0 {
        if draw.n_base > 0 || draw.n_sphere > 0 {
            cmds.push(bind(tile_mapsolo_coarse.unwrap_or(tile_mapsolo)));
        }
        cmds.push(Draw {
            count: n_mapsolo_coarse,
            first: heavy_start,
        });
    }
    if n_mapsolo_fine > 0 {
        if draw.n_base > 0 || draw.n_sphere > 0 || n_mapsolo_coarse > 0 {
            cmds.push(bind(tile_mapsolo));
        }
        cmds.push(Draw {
            count: n_mapsolo_fine,
            first: heavy_start + n_mapsolo_coarse,
        });
    }
    if n_full_coarse > 0 {
        if draw.n_base > 0 || draw.n_sphere > 0 || n_mapsolo > 0 {
            cmds.push(bind(tile_coarse.unwrap_or(tile_heavy)));
        }
        cmds.push(Draw {
            count: n_full_coarse,
            first: heavy_start + n_mapsolo,
        });
    }
    if n_full_fine > 0 {
        if draw.n_base > 0 || draw.n_sphere > 0 || n_mapsolo > 0 || n_full_coarse > 0 {
            cmds.push(bind(tile_heavy));
        }
        cmds.push(Draw {
            count: n_full_fine,
            first: heavy_start + n_mapsolo + n_full_coarse,
        });
    }
    Ok(cmds)
}

/// Each draw of a command list, with the pipeline bound when it runs.
fn bound_draws(cmds: &[LegacyCmd]) -> Vec<SkyRun> {
    let mut bound = None;
    let mut draws = Vec::new();
    for cmd in cmds {
        match *cmd {
            LegacyCmd::Bind(frag, rate) => bound = Some((frag, rate)),
            LegacyCmd::Draw { count, first } => {
                let (frag, rate) = bound.expect("a draw before any bind");
                draws.push(SkyRun {
                    frag,
                    rate,
                    first,
                    count,
                });
            }
        }
    }
    draws
}

/// Every `[u32; N]` with `v[i]` in `0..=max[i]`.
fn grid<const N: usize>(max: [u32; N]) -> impl Iterator<Item = [u32; N]> {
    let total: u32 = max.iter().map(|m| m + 1).product();
    (0..total).map(move |code| {
        let mut rest = code;
        std::array::from_fn(|i| {
            let value = rest % (max[i] + 1);
            rest /= max[i] + 1;
            value
        })
    })
}

/// `SkyDraw::runs` draws what the hand-unrolled `record_sky` drew: the
/// same runs, counts, firstInstance values and bound pipelines, the same
/// first pipeline, and the fullscreen triangle in the same cases. Small
/// counts exhaustively, with the coarse and mapsolo prefixes past their
/// runs to hit the clamps, coarse pipelines on and off, and mapsolo on
/// (`n_mapsolo` free) and off (`n_mapsolo == 0`, what `count_mapsolo`
/// returns then).
#[test]
fn sky_runs_match_legacy_counts() {
    let bodies = [SkyBodyPipe::Full, SkyBodyPipe::NoMap, SkyBodyPipe::Sphere];
    let mut cases = 0u32;
    for [
        quads,
        base,
        body,
        coarse,
        mapsolo,
        n_base,
        n_sphere,
        n_heavy,
        n_coarse,
        n_coarse_far,
        n_mapsolo,
    ] in grid([1, 1, 2, 1, 1, 3, 2, 3, 4, 4, 4])
    {
        if mapsolo == 0 && n_mapsolo != 0 {
            continue;
        }
        let coarse = coarse == 1;
        let draw = SkyDraw {
            quads: quads == 1,
            base: base == 1,
            body: bodies[body as usize],
            n_base,
            n_sphere,
            n_heavy,
            n_coarse,
            n_coarse_far,
            n_mapsolo,
        };
        let runs: Vec<SkyRun> = draw.runs(coarse).into_iter().flatten().collect();
        match legacy_sky_cmds(&draw, coarse) {
            Err(fullscreen) => {
                assert!(runs.is_empty(), "{draw:?} coarse {coarse}");
                assert_eq!(draw.fullscreen_frag(), fullscreen, "{draw:?}");
            }
            Ok(cmds) => {
                assert_eq!(runs, bound_draws(&cmds), "{draw:?} coarse {coarse}");
                assert_eq!(
                    cmds[0],
                    LegacyCmd::Bind(runs[0].frag, runs[0].rate),
                    "{draw:?} coarse {coarse}"
                );
            }
        }
        cases += 1;
    }
    assert_eq!(cases, 2 * 2 * 3 * 2 * (1 + 5) * 4 * 3 * 4 * 5 * 5);
}

/// The runs cut the uploaded tile list where the CPU partition put each
/// class. Built in `FarBodyRing::write` order (`fill_tile_lists` inside
/// the pack, then `partition_mapsolo`, then the two coarse splits), they
/// cover the live list contiguously from slot 0, and every tile in a run
/// has that run's class.
#[test]
fn sky_runs_tile_the_cpu_partition() {
    let view = view_pitched(0.0, 90.0, 1280, 720);
    let frames = TileFrames::build(&view).expect("frames");
    assert!(frames.matches(&view));
    let map_max = [0.0f32; MAX_FAR_MAPS];
    let map_min = [0.0f32; MAX_FAR_MAPS];
    // The planet below the horizon, a sphere on its disc (shared heavy
    // tiles) and a sphere up in the sky (sphere-only tiles).
    let bodies = [
        mapped_down(4.0, 1.0, 0.0, 0.0),
        placed(Vec3::new(0.0, -1.0, -1.0), 0.15, FarShape::Sphere, 2),
        placed(Vec3::new(0.4, 0.3, -1.0), 0.05, FarShape::Sphere, 3),
    ];
    // The sun up and to the left of the view axis keeps the base tiles
    // around its disc at 1x1. The moon is behind the camera.
    let query = coarse_query(Vec3::new(-0.4, 0.4, -1.0), false);
    for [mapsolo, coarse] in grid([1, 1]) {
        let (mapsolo, coarse) = (mapsolo == 1, coarse == 1);
        let mut table = super::pack_table(&bodies, Some(view), &map_max);
        let mut draw = SkyDraw::from_table(&table, true);
        assert!(draw.quads);
        draw.n_mapsolo = if mapsolo {
            partition_mapsolo(&mut table)
        } else {
            0
        };
        if coarse {
            draw.n_coarse = split_coarse_base(&mut table, &frames, &view, &query);
            draw.n_coarse_far = split_coarse_far(&mut table, &frames, &view, &map_min);
        }
        let label = format!("mapsolo {mapsolo} coarse {coarse} {draw:?}");
        let runs: Vec<SkyRun> = draw.runs(coarse).into_iter().flatten().collect();
        let n = used_tiles(&table);
        let mut next = 0u32;
        for run in &runs {
            assert_eq!(run.first, next, "{label}");
            next += run.count;
        }
        assert_eq!(next as usize, n, "{label}");
        let mut live = table.tile_index[..n].to_vec();
        live.sort_unstable();
        assert!(
            live.iter().enumerate().all(|(i, &index)| index == i as u32),
            "{label}"
        );

        let heavy = heavy_mask(&table);
        for run in &runs {
            let end = (run.first + run.count) as usize;
            for &index in &table.tile_index[run.first as usize..end] {
                let mask = table.tile_mask[index as usize];
                let solo = mask_is_mapsolo(&table, mask);
                let ok = match (run.frag, run.rate) {
                    (SkyFrag::Base, _) => mask == 0,
                    (SkyFrag::Sphere, _) => mask != 0 && mask & heavy == 0,
                    (SkyFrag::MapSolo, _) => solo,
                    (_, SkyRate::Coarse) => !mapsolo && solo,
                    (_, SkyRate::Fine) => mask & heavy != 0 && !(mapsolo && solo),
                };
                assert!(ok, "{label}: tile {index} mask {mask:#x} in {run:?}");
            }
        }
        let has = |frag: SkyFrag, rate: SkyRate| {
            runs.iter().any(|run| run.frag == frag && run.rate == rate)
        };
        assert!(has(SkyFrag::Base, SkyRate::Fine), "{label}");
        assert!(has(SkyFrag::Sphere, SkyRate::Fine), "{label}");
        assert!(has(SkyFrag::Full, SkyRate::Fine), "{label}");
        assert_eq!(has(SkyFrag::Base, SkyRate::Coarse), coarse, "{label}");
        assert_eq!(has(SkyFrag::MapSolo, SkyRate::Fine), mapsolo, "{label}");
        assert_eq!(
            has(SkyFrag::MapSolo, SkyRate::Coarse),
            mapsolo && coarse,
            "{label}"
        );
        assert_eq!(
            has(SkyFrag::Full, SkyRate::Coarse),
            !mapsolo && coarse,
            "{label}"
        );
    }
}

/// Every pixel of a lo-sphere interior tile meets the datum. Altitudes run
/// from 10 m to 1e6 m above the lo sphere; the horizon lane stays at 1 so
/// only the lo disc can admit a tile.
#[test]
fn lo_interior_pixels_hit_the_mapped_surface() {
    let radius = 31_017_520.0f32;
    let g = 9u32;
    let datum = datum_from(g, |d| {
        radius * (0.004 * (d.x * 3.0).sin() * (d.y * 2.0 + 0.4).cos() + 0.002 * (d.z * 5.0).sin())
    });
    let min_off = datum.iter().copied().fold(f32::INFINITY, f32::min);
    let max_off = datum.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min_off < 0.0 && max_off > 0.0,
        "relief {min_off}..{max_off}"
    );
    // Point the deepest sample at the eye. The eye is then `altitude`
    // above the surface as well as above the lo sphere, so it is outside
    // the star body and a lo-disc ray has a forward hit.
    let valley = lowland_up(g, &datum);
    let valley_rot = Quat::from_rotation_arc(valley, Vec3::Y);
    let mut map_min = [0.0f32; MAX_FAR_MAPS];
    let mut map_max = [0.0f32; MAX_FAR_MAPS];
    map_min[0] = min_off;
    map_max[0] = max_off;

    let mut state = 0xA11C_E5EDu32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let unit = |rng: &mut dyn FnMut() -> u32| rng() as f32 / u32::MAX as f32;
    // 10 m through 1e6 m, plus a couple of draws inside that range.
    let mut altitudes = vec![10.0f32, 1_000_000.0];
    for _ in 0..6 {
        let t = unit(&mut next);
        // Log span so both the near ground and the far approach show up.
        let log_h = 10.0f32.ln() + t * ((1.0e6f32).ln() - 10.0f32.ln());
        altitudes.push(log_h.exp());
    }
    let width = 192u32;
    let height = 108u32;
    let mut checked = 0u32;
    let mut interiors = 0u32;
    for (trial, &altitude) in altitudes.iter().enumerate() {
        let pitch = -75.0 + unit(&mut next) * 50.0;
        let yaw = unit(&mut next) * std::f32::consts::TAU;
        let distance = radius + min_off + altitude;
        assert!(distance > 0.0);
        let mut body = mapped_down(distance, radius, 1.0, 0.0);
        // Yaw around the valley so the relief turns under a fixed eye.
        body.rotation = Quat::from_rotation_y(yaw) * valley_rot;
        let rho_lo = (radius + min_off) / distance;
        assert!(
            rho_lo > 0.0 && rho_lo < 1.0,
            "alt {altitude} rho_lo {rho_lo}"
        );
        let view = view_pitched(pitch, 70.0, width, height);
        let Some(frames) = TileFrames::build(&view) else {
            continue;
        };
        let mut table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
        if table.header[0] == 0 {
            continue;
        }
        let (start, end) = heavy_run(&table);
        if end <= start {
            continue;
        }
        let n = split_coarse_far(&mut table, &frames, &view, &map_min);
        if n == 0 {
            continue;
        }
        interiors += n;
        let inv = view.view_proj.inverse();
        let tile_px = table.header[1];
        let tiles_x = table.header[2];
        let rho = radius / distance;
        for &index in &table.tile_index[start..start + n as usize] {
            let [x0, y0, x1, y1] = tile_rect(index, tile_px, tiles_x, width, height);
            for y in y0..y1 {
                for x in x0..x1 {
                    let ray = ray_at(&inv, width, height, x as f32 + 0.5, y as f32 + 0.5);
                    if !ray.is_finite() {
                        continue;
                    }
                    let hit = ray_mapped_fast(
                        ray,
                        body.dir,
                        rho,
                        distance,
                        body.rotation,
                        1.0,
                        g,
                        &datum,
                        min_off,
                        max_off,
                    );
                    let facing = ray.dot(body.dir);
                    let lo_disc = facing * facing - (1.0 - rho_lo * rho_lo);
                    assert!(
                        hit.is_some(),
                        "trial {trial} alt {altitude} pitch {pitch:.1} pixel {x},{y} tile {index} missed; facing {facing} rho {rho} rho_lo {rho_lo} lo_disc {lo_disc} ray {ray:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(interiors > 0, "no lo-sphere interior tile in the sweep");
    assert!(checked > 1000, "checked only {checked} interior pixels");
}
