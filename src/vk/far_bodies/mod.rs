//! Host ring of far-body records for the sky pass (set 0, binding 2).
//! One coherent buffer per frame-in-flight, written before the sky draw.
//! The record matches `FarGpu` in `shaders/far_table.slang` (160 bytes, std430).
//! A 16-byte cone sits in front of each record so a pixel can reject the body
//! before loading it. A tile mask follows the records: one bit per kept body,
//! one word per screen tile, so a sky pixel skips bodies that miss its tile.
//! After the mask, compact tile-index lists feed the instanced tile-quad
//! vertex shader: mask == 0, then sphere-only tiles, then tiles that meet a
//! cube, rounded or mapped body. When mapsolo is on, that heavy run is stably
//! split into tiles whose mask is exactly one Mapped body, then the rest.
//! When a coarse-shading query is passed, the base run is stably split into
//! tiles the sun and moon discs miss, then the rest, and the heavy run is
//! split into mapped-interior tiles, then the rest. With mapsolo on, that
//! interior is a prefix of the single-Mapped run and is drawn at 2×2 on the
//! loop-free fragment. A mapped-interior tile lies inside the horizon disc
//! or, with the eye outside the lo sphere, inside that sphere's disc. The
//! edge band stays at full rate. Both use the fixed-point march.
//! Two azimuthal horizon tables follow the body records. Each is 256 sines
//! of elevation around the local up of a Mapped body the eye is inside the
//! hi+air ball of. A tile whose cone sits above its table loses that body's
//! bit; the shader skips a ray above the same table.
//! `list_header.x` stays the whole base count; the shader still indexes by
//! instance id.
//!
//! Submodules: `table` (the GPU layout, packing and upload bytes), `view`
//! (the view, its basis, the frustum cull), `cones` (angular extents of a
//! body), `tiles` (the tile grid, masks and tile-index runs), `sky_draw` (the
//! coarse and mapsolo splits, [`SkyDraw`]), `horizon` and `horizon_build`
//! (the dip and the azimuthal horizon tables), `debug` (`VOXEL_SKY_DEBUG`).

use ash::vk;

use crate::far_body::{FarBody, MAX_FAR_BODIES, MAX_FAR_MAPS};
use crate::rev::{FrameSlot, PerSlot};
use crate::vk::buffers::HostBuffer;

mod cones;
mod debug;
mod horizon;
mod horizon_build;
mod sky_draw;
mod table;
mod tiles;
mod view;

pub(crate) use sky_draw::{SkyCoarseQuery, SkyDraw, stars_drawn};
pub(crate) use view::{FarView, far_view};

use cones::unit_dir;
use debug::log_sky_debug;
use horizon::{HorizonCache, HorizonSlot, cache_dest, horizon_cache_hit, horizon_candidates};
use horizon_build::build_horizon_bins;
use sky_draw::{classify_tiles, publish_far_gauges};
use table::{
    FarTableGpu, HORIZON_BINS, HORIZON_TABLES, MAX_FAR_TILES, list_bytes, nonzero_tiles,
    pack_table_into, table_bytes, zeroed_table,
};
use tiles::TileFrames;

/// `VOXEL_FAR_CULL=0` disables the cone reject, the CPU frustum compaction, and
/// the per-tile mask (every live tile keeps every surviving body, the mapped
/// horizon cone and the horizon-table tile clear included). The shader's
/// per-ray horizon-table skip still runs. Any other value, including unset,
/// leaves culling on. Read once.
fn far_cull_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| !std::env::var("VOXEL_FAR_CULL").is_ok_and(|v| v == "0"))
}

/// `VOXEL_SKY_MAPSOLO=0` draws every heavy tile with the full fragment (or the
/// no-mapped / sphere variant the frame already chose). Any other value,
/// including unset, draws a heavy tile whose mask is exactly one Mapped body
/// with the loop-free fragment. Read once.
fn sky_mapsolo_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| !std::env::var("VOXEL_SKY_MAPSOLO").is_ok_and(|v| v == "0"))
}

/// The far-body A/B switches. [`FarBodyRing::write`] reads them and passes
/// them down, so the host pipeline can be tested with either setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FarSwitches {
    /// [`far_cull_enabled`].
    pub(super) cull: bool,
    /// [`sky_mapsolo_enabled`].
    pub(super) mapsolo: bool,
}

impl FarSwitches {
    fn from_env() -> Self {
        Self {
            cull: far_cull_enabled(),
            mapsolo: sky_mapsolo_enabled(),
        }
    }
}

/// Per-slot far-body SSBO. Identical bytes skip the map write.
/// `tiles` is the view-space tile cones, rebuilt when the projection, the
/// render extent, or the tile size changes and reused across camera turns.
pub(crate) struct FarBodyRing {
    bufs: PerSlot<HostBuffer>,
    /// Previous upload. Boxed: three inline tables would add ~210 KB to
    /// `Renderer`, which lives on the render thread's default stack.
    last: PerSlot<Option<Box<FarTableGpu>>>,
    /// The table the next write packs into: the one the last byte match
    /// left unused, or the upload a slot just replaced.
    spare: Option<Box<FarTableGpu>>,
    tiles: TileFrames,
    draw: PerSlot<SkyDraw>,
    /// Reused across frames. Rebuilt when the eye moves more than 0.1% of
    /// its altitude above the lo sphere, or the body's rotation, radius,
    /// air, pixel angle, or datum generation changes.
    horizon: HorizonCache,
    datum_scratch: Vec<f32>,
    /// The tile-index partitions' scratch, sized for the longest run.
    partition_scratch: Vec<u32>,
}

impl FarBodyRing {
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Self {
        let size = std::mem::size_of::<FarTableGpu>() as u64;
        let make = || {
            let mut b = HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER);
            unsafe { b.maintain(instance, device, physical, size) };
            b
        };
        Self {
            bufs: PerSlot::new(std::array::from_fn(|_| make())),
            last: PerSlot::new(std::array::from_fn(|_| None)),
            spare: None,
            tiles: TileFrames::empty(),
            draw: PerSlot::new(std::array::from_fn(|_| SkyDraw::default())),
            horizon: HorizonCache::new(),
            datum_scratch: Vec::new(),
            partition_scratch: Vec::with_capacity(MAX_FAR_TILES),
        }
    }

    /// Draw the sky recorded for `slot` after [`Self::write`].
    pub(crate) fn sky_draw(&self, slot: FrameSlot) -> SkyDraw {
        self.draw[slot]
    }

    /// Pack and upload. Returns the horizon-dip sine for `sky_bitangent.w`
    /// ([`horizon_dip`]), including when the table bytes are unchanged.
    ///
    /// [`horizon_dip`]: horizon::horizon_dip
    pub(crate) fn write(
        &mut self,
        slot: FrameSlot,
        bodies: &[FarBody],
        view: Option<FarView>,
        maps: &super::far_maps::FarMaps,
        coarse: Option<SkyCoarseQuery>,
        sky_up: glam::Vec3,
    ) -> f32 {
        let switches = FarSwitches::from_env();
        // The frame's one tile-frame match; every pass below reuses it.
        let tiles = view.as_ref().and_then(|view| self.tiles.sync(view));
        let map_max = maps.max_offsets();
        let map_min = maps.min_offsets();
        let mut table = self.spare.take().unwrap_or_else(zeroed_table);
        let (dip, runs) = pack_table_into(
            &mut table,
            bodies,
            view.as_ref(),
            tiles.as_ref(),
            &map_max,
            &map_min,
            sky_up,
            switches.cull,
        );
        publish_horizons(
            &mut self.horizon,
            &mut self.datum_scratch,
            &mut table,
            view.as_ref().map_or(0.0, |view| view.px_max),
            maps,
            &map_max,
            &map_min,
        );
        let draw = classify_tiles(
            &mut table,
            runs,
            tiles.as_ref(),
            coarse.as_ref(),
            &map_min,
            switches,
            &mut self.partition_scratch,
        );
        // `far.tiles` is the full-pipeline tile count (mask != 0).
        debug_assert_eq!(u64::from(table.list_header[1]), nonzero_tiles(&table));
        publish_far_gauges(
            bodies.len().min(MAX_FAR_BODIES) as u64,
            u64::from(table.header[0]),
            u64::from(table.list_header[1]),
        );
        draw.publish_gauges();
        self.draw[slot] = draw;
        if super::uniforms::sky_debug_enabled() {
            log_sky_debug(bodies, &table, view.as_ref(), dip, &map_min, sky_up);
        }
        let bytes = table_bytes(&table);
        let lists = list_bytes(&table);
        if self.last[slot]
            .as_ref()
            .is_some_and(|prev| table_bytes(prev) == bytes && list_bytes(prev) == lists)
        {
            self.spare = Some(table);
            return dip;
        }
        unsafe {
            self.bufs[slot].write(0, bytes);
            self.bufs[slot].write(std::mem::offset_of!(FarTableGpu, list_header) as u64, lists);
        }
        self.spare = self.last[slot].replace(table);
        dip
    }

    pub(crate) fn buffer(&self, slot: FrameSlot) -> vk::Buffer {
        self.bufs[slot]
            .bound()
            .expect("the far-body table is written before the sky pass binds it")
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for buf in self.bufs.iter_mut() {
            unsafe { buf.destroy(device) };
        }
    }
}

/// Fill the two horizon tables from the datum: [`horizon_candidates`], each
/// from the cache when it still holds, else built. `px` is the view's
/// [`FarView::px_max`] (0 with no view). The tiles above a table lose that
/// body's bit later, in [`classify_tiles`].
fn publish_horizons(
    cache: &mut HorizonCache,
    datum_scratch: &mut Vec<f32>,
    table: &mut FarTableGpu,
    px: f32,
    maps: &super::far_maps::FarMaps,
    map_max: &[f32; MAX_FAR_MAPS],
    map_min: &[f32; MAX_FAR_MAPS],
) {
    let cands = horizon_candidates(table, map_max);
    let mut used = [false; HORIZON_TABLES];
    for (out_slot, cand) in cands.iter().flatten().enumerate() {
        let gpu = table.body[cand.index];
        let distance = gpu.distance();
        let radius = gpu.rho() * distance;
        let air = gpu.air().max(0.0);
        let min_off = map_min[cand.map];
        let altitude = distance - (radius + min_off);
        let Some(dir) = unit_dir(gpu.dir()) else {
            continue;
        };
        let eye = -dir * distance;
        let rot_bits = gpu.rot.map(f32::to_bits);
        let Some((g_stamp, generation)) = maps.map_stamp(cand.map) else {
            continue;
        };
        if g_stamp < 2 {
            continue;
        }
        let map_id = cand.map as u32;
        let px_bits = px.to_bits();
        let radius_bits = radius.to_bits();
        let air_bits = air.to_bits();
        let hit = cache.slots.iter().position(|slot| {
            horizon_cache_hit(
                slot,
                map_id,
                generation,
                radius_bits,
                air_bits,
                px_bits,
                rot_bits,
                eye,
            )
        });
        let bins = if let Some(i) = hit {
            used[i] = true;
            cache.slots[i].bins
        } else {
            let Some(g) = maps.copy_datum(cand.map, datum_scratch) else {
                continue;
            };
            let rotation = glam::Quat::from_xyzw(gpu.rot[0], gpu.rot[1], gpu.rot[2], gpu.rot[3]);
            let built =
                build_horizon_bins(g, datum_scratch, rotation, -dir, radius, distance, air, px);
            let dest = cache_dest(cache, &used, map_id);
            cache.slots[dest] = HorizonSlot {
                live: true,
                map: map_id,
                generation,
                radius_bits,
                air_bits,
                px_bits,
                rot_bits,
                eye,
                altitude,
                bins: built,
            };
            used[dest] = true;
            built
        };
        table.horizon_id[out_slot] = cand.index as u32;
        let start = out_slot * HORIZON_BINS;
        table.horizon_sin[start..start + HORIZON_BINS].copy_from_slice(&bins);
    }
}

#[cfg(test)]
mod tests;
