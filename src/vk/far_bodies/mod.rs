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
use horizon::{
    HorizonCache, HorizonSlot, cache_dest, clear_tiles_above_horizon, horizon_cache_hit,
};
use horizon_build::build_horizon_bins;
use sky_draw::{partition_mapsolo, publish_far_gauges, split_coarse_base, split_coarse_far};
use table::{
    FarTableGpu, HORIZON_BINS, HORIZON_TABLES, list_bytes, nonzero_tiles, pack_table_cached,
    table_bytes,
};
use tiles::TileFrames;

/// `VOXEL_FAR_CULL=0` disables the cone reject, the CPU frustum compaction, and
/// the per-tile mask (every tile keeps every surviving body). Any other value,
/// including unset, leaves culling on. Read once.
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

/// Per-slot far-body SSBO. Identical bytes skip the map write.
/// `tiles` is the view-space tile cones, rebuilt when the projection, the
/// render extent, or the tile size changes and reused across camera turns.
pub(crate) struct FarBodyRing {
    bufs: PerSlot<HostBuffer>,
    /// Previous upload. Boxed: three inline tables would add ~210 KB to
    /// `Renderer`, which lives on the render thread's default stack.
    last: PerSlot<Option<Box<FarTableGpu>>>,
    tiles: TileFrames,
    draw: PerSlot<SkyDraw>,
    /// Reused across frames. Rebuilt when the eye moves more than 0.1% of
    /// its altitude above the lo sphere, or the body's rotation, radius,
    /// air, pixel angle, or datum generation changes.
    horizon: HorizonCache,
    datum_scratch: Vec<f32>,
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
            tiles: TileFrames::empty(),
            draw: PerSlot::new(std::array::from_fn(|_| SkyDraw::default())),
            horizon: HorizonCache::new(),
            datum_scratch: Vec::new(),
        }
    }

    /// Draw the sky recorded for `slot` after [`Self::write`].
    pub(crate) fn sky_draw(&self, slot: FrameSlot) -> SkyDraw {
        self.draw[slot]
    }

    /// Fill the two horizon tables from the datum and drop tiles that sit
    /// entirely above them. At most two Mapped bodies, the ones with the
    /// largest reference rho whose hi+air ball contains the eye. A cache hit
    /// skips the cell walk.
    fn publish_horizons(
        &mut self,
        table: &mut FarTableGpu,
        view: Option<&FarView>,
        maps: &super::far_maps::FarMaps,
        map_max: &[f32; MAX_FAR_MAPS],
        map_min: &[f32; MAX_FAR_MAPS],
    ) {
        let kept = (table.header[0] as usize).min(MAX_FAR_BODIES);
        let px = view.map(|v| v.px_max).unwrap_or(0.0);
        struct Cand {
            index: usize,
            rho: f32,
            map: usize,
        }
        let mut cands = Vec::new();
        for k in 0..kept {
            let gpu = &table.body[k];
            let shape = gpu.atmosphere[3];
            if !(3.5..4.5).contains(&shape) {
                continue;
            }
            let map_plus = gpu.seed[2];
            if map_plus == 0 || map_plus as usize > MAX_FAR_MAPS {
                continue;
            }
            let map = (map_plus as usize) - 1;
            let distance = gpu.albedo2[3];
            let rho = gpu.dir_rho[3];
            let air = gpu.albedo1[3];
            let max_off = map_max[map];
            if !(distance > 0.0)
                || !distance.is_finite()
                || !rho.is_finite()
                || !air.is_finite()
                || !max_off.is_finite()
            {
                continue;
            }
            let radius = rho * distance;
            let reach = radius + max_off + air.max(0.0);
            if !(distance <= reach + distance * 1.0e-5) {
                continue;
            }
            cands.push(Cand { index: k, rho, map });
        }
        cands.sort_by(|a, b| b.rho.total_cmp(&a.rho).then(a.index.cmp(&b.index)));
        cands.truncate(HORIZON_TABLES);

        let mut used = [false; HORIZON_TABLES];
        for (out_slot, cand) in cands.iter().enumerate() {
            let gpu = table.body[cand.index];
            let distance = gpu.albedo2[3];
            let radius = gpu.dir_rho[3] * distance;
            let air = gpu.albedo1[3].max(0.0);
            let min_off = map_min[cand.map];
            let altitude = distance - (radius + min_off);
            let Some(dir) = unit_dir(glam::Vec3::new(
                gpu.dir_rho[0],
                gpu.dir_rho[1],
                gpu.dir_rho[2],
            )) else {
                continue;
            };
            let eye = -dir * distance;
            let rot_bits = [
                gpu.rot[0].to_bits(),
                gpu.rot[1].to_bits(),
                gpu.rot[2].to_bits(),
                gpu.rot[3].to_bits(),
            ];
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
            let hit = self.horizon.slots.iter().position(|slot| {
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
                self.horizon.slots[i].bins
            } else {
                let Some(g) = maps.copy_datum(cand.map, &mut self.datum_scratch) else {
                    continue;
                };
                let rotation =
                    glam::Quat::from_xyzw(gpu.rot[0], gpu.rot[1], gpu.rot[2], gpu.rot[3]);
                let built = build_horizon_bins(
                    g,
                    &self.datum_scratch,
                    rotation,
                    -dir,
                    radius,
                    distance,
                    air,
                    px,
                );
                let dest = cache_dest(&self.horizon, &used, map_id);
                self.horizon.slots[dest] = HorizonSlot {
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
        if let Some(view) = view
            && self.tiles.matches(view)
        {
            clear_tiles_above_horizon(table, &self.tiles, view);
        }
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
        if let Some(view) = view.as_ref() {
            self.tiles.rebuild_if_changed(view);
        }
        let map_max = maps.max_offsets();
        let map_min = maps.min_offsets();
        let (mut table, dip) = pack_table_cached(
            bodies,
            view.as_ref(),
            Some(&self.tiles),
            &map_max,
            &map_min,
            sky_up,
        );
        self.publish_horizons(&mut table, view.as_ref(), maps, &map_max, &map_min);
        // `far.tiles` is the full-pipeline tile count (mask != 0).
        debug_assert_eq!(u64::from(table.list_header[1]), nonzero_tiles(&table));
        publish_far_gauges(
            bodies.len().min(MAX_FAR_BODIES) as u64,
            u64::from(table.header[0]),
            u64::from(table.list_header[1]),
        );
        let mut draw = SkyDraw::from_table(&table, far_cull_enabled());
        // Fullscreen sky (no tile grid) stays on the 1×1 triangle. The splits
        // rewrite the uploaded index prefix, so they run before the byte match.
        // Mapsolo comes first: the interior split then pulls its coarse prefix
        // out of that single-Mapped run, and the full-shader tiles stay after it.
        if draw.quads {
            if sky_mapsolo_enabled() {
                let n = partition_mapsolo(&mut table);
                debug_assert_eq!(n, draw.n_mapsolo);
                draw.n_mapsolo = n;
            }
            if let (Some(query), Some(view)) = (coarse.as_ref(), view.as_ref()) {
                if self.tiles.matches(view) {
                    draw.n_coarse = split_coarse_base(&mut table, &self.tiles, view, query);
                    draw.n_coarse_far = split_coarse_far(&mut table, &self.tiles, view, &map_min);
                }
            }
        }
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
            return dip;
        }
        unsafe {
            self.bufs[slot].write(0, bytes);
            self.bufs[slot].write(std::mem::offset_of!(FarTableGpu, list_header) as u64, lists);
        }
        self.last[slot] = Some(table);
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

#[cfg(test)]
mod tests;
