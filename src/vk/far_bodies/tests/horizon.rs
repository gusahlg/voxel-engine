use glam::{Quat, Vec3};

use super::support::{mapped_down, placed, view_pitched};
use crate::color::LinearRgb;
use crate::far_body::{FarBody, FarMapId, FarShape, MAX_FAR_MAPS};
use crate::vk::far_bodies::horizon::{HorizonSlot, clear_tiles_above_horizon, horizon_cache_hit};
use crate::vk::far_bodies::horizon_build::build_horizon_bins;
use crate::vk::far_bodies::table::{HORIZON_BINS, pack_table_cached};
use crate::vk::far_bodies::tiles::TileFrames;
use crate::vk::far_bodies::view::ViewBasis;

/// Reference-radius dip. A zero minimum-offset table leaves every shape,
/// including Mapped, on `radius/distance`.
fn horizon_dip(bodies: &[FarBody], sky_up: Vec3) -> f32 {
    super::horizon_dip(bodies, sky_up, &[0.0; MAX_FAR_MAPS])
}

/// `rho = R/(R+h)` straight down. The exact dip is `sqrt(1-rho²)`; for
/// small `h` that is `sqrt(2h/R)`.
fn dip_below(rho: f32, up: Vec3) -> f32 {
    horizon_dip(
        std::slice::from_ref(&placed(-up, rho, FarShape::Sphere, 1)),
        up,
    )
}

#[test]
fn horizon_dip_is_zero_without_a_ground_body() {
    assert_eq!(horizon_dip(&[], Vec3::Y).to_bits(), 0.0f32.to_bits());
    // Above the viewer, and level with the horizon: neither is ground.
    let above = placed(Vec3::Y, 0.9, FarShape::Sphere, 1);
    let sideways = placed(Vec3::X, 0.9, FarShape::Sphere, 2);
    // Just outside the 60° cone (`dot` a hair under 0.5). The compare is strict.
    let edge = placed(
        Vec3::new(0.87, -0.5, 0.0),
        0.9,
        FarShape::Mapped {
            map: crate::far_body::FarMapId(0),
            horizon: 1.0,
            air: 0.0,
        },
        3,
    );
    assert_eq!(horizon_dip(&[above], Vec3::Y).to_bits(), 0.0f32.to_bits());
    assert_eq!(
        horizon_dip(&[sideways], Vec3::Y).to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(horizon_dip(&[edge], Vec3::Y).to_bits(), 0.0f32.to_bits());
    // Inside the body (rho >= 1), including an inner sphere underfoot.
    let inside = placed(-Vec3::Y, 1.5, FarShape::InnerSphere, 4);
    assert_eq!(horizon_dip(&[inside], Vec3::Y).to_bits(), 0.0f32.to_bits());
    // A body above must not win over "no ground", even with a larger rho
    // than a body that fails the outside test.
    assert_eq!(
        horizon_dip(&[above, inside], Vec3::Y).to_bits(),
        0.0f32.to_bits()
    );
}

#[test]
fn horizon_dip_of_the_body_below_matches_sqrt_2h_over_r() {
    let r = 6_371_000.0f32;
    let h = 10_000.0f32;
    let rho = r / (r + h);
    let s = dip_below(rho, Vec3::Y);
    let approx = (2.0 * h / r).sqrt();
    let exact = (1.0 - rho * rho).max(0.0).sqrt();
    assert!((s - exact).abs() <= exact * 1e-6, "s {s} exact {exact}");
    assert!(
        (s - approx).abs() < 1e-4,
        "small-h dip {s} should track sqrt(2h/R) {approx}"
    );
    assert!(s < 0.5, "10 km on an Earth-sized body is under the clamp");
    // Non-unit up, and a ground body that is not world -Y.
    let up = Vec3::Z * 4.0;
    let s_z = dip_below(rho, up);
    assert_eq!(s_z.to_bits(), s.to_bits());
    assert_eq!(
        horizon_dip(
            std::slice::from_ref(&placed(-Vec3::Y, rho, FarShape::Sphere, 1)),
            up
        )
        .to_bits(),
        0.0f32.to_bits()
    );
}

#[test]
fn horizon_dip_ignores_bodies_above_or_sideways_and_clamps() {
    let ground = placed(-Vec3::Y, 0.98, FarShape::Sphere, 1);
    let above = placed(Vec3::Y, 0.99, FarShape::Sphere, 2);
    let side = placed(Vec3::X, 0.995, FarShape::Cube, 3);
    let s = horizon_dip(&[above, side, ground], Vec3::Y);
    let expect = (1.0 - 0.98 * 0.98f32).sqrt();
    assert!((s - expect).abs() < 1e-6, "{s} vs {expect}");
    // Largest rho wins, not the largest dip. The tiny body would clamp.
    let tiny = placed(
        Vec3::new(0.0, -1.0, 0.1).normalize(),
        0.2,
        FarShape::Sphere,
        4,
    );
    let s_large = horizon_dip(&[tiny, ground], Vec3::Y);
    assert!(
        (s_large - expect).abs() < 1e-6,
        "largest rho, got {s_large}"
    );
    // A small disc underfoot: sqrt(1-rho²) > 0.5, so the lane saturates.
    let s_clamp = dip_below(0.2, Vec3::Y);
    assert_eq!(s_clamp.to_bits(), 0.5f32.to_bits());
    // Just inside the 60° cone still counts.
    let inside_cone = placed(
        Vec3::new((1.0 - 0.6 * 0.6f32).sqrt(), -0.6, 0.0),
        0.97,
        FarShape::Sphere,
        5,
    );
    let s_cone = horizon_dip(&[inside_cone], Vec3::Y);
    let cone_expect = (1.0 - 0.97 * 0.97f32).sqrt();
    assert!((s_cone - cone_expect).abs() < 1e-6, "{s_cone}");
}

#[test]
fn pack_reports_the_same_horizon_dip() {
    let bodies = [
        placed(Vec3::Y, 0.4, FarShape::Sphere, 1),
        placed(-Vec3::Y, 0.96, FarShape::Sphere, 2),
    ];
    let map_min = [0.0; MAX_FAR_MAPS];
    let (table, dip) =
        pack_table_cached(&bodies, None, None, &[0.0; MAX_FAR_MAPS], &map_min, Vec3::Y);
    assert_eq!(table.header[0], 2);
    assert_eq!(dip.to_bits(), horizon_dip(&bodies, Vec3::Y).to_bits());
    assert!(dip > 0.0 && dip < 0.5);
}

/// Lowlands sit inside the reference sphere, so the sky dip has to follow
/// the lo sphere or the gradient edge draws above the silhouette.
#[test]
fn horizon_dip_of_a_mapped_ground_body_uses_the_lo_sphere() {
    let radius = 31_000_000.0f32;
    let distance = radius + 50_000.0;
    let min_off = -278_000.0f32;
    let mut map_min = [0.0f32; MAX_FAR_MAPS];
    map_min[3] = min_off;
    let mapped = |map: u8| FarBody {
        dir: -Vec3::Y,
        distance,
        radius,
        shape: FarShape::Mapped {
            map: FarMapId(map),
            horizon: 0.2,
            air: 12.0,
        },
        rotation: Quat::IDENTITY,
        albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
        atmosphere: LinearRgb([0.1, 0.0, 0.0]),
        seed: 7,
    };
    let reference = FarBody {
        shape: FarShape::Sphere,
        ..mapped(3)
    };
    let lo_sphere = FarBody {
        radius: radius + min_off,
        shape: FarShape::Sphere,
        ..mapped(3)
    };

    let dip_mapped = super::horizon_dip(std::slice::from_ref(&mapped(3)), Vec3::Y, &map_min);
    let dip_ref = super::horizon_dip(std::slice::from_ref(&reference), Vec3::Y, &map_min);
    let dip_lo = super::horizon_dip(std::slice::from_ref(&lo_sphere), Vec3::Y, &map_min);
    assert!(
        dip_mapped > dip_ref,
        "lo sphere {dip_mapped} should dip past the reference sphere {dip_ref}"
    );
    assert_eq!(dip_mapped.to_bits(), dip_lo.to_bits());
    let rho = radius / distance;
    let exact = (1.0 - rho * rho).max(0.0).sqrt();
    assert_eq!(dip_ref.to_bits(), exact.to_bits());

    // A sphere does not read the offset table.
    let mut noisy = [123_456.0f32; MAX_FAR_MAPS];
    noisy[3] = min_off;
    assert_eq!(
        super::horizon_dip(std::slice::from_ref(&reference), Vec3::Y, &noisy).to_bits(),
        dip_ref.to_bits()
    );

    // Unknown id, and a cleared slot whose minimum is still 0.
    assert_eq!(
        super::horizon_dip(
            std::slice::from_ref(&mapped(MAX_FAR_MAPS as u8)),
            Vec3::Y,
            &map_min
        )
        .to_bits(),
        dip_ref.to_bits()
    );
    assert_eq!(
        super::horizon_dip(std::slice::from_ref(&mapped(0)), Vec3::Y, &map_min).to_bits(),
        dip_ref.to_bits()
    );
    // A non-finite offset is not a datum; stay on the reference radius.
    map_min[3] = f32::NAN;
    assert_eq!(
        super::horizon_dip(std::slice::from_ref(&mapped(3)), Vec3::Y, &map_min).to_bits(),
        dip_ref.to_bits()
    );

    map_min[3] = min_off;
    let (_table, packed) = pack_table_cached(
        std::slice::from_ref(&mapped(3)),
        None,
        None,
        &[0.0; MAX_FAR_MAPS],
        &map_min,
        Vec3::Y,
    );
    assert_eq!(packed.to_bits(), dip_mapped.to_bits());
}

#[test]
fn horizon_cache_keeps_a_small_move_and_drops_rotation() {
    let mut slot = HorizonSlot::empty();
    slot.live = true;
    slot.map = 1;
    slot.generation = 4;
    slot.radius_bits = 10.0f32.to_bits();
    slot.air_bits = 2.0f32.to_bits();
    slot.px_bits = 0.0f32.to_bits();
    slot.rot_bits = [0, 0, 0, 1.0f32.to_bits()];
    slot.eye = Vec3::new(0.0, 1_000.0, 0.0);
    slot.altitude = 1_000.0;
    let eye = slot.eye;
    let rot = slot.rot_bits;
    assert!(horizon_cache_hit(
        &slot,
        1,
        4,
        slot.radius_bits,
        slot.air_bits,
        slot.px_bits,
        rot,
        eye,
    ));
    // 0.05% of the altitude stays cached. 0.2% rebuilds.
    let near = eye + Vec3::Y * (0.0005 * slot.altitude);
    assert!(horizon_cache_hit(
        &slot,
        1,
        4,
        slot.radius_bits,
        slot.air_bits,
        slot.px_bits,
        rot,
        near,
    ));
    let far = eye + Vec3::Y * (0.002 * slot.altitude);
    assert!(!horizon_cache_hit(
        &slot,
        1,
        4,
        slot.radius_bits,
        slot.air_bits,
        slot.px_bits,
        rot,
        far,
    ));
    let mut spun = rot;
    spun[0] ^= 1;
    assert!(!horizon_cache_hit(
        &slot,
        1,
        4,
        slot.radius_bits,
        slot.air_bits,
        slot.px_bits,
        spun,
        eye,
    ));
    slot.altitude = 0.0;
    assert!(horizon_cache_hit(
        &slot,
        1,
        4,
        slot.radius_bits,
        slot.air_bits,
        slot.px_bits,
        rot,
        eye,
    ));
    assert!(!horizon_cache_hit(
        &slot,
        1,
        4,
        slot.radius_bits,
        slot.air_bits,
        slot.px_bits,
        rot,
        near,
    ));
}

/// `horizon` is 1, so the scalar cone paints every tile. The eye is inside
/// the hi+air ball only because one cell is a mountain; the local ground
/// is below the eye. Tiles looking up lose the bit. The downward tile keeps it.
#[test]
fn upward_tiles_lose_the_bit_above_the_horizon_table() {
    let radius = 10_000.0f32;
    let air = 10.0f32;
    let altitude = 100.0f32;
    let g = 9u32;
    let gg = g as usize;
    let mut datum = vec![0.0f32; 6 * gg * gg];
    // +X face centre cell. Far from a view that looks along −Z.
    let mountain = 2_000.0f32;
    datum[0 * gg * gg + 4 * gg + 4] = mountain;
    let distance = radius + altitude;
    assert!(distance < radius + mountain + air);
    let mut body = mapped_down(distance, radius, 1.0, air);
    body.rotation = Quat::IDENTITY;
    let mut map_max = [0.0f32; MAX_FAR_MAPS];
    map_max[0] = mountain;
    let view = view_pitched(0.0, 70.0, 640, 360);
    let frames = TileFrames::build(&view).expect("tiles");
    let mut table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
    assert_eq!(table.header[0], 1);
    assert!(
        table.tile_mask[..frames.samples.len()]
            .iter()
            .all(|m| *m == 1),
        "horizon 1 inside the hi ball paints every tile"
    );
    let up = -body.dir;
    let bins = build_horizon_bins(
        g,
        &datum,
        body.rotation,
        up,
        radius,
        distance,
        air,
        view.px_max,
    );
    table.horizon_id[0] = 0;
    table.horizon_sin[..HORIZON_BINS].copy_from_slice(&bins);
    let tiles = frames.view(&view).expect("the frames match the view");
    assert!(clear_tiles_above_horizon(&mut table, &tiles).is_some());
    let basis = ViewBasis::from_view_proj(view.view_proj).expect("basis");
    let up_v = basis.to_view(up).normalize();
    let mut highest = 0usize;
    let mut lowest = 0usize;
    let mut hi = f32::NEG_INFINITY;
    let mut lo = f32::INFINITY;
    for (i, tile) in frames.samples.iter().enumerate() {
        let mu = tile.centre.dot(up_v);
        if mu > hi {
            hi = mu;
            highest = i;
        }
        if mu < lo {
            lo = mu;
            lowest = i;
        }
    }
    assert_eq!(
        table.tile_mask[highest], 0,
        "upward tile mu {hi} kept the bit"
    );
    assert_eq!(
        table.tile_mask[lowest] & 1,
        1,
        "downward tile mu {lo} lost the bit"
    );
}
