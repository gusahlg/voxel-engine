use glam::{Quat, Vec3};

use super::support::{datum_from, view_pitched};
use crate::far_body::mirror::{chart_sample_dir, ray_mapped, ray_mapped_limb};
use crate::vk::far_bodies::horizon::{
    horizon_axes, horizon_azimuth, horizon_bin, ray_above_horizon,
};
use crate::vk::far_bodies::horizon_build::{
    HORIZON_AZ_PAD, build_horizon_bins, fast_asin, fast_atan, fast_atan2,
};
use crate::vk::far_bodies::table::HORIZON_BINS;

#[test]
fn flat_horizon_puts_the_zenith_above_and_keeps_the_nadir() {
    let g = 2u32;
    let datum = vec![0.0f32; 24];
    let radius = 1_000.0f32;
    let air = 10.0f32;
    let distance = radius + air + 100.0;
    let up = Vec3::Y;
    let bins = build_horizon_bins(g, &datum, Quat::IDENTITY, up, radius, distance, air, 0.0);
    let (east, north) = horizon_axes(up).expect("axes");
    assert!(ray_above_horizon(&bins, up, east, north, up));
    assert!(!ray_above_horizon(&bins, up, east, north, -up));
    let rho = radius / distance;
    let hit = ray_mapped(
        -up,
        -up,
        rho,
        distance,
        Quat::IDENTITY,
        1.0,
        g,
        &datum,
        0.0,
        0.0,
    );
    assert!(
        hit.is_some(),
        "nadir misses a flat sphere outside the hi shell"
    );
}

/// No ray that hits the surface or the limb is classified above the table.
/// Relief is a few percent of the radius. Eyes run from 10 m to 2e6 m
/// above the lo sphere. `horizon` is 1 so the scalar plane does not reject.
#[test]
fn no_hit_or_limb_is_above_the_horizon_table() {
    let radius = 31_017_520.0f32;
    let mut state = 0xC0FF_EE01u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let unit = |rng: &mut dyn FnMut() -> u32| rng() as f32 / u32::MAX as f32;
    let mut above_ok = 0u32;
    let mut hits = 0u32;
    for trial in 0..4u32 {
        let g = if trial % 2 == 0 { 5u32 } else { 7u32 };
        let amp = 0.04 + unit(&mut next) * 0.04;
        let datum = datum_from(g, |d| {
            radius
                * amp
                * ((d.x * 3.0 + trial as f32).sin() * (d.y * 2.0).cos()
                    + 0.5 * (d.z * 5.0 + 0.7).sin())
        });
        let min_off = datum.iter().copied().fold(f32::INFINITY, f32::min);
        let max_off = datum.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(min_off < 0.0 && max_off > min_off + radius * 0.02);
        let air = radius * 0.001;
        let yaw = unit(&mut next) * std::f32::consts::TAU;
        let rotation = Quat::from_rotation_y(yaw);
        let px = 2.0e-4f32;
        for alt_i in 0..6 {
            let t = alt_i as f32 / 5.0;
            let log_h = 10.0f32.ln() + t * ((2.0e6f32).ln() - 10.0f32.ln());
            let altitude = log_h.exp();
            let distance = radius + min_off + altitude;
            let rho = radius / distance;
            let dir = -Vec3::Y;
            let up = Vec3::Y;
            let bins = build_horizon_bins(g, &datum, rotation, up, radius, distance, air, px);
            let (east, north) = horizon_axes(up).expect("axes");
            let mut rays = Vec::new();
            rays.push(-up);
            rays.push(up);
            for _ in 0..24 {
                let x = unit(&mut next) * 2.0 - 1.0;
                let y = unit(&mut next) * 2.0 - 1.0;
                let z = unit(&mut next) * 2.0 - 1.0;
                let v = Vec3::new(x, y, z);
                if v.length_squared() > 1.0e-6 {
                    rays.push(v.normalize());
                }
            }
            for ray in rays {
                let above = ray_above_horizon(&bins, up, east, north, ray);
                let hit = ray_mapped(
                    ray, dir, rho, distance, rotation, 1.0, g, &datum, min_off, max_off,
                );
                let limb =
                    ray_mapped_limb(ray, dir, rho, distance, rotation, g, &datum, max_off, air);
                if hit.is_some() {
                    hits += 1;
                }
                if above {
                    assert!(
                        hit.is_none() && !limb,
                        "trial {trial} alt {altitude} above a hit={hit:?} limb {limb} ray {ray:?}"
                    );
                    above_ok += 1;
                }
            }
        }
    }
    assert!(hits > 20, "the sweep barely hit the surface ({hits})");

    // Flat underfoot, one mountain on +X. The eye is outside the local
    // air shell and still inside the mountain's hi ball, so the zenith
    // clears the table while a hit on the mountain does not.
    let g = 9u32;
    let gg = g as usize;
    let mut datum = vec![0.0f32; 6 * gg * gg];
    let mountain = radius * 0.1;
    datum[4 * gg + 4] = mountain;
    let air = 20_000.0f32;
    let rotation = Quat::IDENTITY;
    for altitude in [50_000.0f32, 2.0e6] {
        let distance = radius + altitude;
        assert!(distance > radius + air && distance < radius + mountain + air);
        let up = Vec3::Y;
        let bins = build_horizon_bins(g, &datum, rotation, up, radius, distance, air, 0.0);
        let (east, north) = horizon_axes(up).expect("axes");
        assert!(
            ray_above_horizon(&bins, up, east, north, up),
            "zenith at {altitude} was not above the local ground"
        );
        above_ok += 1;
        let rho = radius / distance;
        for ray in [-up, up, Vec3::X, Vec3::new(1.0, -0.2, 0.0).normalize()] {
            let above = ray_above_horizon(&bins, up, east, north, ray);
            let hit = ray_mapped(
                ray, -up, rho, distance, rotation, 1.0, g, &datum, 0.0, mountain,
            );
            let limb = ray_mapped_limb(ray, -up, rho, distance, rotation, g, &datum, mountain, air);
            if above {
                assert!(
                    hit.is_none() && !limb,
                    "mountain alt {altitude} above a hit ray {ray:?} limb {limb}"
                );
            }
        }
    }
    assert!(
        above_ok > 0,
        "the zenith was never above a directional horizon"
    );
}

/// Home-planet numbers: R = 31,017,520, g = 33, air 20,000, lowland under
/// the eye, a +1M highland 25–30M blocks away toward +X. `px` is the
/// 3440×1440, 70° sky margin.
fn home_horizon_inputs(altitude: f32) -> (f32, f32, f32, f32, Vec<f32>) {
    let radius = 31_017_520.0f32;
    let g = 33u32;
    let mut highland = 0u32;
    let datum = datum_from(g, |d| {
        let arc = d.dot(Vec3::Y).clamp(-1.0, 1.0).acos() * radius;
        // Same atan2(east, north) as the table. 0 is +X.
        let az = d.z.atan2(d.x);
        if (25.0e6..=30.0e6).contains(&arc) && az.abs() <= 0.5 {
            highland += 1;
            1_000_000.0
        } else {
            0.0
        }
    });
    assert!(highland > 8, "highland covered only {highland} samples");
    let view = view_pitched(0.0, 70.0, 3440, 1440);
    (radius, radius + altitude, 20_000.0, view.px_max, datum)
}

/// (a) Open-sky bins stay under +0.1, so the table actually rejects sky.
/// (b) No surface or limb hit on this datum is classified above it.
#[test]
fn open_sky_bins_reject_above_a_distant_highland() {
    let up = Vec3::Y;
    let (east, north) = horizon_axes(up).expect("axes");
    let open_az = horizon_azimuth(-Vec3::X, up, east, north);
    let open_bin = horizon_bin(open_az);
    for altitude in [10.0f32, 10_000.0, 50_000.0] {
        let (radius, distance, air, px, datum) = home_horizon_inputs(altitude);
        let g = 33u32;
        let bins = build_horizon_bins(g, &datum, Quat::IDENTITY, up, radius, distance, air, px);
        for delta in -16..=16 {
            let idx = (open_bin as i32 + delta).rem_euclid(HORIZON_BINS as i32) as usize;
            assert!(
                bins[idx] < 0.1,
                "alt {altitude} open bin {idx} sine {} (px {px})",
                bins[idx]
            );
        }
        assert!(
            ray_above_horizon(&bins, up, east, north, up),
            "alt {altitude} zenith was not rejected"
        );
        let down_open = (-up * 0.5 - Vec3::X * 0.8660254).normalize();
        assert!(
            !ray_above_horizon(&bins, up, east, north, down_open),
            "alt {altitude} a ray 30° below the horizon was rejected; bin {}",
            bins[open_bin]
        );
        let up_open = (up * 0.5 - Vec3::X * 0.8660254).normalize();
        assert!(
            ray_above_horizon(&bins, up, east, north, up_open),
            "alt {altitude} open sky 30° up was kept; bin {}",
            bins[open_bin]
        );

        let rho = radius / distance;
        let min_off = 0.0f32;
        let max_off = 1_000_000.0f32;
        let mut hits = 0u32;
        let mut rays = vec![-up, Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z];
        let gg = g as usize;
        for face in 0..6usize {
            for (j, i) in [(8u32, 16u32), (16, 16), (24, 16), (16, 8), (16, 24)] {
                let s = chart_sample_dir(g, face, i, j);
                let off = datum[face * gg * gg + j as usize * gg + i as usize];
                let point = s * (radius + off);
                let eye = up * distance;
                let to = point - eye;
                if to.length_squared() > 1.0 {
                    rays.push(to.normalize());
                }
            }
        }
        for ray in rays {
            let above = ray_above_horizon(&bins, up, east, north, ray);
            let hit = ray_mapped(
                ray,
                -up,
                rho,
                distance,
                Quat::IDENTITY,
                1.0,
                g,
                &datum,
                min_off,
                max_off,
            );
            let limb = ray_mapped_limb(
                ray,
                -up,
                rho,
                distance,
                Quat::IDENTITY,
                g,
                &datum,
                max_off,
                air,
            );
            if hit.is_some() || limb {
                hits += 1;
                assert!(
                    !above,
                    "alt {altitude} rejected a hit={hit:?} limb {limb} ray {ray:?}"
                );
            }
        }
        assert!(hits > 0, "alt {altitude} never hit the datum or the limb");
    }
}

/// At 50 km above a home-datum lowland, the limb top along every azimuth
/// stays at or under the horizon table. The table still widens by a pixel;
/// the limb is the air the ray crosses.
#[test]
fn lowland_limb_does_not_rise_through_the_horizon_table() {
    let g = 33u32;
    let datum = crate::far_body::mirror::home_datum(g);
    let (up, foot) = crate::far_body::mirror::lowland_foot(g, &datum);
    let radius = 31_017_520.0f32;
    let air = 20_000.0f32;
    let altitude = 50_000.0f32;
    let distance = radius + foot + altitude;
    let px = std::f32::consts::FRAC_PI_2 / 3440.0;
    let bins = build_horizon_bins(g, &datum, Quat::IDENTITY, up, radius, distance, air, px);
    let rho = radius / distance;
    let max_off = datum.iter().copied().fold(f32::MIN, f32::max);
    let dir = -up;
    let (east, north) = horizon_axes(up).expect("axes");
    let mut limbs = 0u32;
    for k in 0..96 {
        let azimuth = k as f32 * std::f32::consts::TAU / 96.0;
        let Some(top) = crate::far_body::mirror::mapped_limb_top_angle(
            dir, azimuth, rho, distance, g, &datum, max_off, air,
        ) else {
            continue;
        };
        limbs += 1;
        let ray = crate::far_body::mirror::ray_from_nadir(dir, azimuth, top as f32);
        let mu = ray.dot(up);
        let bin = horizon_bin(horizon_azimuth(ray, up, east, north));
        assert!(
            mu <= bins[bin] + 1.0e-5,
            "az {azimuth} limb mu {mu} exceeds bin {} ({bin}), foot {foot}",
            bins[bin]
        );
    }
    assert!(limbs > 90, "only {limbs} azimuths drew a limb at 50 km");
}

/// Per-frame cost of the g = 33 table. Release budget is 0.2 ms; debug is
/// only a backstop so a 4^6 fan-out still fails the suite.
#[test]
fn horizon_table_at_g33_builds_within_a_fifth_of_a_millisecond() {
    let up = Vec3::Y;
    let built = [10.0f32, 10_000.0, 50_000.0].map(home_horizon_inputs);
    let time_one = |radius, distance, air, px, datum: &Vec<f32>| {
        std::hint::black_box(build_horizon_bins(
            33,
            datum,
            Quat::IDENTITY,
            up,
            radius,
            distance,
            air,
            px,
        ))
    };
    for (radius, distance, air, px, datum) in &built {
        time_one(*radius, *distance, *air, *px, datum);
    }
    let iters = if cfg!(debug_assertions) { 2 } else { 20 };
    let rounds = if cfg!(debug_assertions) { 1 } else { 5 };
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let start = std::time::Instant::now();
        for _ in 0..iters {
            for (radius, distance, air, px, datum) in &built {
                time_one(*radius, *distance, *air, *px, datum);
            }
        }
        let each = start.elapsed().as_nanos() as f64 / (iters as f64 * built.len() as f64);
        best = best.min(each);
    }
    eprintln!("horizon g=33 build {best:.0} ns");
    let limit = if cfg!(debug_assertions) {
        80_000_000.0
    } else {
        200_000.0
    };
    assert!(
        best < limit,
        "horizon build {best:.0} ns exceeds {limit:.0}"
    );
}

/// The azimuth pad is twice this error, so a fast angle cannot open a gap
/// in the bin range.
#[test]
fn fast_atan_stays_inside_the_azimuth_pad() {
    let mut worst = 0.0f32;
    let ang = |d: f32| {
        let a = d.abs();
        a.min(std::f32::consts::TAU - a)
    };
    for i in 0..=20_000 {
        let z = i as f32 / 20_000.0;
        worst = worst.max(ang(fast_atan(z) - z.atan()));
    }
    for i in 0..64 {
        for j in 0..64 {
            let x = (i as f32 - 32.0) / 8.0;
            let y = (j as f32 - 32.0) / 8.0;
            if x == 0.0 && y == 0.0 {
                continue;
            }
            worst = worst.max(ang(fast_atan2(y, x) - y.atan2(x)));
        }
    }
    for i in 0..=20_000 {
        let s = i as f32 / 20_000.0;
        worst = worst.max(ang(fast_asin(s) - s.asin()));
    }
    assert!(
        worst * 2.0 < HORIZON_AZ_PAD,
        "fast angle error {worst} rad, pad {HORIZON_AZ_PAD}"
    );
}
