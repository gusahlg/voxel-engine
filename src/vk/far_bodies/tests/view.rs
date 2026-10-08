use glam::{Quat, Vec3, Vec4};

use super::support::{
    Rng, kept_seeds, pack_table, placed, random_body, ray_at, view_along_neg_z, view_pitched,
};
use crate::camera::{Camera3D, Lens, WarpMap, WarpStrength};
use crate::color::LinearRgb;
use crate::far_body::mirror::{
    ray_cube, ray_inner_sphere, ray_mapped, ray_rounded, ray_sphere, rounded_rim_band,
};
use crate::far_body::{FarBody, FarShape, MAX_FAR_BODIES, MAX_FAR_MAPS};
use crate::vk::far_bodies::cones::cone_bound;
use crate::vk::far_bodies::tiles::pixel_ndc;
use crate::vk::far_bodies::view::{ViewBasis, far_view, px_max, side_normals};

#[test]
fn px_max_is_twice_the_larger_pixel_angle() {
    let fovy = 60.0f32;
    let w = 3440u32;
    let h = 1440u32;
    let aspect = w as f32 / h as f32;
    let tan_half = (fovy.to_radians() * 0.5).tan();
    let cam = Camera3D {
        position: Vec3::ZERO,
        target: -Vec3::Z,
        up: Vec3::Y,
        fovy,
        lens: Lens::Rectilinear,
    };
    let got = px_max(tan_half, cam.view_proj(aspect), w, h);
    let vert = 2.0 * tan_half / h as f32;
    let horiz = 2.0 * tan_half * aspect / w as f32;
    assert!((vert - horiz).abs() < 1e-5, "{vert} vs {horiz}");
    assert!((got - vert * 2.0).abs() < 1e-5 * got, "{got}");

    let moved = Camera3D {
        position: Vec3::new(80.0, -3.0, 12.0),
        target: Vec3::new(80.0, -3.0, 11.0),
        ..cam
    };
    let got_moved = px_max(tan_half, moved.view_proj(aspect), w, h);
    assert!((got - got_moved).abs() < 1e-4 * got);

    let lens = Lens::WideFov {
        strength: WarpStrength::new(1.5).unwrap(),
    };
    let wide_aspect = aspect * WarpMap::from_lens(lens).fov_scale();
    let wide_cam = Camera3D { lens, ..cam };
    let wide = px_max(tan_half, wide_cam.view_proj(wide_aspect), w, h);
    let wide_horiz = 2.0 * tan_half * wide_aspect / w as f32;
    assert!(wide_horiz > vert * 1.2);
    assert!(
        (wide - wide_horiz * 2.0).abs() < 1e-4 * wide,
        "{wide} vs {wide_horiz}"
    );
}

#[test]
fn frustum_keeps_ahead_and_a_crossing_rim_and_drops_the_rest() {
    let view = view_along_neg_z(60.0, 1920, 1080);
    let forward = Vec3::new(0.0, 0.0, -1.0);
    let back = -forward;
    let ahead = placed(forward, 0.05, FarShape::Sphere, 1);
    let behind = placed(back, 0.05, FarShape::Sphere, 2);

    let right = side_normals(view.view_proj)[1];
    assert!(
        right.dot(forward) > 0.0,
        "forward must sit inside the right plane"
    );
    let on_plane = (forward - right * forward.dot(right)).normalize();
    assert!(right.dot(on_plane).abs() < 1e-4);
    let bound = 1.05 * 0.02;
    let a_sin = (bound + 3.0 * view.px_max).min(1.0);
    let outward = -right;
    let beta_in = (a_sin * 0.5).asin();
    let crossing_dir = (on_plane * beta_in.cos() + outward * beta_in.sin()).normalize();
    assert!(right.dot(crossing_dir) < 0.0);
    assert!(right.dot(crossing_dir) >= -a_sin);
    let crossing = placed(crossing_dir, 0.02, FarShape::Sphere, 3);

    let beta_out = (a_sin + 0.08).min(0.95).asin();
    let outside_dir = (on_plane * beta_out.cos() + outward * beta_out.sin()).normalize();
    assert!(right.dot(outside_dir) < -a_sin);
    let outside = placed(outside_dir, 0.02, FarShape::Sphere, 4);

    let table = pack_table(&[behind, ahead, outside, crossing], Some(view));
    assert_eq!(kept_seeds(&table), vec![1, 3]);
    assert_eq!(table.header[0], 2);
}

#[test]
fn hemisphere_cone_is_kept_only_when_it_meets_the_frustum() {
    let ahead = view_pitched(0.0, 60.0, 1920, 1080);
    // Bound 1 is the facing hemisphere. Straight behind misses a 60° frustum.
    let behind = placed(Vec3::Z, 0.99999, FarShape::Sphere, 1);
    let forward = placed(-Vec3::Z, 0.99999, FarShape::Sphere, 2);
    // Pure +X: the right half of the view is inside the hemisphere.
    let side = placed(Vec3::X, 0.99999, FarShape::Sphere, 3);
    let table = pack_table(&[behind, forward, side], Some(ahead));
    assert_eq!(kept_seeds(&table), vec![2, 3]);
    assert_eq!(table.cone[0][3].to_bits(), 1.0f32.to_bits());
    assert_eq!(table.cone[1][3].to_bits(), 1.0f32.to_bits());

    // Looking steeply up, the whole frustum is above the horizon, so the
    // planet's lower hemisphere does not meet it.
    let up = view_pitched(70.0, 60.0, 1280, 720);
    let planet = placed(-Vec3::Y, 0.99999, FarShape::Sphere, 4);
    let dropped = pack_table(std::slice::from_ref(&planet), Some(up));
    assert_eq!(dropped.header[0], 0);

    // Same planet, looking down: the hemisphere covers the view.
    let down = view_pitched(-70.0, 60.0, 1280, 720);
    let kept = pack_table(std::slice::from_ref(&planet), Some(down));
    assert_eq!(kept.header[0], 1);
    assert_eq!(kept.cone[0][3].to_bits(), 1.0f32.to_bits());
}

#[test]
fn inner_sphere_and_near_camera_rounded_are_always_kept() {
    let view = view_along_neg_z(50.0, 1280, 720);
    let back = Vec3::new(0.0, 0.0, 1.0);
    let wall = FarBody {
        dir: back,
        distance: 2.0,
        radius: 6.0,
        shape: FarShape::InnerSphere,
        rotation: Quat::IDENTITY,
        albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
        atmosphere: LinearRgb([0.0, 0.0, 0.0]),
        seed: 7,
    };
    let rho = 0.96f32;
    let p = 2.0f32;
    let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
    assert!(1.05 * rho_b >= 0.99);
    let round = placed(back, rho, FarShape::Rounded { exponent: p }, 8);
    let table = pack_table(&[wall, round], Some(view));
    assert_eq!(table.header[0], 2);
    assert_eq!(kept_seeds(&table), vec![7, 8]);
    assert_eq!(table.cone[0][3].to_bits(), (-1.0f32).to_bits());
    assert_eq!(table.cone[1][3].to_bits(), (-1.0f32).to_bits());
}

#[test]
fn compaction_preserves_relative_order_and_the_header_count() {
    let view = view_along_neg_z(60.0, 800, 600);
    let forward = Vec3::new(0.0, 0.0, -1.0);
    let back = -forward;
    let wall = FarBody {
        dir: back,
        distance: 1.5,
        radius: 4.0,
        shape: FarShape::InnerSphere,
        rotation: Quat::IDENTITY,
        albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
        atmosphere: LinearRgb([0.0, 0.0, 0.0]),
        seed: 13,
    };
    let bodies = [
        placed(back, 0.04, FarShape::Sphere, 10),
        placed(forward, 0.04, FarShape::Cube, 11),
        placed(back, 0.05, FarShape::Sphere, 12),
        wall,
        placed(forward, 0.03, FarShape::Rounded { exponent: 6.0 }, 14),
    ];
    let table = pack_table(&bodies, Some(view));
    assert_eq!(table.header[0], 3);
    assert_eq!(kept_seeds(&table), vec![11, 13, 14]);
    assert_eq!(table.cone[0][2].to_bits(), forward.z.to_bits());
    assert_eq!(table.body[0].dir_rho[2].to_bits(), forward.z.to_bits());
    assert_eq!(table.cone[2][2].to_bits(), forward.z.to_bits());
}

fn pixel_rejects(ray: Vec3, dir: Vec3, bound: f32, px: f32) -> bool {
    if bound < 0.0 {
        return false;
    }
    let dir = dir.normalize();
    let facing = ray.dot(dir);
    let s2 = ray.cross(dir).length_squared();
    let lim = bound + 3.0 * px;
    facing <= 0.0 || s2 > lim * lim
}

fn solid_hit(ray: Vec3, body: &FarBody) -> bool {
    let dir = body.dir.normalize();
    let rho = body.radius / body.distance;
    match body.shape {
        FarShape::Sphere => ray_sphere(ray, dir, rho).is_some(),
        FarShape::Cube => ray_cube(ray, dir, rho, body.rotation).is_some(),
        FarShape::Rounded { exponent } => {
            ray_rounded(ray, dir, rho, body.rotation, exponent).is_some()
        }
        FarShape::InnerSphere => ray_inner_sphere(ray, dir, body.distance, body.radius).is_some(),
        FarShape::Mapped { horizon, .. } => {
            let datum = [0.0f32; 24];
            ray_mapped(
                ray,
                dir,
                rho,
                body.distance,
                body.rotation,
                horizon,
                2,
                &datum,
                0.0,
                0.0,
            )
            .is_some()
        }
    }
}

/// Front contribution only. `facing <= 0` is past any rim the shader keeps
/// after the cone reject (the rounded antipodal ghost is not a real rim).
fn beyond_rim(ray: Vec3, body: &FarBody, px: f32) -> bool {
    let dir = body.dir.normalize();
    let rho = body.radius / body.distance;
    let facing = ray.dot(dir);
    let s = ray.cross(dir).length();
    match body.shape {
        FarShape::Sphere => {
            let rim = rho + (0.05 * rho).max(1.25 * px);
            facing <= 0.0 || s >= rim
        }
        FarShape::Rounded { exponent } => {
            let (_, rim) = rounded_rim_band(rho, exponent, px);
            facing <= 0.0 || s >= rim
        }
        FarShape::Cube => {
            if facing <= 0.0 {
                return true;
            }
            let rho_rim = rho * 1.035 + 0.5 * px;
            if !(rho_rim > 0.0 && rho_rim < 1.0) {
                return true;
            }
            ray_cube(ray, dir, rho_rim, body.rotation).is_none()
        }
        FarShape::InnerSphere => false,
        FarShape::Mapped { horizon, air, .. } => {
            if ray.dot(-dir) > horizon {
                return true;
            }
            let shell = rho + (air / body.distance).max(px);
            facing <= 0.0 || s >= shell
        }
    }
}

#[test]
fn cull_and_cone_reject_only_drop_rays_with_no_contribution() {
    let mut rng = Rng(0xF4A5_C011);
    let width = 3440u32;
    let height = 1440u32;
    let fovy = 70.0f32;
    let cam = Camera3D {
        position: Vec3::new(4.0, -2.0, 9.0),
        target: Vec3::new(4.0, -2.0, 8.0),
        up: Vec3::Y,
        fovy,
        lens: Lens::Rectilinear,
    };
    let aspect = width as f32 / height as f32;
    let tan_half = (fovy.to_radians() * 0.5).tan();
    let view_proj = cam.view_proj(aspect);
    let view = far_view(tan_half, view_proj, width, height);
    let vert = 2.0 * tan_half / height as f32;
    let row0 = view_proj.transpose().x_axis.truncate().length();
    let horiz = 2.0 / (row0 * width as f32);
    let px_values = [vert, horiz, vert.hypot(horiz)];

    let inv = view_proj.inverse();
    let mut rays = Vec::new();
    let push_ndc = |rays: &mut Vec<Vec3>, x: f32, y: f32| {
        let ray = (inv * Vec4::new(x, y, 0.0, 1.0)).truncate().normalize();
        let inside = side_normals(view_proj)
            .iter()
            .all(|n| n.length_squared() == 0.0 || n.dot(ray) >= -1e-4);
        if inside {
            rays.push(ray);
        }
    };
    push_ndc(&mut rays, 0.0, 0.0);
    for _ in 0..48 {
        push_ndc(&mut rays, rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
    }
    assert!(rays.len() >= 30, "rays inside the frustum: {}", rays.len());

    // The table keeps at most MAX_FAR_BODIES. Anything past that is omitted
    // by capacity, not by the frustum test, so each pack stays within the cap.
    for _batch in 0..3 {
        let bodies: Vec<FarBody> = (0..MAX_FAR_BODIES)
            .map(|i| random_body(&mut rng, i as u32))
            .collect();
        let table = pack_table(&bodies, Some(view));
        let kept = kept_seeds(&table);

        for body in &bodies {
            let culled = !kept.contains(&body.seed);
            let bound = cone_bound(body, &[0.0; MAX_FAR_MAPS]);
            for ray in &rays {
                for px in px_values {
                    if !(culled || pixel_rejects(*ray, body.dir, bound, px)) {
                        continue;
                    }
                    let hit = solid_hit(*ray, body);
                    let past =
                        matches!(body.shape, FarShape::InnerSphere) || beyond_rim(*ray, body, px);
                    let dir = body.dir.normalize();
                    assert!(
                        !hit && past,
                        "seed {} {:?} culled {culled} bound {bound} px {px} hit {hit} past {past} facing {} s {}",
                        body.seed,
                        body.shape,
                        ray.dot(dir),
                        ray.cross(dir).length(),
                    );
                }
            }
        }
    }
}

#[test]
fn view_space_tile_ray_matches_the_sky_unproject() {
    let view = view_pitched(25.0, 75.0, 800, 600);
    let basis = ViewBasis::from_view_proj(view.view_proj).expect("basis");
    let inv = view.view_proj.inverse();
    for &(x, y) in &[
        (0.0, 0.0),
        (400.0, 300.0),
        (799.0, 599.0),
        (10.5, 20.5),
        (0.0, 599.0),
    ] {
        let (nx, ny) = pixel_ndc(x, y, 800.0, 600.0);
        let view_ray = basis.ray_ndc(nx, ny);
        let world = ray_at(&inv, 800, 600, x, y);
        let from_world = basis.to_view(world).normalize();
        assert!(
            (view_ray - from_world).length() < 1e-4,
            "pixel {x},{y} analytic {view_ray} unproject {from_world}"
        );
    }
    // The centre of the framebuffer looks along the camera forward, view −Z.
    let centre = basis.ray_ndc(0.0, 0.0);
    assert!(
        (centre - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5,
        "centre ray {centre}"
    );
}
