use super::mirror::*;
use super::*;

/// The `src/vk/far_bodies.rs` tests import these fixtures through this path.
pub(crate) use super::mirror::{home_datum, lowland_foot, mapped_limb_top_angle, ray_from_nadir};

fn sphere_at(dir: Vec3, distance: f32, radius: f32, seed: u32) -> FarBody {
    FarBody {
        dir,
        distance,
        radius,
        shape: FarShape::Sphere,
        rotation: Quat::IDENTITY,
        albedo: [LinearRgb([1.0, 1.0, 1.0]); 6],
        atmosphere: LinearRgb([0.0, 0.0, 0.0]),
        seed,
    }
}

#[test]
fn rotate_matches_glam() {
    let q = Quat::from_xyzw(0.2, -0.4, 0.1, 0.8).normalize();
    let v = Vec3::new(0.3, -1.2, 0.7);
    let got = rotate(q, v);
    let expect = q * v;
    assert!(
        (got - expect).length() < 1e-5,
        "hand rotate {got} vs glam {expect}"
    );
    let id = rotate(Quat::IDENTITY, v);
    assert_eq!(id.x.to_bits(), v.x.to_bits());
    assert_eq!(id.y.to_bits(), v.y.to_bits());
    assert_eq!(id.z.to_bits(), v.z.to_bits());
}

#[test]
fn centre_ray_hits_sphere_and_cube() {
    let hit = ray_sphere(Vec3::Z, Vec3::Z, 0.25).unwrap();
    assert_eq!(hit.t.to_bits(), 0.75f32.to_bits());
    assert_eq!(hit.normal.z.to_bits(), (-1.0f32).to_bits());
    assert_eq!(hit.normal.x.to_bits(), 0.0f32.to_bits());

    let cube = ray_cube(Vec3::Z, Vec3::Z, 0.25, Quat::IDENTITY).unwrap();
    assert_eq!(cube.face, 5);
    assert_eq!(cube.t.to_bits(), 0.75f32.to_bits());
    assert_eq!(cube.normal.z.to_bits(), (-1.0f32).to_bits());
    let local = rotate(conjugate(Quat::IDENTITY), Vec3::Z * cube.t - Vec3::Z);
    assert_eq!(local.z.to_bits(), (-0.25f32).to_bits());
    assert_eq!(local.x.to_bits(), 0.0f32.to_bits());
    assert_eq!(local.y.to_bits(), 0.0f32.to_bits());
}

#[test]
fn yaw_half_turn_hits_the_opposite_cube_face() {
    // 180° about Y: (x, y, z) → (−x, y, −z). Exact, not `from_rotation_y(PI)`.
    let q = Quat::from_xyzw(0.0, 1.0, 0.0, 0.0);
    let hit = ray_cube(Vec3::Z, Vec3::Z, 0.25, q).unwrap();
    assert_eq!(hit.face, 4);
    assert_eq!(hit.t.to_bits(), 0.75f32.to_bits());
    assert_eq!(hit.normal.x.to_bits(), 0.0f32.to_bits());
    assert_eq!(hit.normal.y.to_bits(), 0.0f32.to_bits());
    assert_eq!(hit.normal.z.to_bits(), (-1.0f32).to_bits());
    let local = rotate(conjugate(q), Vec3::Z * hit.t - Vec3::Z);
    assert_eq!(local.z.to_bits(), 0.25f32.to_bits());
}

#[test]
fn misses_and_inside_draw_nothing() {
    assert!(ray_sphere(Vec3::X, Vec3::Z, 0.25).is_none());
    assert!(ray_cube(Vec3::X, Vec3::Z, 0.25, Quat::IDENTITY).is_none());
    assert!(ray_sphere(Vec3::Z, Vec3::Z, 1.0).is_none());
    assert!(ray_sphere(Vec3::Z, Vec3::Z, 1.5).is_none());
    assert!(ray_cube(Vec3::Z, Vec3::Z, 1.0, Quat::IDENTITY).is_none());
    assert!(ray_sphere(Vec3::Z, Vec3::Z, 0.0).is_none());
    assert!(ray_cube(Vec3::Z, Vec3::Z, 0.0, Quat::IDENTITY).is_none());
}

#[test]
fn grazing_sphere_matches_the_disc() {
    let inside = Vec3::new(0.4, 0.0, (1.0 - 0.16f32).sqrt()).normalize();
    let hit = ray_sphere(inside, Vec3::Z, 0.5).unwrap();
    let err = (inside * hit.t - Vec3::Z).length() - 0.5;
    assert!(err.abs() < 1e-4, "{err}");
    let outside = Vec3::new(0.6, 0.0, (1.0 - 0.36f32).sqrt()).normalize();
    assert!(ray_sphere(outside, Vec3::Z, 0.5).is_none());
}

#[test]
fn billion_block_distance_is_exact_in_normalised_space() {
    let rho = normalised_radius(1.0e9, 2.5e8);
    assert_eq!(rho.to_bits(), 0.25f32.to_bits());
    assert_eq!(normalised_radius(1.0e9, 5.0e8).to_bits(), 0.5f32.to_bits());
    let dir = Vec3::new(-0.2, 0.3, 0.8).normalize();
    // A small offset from the centre ray: still inside a rho of 0.25, for both shapes.
    let ray = (dir + Vec3::new(0.05, -0.02, 0.01)).normalize();
    let sphere_far = ray_sphere(ray, dir, rho).expect("sphere hit");
    let sphere_near = ray_sphere(ray, dir, 0.25).unwrap();
    assert_eq!(sphere_far.t.to_bits(), sphere_near.t.to_bits());
    assert_eq!(
        sphere_far.normal.x.to_bits(),
        sphere_near.normal.x.to_bits()
    );
    assert_eq!(
        sphere_far.normal.y.to_bits(),
        sphere_near.normal.y.to_bits()
    );
    assert_eq!(
        sphere_far.normal.z.to_bits(),
        sphere_near.normal.z.to_bits()
    );
    let cube_far = ray_cube(ray, dir, rho, Quat::IDENTITY).unwrap();
    let cube_near = ray_cube(ray, dir, 0.25, Quat::IDENTITY).unwrap();
    assert_eq!(cube_far.t.to_bits(), cube_near.t.to_bits());
    assert_eq!(cube_far.face, cube_near.face);
    assert_eq!(cube_far.normal.x.to_bits(), cube_near.normal.x.to_bits());
    assert_eq!(cube_far.normal.y.to_bits(), cube_near.normal.y.to_bits());
    assert_eq!(cube_far.normal.z.to_bits(), cube_near.normal.z.to_bits());
    // The intersector never sees the world distance: a centre ray is 1 − rho.
    let centre = ray_sphere(Vec3::Z, Vec3::Z, rho).unwrap();
    assert_eq!(centre.t.to_bits(), (1.0 - rho).to_bits());
}

#[test]
fn store_caps_sorts_and_drops_invalid() {
    let bodies: Vec<FarBody> = (0..40)
        .map(|i| sphere_at(Vec3::new(0.0, 0.0, 2.0), 100.0 + i as f32, 1.0, i))
        .collect();
    let mut out = [FarBody::default(); MAX_FAR_BODIES];
    let n = store(&bodies, &mut out);
    assert_eq!(n, 32);
    assert_eq!(out[0].distance, 131.0);
    assert_eq!(out[31].distance, 100.0);
    assert!(
        out.iter()
            .take(n as usize)
            .all(|b| (0..32).contains(&b.seed))
    );
    assert!((out[0].dir - Vec3::Z).length() < 1e-6);

    let mut nan = vec![
        sphere_at(Vec3::Z, 10.0, 1.0, 1),
        sphere_at(Vec3::Z, 20.0, 1.0, 2),
        sphere_at(Vec3::Z, 30.0, 1.0, 3),
    ];
    nan[1].distance = f32::NAN;
    assert_eq!(store(&nan, &mut out), 2);
    assert_eq!(out[0].distance, 30.0);
    assert_eq!(out[1].distance, 10.0);

    let inside = sphere_at(Vec3::Z, 1.0, 1.0, 4);
    assert_eq!(store(&[inside], &mut out), 0);
    let buried = sphere_at(Vec3::Z, 1.0, 2.0, 5);
    assert_eq!(store(&[buried], &mut out), 0);
    let mut wall = sphere_at(Vec3::Z, 2.0, 5.0, 7);
    wall.shape = FarShape::InnerSphere;
    assert_eq!(store(&[wall], &mut out), 1);
    assert_eq!(out[0].shape, FarShape::InnerSphere);
    assert!(out[0].distance < out[0].radius);
    // Outside the inner sphere is not this shape.
    wall.distance = 8.0;
    assert_eq!(store(&[wall], &mut out), 0);
    let mut zero_dir = sphere_at(Vec3::ZERO, 10.0, 1.0, 6);
    zero_dir.dir = Vec3::ZERO;
    assert_eq!(store(&[zero_dir], &mut out), 0);
}

#[test]
fn inner_sphere_hits_every_ray_and_faces_the_centre() {
    let dir = Vec3::Y;
    let distance = 2.0;
    let radius = 5.0;
    let rho: f32 = radius / distance;
    assert_eq!(rho.to_bits(), 2.5f32.to_bits());
    let rays = [
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Z,
        Vec3::NEG_Z,
        Vec3::new(1.0, 0.3, -0.4).normalize(),
        Vec3::new(-0.2, -1.0, 0.5).normalize(),
    ];
    for ray in rays {
        let hit = ray_inner_sphere(ray, dir, distance, radius).expect("inside hits");
        let point = ray * hit.t;
        let err = (point - dir).length() - rho;
        assert!(err.abs() < 1e-4, "{ray:?} off the wall by {err}");
        let toward = (dir - point).normalize();
        assert!(
            hit.normal.dot(toward) > 0.999,
            "{ray:?} normal {:?} should point at the centre",
            hit.normal
        );
        assert!((hit.normal.length() - 1.0).abs() < 1e-4);
    }
    // Looking at the centre: t = b + sqrt(disc) = 1 + rho, normal back along dir.
    let b = 1.0f32;
    let disc = b * b - (1.0 - rho * rho);
    let t = b + disc.sqrt();
    let hit = ray_inner_sphere(Vec3::Y, dir, distance, radius).unwrap();
    assert_eq!(hit.t.to_bits(), t.to_bits());
    assert_eq!(hit.t.to_bits(), 3.5f32.to_bits());
    assert_eq!(hit.normal.y.to_bits(), (-1.0f32).to_bits());
    assert_eq!(hit.normal.x.to_bits(), 0.0f32.to_bits());
    assert_eq!(hit.face, 0);
    // The near root is behind the camera. This shape takes the far one.
    let near = b - disc.sqrt();
    assert!(near < 0.0);
    assert!(ray_inner_sphere(Vec3::Y, dir, 5.0, 2.0).is_none());
    assert!(ray_inner_sphere(Vec3::Y, dir, 2.0, 2.0).is_none());
    assert!(ray_inner_sphere(Vec3::Y, dir, 0.0, 2.0).is_none());
}

#[test]
fn inner_sphere_draws_behind_the_bodies_inside_it() {
    let mut wall = sphere_at(Vec3::Z, 10.0, 40.0, 7);
    wall.shape = FarShape::InnerSphere;
    let core = sphere_at(Vec3::Z, 10.0, 1.0, 8);
    let far = sphere_at(Vec3::X, 100.0, 1.0, 9);
    let mut out = [FarBody::default(); MAX_FAR_BODIES];
    assert_eq!(store(&[core, far, wall], &mut out), 3);
    assert_eq!(out[0].shape, FarShape::InnerSphere);
    assert_eq!(out[1].distance, 100.0);
    assert_eq!(out[2].seed, 8);
}

/// Each body is ray-tested with its centre at distance 1, so normalised
/// `t` (and a rim's `facing`) cannot order two bodies. The sky shader
/// compares `t * distance` for hits and `facing * distance` for rims and
/// point blobs, and keeps the normalised `t` for shading. The inner
/// sphere's far wall stays the background: a smaller world depth
/// composites over it, and a hit past the shell does not.
#[test]
fn world_depth_orders_bodies_the_normalised_parameter_cannot() {
    let near = ray_sphere(Vec3::Z, Vec3::Z, 0.2).unwrap();
    let far = ray_sphere(Vec3::Z, Vec3::Z, 0.9).unwrap();
    let near_distance = 40.0;
    let far_distance = 4_000.0;
    assert!(near.t > far.t, "normalised t would pick the far body");
    let near_depth = near.t * near_distance;
    let far_depth = far.t * far_distance;
    assert!(near_depth < far_depth);
    assert!((near_depth - 32.0).abs() < 1e-3, "{near_depth}");
    assert!((far_depth - 400.0).abs() < 1e-2, "{far_depth}");

    let wall = ray_inner_sphere(Vec3::Z, Vec3::Z, 100.0, 500.0).unwrap();
    let wall_depth = wall.t * 100.0;
    assert!(
        (wall_depth - 600.0).abs() < 1e-2,
        "far wall world depth {wall_depth}"
    );
    assert!(wall.t > near.t && wall.t > far.t);
    let front = ray_sphere(Vec3::Z, Vec3::Z, 0.5).unwrap();
    assert!(front.t * 80.0 < wall_depth);
    assert!(
        front.t * 2_000.0 > wall_depth,
        "a body past the shell is behind the wall even though its normalised t is smaller"
    );

    // On this ray `facing` is 1. The close rim is still in front of the far hit.
    assert!(1.0 * near_distance < far_depth);
}

fn axis_face(u: Vec3) -> Option<u32> {
    let a = u.abs();
    if a.x > 0.9 && a.y < 1e-4 && a.z < 1e-4 {
        Some(if u.x > 0.0 { 0 } else { 1 })
    } else if a.y > 0.9 && a.x < 1e-4 && a.z < 1e-4 {
        Some(if u.y > 0.0 { 2 } else { 3 })
    } else if a.z > 0.9 && a.x < 1e-4 && a.y < 1e-4 {
        Some(if u.z > 0.0 { 4 } else { 5 })
    } else {
        None
    }
}

/// Face centres, the three edge axes and the space diagonals.
fn rounded_dirs() -> [Vec3; 16] {
    [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
    ]
}

fn outside(q: Vec3, rho: f32, p: f32) -> bool {
    let (s, _) = lp_grad(q, p);
    s + rho * 1.0e-4 >= rho
}

#[test]
fn rounded_hits_analytic_face_edge_and_corner() {
    let rho = 0.3f32;
    let tilt = Quat::from_xyzw(0.2, -0.4, 0.1, 0.8).normalize();
    for p in [2.0f32, 2.17, 10.0] {
        let corner = Vec3::ONE.normalize();
        let radial = rho / lp_grad(corner, p).0;
        let bound = rho * 3.0f32.powf(0.5 - 1.0 / p);
        assert!(
            (radial - bound).abs() < 1e-4,
            "p {p}: corner radius {radial} bound {bound}"
        );
        for rotation in [Quat::IDENTITY, tilt] {
            for dir in rounded_dirs() {
                let u = dir.normalize();
                let outward = rotate(rotation, u);
                // Look straight down this body direction so the sample is the near hit.
                let centre = -outward;
                let reach = rho / lp_grad(u, p).0;
                let point = centre + outward * reach;
                let ray = -outward;
                let hit = ray_rounded(ray, centre, rho, rotation, p)
                    .unwrap_or_else(|| panic!("p {p} dir {u:?} missed"));
                let got = ray * hit.t;
                let err = (got - point).length();
                assert!(
                    err < 2e-4,
                    "p {p} dir {u:?} off by {err} (got {got}, want {point})"
                );
                assert!(
                    hit.normal.dot(outward) > 0.999,
                    "p {p} normal {:?} not along {outward:?}",
                    hit.normal
                );
                assert!((hit.normal.length() - 1.0).abs() < 1e-4);
                let q = rotate(conjugate(rotation), got - centre);
                assert!(outside(q, rho, p), "p {p} analytic hit inside");
                if let Some(face) = axis_face(u) {
                    assert_eq!(hit.face, face, "p {p} face of {u:?}");
                }
            }
        }
        // p = 2 is the sphere the closed form already tests.
        if p == 2.0 {
            let centre = Vec3::Z;
            let ray = Vec3::new(0.12, -0.05, 1.0).normalize();
            let sphere = ray_sphere(ray, centre, 0.4).unwrap();
            let rounded = ray_rounded(ray, centre, 0.4, Quat::IDENTITY, p).unwrap();
            assert!(
                (sphere.t - rounded.t).abs() < 2e-4,
                "{} vs {}",
                sphere.t,
                rounded.t
            );
            assert!(sphere.normal.dot(rounded.normal) > 0.999);
        }
    }
}

#[test]
fn rounded_grazing_never_hits_inside() {
    let perps = [
        Vec3::X,
        Vec3::Y,
        Vec3::new(1.0, 1.0, 0.0).normalize(),
        Vec3::new(-1.0, 2.0, 0.0).normalize(),
        Vec3::new(2.0, -0.5, 0.0).normalize(),
    ];
    for p in [2.0f32, 2.17, 10.0] {
        for rho in [0.35f32, 0.97] {
            let bound = rho * 3.0f32.powf(0.5 - 1.0 / p);
            let centre = Vec3::Z;
            for perp in perps {
                for k in 0..18 {
                    let s = bound * (k as f32) / 16.0;
                    if s >= 1.0 {
                        continue;
                    }
                    let z = (1.0 - s * s).sqrt();
                    let ray = Vec3::new(perp.x * s, perp.y * s, z);
                    if let Some(hit) = ray_rounded(ray, centre, rho, Quat::IDENTITY, p) {
                        let q = rotate(conjugate(Quat::IDENTITY), ray * hit.t - centre);
                        assert!(
                            outside(q, rho, p),
                            "p {p} rho {rho} s {s} hit inside at {:?}",
                            ray * hit.t
                        );
                    }
                }
            }
            // Past the enclosing sphere there is nothing to hit.
            if bound * 1.02 < 1.0 {
                let s = bound * 1.02;
                let ray = Vec3::new(s, 0.0, (1.0 - s * s).sqrt());
                assert!(
                    ray_rounded(ray, centre, rho, Quat::IDENTITY, p).is_none(),
                    "p {p} rho {rho} ray outside the bound hit"
                );
            }
        }
    }
}

#[test]
fn rounded_exponent_is_kept_only_when_real() {
    let mut body = sphere_at(Vec3::Z, 4.0, 1.0, 3);
    body.shape = FarShape::Rounded { exponent: 2.17 };
    let mut out = [FarBody::default(); MAX_FAR_BODIES];
    assert_eq!(store(std::slice::from_ref(&body), &mut out), 1);
    assert_eq!(out[0].shape, FarShape::Rounded { exponent: 2.17 });
    body.shape = FarShape::Rounded { exponent: 1.5 };
    assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
    body.shape = FarShape::Rounded { exponent: f32::NAN };
    assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
    body.shape = FarShape::Rounded {
        exponent: f32::INFINITY,
    };
    assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
}

#[test]
fn far_map_basis_matches_the_chart_table() {
    let (x, y, z) = (Vec3::X, Vec3::Y, Vec3::Z);
    assert_eq!(far_map_basis(0), (-y, x, z));
    assert_eq!(far_map_basis(1), (y, -x, z));
    assert_eq!(far_map_basis(2), (x, y, z));
    assert_eq!(far_map_basis(3), (x, -y, -z));
    assert_eq!(far_map_basis(4), (x, z, -y));
    assert_eq!(far_map_basis(5), (x, -z, y));
}

#[test]
fn far_cube_texel_dir_follows_the_vulkan_cube() {
    let normals = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    let size = 8u32;
    for (face, n) in normals.iter().copied().enumerate() {
        for y in 0..size {
            for x in 0..size {
                let d = far_cube_texel_dir(face, size, x, y);
                assert!((d.length() - 1.0).abs() < 1e-5, "{d:?}");
                let a = d.abs();
                let axis = if a.x >= a.y && a.x >= a.z {
                    0
                } else if a.y >= a.z {
                    1
                } else {
                    2
                };
                assert_eq!(axis, face / 2, "face {face} texel {x},{y} dir {d}");
                assert!(d.dot(n) > 0.0, "face {face} points away");
                let mirror = far_cube_texel_dir(face, size, size - 1 - x, size - 1 - y);
                let sum = d + mirror;
                assert!(
                    sum.cross(n).length() < 1e-4,
                    "face {face} texel {x},{y} not symmetric: {sum}"
                );
            }
        }
    }
    // Image origin of +X is the corner toward +Y and +Z.
    let corner = far_cube_texel_dir(0, 4096, 0, 0);
    let expect = Vec3::new(1.0, 1.0, 1.0).normalize();
    assert!((corner - expect).length() < 1e-3, "{corner} vs {expect}");
}

#[test]
fn far_map_desc_rejects_a_bad_id_or_size() {
    let ok_datum = [0.0f32; 6 * 2 * 2];
    let ok = FarMapDesc {
        datum_res: 2,
        datum: &ok_datum,
        albedo_size: 4,
    };
    assert_eq!(validate_far_map(FarMapId(8), &ok), Err(FarMapError::BadId));
    assert_eq!(
        validate_far_map(FarMapId(255), &ok),
        Err(FarMapError::BadId)
    );
    assert!(validate_far_map(FarMapId(7), &ok).is_ok());

    let mut bad_g = ok;
    bad_g.datum_res = 1;
    assert_eq!(
        validate_far_map(FarMapId(0), &bad_g),
        Err(FarMapError::BadSize)
    );
    bad_g.datum_res = 66;
    assert_eq!(
        validate_far_map(FarMapId(0), &bad_g),
        Err(FarMapError::BadSize)
    );

    let short = FarMapDesc {
        datum_res: 2,
        datum: &[0.0f32; 6],
        albedo_size: 0,
    };
    assert_eq!(
        validate_far_map(FarMapId(0), &short),
        Err(FarMapError::BadSize)
    );

    let mut bad_albedo = ok;
    bad_albedo.albedo_size = 3;
    assert_eq!(
        validate_far_map(FarMapId(0), &bad_albedo),
        Err(FarMapError::BadSize)
    );
    bad_albedo.albedo_size = 4096;
    assert_eq!(
        validate_far_map(FarMapId(0), &bad_albedo),
        Err(FarMapError::BadSize)
    );
    bad_albedo.albedo_size = 0;
    assert!(validate_far_map(FarMapId(0), &bad_albedo).is_ok());
    bad_albedo.albedo_size = 2048;
    assert!(validate_far_map(FarMapId(0), &bad_albedo).is_ok());

    assert_eq!(
        validate_far_map_face(FarMapId(8), 0, 64, Some(4)),
        Err(FarMapError::BadId)
    );
    assert_eq!(
        validate_far_map_face(FarMapId(0), 0, 64, None),
        Err(FarMapError::BadId)
    );
    assert_eq!(
        validate_far_map_face(FarMapId(0), 6, 64, Some(4)),
        Err(FarMapError::BadSize)
    );
    assert_eq!(
        validate_far_map_face(FarMapId(0), 0, 16, Some(4)),
        Err(FarMapError::BadSize)
    );
    assert_eq!(
        validate_far_map_face(FarMapId(0), 0, 64, Some(0)),
        Err(FarMapError::BadSize)
    );
    assert!(validate_far_map_face(FarMapId(0), 5, 64, Some(4)).is_ok());
}

#[test]
fn mapped_is_kept_only_with_a_real_horizon_and_air() {
    let mut body = sphere_at(Vec3::Z, 4.0, 1.0, 1);
    body.shape = FarShape::Mapped {
        map: FarMapId(3),
        horizon: 1.0,
        air: 0.0,
    };
    let mut out = [FarBody::default(); MAX_FAR_BODIES];
    assert_eq!(store(std::slice::from_ref(&body), &mut out), 1);
    body.shape = FarShape::Mapped {
        map: FarMapId(8),
        horizon: 1.0,
        air: 0.0,
    };
    assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
    body.shape = FarShape::Mapped {
        map: FarMapId(0),
        horizon: f32::NAN,
        air: 0.0,
    };
    assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
    body.shape = FarShape::Mapped {
        map: FarMapId(0),
        horizon: 0.0,
        air: -0.01,
    };
    assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
    // Still an outside shape: the viewer must be beyond the reference radius.
    body.shape = FarShape::Mapped {
        map: FarMapId(0),
        horizon: 1.0,
        air: 0.2,
    };
    body.distance = 1.0;
    body.radius = 1.0;
    assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
}

#[test]
fn flat_datum_matches_the_sphere() {
    let dir = Vec3::new(-0.2, 0.4, 0.8).normalize();
    let rho = 0.35f32;
    let distance = 8.0f32;
    let g = 4u32;
    let datum = vec![0.0f32; 6 * 16];
    let mut rays = vec![dir];
    for k in 0..12 {
        let ang = (k as f32) * 0.07;
        let ray = (dir + Vec3::new(ang.sin(), ang * 0.3, 0.0)).normalize();
        rays.push(ray);
    }
    rays.push(Vec3::X);
    rays.push(-dir);
    for ray in rays {
        let sphere = ray_sphere(ray, dir, rho);
        let mapped = ray_mapped(
            ray,
            dir,
            rho,
            distance,
            Quat::IDENTITY,
            1.0,
            g,
            &datum,
            0.0,
            0.0,
        );
        match (sphere, mapped) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert!((a.t - b.t).abs() < 1e-5, "{ray:?} t {} vs {}", a.t, b.t);
                assert!(
                    (a.normal - b.normal).length() < 1e-5,
                    "{ray:?} n {:?} vs {:?}",
                    a.normal,
                    b.normal
                );
            }
            other => panic!("{ray:?} disagreed: {other:?}"),
        }
    }
}

#[test]
fn a_datum_bump_moves_the_hit_closer() {
    let g = 5u32;
    let mut datum = vec![0.0f32; 6 * 25];
    // Centre of the −Z face (the face the +Z centre ray meets).
    let face = 5usize;
    let i = 2usize;
    let j = 2usize;
    datum[face * 25 + j * 5 + i] = 0.2;
    let rho = 0.25f32;
    let distance = 4.0f32;
    let sphere = ray_sphere(Vec3::Z, Vec3::Z, rho).unwrap();
    let hit = ray_mapped(
        Vec3::Z,
        Vec3::Z,
        rho,
        distance,
        Quat::IDENTITY,
        1.0,
        g,
        &datum,
        0.0,
        0.2,
    )
    .expect("bump hits");
    assert!(
        hit.t < sphere.t - 1e-3,
        "bump t {} should be closer than sphere {}",
        hit.t,
        sphere.t
    );
}

#[test]
fn a_limb_ray_that_misses_the_surface_is_a_miss() {
    // max offset 0, the rest sunk, so the hi sphere is the reference and
    // the surface sits inside it. Impact parameter between the two.
    let g = 2u32;
    let mut datum = vec![-0.4f32; 6 * 4];
    datum[2 * 4] = 0.0; // one +Y sample keeps max at 0
    let rho = 0.5f32;
    let distance = 2.0f32;
    let s = 0.45f32;
    let ray = Vec3::new(s, 0.0, (1.0 - s * s).sqrt());
    assert!(ray_sphere(ray, Vec3::Z, rho).is_some());
    assert!(
        ray_mapped(
            ray,
            Vec3::Z,
            rho,
            distance,
            Quat::IDENTITY,
            1.0,
            g,
            &datum,
            -0.4,
            0.0,
        )
        .is_none()
    );
}

#[test]
fn horizon_cull_skips_rays_above_it() {
    let rho = 0.95f32;
    let datum = [0.0f32; 6 * 4];
    let centre = ray_mapped(
        Vec3::Z,
        Vec3::Z,
        rho,
        1.0,
        Quat::IDENTITY,
        -0.8,
        2,
        &datum,
        0.0,
        0.0,
    );
    assert!(centre.is_some());
    let s = (1.0f32 - 0.25).sqrt();
    let raised = Vec3::new(s, 0.0, 0.5);
    assert!(raised.dot(-Vec3::Z) > -0.8);
    assert!(ray_sphere(raised, Vec3::Z, rho).is_some());
    assert!(
        ray_mapped(
            raised,
            Vec3::Z,
            rho,
            1.0,
            Quat::IDENTITY,
            -0.8,
            2,
            &datum,
            0.0,
            0.0,
        )
        .is_none()
    );
    assert!(
        ray_mapped(
            raised,
            Vec3::Z,
            rho,
            1.0,
            Quat::IDENTITY,
            1.0,
            2,
            &datum,
            0.0,
            0.0,
        )
        .is_some()
    );
}

#[test]
fn atan_approx_stays_under_a_datum_cell() {
    let mut max_err = 0.0f64;
    let n = 200_001i32;
    for i in 0..=n {
        let x = -1.0 + 2.0 * f64::from(i) / f64::from(n);
        let got = f64::from(atan_approx(x as f32));
        max_err = max_err.max((got - x.atan()).abs());
    }
    // The stepwise f32 Horner peaks near −0.972. Sweep that neighbourhood.
    for i in 0..20_000 {
        let x = -0.99 + 0.04 * f64::from(i) / 20_000.0;
        let got = f64::from(atan_approx(x as f32));
        max_err = max_err.max((got - x.atan()).abs());
    }
    assert!(
        max_err <= 6.7e-7,
        "atan approx err {max_err} exceeds 6.7e-7 rad"
    );
}

#[test]
fn analytic_normal_matches_the_finite_difference() {
    let g = 33u32;
    let gg = g as usize;
    let mut datum = vec![0.0f32; 6 * gg * gg];
    for face in 0..6 {
        let (tu, n, tv) = far_map_basis(face);
        for j in 0..g {
            for i in 0..g {
                let edge = (g - 1) as f32;
                let xi = 2.0 * i as f32 / edge - 1.0;
                let eta = 2.0 * j as f32 / edge - 1.0;
                let quarter = std::f32::consts::FRAC_PI_4;
                let d = (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
                // Small enough that a half-cell finite difference stays within
                // 1e-3 rad of the analytic slope, and large enough to tilt
                // the normal off the radius.
                let bump = 0.005 * (d.x * 1.3 + d.y * 0.7).sin() * (d.z * 1.1).cos();
                datum[face * gg * gg + j as usize * gg + i as usize] = bump;
            }
        }
    }
    let rho = 0.4f32;
    let distance = 6.0f32;
    let rotation = Quat::from_axis_angle(Vec3::new(0.2, 0.5, 0.8).normalize(), 0.4);
    let mut worst = 0.0f32;
    let mut tilt = 0.0f32;
    for k in 0..64 {
        let z = -1.0 + 2.0 * (k as f32) / 63.0;
        let ang = k as f32 * 0.37;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let dir = Vec3::new(r * ang.cos(), r * ang.sin(), z);
        let fd = mapped_normal(rotation, dir, rho, distance, g, &datum);
        let an = mapped_normal_analytic(rotation, dir, rho, distance, g, &datum);
        let err = fd.dot(an).clamp(-1.0, 1.0).acos();
        worst = worst.max(err);
        // `dir` is the world radius; the normal is world-space too.
        tilt = tilt.max(dir.dot(an).clamp(-1.0, 1.0).acos());
    }
    assert!(tilt > 1e-3, "datum did not tilt the normal ({tilt})");
    assert!(worst < 1e-3, "normal angle {worst} rad");
}

fn sphere_roots_f64(facing: f64, rho: f64) -> Option<(f64, f64)> {
    if !(rho > 0.0) || !rho.is_finite() {
        return None;
    }
    let disc = facing * facing - (1.0 - rho * rho);
    if disc < 0.0 || !disc.is_finite() {
        return None;
    }
    let sd = disc.sqrt();
    Some((facing - sd, facing + sd))
}

/// Residual of the shader's bilinear datum along an f64 ray.
/// `t` is refined in f64; the height sample is the same f32 bilinear the
/// march uses, including the equiangular polynomial.
fn mapped_residual_f64(
    t: f64,
    ray: glam::DVec3,
    center: glam::DVec3,
    rho: f64,
    distance: f64,
    rotation: Quat,
    g: u32,
    datum: &[f32],
) -> f64 {
    let p = ray * t - center;
    let rad = p.length();
    if rad < 1e-12 {
        return -rho;
    }
    let u = p / rad;
    let u32 = Vec3::new(u.x as f32, u.y as f32, u.z as f32);
    let body = rotate(conjugate(rotation), u32);
    let height = f64::from(sample_datum(g, datum, body));
    rad - (rho + height / distance)
}

/// f64 reference intersection. Dense march inside
/// `[max(t_hi_in, 0), min(t_lo_in, t_hi_out)]` (or `t_hi_out` when the ray
/// misses the lo sphere), with a step well under one datum cell projected
/// on the ray, then bisection to `1e-12` in normalised `t`.
fn ray_mapped_reference(
    ray: Vec3,
    dir: Vec3,
    rho: f32,
    distance: f32,
    rotation: Quat,
    g: u32,
    datum: &[f32],
    min_off: f32,
    max_off: f32,
) -> Option<f64> {
    if !(rho > 0.0) || !(distance > 0.0) || !ray.is_finite() || !dir.is_finite() {
        return None;
    }
    let ray_d = ray.as_dvec3().normalize();
    let dir_d = dir.as_dvec3().normalize();
    let rho_d = f64::from(rho);
    let distance_d = f64::from(distance);
    let rho_lo = rho_d + f64::from(min_off) / distance_d;
    let rho_hi = rho_d + f64::from(max_off) / distance_d;
    let facing = ray_d.dot(dir_d);
    let (t_hi_in, t_hi_out) = sphere_roots_f64(facing, rho_hi)?;
    if !(t_hi_out > 0.0) {
        return None;
    }
    let t_start = t_hi_in.max(0.0);
    let t_lo = sphere_roots_f64(facing, rho_lo).and_then(|(t_lo, _)| (t_lo > 0.0).then_some(t_lo));
    let t_end = t_lo.map(|t| t.min(t_hi_out)).unwrap_or(t_hi_out);
    let f_at = |t: f64| mapped_residual_f64(t, ray_d, dir_d, rho_d, distance_d, rotation, g, datum);
    // Lo and hi spheres coincide on a constant datum: the entrance is the hit.
    if !(t_end > t_start) {
        let f0 = f_at(t_start);
        return (t_start > 0.0 && f0.abs() <= 1e-5).then_some(t_start);
    }
    let f_start = f_at(t_start);
    if f_start < -1e-5 {
        return None;
    }
    if f_start.abs() <= 1e-8 {
        return (t_start > 0.0).then_some(t_start);
    }
    // One datum cell is π / (2 (g-1)) radians. Along the ray that is
    // `cell * rad / sin(phi)`. A tenth of that cannot step over the
    // bilinear's single sign change inside a cell.
    let cells = f64::from(g.max(2) - 1);
    let cell = std::f64::consts::FRAC_PI_2 / cells;
    let mut t = t_start;
    let mut prev_f = f_start;
    let mut bracket: Option<(f64, f64)> = None;
    let mut guard = 0u32;
    while t < t_end && guard < 100_000 {
        guard += 1;
        let p = ray_d * t - dir_d;
        let rad = p.length().max(rho_d);
        let sin_phi = if rad > 1e-8 {
            ray_d.cross(p / rad).length()
        } else {
            0.0
        };
        let dt = (cell * rad / sin_phi.max(0.05)) * 0.1;
        let next = (t + dt.max(1e-8)).min(t_end);
        let ft = f_at(next);
        if prev_f * ft <= 0.0 {
            bracket = Some((t, next));
            break;
        }
        t = next;
        prev_f = ft;
    }
    let (mut a, mut b) = bracket?;
    for _ in 0..80 {
        if (b - a).abs() <= 1e-12 {
            break;
        }
        let mid = 0.5 * (a + b);
        let fm = f_at(mid);
        let fa = f_at(a);
        if fa * fm <= 0.0 {
            b = mid;
        } else {
            a = mid;
        }
    }
    let root = 0.5 * (a + b);
    (root > 0.0 && root.is_finite()).then_some(root)
}

#[test]
fn reference_matches_spheres_of_constant_datum() {
    let g = 33u32;
    let gg = g as usize;
    let distance = 4.0f32;
    for offset in [0.0f32, 0.25] {
        let datum = vec![offset; 6 * gg * gg];
        let rho = 0.35f32;
        // Facing stays above the graze of rho 0.35 (disc needs |ray·dir| ≳ 0.94).
        for (ray, dir) in [
            (Vec3::Z, Vec3::Z),
            (Vec3::new(0.25, -0.1, 1.0).normalize(), Vec3::Z),
            (
                Vec3::new(0.15, 0.05, 1.0).normalize(),
                Vec3::new(0.05, -0.08, 1.0).normalize(),
            ),
        ] {
            // Same f64 ray the reference normalises, and the same radius
            // the residual subtracts (f64 division, not the f32 sphere).
            let facing = ray.as_dvec3().normalize().dot(dir.as_dvec3().normalize());
            let radius = f64::from(rho) + f64::from(offset) / f64::from(distance);
            let (t_near, _) = sphere_roots_f64(facing, radius).unwrap();
            let got = ray_mapped_reference(
                ray,
                dir,
                rho,
                distance,
                Quat::IDENTITY,
                g,
                &datum,
                offset,
                offset,
            )
            .unwrap_or_else(|| panic!("reference missed offset {offset} ray {ray:?}"));
            assert!(
                (got - t_near).abs() < 1e-9,
                "offset {offset} reference {got} analytic {t_near}"
            );
        }
    }
}

#[test]
fn reference_straight_down_is_the_local_altitude() {
    let g = 33u32;
    let datum = home_datum(g);
    let min_off = datum.iter().copied().fold(f32::MAX, f32::min);
    let max_off = datum.iter().copied().fold(f32::MIN, f32::max);
    assert!(min_off > -278_001.0 && max_off < 1_040_001.0);
    assert!(max_off - min_off > 500_000.0, "harmonics collapsed");
    let up = Vec3::Y;
    let local = sample_datum(g, &datum, up);
    for altitude in [10_000.0f32, 50_000.0, 1.0e5, 1.0e7] {
        let distance = (HOME_RADIUS + f64::from(local) + f64::from(altitude)) as f32;
        let rho = (HOME_RADIUS as f32) / distance;
        let dir = -up;
        let got = ray_mapped_reference(
            dir,
            dir,
            rho,
            distance,
            Quat::IDENTITY,
            g,
            &datum,
            min_off,
            max_off,
        )
        .expect("straight down hits");
        // f32 radius/distance, not the f64 altitude ratio: R is not an f32 integer.
        let r = f64::from(rho) + f64::from(local) / f64::from(distance);
        let expect = 1.0 - r;
        assert!(
            (got - expect).abs() < 1e-9,
            "alt {altitude} reference {got} 1 - r(local) {expect}"
        );
    }
}

/// Outermost reference hit, sweeping from nadir toward the anti-centre.
fn reference_horizon_angle(
    dir: Vec3,
    azimuth: f32,
    rho: f32,
    distance: f32,
    g: u32,
    datum: &[f32],
    min_off: f32,
    max_off: f32,
) -> f64 {
    let hits = |angle: f64| {
        let ray = ray_from_nadir(dir, azimuth, angle as f32);
        ray_mapped_reference(
            ray,
            dir,
            rho,
            distance,
            Quat::IDENTITY,
            g,
            datum,
            min_off,
            max_off,
        )
        .is_some()
    };
    assert!(hits(0.0), "nadir missed");
    let mut lo = 0.0f64;
    let mut hi = std::f64::consts::PI;
    if hits(hi) {
        return hi;
    }
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if hits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Fixed-point march ([`ray_mapped_fast`]) against the f64 reference on a
/// home-datum sweep.
///
/// Measured on this sweep (208 rays: altitudes 10 km, 50 km, 100 km and
/// 10,000 km, four azimuths, nadir through the horizon band):
/// worst relative `|Δt|/t` = 8.133e-4, at 100 km, azimuth 0, 0.002 rad
/// inside the reference horizon. Grazing horizon misses = 11, each the
/// sample 2e-4 rad inside that horizon (the reference hits and the
/// fixed-point march misses). A disagreement any farther away, or a
/// fixed-point hit where the reference misses, fails. Misses that close
/// to the horizon are the limb's job.
#[test]
fn mapped_solver_matches_the_reference_near_the_ground() {
    // Just above the measured 8.133e-4. The march is deterministic, so
    // the slack is one digit of the reported figure.
    const FAST_WORST_REL: f64 = 8.14e-4;
    /// The sweep's closest interior sample. Every measured miss lands here.
    const FAST_GRAZE_RAD: f64 = 2.0e-4;
    const FAST_GRAZING_MISSES: u32 = 11;

    let g = 33u32;
    let datum = home_datum(g);
    let min_off = datum.iter().copied().fold(f32::MAX, f32::min);
    let max_off = datum.iter().copied().fold(f32::MIN, f32::max);
    let local = sample_datum(g, &datum, Vec3::Y);
    let radius = HOME_RADIUS as f32;
    let mut worst = 0.0f64;
    let mut worst_where = String::new();
    let mut grazing_misses = 0u32;
    let mut other_misses = 0u32;
    let mut compared = 0u32;
    for altitude in [10_000.0f32, 50_000.0, 1.0e5, 1.0e7] {
        let distance = radius + local + altitude;
        let rho = radius / distance;
        let dir = -Vec3::Y;
        for azimuth in [0.0f32, 1.3, 2.6, 4.2] {
            let horizon =
                reference_horizon_angle(dir, azimuth, rho, distance, g, &datum, min_off, max_off);
            let mut angles = vec![0.0f64, 0.4, 0.8, 1.2217304763960306];
            for delta in [
                -0.2, -0.05, -0.02, -0.01, -0.005, -0.002, -2.0e-4, 2.0e-4, 0.02,
            ] {
                let angle = horizon + delta;
                if angle > 0.0 && angle < std::f64::consts::PI {
                    angles.push(angle);
                }
            }
            for angle in angles {
                let ray = ray_from_nadir(dir, azimuth, angle as f32);
                let reference = ray_mapped_reference(
                    ray,
                    dir,
                    rho,
                    distance,
                    Quat::IDENTITY,
                    g,
                    &datum,
                    min_off,
                    max_off,
                );
                let got = ray_mapped_fast(
                    ray,
                    dir,
                    rho,
                    distance,
                    Quat::IDENTITY,
                    1.0,
                    g,
                    &datum,
                    min_off,
                    max_off,
                );
                let from_horizon = (angle - horizon).abs();
                compared += 1;
                match (reference, got) {
                    (None, None) => {}
                    (Some(t_ref), Some(hit)) => {
                        let rel = (f64::from(hit.t) - t_ref).abs() / t_ref;
                        if rel > worst {
                            worst = rel;
                            worst_where = format!(
                                "alt {altitude} az {azimuth} angle {angle} \
                                 from_horizon {from_horizon} t {t_ref} got {}",
                                hit.t
                            );
                        }
                    }
                    (Some(_), None) if from_horizon <= FAST_GRAZE_RAD => {
                        grazing_misses += 1;
                    }
                    (reference, got) => {
                        other_misses += 1;
                        if other_misses <= 8 {
                            eprintln!(
                                "hit/miss alt {altitude} az {azimuth} angle {angle} \
                                 from_horizon {from_horizon} horizon {horizon} \
                                 ref {reference:?} got {got:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        other_misses == 0 && grazing_misses <= FAST_GRAZING_MISSES && worst <= FAST_WORST_REL,
        "fixed-point march disagrees with the f64 reference: {other_misses} \
         non-grazing hit/miss mismatches over {compared} rays, \
         {grazing_misses} grazing horizon misses (bound {FAST_GRAZING_MISSES}, \
         within {FAST_GRAZE_RAD} rad), worst relative |Δt|/t = {worst} \
         (bound {FAST_WORST_REL}) at {worst_where}"
    );
}

/// Rays inside the lo-sphere disc hit far from a graze, so the five-sample
/// march and the bracket reference name the same surface.
#[test]
fn fast_march_matches_robust_inside_the_lo_sphere_disc() {
    let g = 33u32;
    let datum = home_datum(g);
    let min_off = datum.iter().copied().fold(f32::MAX, f32::min);
    let max_off = datum.iter().copied().fold(f32::MIN, f32::max);
    let radius = HOME_RADIUS as f32;
    let local = sample_datum(g, &datum, Vec3::Y);
    let dir = -Vec3::Y;
    let mut worst = 0.0f64;
    let mut worst_where = String::new();
    let mut compared = 0u32;
    // Eye altitude above the local surface, 10 m through 1e6 m.
    for altitude in [10.0f32, 100.0, 1_000.0, 10_000.0, 1.0e5, 1.0e6] {
        let distance = radius + local + altitude;
        let rho = radius / distance;
        let rho_lo = rho + min_off / distance;
        assert!(
            rho_lo > 0.0 && rho_lo < 1.0,
            "alt {altitude} rho_lo {rho_lo} is not an exterior lo sphere"
        );
        // Half-angle of the lo-sphere disc. A unit ray at angle θ from the
        // centre hits that sphere when sin θ < rho_lo. Stay strictly inside.
        let limb = f64::from(rho_lo).asin();
        for azimuth in [0.0f32, 1.1, 2.4, 3.7, 5.0] {
            for frac in [0.0f64, 0.2, 0.45, 0.7, 0.9, 0.98] {
                let angle = limb * frac;
                let ray = ray_from_nadir(dir, azimuth, angle as f32);
                let robust = ray_mapped(
                    ray,
                    dir,
                    rho,
                    distance,
                    Quat::IDENTITY,
                    1.0,
                    g,
                    &datum,
                    min_off,
                    max_off,
                );
                let fast = ray_mapped_fast(
                    ray,
                    dir,
                    rho,
                    distance,
                    Quat::IDENTITY,
                    1.0,
                    g,
                    &datum,
                    min_off,
                    max_off,
                );
                let (Some(robust), Some(fast)) = (robust, fast) else {
                    panic!(
                        "interior ray missed alt {altitude} az {azimuth} \
                         frac {frac} angle {angle} robust {robust:?} fast {fast:?}"
                    );
                };
                compared += 1;
                let rel =
                    (f64::from(fast.t) - f64::from(robust.t)).abs() / f64::from(robust.t).abs();
                if rel > worst {
                    worst = rel;
                    worst_where = format!(
                        "alt {altitude} az {azimuth} frac {frac} \
                         robust {} fast {}",
                        robust.t, fast.t
                    );
                }
            }
        }
    }
    assert!(
        compared > 100 && worst <= 1.0e-6,
        "fast march disagrees inside the lo disc: {compared} rays, \
         worst relative |Δt|/t = {worst} at {worst_where}"
    );
}

/// Cube LOD the shader derives from the pixel footprint. `ndot` is
/// `|ray · n|`. The face-centre texel is `(π/2) R / albedo_size`, and the
/// result is clamped to the mip range `[0, log2(albedo_size)]`.
fn mapped_albedo_lod(
    px: f32,
    t: f32,
    distance: f32,
    radius: f32,
    ndot: f32,
    albedo_size: u32,
) -> f32 {
    let nd = ndot.abs().max(0.05);
    let footprint = px * t * distance / nd;
    let texel = (std::f32::consts::FRAC_PI_2 * radius) / albedo_size as f32;
    let lod = (footprint / texel).max(1e-20).log2();
    let mip_max = (albedo_size as f32).log2();
    lod.clamp(0.0, mip_max)
}

#[test]
fn albedo_lod_rises_toward_the_horizon() {
    let px = 0.01f32;
    let t = 0.5f32;
    let distance = 1.0e7f32;
    let radius = 3.101752e7f32;
    let albedo = 1024u32;
    let down = mapped_albedo_lod(px, t, distance, radius, 1.0, albedo);
    let graze = mapped_albedo_lod(px, t, distance, radius, 0.05, albedo);
    let flatter = mapped_albedo_lod(px, t, distance, radius, 0.01, albedo);
    let levels = std::f32::consts::LN_2;
    let expect = (20.0f32).ln() / levels;
    assert!(
        (graze - down - expect).abs() < 1e-4,
        "nadir {down} graze {graze} expected +{expect}"
    );
    assert!((flatter - graze).abs() < 1e-5, "ndot below 0.05 must clamp");
    assert!(down > 0.0 && graze < (albedo as f32).log2());
    let tiny = mapped_albedo_lod(1e-6, 1e-4, distance, radius, 1.0, albedo);
    let huge = mapped_albedo_lod(1.0, 10.0, distance, radius, 0.05, albedo);
    assert_eq!(tiny, 0.0);
    assert_eq!(huge, (albedo as f32).log2());
}

/// Closed-form tangent of the outer air sphere, or `1` when that sphere
/// contains the camera (the game then disables the scalar cone).
fn outer_horizon_sine(rho_cap: f32) -> f32 {
    if rho_cap > 0.0 && rho_cap < 1.0 {
        -(1.0 - rho_cap * rho_cap).max(0.0).sqrt()
    } else {
        1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SilhouetteClass {
    Hit,
    Limb,
    Miss,
}

/// Highland at the horizon: every fixed-point ray hits, or the air chord
/// covers the graze the march missed. A miss next to the silhouette is a
/// hole. No limb past the air cap.
#[test]
fn highland_silhouette_is_continuous() {
    let g = 33u32;
    let radius = HOME_RADIUS as f32;
    let air = 20_000.0f32;
    let px = 1.5e-3f32;
    let dir = -Vec3::Y;
    let datum = horizon_highland(g, 0.03, 0.55, 0.45, 1_000_000.0);
    let max_off = datum.iter().copied().fold(0.0f32, f32::max);
    let min_off = 0.0f32;
    assert_eq!(sample_datum(g, &datum, Vec3::Y), 0.0);
    let mut inside_empty = 0u32;
    let mut limb_above = 0u32;
    let mut holes = 0u32;
    let mut hole_where = String::new();
    let mut gaps = 0u32;
    let mut gap_where = String::new();
    let mut limb_n = 0u32;
    let mut chord_jump = 0u32;
    for altitude in [50_000.0f32, 2_000_000.0] {
        let distance = radius + altitude;
        let rho = radius / distance;
        let rho_cap = rho + (max_off + air) / distance;
        let horizon = outer_horizon_sine(rho_cap);
        for azimuth in [0.0f32, 0.35, 0.55, 1.2] {
            let horizon_ang =
                reference_horizon_angle(dir, azimuth, rho, distance, g, &datum, min_off, max_off);
            let mut row: Vec<(SilhouetteClass, f32, f32, f32, f32)> = Vec::new();
            for k in -40..=80 {
                let angle = horizon_ang + f64::from(k) * 2.0e-4;
                if !(angle > 0.0 && angle < std::f64::consts::PI) {
                    continue;
                }
                let ray = ray_from_nadir(dir, azimuth, angle as f32);
                let facing = ray.dot(dir);
                let above = ray.dot(-dir) > horizon;
                let hit = if above {
                    None
                } else {
                    ray_mapped_fast(
                        ray,
                        dir,
                        rho,
                        distance,
                        Quat::IDENTITY,
                        horizon,
                        g,
                        &datum,
                        min_off,
                        max_off,
                    )
                };
                let chord_world = if above {
                    0.0
                } else {
                    ray_mapped_limb_chord(
                        ray,
                        dir,
                        rho,
                        distance,
                        Quat::IDENTITY,
                        g,
                        &datum,
                        max_off,
                        air,
                    )
                };
                // Normalised, so the jump check stays on the same scale as
                // `px`. The world chord is what the shader integrates.
                let chord = if distance > 0.0 {
                    chord_world / distance
                } else {
                    0.0
                };
                let s = ray.cross(dir).length();
                let r_local = if facing > 0.0 {
                    mapped_radius_at(ray, dir, facing, rho, distance, Quat::IDENTITY, g, &datum)
                } else {
                    0.0
                };
                let class = if hit.is_some() {
                    SilhouetteClass::Hit
                } else if chord_world > 0.0 {
                    SilhouetteClass::Limb
                } else {
                    SilhouetteClass::Miss
                };
                if facing > 0.0 && s + 1e-5 < r_local && class == SilhouetteClass::Miss {
                    inside_empty += 1;
                }
                if class == SilhouetteClass::Limb && s > rho_cap + 1e-4 {
                    limb_above += 1;
                }
                if class == SilhouetteClass::Limb {
                    limb_n += 1;
                }
                row.push((class, facing, chord, angle as f32, chord_world));
            }
            for w in row.windows(3) {
                let (a, _, _, _, _) = w[0];
                let (b, facing, _, ang, chord_w) = w[1];
                let (c, _, _, _, _) = w[2];
                if b == SilhouetteClass::Miss
                    && a != SilhouetteClass::Miss
                    && c != SilhouetteClass::Miss
                {
                    holes += 1;
                    if hole_where.is_empty() {
                        hole_where = format!(
                            "alt {altitude} az {azimuth} ang {ang} facing {facing} chord {chord_w} href {horizon_ang} {:?}->{:?}->{:?}",
                            a, b, c
                        );
                    }
                }
                if a == SilhouetteClass::Hit && b == SilhouetteClass::Miss && facing > 0.0 {
                    gaps += 1;
                    if gap_where.is_empty() {
                        gap_where = format!(
                            "alt {altitude} az {azimuth} ang {ang} facing {facing} chord {chord_w} href {horizon_ang} {:?}->{:?}->{:?}",
                            a, b, c
                        );
                    }
                }
            }
            for w in row.windows(2) {
                let (a, _, ca, _, _) = w[0];
                let (b, _, cb, _, _) = w[1];
                if a == SilhouetteClass::Limb && b == SilhouetteClass::Limb {
                    let scale = ca.max(cb).max(px);
                    if (ca - cb).abs() > 8.0 * scale {
                        chord_jump += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        inside_empty, 0,
        "rays inside the local surface with no hit and no limb"
    );
    assert_eq!(limb_above, 0, "limb rays past the outer air sphere");
    assert_eq!(
        gaps, 0,
        "hit then miss with no limb between the surface and space ({gap_where})"
    );
    assert_eq!(holes, 0, "limb band with a miss hole in it ({hole_where})");
    assert_eq!(chord_jump, 0, "limb chord jumped between adjacent rays");
    assert!(limb_n > 0, "expected some limb rays on the highland");
}

/// `true` when a 25 km march along the cap chord finds air. Same eye and
/// facing gates as [`ray_mapped_limb_chord`]. The segment test is the
/// analytic ray radius against the linearly interpolated air top, so a
/// graze thinner than the step is not missed.
fn dense_crosses_air(
    ray: Vec3,
    dir: Vec3,
    rho: f32,
    distance: f32,
    g: u32,
    datum: &[f32],
    max_off: f32,
    air: f32,
) -> bool {
    if !(distance > 0.0) || !(air > 0.0) || !ray.is_finite() || !dir.is_finite() {
        return false;
    }
    let facing = ray.dot(dir);
    if !(facing > 0.0) {
        return false;
    }
    if eye_inside_mapped_air(dir, rho, distance, Quat::IDENTITY, g, datum, air) {
        return false;
    }
    let rho_cap = rho + (max_off + air) / distance;
    let Some((t0, t1)) = cap_chord(facing, rho_cap) else {
        return false;
    };
    let step = (25_000.0 / distance).max(1.0e-8);
    let ca = facing.clamp(t0, t1);
    let (mut prev_rad, mut prev_r) =
        limb_air_at(ray, dir, t0, rho, distance, Quat::IDENTITY, g, datum, air);
    let mut prev_t = t0;
    for _ in 0..20_000 {
        let mut next = prev_t + step;
        if prev_t < ca && next > ca {
            next = ca;
        }
        if next >= t1 {
            next = t1;
        }
        if !(next > prev_t) {
            break;
        }
        let (rad, r_air) =
            limb_air_at(ray, dir, next, rho, distance, Quat::IDENTITY, g, datum, air);
        if limb_segment_t(prev_t, next, prev_rad, rad, prev_r, r_air) > 0.0 {
            return true;
        }
        let done = next >= t1;
        prev_t = next;
        prev_rad = rad;
        prev_r = r_air;
        if done {
            break;
        }
    }
    false
}

fn dense_air_top_angle(
    dir: Vec3,
    azimuth: f32,
    rho: f32,
    distance: f32,
    g: u32,
    datum: &[f32],
    max_off: f32,
    air: f32,
) -> Option<f64> {
    limb_boundary_angle(&|theta| {
        let ray = ray_from_nadir(dir, azimuth, theta as f32);
        dense_crosses_air(ray, dir, rho, distance, g, datum, max_off, air)
    })
}

/// Constant datum: the crossed-air chord is the outer-sphere chord. A ray
/// that dips under the surface still counts the interior, so a graze the
/// march missed does not open a gap. The eye inside the shell draws none.
#[test]
fn constant_datum_limb_chord_matches_the_outer_sphere() {
    let g = 4u32;
    let h = 100_000.0f32;
    let air = 20_000.0f32;
    let radius = 1_000_000.0f32;
    let distance = 2_000_000.0f32;
    let datum = vec![h; 6 * g as usize * g as usize];
    let rho = radius / distance;
    let dir = Vec3::Z;
    let r_air = rho + (h + air) / distance;
    let r_surf = rho + h / distance;
    let ray_at = |s: f32| {
        let facing = (1.0 - s * s).max(0.0).sqrt();
        (dir * facing + Vec3::X * s).normalize()
    };
    let analytic = |ray: Vec3| {
        let s = ray.cross(dir).length();
        let facing = ray.dot(dir);
        let disc = r_air * r_air - s * s;
        assert!(disc > 0.0, "s {s} r_air {r_air}");
        let sd = disc.sqrt();
        assert!(facing - sd > 0.0, "near root is behind the camera");
        2.0 * sd * distance
    };
    // Between the surface and the air top: the shell chord is the outer chord.
    let between = ray_at(0.5 * (r_surf + r_air));
    let got = ray_mapped_limb_chord(
        between,
        dir,
        rho,
        distance,
        Quat::IDENTITY,
        g,
        &datum,
        h,
        air,
    );
    let expect = analytic(between);
    assert!(
        (got - expect).abs() < 1.0,
        "shell ray chord {got} analytic {expect}"
    );
    // Under the surface: the interior counts, so this is the full outer
    // chord, not the thin shell between r_surf and r_air.
    let under = ray_at(r_surf * 0.85);
    let got_under =
        ray_mapped_limb_chord(under, dir, rho, distance, Quat::IDENTITY, g, &datum, h, air);
    let expect_under = analytic(under);
    let s_under = under.cross(dir).length();
    let shell_only = 2.0
        * ((r_air * r_air - s_under * s_under).sqrt()
            - (r_surf * r_surf - s_under * s_under).max(0.0).sqrt())
        * distance;
    assert!(
        (got_under - expect_under).abs() < 1.0,
        "under-surface chord {got_under} analytic {expect_under}"
    );
    assert!(
        got_under > shell_only + 1_000.0,
        "under-surface chord {got_under} collapsed to the shell {shell_only}"
    );
    // Inside the cap, outside the air.
    let miss = ray_at(r_air + 0.02);
    let got_miss = ray_mapped_limb_chord(
        miss,
        dir,
        rho,
        distance,
        Quat::IDENTITY,
        g,
        &datum,
        h + 500_000.0,
        air,
    );
    assert_eq!(got_miss, 0.0, "a ray outside the air drew a limb");
    // Eye 5 km above the datum, air 20 km: no limb on any ray.
    let distance_in = radius + h + 5_000.0;
    let rho_in = radius / distance_in;
    let got_in = ray_mapped_limb_chord(
        dir,
        dir,
        rho_in,
        distance_in,
        Quat::IDENTITY,
        g,
        &datum,
        h,
        air,
    );
    assert_eq!(got_in, 0.0, "eye inside the air drew a limb");
}

/// Eye 10 km above a home-datum lowland draws no limb. At 50 km the limb's
/// top along every azimuth is the true top of the air the ray crosses,
/// within 2 pixels of a 3440-wide 90° view.
#[test]
fn lowland_limb_matches_the_air_the_ray_crosses() {
    let g = 33u32;
    let datum = home_datum(g);
    let max_off = datum.iter().copied().fold(f32::MIN, f32::max);
    let min_off = datum.iter().copied().fold(f32::MAX, f32::min);
    let (up, foot) = lowland_foot(g, &datum);
    assert!(
        foot < min_off + 50_000.0,
        "foot {foot} is not a lowland (datum {min_off}..{max_off})"
    );
    let radius = HOME_RADIUS as f32;
    let air = 20_000.0f32;
    let dir = -up;
    let px = f64::from(std::f32::consts::FRAC_PI_2 / 3440.0);

    let distance_in = radius + foot + 10_000.0;
    let rho_in = radius / distance_in;
    assert!(eye_inside_mapped_air(
        dir,
        rho_in,
        distance_in,
        Quat::IDENTITY,
        g,
        &datum,
        air,
    ));
    for k in 0..48 {
        let azimuth = k as f32 * std::f32::consts::TAU / 48.0;
        for j in 0..48 {
            let angle = (j as f32 + 0.5) * std::f32::consts::PI / 48.0;
            let ray = ray_from_nadir(dir, azimuth, angle);
            let chord = ray_mapped_limb_chord(
                ray,
                dir,
                rho_in,
                distance_in,
                Quat::IDENTITY,
                g,
                &datum,
                max_off,
                air,
            );
            assert_eq!(
                chord, 0.0,
                "alt 10000 az {azimuth} ang {angle} drew a limb of {chord}"
            );
        }
    }

    let distance = radius + foot + 50_000.0;
    let rho = radius / distance;
    assert!(!eye_inside_mapped_air(
        dir,
        rho,
        distance,
        Quat::IDENTITY,
        g,
        &datum,
        air,
    ));
    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    for k in 0..96 {
        let azimuth = k as f32 * std::f32::consts::TAU / 96.0;
        let got = mapped_limb_top_angle(dir, azimuth, rho, distance, g, &datum, max_off, air)
            .map(|angle| angle - std::f64::consts::FRAC_PI_2);
        let expect = dense_air_top_angle(dir, azimuth, rho, distance, g, &datum, max_off, air)
            .map(|angle| angle - std::f64::consts::FRAC_PI_2);
        let (Some(got), Some(expect)) = (got, expect) else {
            panic!("az {azimuth} limb {got:?} true air {expect:?}");
        };
        let pixels = (got - expect).abs() / px;
        if pixels > worst {
            worst = pixels;
            worst_at = format!("az {azimuth} limb {got} true {expect}");
        }
    }
    assert!(
        worst <= 2.0,
        "limb top missed the air the ray crosses by {worst} px at {worst_at}"
    );
}

/// Source of the shader function whose definition starts with `head`, from
/// its opening brace to the matching close.
fn shader_fn<'a>(src: &'a str, head: &str) -> &'a str {
    let at = src
        .find(head)
        .unwrap_or_else(|| panic!("far_body.slang has no `{head}`"));
    let open = at + src[at..].find('{').expect("function body");
    let mut depth = 0u32;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &src[open..=open + i];
                }
            }
            _ => {}
        }
    }
    panic!("`{head}` has no closing brace")
}

/// The literal after each `marker` in `text`, in order. At least one.
fn literals_after<'a>(text: &'a str, marker: &str) -> Vec<&'a str> {
    let found: Vec<&str> = text
        .match_indices(marker)
        .map(|(at, _)| {
            let rest = &text[at + marker.len()..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+')))
                .unwrap_or(rest.len());
            &rest[..end]
        })
        .collect();
    assert!(!found.is_empty(), "far_body.slang has no `{marker}`");
    found
}

/// The literal after the one `marker` in `text`.
fn literal_after<'a>(text: &'a str, marker: &str) -> &'a str {
    let found = literals_after(text, marker);
    assert_eq!(found.len(), 1, "`{marker}` is not unique: {found:?}");
    found[0]
}

fn float_bits(lit: &str) -> u32 {
    lit.parse::<f32>()
        .unwrap_or_else(|_| panic!("`{lit}` is not a float"))
        .to_bits()
}

fn uint(lit: &str) -> u32 {
    lit.trim_end_matches('u')
        .parse()
        .unwrap_or_else(|_| panic!("`{lit}` is not an integer"))
}

/// The shader writes these out by hand. The mirror only mirrors it while they
/// agree, so read each one back out of the shader source.
#[test]
fn mirror_constants_match_the_shader() {
    let src = include_str!("../../shaders/far_body.slang");

    // Fixed-point march budget and shallow-slope settle.
    assert_eq!(
        uint(literal_after(src, "static const int FAR_MAP_CAP = ")),
        MAPPED_EVAL_CAP,
        "FAR_MAP_CAP"
    );
    assert_eq!(
        float_bits(literal_after(src, "static const float FAR_MAP_FTOL = ")),
        MAPPED_F_TOL.to_bits(),
        "FAR_MAP_FTOL"
    );

    // Equiangular chart atan, c6 first.
    let atan = shader_fn(src, "float far_atan_approx(float x)");
    let bits: Vec<u32> = literals_after(atan, "asfloat(0x")
        .iter()
        .map(|lit| u32::from_str_radix(lit.trim_end_matches('u'), 16).expect("hex bits"))
        .collect();
    assert_eq!(bits, ATAN_BITS, "far_atan_approx coefficients");

    // Limb sample offsets, indexed 0 ..= 7, and the loop that walks them.
    let delta = shader_fn(src, "float far_limb_delta(int i)");
    assert!(delta.contains("if (i <= 0)"), "far_limb_delta first guard");
    let guards: Vec<u32> = literals_after(delta, "if (i == ")
        .into_iter()
        .map(uint)
        .collect();
    let expect: Vec<u32> = (1..LIMB_SAMPLES as u32 - 1).collect();
    assert_eq!(guards, expect, "far_limb_delta guards");
    let deltas: Vec<u32> = literals_after(delta, "return ")
        .into_iter()
        .map(float_bits)
        .collect();
    let expect: Vec<u32> = (0..LIMB_SAMPLES)
        .map(|i| limb_delta_m(i).to_bits())
        .collect();
    assert_eq!(deltas, expect, "far_limb_delta offsets");
    let chord = shader_fn(src, "float far_mapped_air_chord(");
    assert_eq!(
        uint(literal_after(chord, "for (int i = 0; i < ")) as usize,
        LIMB_SAMPLES,
        "far_mapped_air_chord samples"
    );

    // Rounded body: march budget and pads.
    let rounded = shader_fn(src, "bool far_ray_rounded(");
    assert_eq!(
        uint(literal_after(rounded, "step < ")),
        ROUNDED_STEPS,
        "far_ray_rounded steps"
    );
    for (marker, value) in [
        ("if (s <= rho * ", ROUNDED_SNAP),
        ("sd <= rhoB * ", ROUNDED_GRAZE),
        ("s >= rho && s <= rho * ", ROUNDED_GRAZE_BAND),
        ("sL <= rho * (1.0 + ", ROUNDED_SETTLE),
    ] {
        assert_eq!(
            float_bits(literal_after(rounded, marker)),
            value.to_bits(),
            "far_ray_rounded `{marker}`"
        );
    }
    // Every copy of the enclosing-sphere bound, not only the intersection's.
    for lit in literals_after(src, "pow(3.0, 0.5 - 1.0 / p) * (1.0 + ") {
        assert_eq!(
            float_bits(lit),
            ROUNDED_BOUND_PAD.to_bits(),
            "rounded bound pad"
        );
    }

    // Cube slabs.
    let slab = shader_fn(src, "bool far_slab(");
    assert_eq!(
        float_bits(literal_after(slab, "abs(d) < ")),
        SLAB_EPS.to_bits(),
        "far_slab epsilon"
    );
    let cube = shader_fn(src, "bool far_ray_cube(");
    assert_eq!(
        float_bits(literal_after(cube, "float tMax = ")),
        T_FAR.to_bits(),
        "far_ray_cube far t"
    );

    // Composite start depth. The single-mapped variant tests its hit and limb
    // against the same value the loop starts `bestDepth` at, so a mapsolo tile
    // keeps the loop's pixels.
    let solo = shader_fn(src, "float4 far_mapped_solo(");
    let sentinels = literals_after(src, "float bestDepth = ");
    let solo_tests = literals_after(solo, "distance < ");
    assert_eq!(solo_tests.len(), 2, "far_mapped_solo depth tests");
    for lit in sentinels.into_iter().chain(solo_tests) {
        assert_eq!(float_bits(lit), T_FAR.to_bits(), "far depth sentinel");
    }
}

/// The shaders size the far table and the map arrays with generated constants
/// whose Rust twins are asserted equal at compile time. The cube sampler
/// switch is still written out by hand: one constant-index arm per map slot.
#[test]
fn shader_map_slots_match_the_host() {
    let src = include_str!("../../shaders/far_body.slang");
    assert!(src.contains("SamplerCube farAlbedo[MAX_FAR_MAPS];"));
    let cube = shader_fn(src, "float3 far_cube(");
    let guards: Vec<u32> = literals_after(cube, "if (map == ")
        .into_iter()
        .map(uint)
        .collect();
    let expect: Vec<u32> = (0..MAX_FAR_MAPS as u32 - 1).collect();
    assert_eq!(guards, expect, "far_cube guards");
    let arms: Vec<u32> = literals_after(cube, "farAlbedo[")
        .into_iter()
        .map(uint)
        .collect();
    let expect: Vec<u32> = (0..MAX_FAR_MAPS as u32).collect();
    assert_eq!(arms, expect, "far_cube arms");
}
