use glam::{Quat, Vec3};

use super::support::{pack_table, placed, sample, view_along_neg_z};
use crate::color::LinearRgb;
use crate::far_body::{FarBody, FarMapId, FarShape, MAX_FAR_BODIES, MAX_FAR_MAPS, store};
use crate::vk::far_bodies::table::{
    FarBodyGpu, HORIZON_BINS, HORIZON_TABLES, MAX_FAR_TILES, ShapeCode, pack_one, table_bytes,
};
use crate::vk::far_bodies::tiles::used_tiles;

#[test]
fn packs_a_billion_block_sphere_exactly() {
    let body = FarBody {
        dir: Vec3::new(0.0, 0.0, 4.0),
        distance: 1.0e9,
        radius: 2.5e8,
        shape: FarShape::Sphere,
        rotation: Quat::IDENTITY,
        albedo: [LinearRgb([0.2, 0.3, 0.4]); 6],
        atmosphere: LinearRgb([0.0, 0.1, 0.0]),
        seed: 0xA11CE,
    };
    let mut slot = [FarBody::default(); MAX_FAR_BODIES];
    let n = store(std::slice::from_ref(&body), &mut slot);
    assert_eq!(n, 1);
    let table = pack_table(&slot[..n as usize], None);
    assert_eq!(table.header[0], 1);
    assert_eq!(table.header[1], 0);
    let gpu = &table.body[0];
    assert_eq!(gpu.dir_rho[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(gpu.dir_rho[1].to_bits(), 0.0f32.to_bits());
    assert_eq!(gpu.dir_rho[2].to_bits(), 1.0f32.to_bits());
    assert_eq!(gpu.dir_rho[3].to_bits(), 0.25f32.to_bits());
    assert_eq!(gpu.rot, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(gpu.seed[0], 0xA11CE);
    assert_eq!(gpu.atmosphere[3].to_bits(), 1.0f32.to_bits());
    assert_eq!(gpu.albedo0[0].to_bits(), 0.2f32.to_bits());

    let mut cube = body;
    cube.shape = FarShape::Cube;
    cube.distance = 4.0;
    cube.radius = 1.0;
    let packed = pack_one(&cube);
    assert_eq!(packed.atmosphere[3].to_bits(), 0.0f32.to_bits());
    assert_eq!(packed.dir_rho[3].to_bits(), 0.25f32.to_bits());

    let mut wall = body;
    wall.shape = FarShape::InnerSphere;
    wall.dir = Vec3::Z;
    wall.distance = 2.0;
    wall.radius = 5.0;
    let packed = pack_one(&wall);
    assert_eq!(packed.atmosphere[3].to_bits(), 2.0f32.to_bits());
    assert_eq!(packed.dir_rho[3].to_bits(), 2.5f32.to_bits());
    assert_eq!(packed.dir_rho[2].to_bits(), 1.0f32.to_bits());
    assert_eq!(packed.seed[1], 0);
}

#[test]
fn packs_rounded_exponent_round_trip() {
    let body = FarBody {
        dir: Vec3::new(0.0, 0.0, 8.0),
        distance: 8.0,
        radius: 2.0,
        shape: FarShape::Rounded { exponent: 2.17 },
        rotation: Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
        albedo: [LinearRgb([0.4, 0.5, 0.6]); 6],
        atmosphere: LinearRgb([0.1, 0.2, 0.3]),
        seed: 0xB0D1,
    };
    let mut slot = [FarBody::default(); MAX_FAR_BODIES];
    let n = store(std::slice::from_ref(&body), &mut slot);
    assert_eq!(n, 1);
    assert_eq!(slot[0].shape, FarShape::Rounded { exponent: 2.17 });
    let table = pack_table(&slot[..n as usize], None);
    let gpu = &table.body[0];
    assert_eq!(gpu.atmosphere[3].to_bits(), 3.0f32.to_bits());
    assert_eq!(f32::from_bits(gpu.seed[1]).to_bits(), 2.17f32.to_bits());
    assert_eq!(gpu.seed[0], 0xB0D1);
    assert_eq!(gpu.seed[2], 0);
    assert_eq!(gpu.dir_rho[3].to_bits(), 0.25f32.to_bits());
    assert_eq!(gpu.rot, [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(std::mem::size_of::<FarBodyGpu>(), 160);

    let again = f32::from_bits(pack_one(&slot[0]).seed[1]);
    assert_eq!(again.to_bits(), 2.17f32.to_bits());
}

#[test]
fn every_shape_packs_world_distance_in_albedo2_w() {
    let shapes = [
        (FarShape::Sphere, 12.5, 1.0),
        (FarShape::Cube, 12.5, 1.0),
        (FarShape::InnerSphere, 2.0, 8.0),
        (FarShape::Rounded { exponent: 3.0 }, 12.5, 1.0),
        (
            FarShape::Mapped {
                map: FarMapId(1),
                horizon: 0.1,
                air: 0.2,
            },
            12.5,
            1.0,
        ),
    ];
    for (shape, distance, radius) in shapes {
        let gpu = pack_one(&sample(shape, distance, radius));
        assert_eq!(
            gpu.albedo2[3].to_bits(),
            distance.to_bits(),
            "albedo2.w is the world distance"
        );
    }
}

/// The named shape codes are the ones `far_table.slang` lists, each packed
/// shape decodes to its own code, and the named lanes read what
/// `pack_one` wrote.
#[test]
fn shape_codes_and_lanes_match_the_packed_record() {
    let src = include_str!("../../../../shaders/far_table.slang");
    let listed = src
        .split("atmosphere.a is the shape:")
        .nth(1)
        .and_then(|rest| rest.split('.').next())
        .expect("far_table.slang lists the shape codes");
    let listed = listed.replace("//", " ");
    let listed: Vec<&str> = listed.split_whitespace().collect();
    let names = ["cube", "sphere", "inner sphere", "rounded", "mapped"];
    let expect: Vec<String> = ShapeCode::ALL
        .iter()
        .zip(names)
        .map(|(code, name)| format!("{} {name}", code.lane()))
        .collect();
    assert_eq!(listed.join(" "), expect.join(", "));

    let mapped = FarShape::Mapped {
        map: FarMapId(5),
        horizon: -0.25,
        air: 0.4,
    };
    let shapes = [
        (FarShape::Cube, ShapeCode::Cube),
        (FarShape::Sphere, ShapeCode::Sphere),
        (FarShape::InnerSphere, ShapeCode::InnerSphere),
        (FarShape::Rounded { exponent: 3.0 }, ShapeCode::Rounded),
        (mapped, ShapeCode::Mapped),
    ];
    for (shape, code) in shapes {
        let gpu = pack_one(&sample(shape, 4.0, 1.0));
        assert_eq!(gpu.atmosphere[3].to_bits(), code.lane().to_bits());
        assert_eq!(gpu.shape(), Some(code));
        assert_eq!(gpu.is_mapped(), code == ShapeCode::Mapped);
        let light = matches!(code, ShapeCode::Sphere | ShapeCode::InnerSphere);
        assert_eq!(gpu.is_light(), light);
        assert_eq!(gpu.dir(), Vec3::Z);
        assert_eq!(gpu.rho().to_bits(), 0.25f32.to_bits());
        assert_eq!(gpu.distance().to_bits(), 4.0f32.to_bits());
        if code == ShapeCode::Mapped {
            assert_eq!(gpu.horizon().to_bits(), (-0.25f32).to_bits());
            assert_eq!(gpu.air().to_bits(), 0.4f32.to_bits());
            assert_eq!(gpu.map_index(), Some(5));
        } else {
            assert_eq!((gpu.horizon(), gpu.air()), (0.0, 0.0));
            assert_eq!(gpu.map_index(), None);
        }
    }
    // Off-code lanes decode to nothing and stay heavy, as before the names.
    let mut odd = pack_one(&sample(FarShape::Sphere, 4.0, 1.0));
    for lane in [f32::NAN, -0.6, 4.5, 7.0] {
        odd.atmosphere[3] = lane;
        assert_eq!(odd.shape(), None, "{lane}");
        assert!(!odd.is_mapped() && !odd.is_light(), "{lane}");
    }
    odd.seed[2] = MAX_FAR_MAPS as u32 + 1;
    assert_eq!(odd.map_index(), None);
}

#[test]
fn shader_tile_mask_length_matches_the_host_cap() {
    // The arrays are sized by the generated constants, and each Rust twin
    // is asserted equal to its generated value at compile time.
    let src = include_str!("../../../../shaders/far_table.slang");
    for decl in [
        "float4 cone[MAX_FAR_BODIES]",
        "FarGpu body[MAX_FAR_BODIES]",
        "uint tile_mask[MAX_FAR_TILES]",
        "uint tile_index[MAX_FAR_TILES]",
    ] {
        assert!(src.contains(decl), "far_table.slang has no `{decl}`");
    }
    assert!(src.contains("uint4 list_header"));
    let horizon = format!("float horizon_sin[{HORIZON_TABLES} * FAR_HORIZON_BINS]");
    assert!(
        src.contains(&horizon),
        "horizon table length drifted from {HORIZON_TABLES} * {HORIZON_BINS}"
    );
    assert!(src.contains("uint4 horizon_id"));
    let vert = include_str!("../../../../shaders/sky_tile.vert.slang");
    assert!(
        vert.contains("(xf / float(width)) * 2.0 - 1.0"),
        "tile ndc x drifted from framebuffer_ndc"
    );
    assert!(
        vert.contains("1.0 - (yf / float(height)) * 2.0"),
        "tile ndc y drifted from framebuffer_ndc"
    );
    assert!(vert.contains("min(x0 + tilePx, width)"));
}

#[test]
fn identical_used_prefix_ignores_the_mask_tail() {
    let view = view_along_neg_z(60.0, 800, 600);
    let body = placed(-Vec3::Z, 0.05, FarShape::Sphere, 1);
    let mut a = pack_table(std::slice::from_ref(&body), Some(view));
    let b = a;
    let n = used_tiles(&a);
    assert!(n > 0 && n < MAX_FAR_TILES);
    assert_eq!(table_bytes(&a), table_bytes(&b));
    a.tile_mask[n] = 0xFFFF_FFFF;
    assert_eq!(table_bytes(&a), table_bytes(&b));
    a.tile_mask[0] ^= 1;
    assert_ne!(table_bytes(&a), table_bytes(&b));
}

#[test]
fn packed_horizon_tables_do_not_name_a_body() {
    let table = pack_table(&[], None);
    assert_eq!(table.horizon_id[0], u32::MAX);
    assert_eq!(table.horizon_id[1], u32::MAX);
    assert_eq!(table.horizon_id[2], HORIZON_BINS as u32);
    assert!(table.horizon_sin.iter().all(|s| *s == 1.0));
}
