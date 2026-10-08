use super::support::{pack_table, sample};
use crate::far_body::{FarMapId, FarShape, MAX_FAR_BODIES, MAX_FAR_MAPS};
use crate::vk::far_bodies::table::FarTableGpu;

#[test]
fn cone_bound_matches_what_the_shader_can_draw() {
    let sphere = sample(FarShape::Sphere, 1.0, 0.25);
    let table = pack_table(std::slice::from_ref(&sphere), None);
    assert_eq!(table.header[0], 1);
    assert_eq!(table.cone[0][0].to_bits(), 0.0f32.to_bits());
    assert_eq!(table.cone[0][1].to_bits(), 0.0f32.to_bits());
    assert_eq!(table.cone[0][2].to_bits(), 1.0f32.to_bits());
    assert_eq!(table.cone[0][3].to_bits(), (1.05f32 * 0.25).to_bits());
    assert_eq!(table.cone[0][0..3], table.body[0].dir_rho[0..3]);

    // 1.05 * rho < 1 stays the rim sine, even past the old 0.99 sentinel.
    let near = sample(FarShape::Sphere, 1.0, 0.95);
    let near_bound = 1.05f32 * 0.95;
    assert!(near_bound > 0.99 && near_bound < 1.0);
    assert_eq!(
        pack_table(std::slice::from_ref(&near), None).cone[0][3].to_bits(),
        near_bound.to_bits()
    );
    // 1.05 * rho >= 1 is the facing hemisphere, never the sentinel.
    let home = sample(FarShape::Sphere, 1.0, 0.99999);
    assert!(1.05 * 0.99999 >= 1.0);
    assert_eq!(
        pack_table(std::slice::from_ref(&home), None).cone[0][3].to_bits(),
        1.0f32.to_bits()
    );
    let over = sample(FarShape::Sphere, 1.0, 0.96);
    assert!(1.05 * 0.96 >= 1.0);
    assert_eq!(
        pack_table(std::slice::from_ref(&over), None).cone[0][3].to_bits(),
        1.0f32.to_bits()
    );

    let cube = sample(FarShape::Cube, 1.0, 0.2);
    let cube_bound = 3.0f32.sqrt() * 1.035 * 0.2;
    assert!(cube_bound < 0.99);
    assert_eq!(
        pack_table(std::slice::from_ref(&cube), None).cone[0][3].to_bits(),
        cube_bound.to_bits()
    );
    // Inside the corner sphere the cube surrounds the camera: sentinel.
    let fat = sample(FarShape::Cube, 1.0, 0.6);
    let fat_bound = 3.0f32.sqrt() * 1.035 * 0.6;
    assert!(fat_bound >= 0.99);
    assert_eq!(
        pack_table(std::slice::from_ref(&fat), None).cone[0][3].to_bits(),
        (-1.0f32).to_bits()
    );

    let rounded = sample(FarShape::Rounded { exponent: 4.0 }, 1.0, 0.2);
    let p = 4.0f32;
    let rho_b = 0.2 * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
    assert_eq!(
        pack_table(std::slice::from_ref(&rounded), None).cone[0][3].to_bits(),
        (1.05 * rho_b).to_bits()
    );

    // The shader clamps the exponent to 32; the cone uses that same p.
    let steep = sample(FarShape::Rounded { exponent: 80.0 }, 1.0, 0.1);
    let p32 = 32.0f32;
    let rho_b32 = 0.1 * 3.0f32.powf(0.5 - 1.0 / p32) * (1.0 + 2.0e-4);
    assert!(1.05 * rho_b32 < 0.99);
    assert_eq!(
        pack_table(std::slice::from_ref(&steep), None).cone[0][3].to_bits(),
        (1.05 * rho_b32).to_bits()
    );

    let wall = sample(FarShape::InnerSphere, 2.0, 5.0);
    assert_eq!(
        pack_table(std::slice::from_ref(&wall), None).cone[0][3].to_bits(),
        (-1.0f32).to_bits()
    );

    let mut map_max = [0.0f32; MAX_FAR_MAPS];
    map_max[3] = 0.1;
    let mapped = sample(
        FarShape::Mapped {
            map: FarMapId(3),
            horizon: -0.25,
            air: 0.4,
        },
        4.0,
        1.0,
    );
    let table = super::pack_table(std::slice::from_ref(&mapped), None, &map_max);
    let hi = ((1.0f32 + 0.1) / 4.0).max(0.0);
    let shell = (0.4f32 / 4.0).max(0.0);
    assert!(hi + shell < 0.99);
    assert_eq!(table.cone[0][3].to_bits(), (hi + shell).to_bits());
    let gpu = &table.body[0];
    assert_eq!(gpu.atmosphere[3].to_bits(), 4.0f32.to_bits());
    assert_eq!(gpu.seed[2], 4);
    assert_eq!(gpu.seed[1], 0);
    assert_eq!(gpu.seed[3], 0);
    assert_eq!(gpu.albedo0[3].to_bits(), (-0.25f32).to_bits());
    assert_eq!(gpu.albedo1[3].to_bits(), 0.4f32.to_bits());
    assert_eq!(gpu.albedo2[3].to_bits(), 4.0f32.to_bits());
    assert_eq!(gpu.dir_rho[3].to_bits(), 0.25f32.to_bits());
    map_max[3] = 10.0;
    assert_eq!(
        super::pack_table(std::slice::from_ref(&mapped), None, &map_max).cone[0][3].to_bits(),
        (-1.0f32).to_bits()
    );

    assert_eq!(std::mem::offset_of!(FarTableGpu, cone), 16);
    assert_eq!(
        std::mem::offset_of!(FarTableGpu, body),
        16 + 16 * MAX_FAR_BODIES
    );
}

#[test]
fn mapped_cone_widens_by_the_air_shell() {
    let mut map_max = [0.0f32; MAX_FAR_MAPS];
    map_max[2] = 30.0;
    let distance = 1_000.0f32;
    let radius = 200.0f32;
    let air = 40.0f32;
    let hi = ((radius + map_max[2]) / distance).max(0.0);
    let shell = (air / distance).max(0.0);
    assert!(hi + shell < 0.99, "precondition {hi} {shell}");
    // 1080p at 60° is about a milliradian per pixel. This shell is many
    // times the shader's 3 px margin, so the margin cannot stand in for it.
    let px = 2.0 * (60.0f32.to_radians() * 0.5).tan() / 1080.0;
    assert!(
        shell > 3.0 * px,
        "shell {shell} must exceed the 3 px margin {px}"
    );

    let bare = sample(
        FarShape::Mapped {
            map: FarMapId(2),
            horizon: 1.0,
            air: 0.0,
        },
        distance,
        radius,
    );
    assert_eq!(
        super::pack_table(std::slice::from_ref(&bare), None, &map_max).cone[0][3].to_bits(),
        hi.to_bits(),
        "no air leaves the hi radius"
    );

    let thick = sample(
        FarShape::Mapped {
            map: FarMapId(2),
            horizon: 1.0,
            air,
        },
        distance,
        radius,
    );
    assert_eq!(
        super::pack_table(std::slice::from_ref(&thick), None, &map_max).cone[0][3].to_bits(),
        (hi + shell).to_bits()
    );

    // hi itself is inside the sine limit; the shell crosses it.
    let close_distance = 100.0f32;
    let close_radius = 50.0f32;
    let close_air = 25.0f32;
    let close_hi = ((close_radius + map_max[2]) / close_distance).max(0.0);
    let close_bound = close_hi + (close_air / close_distance).max(0.0);
    assert!(
        close_hi < 0.99 && close_bound >= 0.99,
        "{close_hi} {close_bound}"
    );
    let close = sample(
        FarShape::Mapped {
            map: FarMapId(2),
            horizon: 1.0,
            air: close_air,
        },
        close_distance,
        close_radius,
    );
    assert_eq!(
        super::pack_table(std::slice::from_ref(&close), None, &map_max).cone[0][3].to_bits(),
        (-1.0f32).to_bits()
    );
}
