/// Visual smoke test for the engine: a sine-hill field of colored cubes as a
/// retained mesh, an orbiting camera, immediate cubes/wires, and the 2D
/// overlay. Keys: F fullscreen, V vsync, M cycle MSAA, Esc quit.
use voxel_engine::{Camera3D, Color, Config, Key, MeshData, Vec3, Vertex};

fn push_cube(data: &mut MeshData, min: Vec3, max: Vec3, color: Color) {
    let c = [color.r, color.g, color.b, color.a];
    // Same CCW-from-outside winding the engine uses for immediate cubes.
    let faces: [[[f32; 3]; 4]; 6] = [
        [[min.x, max.y, min.z], [min.x, max.y, max.z], [max.x, max.y, max.z], [max.x, max.y, min.z]],
        [[min.x, min.y, min.z], [max.x, min.y, min.z], [max.x, min.y, max.z], [min.x, min.y, max.z]],
        [[max.x, min.y, min.z], [max.x, max.y, min.z], [max.x, max.y, max.z], [max.x, min.y, max.z]],
        [[min.x, min.y, min.z], [min.x, min.y, max.z], [min.x, max.y, max.z], [min.x, max.y, min.z]],
        [[min.x, min.y, max.z], [max.x, min.y, max.z], [max.x, max.y, max.z], [min.x, max.y, max.z]],
        [[min.x, min.y, min.z], [min.x, max.y, min.z], [max.x, max.y, min.z], [max.x, min.y, min.z]],
    ];
    for face in faces {
        let base = data.vertices.len() as u32;
        for corner in face {
            data.vertices.push(Vertex { pos: corner, color: c });
        }
        data.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn main() {
    env_logger::init();

    let mut terrain = None;
    let mut angle = 0.0f32;

    voxel_engine::run(
        Config {
            title: "voxel_engine demo".into(),
            target_fps: 0,
            ..Config::default()
        },
        move |eng| {
            if eng.should_close() || eng.is_key_pressed(Key::Escape) {
                return false;
            }
            if eng.is_key_pressed(Key::F) {
                let now = !eng.fullscreen();
                eng.set_fullscreen(now);
            }
            if eng.is_key_pressed(Key::V) {
                let now = !eng.vsync();
                eng.set_vsync(now);
            }
            if eng.is_key_pressed(Key::M) {
                let next = if eng.msaa() >= eng.max_msaa() { 1 } else { eng.msaa() * 2 };
                eng.set_msaa(next);
            }

            if terrain.is_none() {
                let mut data = MeshData::default();
                for x in -24i32..24 {
                    for z in -24i32..24 {
                        let h = ((x as f32 * 0.35).sin() + (z as f32 * 0.3).cos()) * 3.0;
                        let h = h.round();
                        let shade = (140.0 + h * 12.0).clamp(60.0, 235.0) as u8;
                        let color = if (x + z) % 2 == 0 {
                            Color::new(40, shade, 60, 255)
                        } else {
                            Color::new(50, shade, 80, 255)
                        };
                        push_cube(
                            &mut data,
                            Vec3::new(x as f32, h - 1.0, z as f32),
                            Vec3::new(x as f32 + 1.0, h, z as f32 + 1.0),
                            color,
                        );
                    }
                }
                terrain = eng.upload_mesh(&data);
            }

            angle += eng.frame_time() * 0.4;
            let cam = Camera3D {
                position: Vec3::new(angle.cos() * 34.0, 20.0, angle.sin() * 34.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                up: Vec3::Y,
                fovy: 70.0,
            };

            let vsync = eng.vsync();
            let msaa = eng.msaa();
            let fullscreen = eng.fullscreen();

            let mut frame = eng.begin_frame(Color::SKYBLUE);
            {
                let mut f3 = frame.begin_3d(&cam);
                if let Some(handle) = terrain {
                    f3.draw_mesh(handle);
                }
                f3.draw_cube(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(2.0), Color::RED);
                f3.draw_cube_wires(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(2.2), Color::BLACK);
            }
            frame.draw_rect(8, 8, 340, 76, Color::new(0, 0, 0, 150));
            frame.draw_fps(16, 14);
            frame.draw_text(
                &format!("vsync {vsync} msaa {msaa}x fullscreen {fullscreen}"),
                16,
                38,
                16,
                Color::RAYWHITE,
            );
            frame.draw_text("F fullscreen  V vsync  M msaa  Esc quit", 16, 60, 16, Color::GRAY);
            true
        },
    );
}
