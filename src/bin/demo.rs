/// Visual smoke test for the packed-vertex world-mesh path. Builds ONE
/// chunk-local mesh via the typed `MeshVertex` API (integer coords ≤16³, a
/// `Normal` per face, a texture layer) and draws it as a 2×2 grid of chunks,
/// each 16 units apart via a per-draw offset — so the tiles only line up
/// seamlessly if every draw reads its own offsets-SSBO slot. Plus a flat floor
/// quad spanning the full 16 units to exercise UV-from-position + REPEAT
/// tiling, one darkened region to prove the AO/light bits modulate output, an
/// orbiting camera, immediate debug cube/wires, and the 2D overlay.
/// Keys: F fullscreen, V vsync, M cycle MSAA, Esc quit.
use voxel_engine::{
    Ao, Camera3D, Color, Config, Key, Light, MeshData, MeshHandle, MeshVertex, Normal, Pass,
    SkyDesc, Vec3,
};

const CHUNK: u8 = 16;
/// Texture layer sampled by the demo (layer 0 is the white layer).
const CHECKER_LAYER: u8 = 1;
/// Translucent water layer (alpha < 1), used by the transparent-pass mesh.
const WATER_LAYER: u8 = 2;
/// World height of the water plane. Deliberately fractional: the transparent
/// pass tests depth but doesn't depth-sort (v1), so a water plane coplanar with
/// the integer block tops would z-fight. Sitting it between integer heights
/// keeps it strictly in front of the terrain it covers.
const WATER_LEVEL: f32 = 4.5;

/// A quad, corners wound CCW seen from outside, with neutral (full-bright) AO/light.
fn push_quad(data: &mut MeshData, corners: [[u8; 3]; 4], normal: Normal, layer: u8) {
    push_quad_lit(data, corners, normal, layer, Ao::NONE, Light::FULL);
}

fn push_quad_lit(
    data: &mut MeshData,
    corners: [[u8; 3]; 4],
    normal: Normal,
    layer: u8,
    ao: Ao,
    light: Light,
) {
    data.quad(corners.map(|c| MeshVertex::new(c, normal, layer, ao, light)));
}

/// A unit cube whose top sits at `y` (so it occupies `y-1..y`), all 6 faces.
/// `lit` optionally overrides AO/light to prove the word-1 bits modulate.
fn push_cube(data: &mut MeshData, x: u8, y: u8, z: u8, layer: u8, lit: Option<(Ao, Light)>) {
    let (x1, y1, z1) = (x + 1, y, z + 1);
    let y0 = y - 1;
    // 8 corners.
    let faces: [([[u8; 3]; 4], Normal); 6] = [
        // +Y top
        (
            [[x, y1, z], [x, y1, z1], [x1, y1, z1], [x1, y1, z]],
            Normal::PosY,
        ),
        // -Y bottom
        (
            [[x, y0, z], [x1, y0, z], [x1, y0, z1], [x, y0, z1]],
            Normal::NegY,
        ),
        // +X
        (
            [[x1, y0, z], [x1, y1, z], [x1, y1, z1], [x1, y0, z1]],
            Normal::PosX,
        ),
        // -X
        (
            [[x, y0, z], [x, y0, z1], [x, y1, z1], [x, y1, z]],
            Normal::NegX,
        ),
        // +Z
        (
            [[x, y0, z1], [x1, y0, z1], [x1, y1, z1], [x, y1, z1]],
            Normal::PosZ,
        ),
        // -Z
        (
            [[x, y0, z], [x, y1, z], [x1, y1, z], [x1, y0, z]],
            Normal::NegZ,
        ),
    ];
    for (corners, normal) in faces {
        match lit {
            Some((ao, light)) => push_quad_lit(data, corners, normal, layer, ao, light),
            None => push_quad(data, corners, normal, layer),
        }
    }
}

/// Builds one 16×16 chunk: a full-span floor quad (UV tiling proof) plus a
/// sine-hill of 1-thick tiles. The near-origin quadrant is darkened.
fn build_chunk() -> MeshData {
    let mut data = MeshData::new(Pass::Opaque);
    // Floor spanning the whole chunk — uv runs 0..16, tiling the checker.
    push_quad(
        &mut data,
        [[0, 0, 0], [0, 0, CHUNK], [CHUNK, 0, CHUNK], [CHUNK, 0, 0]],
        Normal::PosY,
        CHECKER_LAYER,
    );
    for x in 0..CHUNK {
        for z in 0..CHUNK {
            let h = ((x as f32 * 0.6).sin() + (z as f32 * 0.5).cos()) * 2.0;
            let y = (h.round() as i32 + 4).clamp(1, 8) as u8;
            // Darken one quadrant to prove AO/light modulation.
            let lit = (x < CHUNK / 2 && z < CHUNK / 2).then_some((Ao::new(0), Light::new(5, 5)));
            push_cube(&mut data, x, y, z, CHECKER_LAYER, lit);
        }
    }
    data
}

/// A flat translucent water plane (transparent pass), meshed at local `y=0` and
/// lifted to `WATER_LEVEL` by the per-draw offset. Drawn after all opaque
/// geometry so it blends over the terrain.
fn build_water() -> MeshData {
    let mut data = MeshData::new(Pass::Transparent);
    push_quad(
        &mut data,
        [[0, 0, 0], [0, 0, CHUNK], [CHUNK, 0, CHUNK], [CHUNK, 0, 0]],
        Normal::PosY,
        WATER_LAYER,
    );
    data
}

/// Layer 0: all white (engine contract). Layer 1: a 4×4-cell two-tone checker.
/// Layer 2: a semi-transparent blue for the water plane (alpha < 255).
fn block_texture_layers(size: u32) -> Vec<Vec<u8>> {
    let n = (size * size * 4) as usize;
    let white = vec![255u8; n];
    let mut checker = Vec::with_capacity(n);
    for y in 0..size {
        for x in 0..size {
            let v = if ((x / 4) + (y / 4)) % 2 == 0 {
                255
            } else {
                150
            };
            checker.extend_from_slice(&[v, v, v, 255]);
        }
    }
    let mut water = Vec::with_capacity(n);
    for _ in 0..(size * size) {
        water.extend_from_slice(&[40, 90, 200, 120]);
    }
    vec![white, checker, water]
}

fn main() {
    env_logger::init();

    let mut chunk: Option<MeshHandle> = None;
    let mut water: Option<MeshHandle> = None;
    let mut angle = 0.0f32;

    voxel_engine::run(
        Config {
            title: "voxel_engine demo".into(),
            target_fps: 0,
            vsync: false,
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
                let next = if eng.msaa() >= eng.max_msaa() {
                    1
                } else {
                    eng.msaa() * 2
                };
                eng.set_msaa(next);
            }
            if eng.is_key_pressed(Key::C) {
                let now = !eng.cull_faces();
                eng.set_cull_faces(now);
            }

            if chunk.is_none() {
                eng.set_block_textures(16, &block_texture_layers(16));
                chunk = Some(eng.upload_mesh(&build_chunk()).expect("chunk upload"));
                water = Some(eng.upload_mesh(&build_water()).expect("water upload"));
            }

            angle += eng.frame_time() * 0.4;
            let center = Vec3::new(CHUNK as f32, 4.0, CHUNK as f32);
            let cam = Camera3D {
                position: center + Vec3::new(angle.cos() * 44.0, 30.0, angle.sin() * 44.0),
                target: center + Vec3::new(0.0, 8.0, 0.0),
                up: Vec3::Y,
                fovy: 70.0,
            };

            let vsync = eng.vsync();
            let msaa = eng.msaa();
            let fullscreen = eng.fullscreen();
            let cull = eng.cull_faces();

            let mut frame = eng.begin_frame(Color::SKYBLUE);
            {
                let mut f3 = frame.begin_3d(&cam);
                // Procedural sky background: an afternoon palette with the sun
                // low in the west so the horizon glow and disc are both visible.
                f3.set_sky(SkyDesc {
                    sun_dir: Vec3::new(0.6, 0.35, 0.2).normalize(),
                    zenith: Color::rgb(71, 128, 224),
                    horizon: Color::rgb(168, 204, 240),
                    sun_tint: Color::rgb(240, 150, 70),
                    exposure: 1.0,
                    sun_angular_radius: 0.03,
                });
                if let Some(handle) = chunk {
                    // 2×2 grid, each chunk offset by 16 — seamless only if each
                    // draw reads its own per-draw offset slot.
                    for gx in 0..2 {
                        for gz in 0..2 {
                            let off = Vec3::new(
                                (gx * CHUNK as i32) as f32,
                                0.0,
                                (gz * CHUNK as i32) as f32,
                            );
                            f3.draw_mesh(handle, off, 1.0);
                        }
                    }
                    // Same mesh at scale 2 beside the grid: double size, seamless
                    // frustum cull — proves DrawOffset.scale threads shader + cull.
                    f3.draw_mesh(handle, Vec3::new(-2.0 * CHUNK as f32, 0.0, 0.0), 2.0);
                }
                // Translucent water over each grid chunk, lifted to WATER_LEVEL
                // (fractional, so it never coplanar-z-fights the block tops),
                // drawn after all opaque.
                if let Some(handle) = water {
                    for gx in 0..2 {
                        for gz in 0..2 {
                            let off = Vec3::new(
                                (gx * CHUNK as i32) as f32,
                                WATER_LEVEL,
                                (gz * CHUNK as i32) as f32,
                            );
                            f3.draw_mesh(handle, off, 1.0);
                        }
                    }
                }
                f3.draw_cube(
                    center + Vec3::new(0.0, 10.0, 0.0),
                    Vec3::splat(2.0),
                    Color::RED,
                );
                f3.draw_cube_wires(
                    center + Vec3::new(0.0, 10.0, 0.0),
                    Vec3::splat(2.2),
                    Color::BLACK,
                );
            }
            frame.draw_rect(8, 8, 360, 76, Color::new(0, 0, 0, 150));
            frame.draw_fps(16, 14);
            frame.draw_text(
                &format!("vsync {vsync} msaa {msaa}x fullscreen {fullscreen} cull {cull}"),
                16,
                38,
                16,
                Color::RAYWHITE,
            );
            frame.draw_text(
                "F fullscreen  V vsync  M msaa  C cull  Esc quit",
                16,
                60,
                16,
                Color::GRAY,
            );
            true
        },
    );
}
