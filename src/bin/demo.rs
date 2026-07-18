/// Smoke test: packed-vertex mesh via typed API, 2×2 grid with offset-per-draw,
/// flat floor, orbiting camera, debug overlay. Keys: F fullscreen, V vsync, M MSAA, Esc quit.
use voxel_engine::{
    Ao, Camera3D, Color, Config, Detail, Key, Light, MeshData, MeshVertex, Normal, Pass, SkyDesc, Vec3,
};

const CHUNK: u8 = 16;
/// Texture layer sampled by the demo (layer 0 is the white layer).
const CHECKER_LAYER: u16 = 1;
/// Translucent water layer (alpha < 1), used by the transparent-pass mesh.
const WATER_LAYER: u16 = 2;
/// World height of the water plane. Deliberately fractional: the transparent
/// pass tests depth but doesn't depth-sort (v1), so a water plane coplanar with
/// the integer block tops would z-fight. Sitting it between integer heights
/// keeps it strictly in front of the terrain it covers.
const WATER_LEVEL: f32 = 4.5;

/// Quad with CCW winding; neutral daylight (no baked light/AO).
fn push_quad(data: &mut MeshData, corners: [[u8; 3]; 4], normal: Normal, layer: u16) {
    data.quad(corners.map(|c| MeshVertex::new(c, normal, layer, Ao::NONE, Light::DAY, false)));
}

/// A unit cube whose top sits at `y` (so it occupies `y-1..y`), all 6 faces.
fn push_cube(data: &mut MeshData, x: u8, y: u8, z: u8, layer: u16) {
    let (x1, y1, z1) = (x + 1, y, z + 1);
    let y0 = y - 1;
    let faces: [([[u8; 3]; 4], Normal); 6] = [
        (
            [[x, y1, z], [x, y1, z1], [x1, y1, z1], [x1, y1, z]],
            Normal::PosY,
        ),
        (
            [[x, y0, z], [x1, y0, z], [x1, y0, z1], [x, y0, z1]],
            Normal::NegY,
        ),
        (
            [[x1, y0, z], [x1, y1, z], [x1, y1, z1], [x1, y0, z1]],
            Normal::PosX,
        ),
        (
            [[x, y0, z], [x, y0, z1], [x, y1, z1], [x, y1, z]],
            Normal::NegX,
        ),
        (
            [[x, y0, z1], [x1, y0, z1], [x1, y1, z1], [x, y1, z1]],
            Normal::PosZ,
        ),
        (
            [[x, y0, z], [x, y1, z], [x1, y1, z], [x1, y0, z]],
            Normal::NegZ,
        ),
    ];
    for (corners, normal) in faces {
        push_quad(data, corners, normal, layer);
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
            push_cube(&mut data, x, y, z, CHECKER_LAYER);
        }
    }
    data
}

/// A flat translucent water plane (transparent pass), meshed at local `y=0` and
/// lifted to `WATER_LEVEL` by the per-draw offset. Drawn after all opaque
/// geometry so it blends over the terrain.
fn build_water() -> MeshData {
    let mut data = MeshData::new(Pass::Blend);
    push_quad(
        &mut data,
        [[0, 0, 0], [0, 0, CHUNK], [CHUNK, 0, CHUNK], [CHUNK, 0, 0]],
        Normal::PosY,
        WATER_LAYER,
    );
    data
}

/// Layer 0: all white. Layer 1: a 4×4-cell two-tone checker.
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

/// Sun direction shared by the sky disc ([`SkyDesc`]) and the lighting UBO, so
/// the disc and the terrain shading agree.
const SUN_DIR: Vec3 = Vec3::new(0.6, 0.35, 0.2);

/// A daytime lighting block for the demo, passed as `Lighting::Composed` to
/// `begin_3d`. The mesh/sky/water shaders read ALL their lighting from this
/// per-frame UBO; this drives the real lit path — sun-tinted skylight, a
/// blue-gradient sky, a modest ambient floor, no fog. (`Lighting::FullBright`
/// would instead give a flat lit neutral.)
fn daytime_uniforms() -> voxel_engine::skeleton::FrameUniformsGpu {
    let sun = SUN_DIR.normalize();
    voxel_engine::skeleton::FrameUniformsGpu {
        sun_dir_elev: [sun.x, sun.y, sun.z, sun.y.asin()],
        // Bright warm sun (linear), day_night_mix = 1.0 (full day).
        light: [1.25, 1.15, 1.0, 1.0],
        // Sky anchors (linear): deep blue zenith, pale horizon; w = turbidity.
        zenith: [0.09, 0.22, 0.45, 2.0],
        horizon: [0.55, 0.65, 0.80, 0.001], // w = fog density
        // No blocklight; ambient floor keeps shadowed faces off pure black.
        candle: [0.0, 0.0, 0.0, 0.30],
        exposure_dither: [1.0, 0.0, 0.0, 0.0],
        extras: [1.0, 0.0, 0.0, 0.0], // x = stars gain (gated by RenderFlags::stars)
        // Static demo scene: no animated water/clouds, so time/camera phase stay zero.
        anim: [0.0; 4],
    }
}

fn main() {
    env_logger::init();

    // Upload meshes once; GPU records drive draws while resident+visible.
    // `big` exercises upload_mesh (Tracked) with set_mesh_placement.
    let mut uploaded = false;
    let mut angle = 0.0f32;
    // High-FOV cylindrical warp strength, cycled with G. Seeded from VOXEL_WARP
    // so headless screenshot runs can pick a value without a keypress.
    let mut warp_ratio: f32 = std::env::var("VOXEL_WARP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    // Vertical FOV in degrees, zoomed with the scroll wheel and clamped to a
    // usable range (telephoto .. wide) so you can't invert or flatten the lens.
    let mut fovy: f32 = 70.0;
    // VOXEL_AUTOSHOT=1: grab one screenshot after warm-up, then quit — used to
    // verify the warp offscreen. Off by default (interactive run).
    let autoshot = std::env::var("VOXEL_AUTOSHOT").is_ok();
    let mut frame_n: u32 = 0;

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
            // Headless verification: request a capture of the settled scene
            // (queued now, written when this frame submits), then quit shortly
            // after. `eng` is unborrowed here, before `begin_frame`.
            frame_n += 1;
            if autoshot {
                if frame_n == 30 {
                    if let Some(p) = eng.screenshot() {
                        log::info!("autoshot -> {}", p.display());
                    }
                }
                if frame_n >= 36 {
                    return false;
                }
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
            if eng.is_key_pressed(Key::G) {
                // Cycle 0 → 0.5 → 1.0 → 0 …
                warp_ratio = match warp_ratio {
                    r if r < 0.25 => 0.5,
                    r if r < 0.75 => 1.0,
                    _ => 0.0,
                };
            }

            if !uploaded {
                uploaded = true;
                eng.set_block_textures(16, &block_texture_layers(16));
                let chunk_data = build_chunk();
                let water_data = build_water();
                for gx in 0..2i32 {
                    for gz in 0..2i32 {
                        let block =
                            voxel_engine::IVec3::new(gx * CHUNK as i32, 0, gz * CHUNK as i32);
                        let placed = voxel_engine::MeshPlacement::terrain(block, Detail::FULL);
                        eng.upload_mesh_placed(&chunk_data, placed)
                            .expect("chunk upload");
                        // Water at fractional WATER_LEVEL via local_off.
                        let wet = voxel_engine::MeshPlacement {
                            block,
                            local_off: Vec3::new(0.0, WATER_LEVEL, 0.0),
                            detail: Detail::FULL,
                        };
                        eng.upload_mesh_placed(&water_data, wet)
                            .expect("water upload");
                    }
                }
                // Scale-2 mesh tests LOD threading and placement patching.
                let handle = eng.upload_mesh(&chunk_data).expect("big chunk upload");
                eng.set_mesh_placement(
                    handle,
                    voxel_engine::MeshPlacement::terrain(
                        voxel_engine::IVec3::new(-2 * CHUNK as i32, 0, 0),
                        Detail::new(1),
                    ),
                );
            }

            // Scroll to zoom: each notch nudges the vertical FOV, scrolling up
            // (positive) narrows toward telephoto. Clamped so the lens stays sane.
            const FOVY_MIN: f32 = 20.0;
            const FOVY_MAX: f32 = 160.0;
            fovy = (fovy - eng.mouse_wheel() * 4.0).clamp(FOVY_MIN, FOVY_MAX);

            angle += eng.frame_time() * 0.4;
            let center = Vec3::new(CHUNK as f32, 4.0, CHUNK as f32);
            let cam = Camera3D {
                position: center + Vec3::new(angle.cos() * 44.0, 30.0, angle.sin() * 44.0),
                target: center + Vec3::new(0.0, 8.0, 0.0),
                up: Vec3::Y,
                fovy,
                lens: voxel_engine::WarpStrength::new(warp_ratio)
                    .map_or(voxel_engine::Lens::Rectilinear, |strength| {
                        voxel_engine::Lens::WideFov { strength }
                    }),
            };

            let vsync = eng.vsync();
            let msaa = eng.msaa();
            let fullscreen = eng.fullscreen();
            let cull = eng.cull_faces();
            let fps = eng.fps();

            let mut frame = eng.begin_frame(Color::SKYBLUE.to_linear());
            {
                // Demo draws in absolute coordinates (no camera rebase): the
                // render-space origin is the world origin, so eye = ZERO — the
                // camera's own translation already lives in the view matrix.
                let mut f3 = frame.begin_3d(
                    &cam,
                    voxel_engine::DVec3::ZERO,
                    voxel_engine::Lighting::Composed(daytime_uniforms()),
                );
                // Procedural sky background: sun low in the west so the disc is
                // visible. The gradient/glow colours come from the
                // per-frame UBO (unset here → neutral); this smoke test only
                // exercises the sun geometry + disc tint (approx. warm linear).
                f3.set_sky(SkyDesc {
                    sun_dir: SUN_DIR.normalize(),
                    sun_tint: voxel_engine::LinearRgb([0.87, 0.30, 0.06]),
                    sun_angular_radius: 0.03,
                });
                // All meshes draw automatically from persistent records.
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
            frame.draw_text(&format!("{fps} FPS"), 16, 14, 20, Color::LIME);
            frame.draw_text(
                &format!("vsync {vsync} msaa {msaa}x fullscreen {fullscreen} cull {cull}"),
                16,
                38,
                16,
                Color::RAYWHITE,
            );
            frame.draw_text(
                &format!("F fullscreen  V vsync  M msaa  C cull  G warp {warp_ratio:.1}  Esc quit"),
                16,
                60,
                16,
                Color::GRAY,
            );
            true
        },
    );
}
