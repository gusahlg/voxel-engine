/// Smoke test: packed-vertex mesh via typed API, 2×2 grid with offset-per-draw,
/// flat floor, orbiting camera, debug overlay. Keys: F fullscreen, V vsync,
/// A MSAA, M materials (array vs procedural), Esc quit.
/// `VOXEL_DEMO_FAR=1` pins that startup view and draws one of each far shape.
/// `VOXEL_DEMO_RECREATE_CYCLE=1` after ~2s steps vsync, MSAA, render scale, and
/// fullscreen (every ~60 frames) and quits, so a validation run hits every
/// recreate plan.
/// `VOXEL_DEMO_VRS=1` turns on `RenderFlags::vrs`, `VOXEL_DEMO_MSAA=<n>` starts
/// at that sample count, and `VOXEL_DEMO_FULLSCREEN=1` starts fullscreen, so a
/// validation run can cover VRS + MSAA at the output's full resolution.
/// `VOXEL_DEMO_UPLOAD_CYCLE=1` re-uploads far map 0, block textures and the
/// material table every ~60 frames (or only the parts in a comma list of
/// `far`, `set`, `append`, `mat`), so a validation run hits the overwrite,
/// append and grow upload paths.
use voxel_engine::{
    Ao, Camera3D, Color, Config, Detail, FarBody, FarMapDesc, FarMapId, FarShape, Key, Light,
    LinearRgb, MATERIAL_FLAG_PROCEDURAL, MaterialDesc, MeshData, MeshVertex, Normal, Pass, Quat,
    SkyDesc, Vec3, far_cube_texel_dir, far_map_basis,
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

fn demo_array_materials() -> [MaterialDesc; 3] {
    [MaterialDesc::ARRAY_LAYER; 3]
}

fn demo_procedural_materials() -> [MaterialDesc; 3] {
    [
        // Layer 0: immediate cubes stay white (rgb == rgb2).
        MaterialDesc {
            rgb: [255, 255, 255],
            rgb2: [255, 255, 255],
            flags: MATERIAL_FLAG_PROCEDURAL,
            ..MaterialDesc::ARRAY_LAYER
        },
        // Layer 1: two-tone grey matching the checker texels, with grain.
        MaterialDesc {
            rgb: [255, 255, 255],
            rgb2: [150, 150, 150],
            frequency: 48,
            roughness: 160,
            flags: MATERIAL_FLAG_PROCEDURAL,
            ..MaterialDesc::ARRAY_LAYER
        },
        // Layer 2: water — same sRGB/alpha as the uploaded texture layer.
        MaterialDesc {
            rgb: [40, 90, 200],
            rgb2: [20, 50, 140],
            frequency: 24,
            roughness: 96,
            alpha: 120,
            flags: MATERIAL_FLAG_PROCEDURAL,
            ..MaterialDesc::ARRAY_LAYER
        },
    ]
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

/// Vertical FOV the window opens at, in degrees. The far showcase is framed
/// for this lens and the default 1280×720 window.
const DEMO_START_FOVY: f32 = 70.0;

/// Shared distance for the showcase. `rho = radius / distance`.
const FAR_DEMO_DISTANCE: f32 = 1_000.0;
const FAR_DATUM_RES: u32 = 33;
const FAR_ALBEDO_SIZE: u32 = 64;
/// Reference radius fraction for the mapped planet (`rho`).
const FAR_MAPPED_RHO: f32 = 0.12;

/// Next non-colliding `screenshots/watt-<utc-secs>.png`. Autoshot cannot use
/// [`Engine::screenshot`](voxel_engine::Engine::screenshot): that encode is
/// detached, and quitting the process drops it before the rename.
fn autoshot_path() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from("screenshots");
    let _ = std::fs::create_dir_all(&dir);
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut path = dir.join(format!("watt-{secs}.png"));
    let mut n = 1u32;
    while path.exists() {
        path = dir.join(format!("watt-{secs}-{n}.png"));
        n += 1;
    }
    path
}

/// `VOXEL_DEMO_MAPPED_RHO` when it parses as a finite value in `(0, 1)`.
fn demo_mapped_override() -> Option<f32> {
    let text = std::env::var("VOXEL_DEMO_MAPPED_RHO").ok()?;
    let rho = text.parse::<f32>().ok()?;
    (rho.is_finite() && rho > 0.0 && rho < 1.0).then_some(rho)
}

/// Showcase planet rho. Unset (or an unusable override) keeps [`FAR_MAPPED_RHO`].
fn demo_mapped_rho() -> f32 {
    demo_mapped_override().unwrap_or(FAR_MAPPED_RHO)
}

/// Horizon sine for the showcase planet. An override uses the hi-sphere limb
/// (`-sqrt(1 - rho_hi²)`, rho_hi = rho × 1.12) so interior tiles exist. The
/// datum's largest bump is `0.12 * radius`. Unset keeps 1, which disables the
/// cull and is what the showcase tests lock.
fn demo_mapped_horizon(rho: f32) -> f32 {
    if demo_mapped_override().is_none() {
        return 1.0;
    }
    let rho_hi = (rho * 1.12).clamp(0.0, 0.999);
    -(1.0 - rho_hi * rho_hi).sqrt()
}

/// NDC on the startup frame. `y > 0` is the upper half of the view (the
/// projection is GL-style y-up). Order: sphere, cube, rounded, mapped, point.
/// Staggered in y so the mapped bulge and the cube's corners stay apart.
const FAR_LAYOUT_NDC: [(f32, f32); 5] = [
    (-0.72, 0.62),
    (-0.36, 0.74),
    (0.02, 0.62),
    (0.44, 0.70),
    (0.82, 0.62),
];

/// sRGB base colour of each albedo face: +X, −X, +Y, −Y, +Z, −Z.
const FAR_FACE_SRGB: [[u8; 3]; 6] = [
    [230, 48, 48],
    [40, 210, 220],
    [48, 210, 72],
    [220, 48, 210],
    [64, 96, 235],
    [235, 200, 48],
];

/// Same orbit `main` draws. `angle` 0 is the startup pose: eye at
/// focus + (44, 30, 0), looking toward focus + (0, 8, 0).
fn demo_camera(angle: f32, fovy: f32, warp_ratio: f32) -> Camera3D {
    let center = Vec3::new(CHUNK as f32, 4.0, CHUNK as f32);
    Camera3D {
        position: center + Vec3::new(angle.cos() * 44.0, 30.0, angle.sin() * 44.0),
        target: center + Vec3::new(0.0, 8.0, 0.0),
        up: Vec3::Y,
        fovy,
        lens: voxel_engine::WarpStrength::new(warp_ratio)
            .map_or(voxel_engine::Lens::Rectilinear, |strength| {
                voxel_engine::Lens::WideFov { strength }
            }),
    }
}

fn demo_aspect() -> f32 {
    let cfg = Config::default();
    cfg.width as f32 / cfg.height as f32
}

/// `(right, up, forward)` of [`Camera3D::view`] at startup. `look_at_rh`
/// uses right = forward × up.
fn startup_camera_basis() -> (Vec3, Vec3, Vec3) {
    let cam = demo_camera(0.0, DEMO_START_FOVY, 0.0);
    let forward = (cam.target - cam.position).normalize();
    let right = forward.cross(cam.up).normalize();
    let up = right.cross(forward);
    (right, up, forward)
}

/// World direction of an NDC point on the startup view. `ndc_y > 0` is up.
fn startup_view_dir(ndc_x: f32, ndc_y: f32) -> Vec3 {
    let (right, up, forward) = startup_camera_basis();
    let tan_half = (DEMO_START_FOVY.to_radians() * 0.5).tan();
    let x = ndc_x * tan_half * demo_aspect();
    let y = ndc_y * tan_half;
    (right * x + up * y + forward).normalize()
}

/// Body-space direction of datum sample `(i, j)` on `face`.
///
/// [`FarMapDesc`] stores an equiangular chart: `xi = (4/π) atan(dot(d, tu) /
/// dot(d, n))` and eta the same with `tv`, with sample `i` at
/// `xi = 2i/(g-1) - 1`. Inverting that is `d ∥ n + tu tan(xi·π/4) + tv tan(eta·π/4)`.
fn far_chart_dir(face: usize, g: u32, i: u32, j: u32) -> Vec3 {
    let (tu, n, tv) = far_map_basis(face);
    let edge = (g - 1) as f32;
    let xi = 2.0 * i as f32 / edge - 1.0;
    let eta = 2.0 * j as f32 / edge - 1.0;
    let quarter = std::f32::consts::FRAC_PI_4;
    (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize()
}

/// `0.08 * radius * (sin(3x) cos(2y) + 0.5 sin(5z))` on the chart directions.
fn far_showcase_datum(radius: f32) -> Vec<f32> {
    let g = FAR_DATUM_RES;
    let mut datum = Vec::with_capacity(6 * g as usize * g as usize);
    for face in 0..6 {
        for j in 0..g {
            for i in 0..g {
                let d = far_chart_dir(face, g, i, j);
                let bump = (3.0 * d.x).sin() * (2.0 * d.y).cos() + 0.5 * (5.0 * d.z).sin();
                datum.push(0.08 * radius * bump);
            }
        }
    }
    datum
}

/// 1 on the open face, ~0.28 on a latitude or longitude line.
fn lat_lon_grid(dir: Vec3) -> f32 {
    let lat01 = dir.y.clamp(-1.0, 1.0).asin() / std::f32::consts::PI + 0.5;
    let lon01 = dir.z.atan2(dir.x).rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
    let on_line = |t: f32, cells: f32| {
        let f = (t * cells).rem_euclid(1.0);
        f < 0.10 || f > 0.90
    };
    if on_line(lat01, 6.0) || on_line(lon01, 12.0) {
        0.28
    } else {
        1.0
    }
}

/// 64×64 RGBA8 sRGB. Base colour of the face times a lat/lon grid, so a seam
/// or a swapped face breaks the graticule.
fn far_showcase_face(face: usize) -> Vec<u8> {
    let base = FAR_FACE_SRGB[face];
    let size = FAR_ALBEDO_SIZE;
    let mut bytes = vec![0u8; size as usize * size as usize * 4];
    for y in 0..size {
        for x in 0..size {
            let scale = lat_lon_grid(far_cube_texel_dir(face, size, x, y));
            let i = (y * size + x) as usize * 4;
            for c in 0..3 {
                bytes[i + c] = (base[c] as f32 * scale).round().clamp(0.0, 255.0) as u8;
            }
            bytes[i + 3] = 255;
        }
    }
    bytes
}

fn far_face_linear() -> [LinearRgb; 6] {
    FAR_FACE_SRGB.map(|c| Color::rgb(c[0], c[1], c[2]).to_linear())
}

/// 30° about X, then 30° about Y. Body space into world.
fn far_cube_rotation() -> Quat {
    let a = 30.0_f32.to_radians();
    Quat::from_rotation_y(a) * Quat::from_rotation_x(a)
}

fn far_showcase_bodies() -> [FarBody; 5] {
    let distance = FAR_DEMO_DISTANCE;
    let at = |slot: usize,
              rho: f32,
              shape: FarShape,
              rotation: Quat,
              albedo: [LinearRgb; 6],
              atmosphere: LinearRgb,
              seed: u32| FarBody {
        dir: startup_view_dir(FAR_LAYOUT_NDC[slot].0, FAR_LAYOUT_NDC[slot].1),
        distance,
        radius: rho * distance,
        shape,
        rotation,
        albedo,
        atmosphere,
        seed,
    };
    let mut sphere_alb = [Color::rgb(255, 210, 90).to_linear(); 6];
    sphere_alb[1] = Color::rgb(70, 200, 225).to_linear();
    let cube_alb = [
        Color::WHITE,
        Color::RED,
        Color::GREEN,
        Color::ORANGE,
        Color::SKYBLUE,
        Color::MAGENTA,
    ]
    .map(Color::to_linear);
    let rounded_alb = [
        Color::SALMON,
        Color::PURPLE,
        Color::PINK,
        Color::RAYWHITE,
        Color::YELLOW,
        Color::GOLD,
    ]
    .map(Color::to_linear);
    let mapped_rho = demo_mapped_rho();
    let mapped_r = mapped_rho * distance;
    let hot = LinearRgb([3.2, 2.8, 1.2]);
    [
        at(
            0,
            0.08,
            FarShape::Sphere,
            Quat::IDENTITY,
            sphere_alb,
            LinearRgb([1.05, 0.48, 0.16]),
            1,
        ),
        at(
            1,
            0.05,
            FarShape::Cube,
            far_cube_rotation(),
            cube_alb,
            LinearRgb([0.95, 0.40, 0.85]),
            2,
        ),
        at(
            2,
            0.06,
            FarShape::Rounded { exponent: 4.0 },
            Quat::IDENTITY,
            rounded_alb,
            LinearRgb([0.35, 0.95, 0.55]),
            3,
        ),
        at(
            3,
            mapped_rho,
            FarShape::Mapped {
                map: FarMapId(0),
                horizon: demo_mapped_horizon(mapped_rho),
                air: 0.02 * mapped_r,
            },
            Quat::IDENTITY,
            far_face_linear(),
            LinearRgb([0.45, 0.72, 1.35]),
            4,
        ),
        at(
            4,
            1.0e-4,
            FarShape::Sphere,
            Quat::IDENTITY,
            [hot; 6],
            LinearRgb([1.2, 1.0, 0.7]),
            5,
        ),
    ]
}

/// Datum and six albedo faces for map 0. Call once, before the first frame
/// that submits far bodies; the upload lands over the following frames.
fn install_far_showcase(eng: &mut voxel_engine::Engine) {
    let radius = demo_mapped_rho() * FAR_DEMO_DISTANCE;
    let datum = far_showcase_datum(radius);
    eng.set_far_map(
        FarMapId(0),
        &FarMapDesc {
            datum_res: FAR_DATUM_RES,
            datum: &datum,
            albedo_size: FAR_ALBEDO_SIZE,
        },
    )
    .expect("far map");
    for face in 0..6 {
        let rgba = far_showcase_face(face);
        eng.set_far_map_face(FarMapId(0), face, &rgba)
            .expect("far map face");
    }
}

/// Uploads `VOXEL_DEMO_UPLOAD_CYCLE` repeats: `1` or `all` for every part,
/// else a comma list of `far`, `set`, `append` and `mat`.
#[derive(Clone, Copy, Default)]
struct UploadCycle {
    far: bool,
    set: bool,
    append: bool,
    mat: bool,
}

impl UploadCycle {
    fn from_env() -> Option<Self> {
        let text = std::env::var("VOXEL_DEMO_UPLOAD_CYCLE").ok()?;
        match text.trim() {
            "0" | "" => return None,
            "1" | "all" => {
                return Some(Self {
                    far: true,
                    set: true,
                    append: true,
                    mat: true,
                });
            }
            _ => {}
        }
        let mut parts = Self::default();
        for part in text.split(',') {
            match part.trim() {
                "far" => parts.far = true,
                "set" => parts.set = true,
                "append" => parts.append = true,
                "mat" => parts.mat = true,
                other => log::warn!("VOXEL_DEMO_UPLOAD_CYCLE: unknown part {other:?}"),
            }
        }
        Some(parts)
    }
}

/// One step of `VOXEL_DEMO_UPLOAD_CYCLE`. `far` re-sets far map 0 (a pending
/// replacement cube and a datum overwrite). `set` rewrites block-texture
/// layer 1 in place and `append` adds one layer (a grow once the palette
/// passes its capacity; past 100 layers the palette drops back to the three
/// base layers in place). `mat` edits the material table: a set on even steps,
/// an append on odd ones.
fn demo_upload_step(
    eng: &mut voxel_engine::Engine,
    parts: UploadCycle,
    step: u32,
    layers: &mut Vec<Vec<u8>>,
    procedural: &mut bool,
) {
    log::info!("demo upload cycle: step {step}");
    if parts.far {
        install_far_showcase(eng);
    }
    let mut set = parts.set;
    if layers.len() >= 100 {
        layers.truncate(3);
        set = true;
    }
    if parts.set {
        let tint = (step % 4) as u8 * 40;
        for px in layers[CHECKER_LAYER as usize].chunks_exact_mut(4) {
            px[2] = px[0].saturating_sub(tint);
        }
    }
    if set {
        eng.set_block_textures(16, layers);
    }
    if parts.append {
        let shade = (step % 200) as u8;
        let extra: Vec<u8> = [shade, 255 - shade, 128, 255].repeat(16 * 16);
        eng.append_block_textures(std::slice::from_ref(&extra));
        layers.push(extra);
    }
    if !parts.mat {
        return;
    }
    if step.is_multiple_of(2) {
        *procedural = !*procedural;
        if *procedural {
            eng.set_material_descs(&demo_procedural_materials());
        } else {
            eng.set_material_descs(&demo_array_materials());
        }
    } else {
        eng.append_material_descs(&[MaterialDesc::ARRAY_LAYER]);
    }
}

/// One step of `VOXEL_DEMO_RECREATE_CYCLE`. MSAA steps the way the A key does,
/// then returns to the count from before that step.
fn demo_recreate_step(eng: &mut voxel_engine::Engine, step: u32, msaa_before: &mut u32) {
    match step {
        0 => {
            log::info!("demo recreate cycle: vsync on");
            eng.set_vsync(true);
        }
        1 => {
            log::info!("demo recreate cycle: vsync off");
            eng.set_vsync(false);
        }
        2 => {
            *msaa_before = eng.msaa();
            let next = if eng.msaa() >= eng.max_msaa() {
                1
            } else {
                eng.msaa() * 2
            };
            log::info!("demo recreate cycle: msaa {next}");
            eng.set_msaa(next);
        }
        3 => {
            log::info!("demo recreate cycle: msaa {}", *msaa_before);
            eng.set_msaa(*msaa_before);
        }
        4 => {
            log::info!("demo recreate cycle: render scale 0.5");
            eng.set_render_scale(0.5);
        }
        5 => {
            log::info!("demo recreate cycle: render scale 1.0");
            eng.set_render_scale(1.0);
        }
        6 => {
            log::info!("demo recreate cycle: fullscreen on");
            eng.set_fullscreen(true);
        }
        7 => {
            log::info!("demo recreate cycle: fullscreen off");
            eng.set_fullscreen(false);
        }
        _ => {}
    }
}

fn main() {
    env_logger::init();

    // Upload meshes once; GPU records drive draws while resident+visible.
    // `big` exercises upload_mesh (Tracked) with set_mesh_placement.
    let mut uploaded = false;
    let mut procedural_mats = false;
    let mut angle = 0.0f32;
    // High-FOV cylindrical warp strength, cycled with G. Seeded from VOXEL_WARP
    // so headless screenshot runs can pick a value without a keypress.
    let mut warp_ratio: f32 = std::env::var("VOXEL_WARP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    // Vertical FOV in degrees, zoomed with the scroll wheel and clamped to a
    // usable range (telephoto .. wide) so you can't invert or flatten the lens.
    let mut fovy: f32 = DEMO_START_FOVY;
    // VOXEL_AUTOSHOT=1: grab one screenshot after warm-up, then quit — used to
    // verify the warp offscreen. Off by default (interactive run).
    let autoshot = std::env::var("VOXEL_AUTOSHOT").is_ok();
    // Far-body showcase. Unset leaves the orbit, the sky, and the scene as they are.
    let far_demo = std::env::var("VOXEL_DEMO_FAR").is_ok();
    // After ~2s, every ~60 frames: vsync on, off, MSAA up, MSAA back,
    // scale 0.5, scale 1, fullscreen on, fullscreen off, then quit.
    let recreate_cycle = std::env::var("VOXEL_DEMO_RECREATE_CYCLE").is_ok();
    let mut cycle_origin: Option<std::time::Instant> = None;
    let mut cycle_started = false;
    let mut cycle_frames: u32 = 0;
    let mut cycle_step: u32 = 0;
    let mut cycle_msaa: u32 = 1;
    let mut frame_n: u32 = 0;
    // Every ~60 frames after the first uploads: far map, block textures and
    // materials again (see `demo_upload_step`).
    let upload_cycle = UploadCycle::from_env();
    let mut upload_step: u32 = 0;
    let mut upload_layers = block_texture_layers(16);
    // Test aids: VRS (classifier + rate attachment), a starting MSAA count,
    // and a fullscreen start.
    let flags = voxel_engine::RenderFlags {
        vrs: std::env::var("VOXEL_DEMO_VRS").is_ok_and(|v| v != "0"),
        ..voxel_engine::RenderFlags::default()
    };
    let msaa = std::env::var("VOXEL_DEMO_MSAA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let fullscreen = std::env::var("VOXEL_DEMO_FULLSCREEN").is_ok_and(|v| v != "0");

    voxel_engine::run(
        Config {
            title: "voxel_engine demo".into(),
            target_fps: 0,
            vsync: false,
            msaa,
            flags,
            fullscreen,
            ..Config::default()
        },
        move |eng| {
            if eng.should_close() || eng.is_key_pressed(Key::Escape) {
                return false;
            }
            // Headless verification: capture the settled scene, then quit.
            // `screenshot` queues a detached encode that process exit kills
            // before the rename, so this blocks on `screenshot_to` (the reply
            // arrives only after the PNG is on disk). `eng` is unborrowed
            // here, before `begin_frame`; the capture re-presents the last
            // submitted frame.
            frame_n += 1;
            if autoshot && frame_n == 30 {
                let path = autoshot_path();
                match voxel_engine::screenshot_to(eng, &path) {
                    Ok(()) => log::info!("autoshot -> {}", path.display()),
                    Err(e) => log::error!("autoshot failed ({}): {e}", path.display()),
                }
                return false;
            }
            if recreate_cycle {
                let origin = cycle_origin.get_or_insert_with(std::time::Instant::now);
                if origin.elapsed().as_secs_f32() >= 2.0 {
                    let fire = if !cycle_started {
                        cycle_started = true;
                        cycle_frames = 0;
                        true
                    } else {
                        cycle_frames += 1;
                        if cycle_frames >= 60 {
                            cycle_frames = 0;
                            true
                        } else {
                            false
                        }
                    };
                    if fire {
                        // Eight steps, then one more 60-frame gap so the last
                        // fullscreen restore is applied and presented, then quit.
                        if cycle_step >= 8 {
                            return false;
                        }
                        demo_recreate_step(eng, cycle_step, &mut cycle_msaa);
                        cycle_step += 1;
                    }
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
            if eng.is_key_pressed(Key::A) {
                let next = if eng.msaa() >= eng.max_msaa() {
                    1
                } else {
                    eng.msaa() * 2
                };
                eng.set_msaa(next);
            }
            if eng.is_key_pressed(Key::M) {
                procedural_mats = !procedural_mats;
                if procedural_mats {
                    eng.set_material_descs(&demo_procedural_materials());
                } else {
                    eng.set_material_descs(&demo_array_materials());
                }
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
                if far_demo {
                    install_far_showcase(eng);
                }
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
                            cage: None,
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
            if let Some(parts) = upload_cycle
                && frame_n > 60
                && frame_n.is_multiple_of(60)
            {
                demo_upload_step(
                    eng,
                    parts,
                    upload_step,
                    &mut upload_layers,
                    &mut procedural_mats,
                );
                upload_step += 1;
            }

            // Scroll to zoom: each notch nudges the vertical FOV, scrolling up
            // (positive) narrows toward telephoto. Clamped so the lens stays sane.
            const FOVY_MIN: f32 = 20.0;
            const FOVY_MAX: f32 = 160.0;
            fovy = (fovy - eng.mouse_wheel() * 4.0).clamp(FOVY_MIN, FOVY_MAX);

            // The showcase is framed for the startup pose. Leave the orbit
            // alone unless that pose is what the frame has to show.
            if !far_demo {
                angle += eng.frame_time() * 0.4;
            }
            let center = Vec3::new(CHUNK as f32, 4.0, CHUNK as f32);
            let cam = demo_camera(angle, fovy, warp_ratio);

            let vsync = eng.vsync();
            let msaa = eng.msaa();
            let fullscreen = eng.fullscreen();
            let cull = eng.cull_faces();
            let fps = eng.fps();
            let mats = if procedural_mats { "proc" } else { "array" };

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
                // Far bodies composite in the sky pass. The sky above is
                // already set; this only adds the showcase when asked.
                if far_demo {
                    f3.set_far_bodies(&far_showcase_bodies());
                }
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
            frame.draw_rect(8, 8, 420, 76, Color::new(0, 0, 0, 150));
            frame.draw_text(&format!("{fps} FPS"), 16, 14, 20, Color::LIME);
            frame.draw_text(
                &format!(
                    "vsync {vsync} msaa {msaa}x fullscreen {fullscreen} cull {cull} mats {mats}"
                ),
                16,
                38,
                16,
                Color::RAYWHITE,
            );
            frame.draw_text(
                &format!(
                    "F fullscreen  V vsync  A msaa  M mats  C cull  G warp {warp_ratio:.1}  Esc quit"
                ),
                16,
                60,
                16,
                Color::GRAY,
            );
            true
        },
    );
}

#[cfg(test)]
fn project_startup(dir: Vec3) -> (f32, f32) {
    let cam = demo_camera(0.0, DEMO_START_FOVY, 0.0);
    let p = cam.position + dir * 1.0e5;
    let clip = cam.view_proj(demo_aspect()) * p.extend(1.0);
    assert!(clip.w > 0.0, "direction is behind the startup camera");
    let ndc = clip.truncate() / clip.w;
    (ndc.x, ndc.y)
}

/// Angular radius of the drawable bound (disc, corners, datum bulge, air).
#[cfg(test)]
fn showcase_ang(body: &FarBody, max_off: f32) -> f32 {
    let rho = body.radius / body.distance;
    let bound = match body.shape {
        FarShape::Sphere => rho * 1.05,
        FarShape::Cube => rho * 3.0f32.sqrt() * 1.035,
        FarShape::Rounded { exponent } => {
            rho * 3.0f32.powf(0.5 - 1.0 / exponent) * (1.0 + 2.0e-4) * 1.05
        }
        FarShape::Mapped { air, .. } => (body.radius + max_off + air) / body.distance,
        FarShape::InnerSphere => rho,
    };
    bound.asin()
}

#[cfg(test)]
fn assert_on_screen(ndc_x: f32, ndc_y: f32, ang: f32) {
    let tan_half = (DEMO_START_FOVY.to_radians() * 0.5).tan();
    let aspect = demo_aspect();
    let theta = (ndc_y * tan_half).atan();
    let phi = (ndc_x * tan_half * aspect).atan();
    let top = (theta + ang).tan() / tan_half;
    let bot = (theta - ang).tan() / tan_half;
    let left = (phi - ang).tan() / (tan_half * aspect);
    let right = (phi + ang).tan() / (tan_half * aspect);
    assert!(
        bot > 0.25 && top < 0.99,
        "vertical extent {bot}..{top} leaves the upper half"
    );
    assert!(
        left > -0.99 && right < 0.99,
        "horizontal extent {left}..{right} leaves the frame"
    );
}

#[test]
fn startup_basis_is_the_orbit_at_angle_zero() {
    let (right, up, forward) = startup_camera_basis();
    let expect_f = Vec3::new(-44.0, -22.0, 0.0).normalize();
    assert!((forward - expect_f).length() < 1e-5, "forward {forward:?}");
    // Forward lies in the XY plane toward −X, so forward × +Y is −Z.
    // That −Z is screen-right: an independent check from the cross product
    // written in `startup_camera_basis`.
    assert!(
        (right - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4,
        "right {right:?}"
    );
    assert!(up.y > 0.8, "screen up should lift world Y, got {up:?}");
    let (cx, cy) = project_startup(forward);
    assert!(
        cx.abs() < 1e-4 && cy.abs() < 1e-4,
        "center NDC ({cx}, {cy})"
    );
    let (rx, _) = project_startup(startup_view_dir(0.5, 0.0));
    assert!(rx > 0.45, "screen-right projected to {rx}");
    let (_, uy) = project_startup(startup_view_dir(0.0, 0.5));
    assert!(uy > 0.45, "screen-up projected to {uy}");
}

#[test]
fn far_showcase_fits_the_upper_half() {
    let bodies = far_showcase_bodies();
    let datum = far_showcase_datum(bodies[3].radius);
    let max_off = datum.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut dirs = [Vec3::ZERO; 5];
    for (i, body) in bodies.iter().enumerate() {
        let (nx, ny) = FAR_LAYOUT_NDC[i];
        let (px, py) = project_startup(body.dir);
        assert!(
            (px - nx).abs() < 1e-3 && (py - ny).abs() < 1e-3,
            "body {i} projected to ({px}, {py}), wanted ({nx}, {ny})"
        );
        assert_on_screen(nx, ny, showcase_ang(body, max_off));
        dirs[i] = body.dir.normalize();
    }
    for i in 0..5 {
        for j in (i + 1)..5 {
            let sep = dirs[i].dot(dirs[j]).clamp(-1.0, 1.0).acos();
            let need = showcase_ang(&bodies[i], max_off) + showcase_ang(&bodies[j], max_off);
            assert!(
                sep > need,
                "bodies {i} and {j} overlap: sep {sep} need {need}"
            );
        }
    }
}

#[test]
fn far_showcase_matches_the_shape_contract() {
    let bodies = far_showcase_bodies();
    let expect_rho = [0.08f32, 0.05, 0.06, FAR_MAPPED_RHO, 1.0e-4];
    for (body, rho) in bodies.iter().zip(expect_rho) {
        assert!((body.distance - FAR_DEMO_DISTANCE).abs() < 1e-3);
        assert!((body.radius / body.distance - rho).abs() < 1e-6);
    }
    assert!(matches!(bodies[0].shape, FarShape::Sphere));
    assert!(matches!(bodies[1].shape, FarShape::Cube));
    assert!(matches!(
        bodies[2].shape,
        FarShape::Rounded { exponent } if (exponent - 4.0).abs() < 1e-6
    ));
    match bodies[3].shape {
        FarShape::Mapped { map, horizon, air } => {
            assert_eq!(map, FarMapId(0));
            assert!((horizon - 1.0).abs() < 1e-6);
            assert!((air - 0.02 * bodies[3].radius).abs() < 1e-3);
        }
        other => panic!("expected mapped, got {other:?}"),
    }
    assert!(matches!(bodies[4].shape, FarShape::Sphere));
    let a = 30.0_f32.to_radians();
    let cube_q = Quat::from_rotation_y(a) * Quat::from_rotation_x(a);
    assert!((bodies[1].rotation.dot(cube_q).abs() - 1.0).abs() < 1e-5);
    for body in [&bodies[0], &bodies[2], &bodies[3], &bodies[4]] {
        assert!((body.rotation.dot(Quat::IDENTITY).abs() - 1.0).abs() < 1e-5);
    }
    // Sphere uses two tones; atmospheres are non-black so the limb draws.
    assert!(bodies[0].albedo[0].0 != bodies[0].albedo[1].0);
    for body in &bodies {
        let c = body.atmosphere.0;
        assert!(c[0] + c[1] + c[2] > 0.1, "black atmosphere draws no rim");
    }
    let blue = bodies[3].atmosphere.0;
    assert!(
        blue[2] > blue[0] && blue[2] > blue[1],
        "mapped air {blue:?}"
    );

    let g = FAR_DATUM_RES;
    let datum = far_showcase_datum(bodies[3].radius);
    assert_eq!(datum.len(), 6 * g as usize * g as usize);
    assert!(datum.iter().all(|v| v.is_finite()));
    let (face, i, j) = (2usize, 10u32, 21u32);
    let d = far_chart_dir(face, g, i, j);
    let (tu, n, tv) = far_map_basis(face);
    let xi = (4.0 / std::f32::consts::PI) * (d.dot(tu) / d.dot(n)).atan();
    let eta = (4.0 / std::f32::consts::PI) * (d.dot(tv) / d.dot(n)).atan();
    let xi_g = 2.0 * i as f32 / (g - 1) as f32 - 1.0;
    let eta_g = 2.0 * j as f32 / (g - 1) as f32 - 1.0;
    assert!((xi - xi_g).abs() < 1e-4, "{xi} vs {xi_g}");
    assert!((eta - eta_g).abs() < 1e-4, "{eta} vs {eta_g}");
    let bump = (3.0 * d.x).sin() * (2.0 * d.y).cos() + 0.5 * (5.0 * d.z).sin();
    let idx = face * (g * g) as usize + j as usize * g as usize + i as usize;
    assert!((datum[idx] - 0.08 * bodies[3].radius * bump).abs() < 1e-3);

    let face0 = far_showcase_face(0);
    let face2 = far_showcase_face(2);
    assert_eq!(
        face0.len(),
        FAR_ALBEDO_SIZE as usize * FAR_ALBEDO_SIZE as usize * 4
    );
    let dir = far_cube_texel_dir(0, FAR_ALBEDO_SIZE, 0, 0);
    let scale = lat_lon_grid(dir);
    let base = FAR_FACE_SRGB[0];
    assert_eq!(face0[0], (base[0] as f32 * scale).round() as u8);
    assert_ne!(&face0[..3], &face2[..3]);
    let mut saw_line = false;
    let mut saw_field = false;
    for px in face0.chunks(4) {
        if px[0] == base[0] {
            saw_field = true;
        }
        if px[0] < base[0] {
            saw_line = true;
        }
    }
    assert!(
        saw_line && saw_field,
        "lat/lon grid did not modulate the face"
    );
}
