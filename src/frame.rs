/// Per-frame draw recording: `Frame` (2D overlay + frame lifecycle) and
/// `Frame3D` (world rendering inside a `begin_3d` scope). Everything records
/// into reused CPU lists; submission happens when the `Frame` drops.
use glam::{DVec3, Mat3, Mat4, Vec2, Vec3};

use crate::camera::{Aspect, Camera3D, WarpMap};
use crate::color::{Color, LinearRgb};
use crate::engine::Engine;
use crate::far_body::{FarBody, MAX_FAR_BODIES};
use crate::font;
use crate::mesh::DebugVertex;
use crate::vk::pipeline::{EyeSplit, Vertex2D};
use crate::vk::taa::JitterOffset;
use crate::vk::uniforms::{FrameUniformsGpu, LOD_MORPH_FLAG, LocalFrame, LodMorphGpu};

/// Sky-pass-private state, set by the app inside a `begin_3d` scope: sun
/// geometry plus the disc tint. The sky COLOURS (zenith/horizon gradient, fog
/// glow) are NOT here — the sky fragment reads them from the per-frame
/// `FrameUniforms` UBO, the SAME linear source the terrain fog reads, so the two
/// can never diverge (one source of truth for sky data). The engine adds the inverse
/// view-projection at record time, so the app never touches a matrix.
#[derive(Clone, Copy, PartialEq)]
pub struct SkyDesc {
    pub sun_dir: Vec3,
    /// Linear disc/glow tint (no OETF on this path); the analytic sun disc adds
    /// it around the sun direction.
    pub sun_tint: LinearRgb,
    /// Angular radius of the sun disc, in radians; the shader derives the disc's
    /// core/rim edge cosines from it (generated `SUN_DISC_*` cone constants).
    pub sun_angular_radius: f32,
}

/// A point light standing in for the sun. When set on a frame, terrain,
/// impostors and the sky halo use `dir` and `color`. The real sun disc stays
/// hidden unless `show_disc`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SunOverride {
    /// World-space direction from the viewer toward the light. Not required to be unit.
    pub dir: Vec3,
    /// Linear light colour, unclamped.
    pub color: LinearRgb,
    /// Draw the analytic sun disc along `dir` in `color`. Off hides the real disc.
    pub show_disc: bool,
}

fn finite_unit(v: Vec3) -> Option<Vec3> {
    if !v.is_finite() {
        return None;
    }
    let l2 = v.length_squared();
    if !(l2 > 1e-12) {
        return None;
    }
    Some(v * l2.sqrt().recip())
}

fn color_finite(c: LinearRgb) -> bool {
    c.0.iter().all(|ch| ch.is_finite())
}

/// Replace the sun direction and colour. `light.w` stays 1 so the night flip
/// does not reverse `dir`. A non-finite override leaves `u` untouched.
fn apply_sun_override(mut u: FrameUniformsGpu, over: SunOverride, up: Vec3) -> FrameUniformsGpu {
    if !color_finite(over.color) {
        return u;
    }
    let Some(dir) = finite_unit(over.dir) else {
        return u;
    };
    let up = finite_unit(up).unwrap_or(Vec3::Y);
    u.sun_dir_elev = [dir.x, dir.y, dir.z, dir.dot(up)];
    u.light[0] = over.color.0[0];
    u.light[1] = over.color.0[1];
    u.light[2] = over.color.0[2];
    u.light[3] = 1.0;
    u.prepare_derived();
    u
}

/// Sky descriptor after the override. `None` returns `desc` unchanged.
pub(crate) fn sky_with_override(desc: SkyDesc, over: Option<SunOverride>) -> SkyDesc {
    let Some(over) = over else {
        return desc;
    };
    if !color_finite(over.color) {
        return desc;
    }
    let Some(dir) = finite_unit(over.dir) else {
        return desc;
    };
    SkyDesc {
        sun_dir: dir,
        sun_tint: if over.show_disc {
            over.color
        } else {
            LinearRgb([0.0, 0.0, 0.0])
        },
        sun_angular_radius: desc.sun_angular_radius,
    }
}

/// Full-res coverage box, camera-relative. A fragment is covered when
/// `abs(world)` is strictly inside `half` on every axis. A non-positive
/// component covers nothing on that axis.
#[derive(Clone, Copy, PartialEq)]
pub struct CoverageVolume {
    pub half: Vec3,
}

/// One detail level's LOD-morph band, in blocks, around the morph eye.
///
/// `half` is the band box's half-extent. `start` is the smoothstep edge in
/// units of that box (`d = 0` at the eye, `d = 1` on the box surface). A
/// non-positive `half` component disables morphing for the level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodMorph {
    /// Half-extent of the band, in blocks.
    pub half: Vec3,
    /// Smoothstep start. The blend is 0 at and inside `start`, and 1 at the box edge.
    pub start: f32,
}

/// Bands for one frame. `count == 0` is morphing off (eye and bands zero).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LodMorphFrame {
    eye: DVec3,
    bands: [LodMorph; 16],
    count: u8,
}

impl LodMorphFrame {
    fn off() -> Self {
        Self {
            eye: DVec3::ZERO,
            bands: [LodMorph {
                half: Vec3::ZERO,
                start: 0.0,
            }; 16],
            count: 0,
        }
    }

    /// Empty `bands` is the off state, whatever `eye` was. Entries past 16 are dropped.
    pub(crate) fn from_api(eye: DVec3, bands: &[LodMorph]) -> Self {
        if bands.is_empty() {
            return Self::off();
        }
        let mut out = Self::off();
        let n = bands.len().min(16);
        out.bands[..n].copy_from_slice(&bands[..n]);
        out.count = n as u8;
        out.eye = eye;
        out
    }

    /// GPU tail. The eye is split like the camera (`EyeSplit::of`) only while morphing is on.
    pub(crate) fn to_gpu(self) -> LodMorphGpu {
        if self.count == 0 {
            return LodMorphGpu::default();
        }
        let split = EyeSplit::of(self.eye);
        let mut gpu = LodMorphGpu::default();
        gpu.eye_block = [
            split.block[0],
            split.block[1],
            split.block[2],
            LOD_MORPH_FLAG,
        ];
        gpu.eye_frac = [split.frac[0], split.frac[1], split.frac[2], 0.0];
        for i in 0..self.count as usize {
            let b = self.bands[i];
            gpu.bands[i] = [b.half.x, b.half.y, b.half.z, b.start];
        }
        gpu
    }
}

/// Per-mesh style for [`Engine::set_mesh_style`](crate::Engine::set_mesh_style) — the
/// typed form of the shader's per-draw mode bits, so mode bits must go through Rust
/// and can't bypass the GPU via raw bit patterns. `Default` is plain textured.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct FadeStyle {
    /// Replace texturing with the draw's `flat_rgba` colour.
    pub flat_color: bool,
}

impl FadeStyle {
    /// The shader-side bit encoding (`1` = flat color).
    pub(crate) fn bits(self) -> u32 {
        self.flat_color as u32
    }
}

/// This 3D scope's lighting, a REQUIRED argument to [`Frame::begin_3d`]. The
/// mesh, sky, and water shaders read all of their lighting from the per-frame
/// UBO, so a 3D pass with no lighting renders pure black — making it a required
/// parameter (rather than an optional setter) removes that failure mode at the
/// type level: there is no way to open a 3D scope without deciding lighting.
pub enum Lighting {
    /// App-composed per-frame uniforms (`FrameSnapshot` → [`FrameUniformsGpu`]):
    /// the single CPU truth for shader-side sky, fog, candle/ambient, and shadow
    /// evaluation. Passed through the CPU feature gates (`gate_uniforms`, `RenderFlags`)
    /// before it reaches the GPU, so a disabled feature is neutralized once, at
    /// this producer→GPU chokepoint, for every consumer.
    Composed(FrameUniformsGpu),
    /// A fixed lit neutral ([`FrameUniformsGpu::full_bright`]): unit ambient
    /// floor, valid sun, no fog. For smoke tests and apps that don't compose
    /// lighting yet — geometry renders lit instead of black. A deliberate,
    /// named choice, never a silent fallback.
    FullBright,
}

/// Applies the env feature gates to composed uniforms — the ONE producer→GPU
/// chokepoint, so terrain, sky, fog, and water all see the same disabled state.
fn gate_uniforms(f: &crate::engine::RenderFlags, mut u: FrameUniformsGpu) -> FrameUniformsGpu {
    if !f.fog {
        u.horizon[3] = 0.0;
    }
    if !f.blocklight {
        u.candle[..3].fill(0.0);
    }
    if !f.ambient {
        u.candle[3] = 0.0;
    }
    if !f.sunlight {
        u.light[..3].fill(0.0);
    }
    if !f.exposure {
        u.exposure_dither[0] = 1.0;
    }
    if !f.water_anim {
        // Freeze the animation phase: water still renders (tint, reflection),
        // its surface just holds still.
        u.anim[0] = 0.0;
    }
    if !f.stars {
        // Zero the stars gain; sky.frag skips the starfield evaluation.
        u.extras[0] = 0.0;
    }
    u.prepare_derived();
    u
}

/// The current 3D scope's state: either a 3D scene exists with ALL its data,
/// or none of it does (`DrawLists::scene: Option<Scene3D>`) — replaces a
/// `has_3d` bool plus two comment-enforced-coupled `Option`s that could
/// disagree by construction.
#[derive(Clone, Copy)]
pub(crate) struct Scene3D {
    pub view_proj: Mat4,
    /// Retained so the render thread can fit the shadow cascades around this
    /// frame's frustum.
    pub camera: Camera3D,
    /// This frame's lighting uniforms, resolved from the required [`Lighting`]
    /// argument to [`Frame::begin_3d`] — a 3D scene always carries lighting.
    /// Written into the per-frame UBO (set 0, binding 2) each frame; the single
    /// source of truth for shader-side sky, fog, candle/ambient, and shadow
    /// evaluation, and for avatar key lighting (see [`KeyLight`]).
    pub frame_uniforms: FrameUniformsGpu,
    /// Camera world position; feeds six-way face culling (which needs the
    /// camera in each mesh's local frame).
    pub cam_pos: Vec3,
    /// World-space position of the RENDER-SPACE ORIGIN (the rebase point), in
    /// f64: a camera-at-origin app passes its true eye (the translation TAA
    /// reprojection needs lives ONLY here); an app whose draws use absolute
    /// world coordinates passes `DVec3::ZERO`.
    pub eye: DVec3,
    /// `tan(fovy/2)` for the current 3D camera; the renderer derives the
    /// vertical focal length (`0.5*height/tan_half`) for VRS depth thresholding.
    pub fovy_tan_half: f32,
    /// Wide-FOV lens for this scope. `Identity` in rectilinear mode.
    pub warp_map: WarpMap,
    /// Sub-pixel camera jitter (PIXELS, +/-0.5) for this 3D scope, injected here
    /// — the sole injection point. `view_proj` above stays CLEAN so culling
    /// and TAA reprojection never see the jitter.
    pub jitter: JitterOffset,
    /// Key light for oriented debug boxes, derived once per `begin_3d`.
    pub(crate) key_light: KeyLight,
}

/// CPU-side draw lists for one frame. Vec capacities persist across frames.
///
/// Blocking capture ([`crate::screenshot_to`]) re-presents the last completed
/// pooled snapshot rather than cloning these lists every frame.
pub(crate) struct DrawLists {
    pub clear: LinearRgb,
    /// `Some` for exactly the frames between `begin_3d` and the next `reset`;
    /// `None` on pure-2D frames.
    pub scene: Option<Scene3D>,
    /// Procedural sky palette for this frame's background pass, or `None` to
    /// leave the flat clear colour showing. Set via [`Frame3D::set_sky`]; unlike
    /// `Scene3D`'s fields this is set AFTER `begin_3d` and must survive a second
    /// `begin_3d` call within the same frame, so it stays outside the
    /// atomically-replaced scene.
    pub sky: Option<SkyDesc>,
    /// Local sky frame for this scope. `None` is up `+Y` and altitude `0`.
    /// Set via [`Frame3D::set_local_frame`]; same post-`begin_3d` lifetime as `sky`.
    pub local: Option<LocalFrame>,
    /// Far-body impostors for the sky pass. `far_count` is the live prefix;
    /// the tail is stale and never read. Same post-`begin_3d` lifetime as `sky`.
    pub far_bodies: [FarBody; MAX_FAR_BODIES],
    pub far_count: u32,
    /// Point light standing in for the sun. `None` leaves the composed sun alone.
    /// Set via [`Frame3D::set_sun_override`]; same post-`begin_3d` lifetime as `sky`.
    pub sun_override: Option<SunOverride>,
    /// Chunk→LOD box half-extents. A non-positive component disables that
    /// axis. LOD tiles hard-discard inside the box. Set via
    /// [`Frame3D::set_lod_clip`]; same post-`begin_3d` lifetime as `sky`.
    pub lod_half: Vec3,
    /// Per-detail LOD morph bands. Default and [`Self::reset`] are off.
    /// Set via [`Frame3D::set_lod_morph`]; same post-`begin_3d` lifetime as `sky`.
    pub lod_morph: LodMorphFrame,
    /// Debug-flat override (`DebugView::TerrainKey`): when set, every 3D mesh
    /// fragment outputs this flat key colour while still writing depth.
    /// `None` renders normally. Set via [`Frame3D::set_debug_flat`]; same
    /// post-`begin_3d` lifetime as `sky`.
    pub debug_flat: Option<Color>,
    pub cube_verts: Vec<DebugVertex>,
    pub line_verts: Vec<DebugVertex>,
    /// Translucent ground decals (contact shadows), drawn with the blended,
    /// depth-read-only debug pipeline after the opaque cubes.
    pub shadow_verts: Vec<DebugVertex>,
    pub verts_2d: Vec<Vertex2D>,
    /// Minimap verts (drawn by `tris2d_tex` pipeline).
    pub tex_verts_2d: Vec<Vertex2D>,
}

impl DrawLists {
    pub fn new() -> Self {
        Self {
            clear: LinearRgb([0.0, 0.0, 0.0]),
            scene: None,
            sky: None,
            local: None,
            far_bodies: [FarBody::default(); MAX_FAR_BODIES],
            far_count: 0,
            sun_override: None,
            lod_half: Vec3::ZERO,
            lod_morph: LodMorphFrame::off(),
            debug_flat: None,
            cube_verts: Vec::new(),
            line_verts: Vec::new(),
            shadow_verts: Vec::new(),
            verts_2d: Vec::new(),
            tex_verts_2d: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.scene = None;
        self.sky = None;
        self.local = None;
        self.far_count = 0;
        self.sun_override = None;
        self.lod_half = Vec3::ZERO;
        self.lod_morph = LodMorphFrame::off();
        self.debug_flat = None;
        self.cube_verts.clear();
        self.line_verts.clear();
        self.shadow_verts.clear();
        self.verts_2d.clear();
        self.tex_verts_2d.clear();
    }

    /// Resolved local frame: the value from [`Frame3D::set_local_frame`], or
    /// `+Y` / altitude 0 when the app left it unset.
    pub(crate) fn local_frame(&self) -> LocalFrame {
        self.local.unwrap_or_default()
    }

    /// Live far bodies, far to near. The tail past `far_count` is stale.
    pub(crate) fn far_slice(&self) -> &[FarBody] {
        &self.far_bodies[..self.far_count as usize]
    }

    /// Sun direction and colour for this frame. With no override this is the
    /// `begin_3d` packet, bit for bit. With no 3D scene, full-bright.
    pub(crate) fn lit_uniforms(&self) -> FrameUniformsGpu {
        let Some(scene) = &self.scene else {
            return FrameUniformsGpu::full_bright();
        };
        match self.sun_override {
            None => scene.frame_uniforms,
            Some(over) => apply_sun_override(scene.frame_uniforms, over, self.local_frame().up),
        }
    }

    /// Sky descriptor for the pass, with the real disc hidden unless the
    /// override asks for it. `None` when the frame set no sky.
    pub(crate) fn sky_for_pass(&self) -> Option<SkyDesc> {
        self.sky.map(|desc| sky_with_override(desc, self.sun_override))
    }
}

/// Monotone jitter-sequence index, advanced once per [`Frame::begin_3d`]. Indexes
/// the shared Halton table (`jitter_at`), so consecutive frames decorrelate.
static JITTER_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct Frame<'e> {
    pub(crate) eng: &'e mut Engine,
}

impl<'e> Frame<'e> {
    /// Starts the 3D pass. Drop the returned scope (or let it fall out of a
    /// block) before drawing 2D overlays on top.
    ///
    /// All 3D geometry in a frame shares one camera: calling `begin_3d` a
    /// second time replaces the camera for every 3D draw already recorded
    /// this frame (frustum culling, however, uses each scope's own camera).
    ///
    /// `eye` is the world-space position of the RENDER-SPACE ORIGIN in f64
    /// (see [`Scene3D::eye`]): a camera-at-origin app passes its true eye
    /// (the translation TAA reprojection needs lives ONLY here); an app whose
    /// draws use absolute world coordinates passes `DVec3::ZERO`. Taking it as
    /// a parameter — rather than an optional setter — makes forgetting it
    /// unrepresentable.
    ///
    /// `light` is this scope's lighting ([`Lighting`]): the mesh, sky, and water
    /// shaders read ALL their lighting from the per-frame UBO, so a 3D scope with
    /// no lighting would render pure black. Taking it as a required parameter —
    /// same rationale as `eye` — makes the black-scene bug unrepresentable:
    /// there is no way to open a 3D pass without deciding lighting. Pass
    /// [`Lighting::FullBright`] for a lit neutral, or [`Lighting::Composed`] with
    /// the app's own [`FrameUniformsGpu`].
    pub fn begin_3d(&mut self, cam: &Camera3D, eye: DVec3, light: Lighting) -> Frame3D<'_, 'e> {
        let w = self.eng.client.screen_width().max(1) as f32;
        let h = self.eng.client.screen_height().max(1) as f32;
        // Wide-FOV renders a *wider* rectilinear source (horizontal only, so the
        // vertical fov — and thus `fovy_tan_half`/VRS — is unchanged). The tonemap
        // resample compresses the periphery back to the presented frame. Culling
        // must use this widened frustum or the extra periphery is culled away.
        let warp_map = WarpMap::from_lens(cam.lens);
        let source_aspect = Aspect(w / h).source(&warp_map);
        let view_proj = cam.view_proj(source_aspect.get());
        let seq = JITTER_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Jitter and TAA resolve are coupled — toggling one breaks temporal
        // stability. Jitter is applied every rendered frame; the resolve runs
        // at present time at output resolution.
        let jitter = if self.eng.flags.taa {
            crate::skeleton::jitter_at(seq)
        } else {
            JitterOffset::ZERO
        };
        let frame_uniforms = match light {
            Lighting::Composed(u) => {
                if self.eng.last_composed == Some(u) && self.eng.last_gate_flags == self.eng.flags {
                    self.eng
                        .last_gated
                        .expect("gated uniforms cached with last_composed")
                } else {
                    let gated = gate_uniforms(&self.eng.flags, u);
                    self.eng.last_composed = Some(u);
                    self.eng.last_gate_flags = self.eng.flags;
                    self.eng.last_gated = Some(gated);
                    gated
                }
            }
            Lighting::FullBright => {
                if self.eng.last_composed.is_none()
                    && self.eng.last_gated.is_some()
                    && self.eng.last_gate_flags == self.eng.flags
                {
                    self.eng.last_gated.expect("full-bright uniforms cached")
                } else {
                    let gated = FrameUniformsGpu::full_bright();
                    self.eng.last_composed = None;
                    self.eng.last_gate_flags = self.eng.flags;
                    self.eng.last_gated = Some(gated);
                    gated
                }
            }
        };
        self.eng.lists.scene = Some(Scene3D {
            view_proj,
            camera: *cam,
            frame_uniforms,
            cam_pos: cam.position,
            eye,
            fovy_tan_half: (cam.fovy.to_radians() * 0.5).tan(),
            warp_map,
            jitter,
            key_light: KeyLight::from_uniforms(frame_uniforms),
        });
        Frame3D { frame: self }
    }

    pub fn screen_width(&self) -> i32 {
        self.eng.screen_width()
    }

    pub fn screen_height(&self) -> i32 {
        self.eng.screen_height()
    }

    pub fn measure_text(&self, text: &str, font_size: i32) -> i32 {
        font::measure_text(text, font_size)
    }

    pub fn draw_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: Color) {
        let uv = font::white_uv();
        push_quad_2d(
            &mut self.eng.lists.verts_2d,
            [x as f32, y as f32],
            [(x + w) as f32, (y + h) as f32],
            uv,
            uv,
            color,
        );
    }

    pub fn draw_line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, color: Color) {
        let a = Vec2::new(x1 as f32, y1 as f32);
        let b = Vec2::new(x2 as f32, y2 as f32);
        let dir = b - a;
        if dir.length_squared() < 1e-6 {
            return;
        }
        // 1px-thick quad around the segment.
        let n = Vec2::new(-dir.y, dir.x).normalize() * 0.5;
        let uv = font::white_uv();
        let c = [color.r, color.g, color.b, color.a];
        let corners = [a + n, b + n, b - n, a - n];
        let v = &mut self.eng.lists.verts_2d;
        for idx in [0usize, 3, 2, 0, 2, 1] {
            v.push(Vertex2D {
                pos: corners[idx].to_array(),
                uv,
                color: c,
            });
        }
    }

    pub fn draw_text(&mut self, text: &str, x: i32, y: i32, font_size: i32, color: Color) {
        let size = font_size.max(1) as f32;
        let mut pen_x = x as f32;
        let mut pen_y = y as f32;
        for ch in text.chars() {
            if ch == '\n' {
                pen_x = x as f32;
                pen_y += size;
                continue;
            }
            if ch != ' ' {
                let (uv_min, uv_max) = font::glyph_uv(ch);
                push_quad_2d(
                    &mut self.eng.lists.verts_2d,
                    [pen_x, pen_y],
                    [pen_x + size, pen_y + size],
                    uv_min,
                    uv_max,
                    color,
                );
            }
            pen_x += size;
        }
    }

    /// Rotated textured quad; `tint` white means unmodified.
    pub fn draw_minimap(&mut self, center: [f32; 2], radius_px: f32, rotation: f32, tint: Color) {
        let c = [tint.r, tint.g, tint.b, tint.a];
        let center = Vec2::from(center);
        let (sin, cos) = rotation.sin_cos();
        let r = radius_px;
        // Vertex order matches push_quad_2d's TL, BL, BR, TR.
        let corners = [
            (Vec2::new(-r, -r), [0.0, 0.0]),
            (Vec2::new(-r, r), [0.0, 1.0]),
            (Vec2::new(r, r), [1.0, 1.0]),
            (Vec2::new(r, -r), [1.0, 0.0]),
        ];
        let verts: [Vertex2D; 4] = corners.map(|(o, uv)| {
            let pos = center + Vec2::new(o.x * cos - o.y * sin, o.x * sin + o.y * cos);
            Vertex2D {
                pos: pos.into(),
                uv,
                color: c,
            }
        });
        let [tl, bl, br, tr] = verts;
        self.eng
            .lists
            .tex_verts_2d
            .extend_from_slice(&[tl, bl, br, tl, br, tr]);
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        // Don't submit GPU work during a panic unwind: a failing Vulkan call
        // here would double-panic and abort, hiding the original error.
        if std::thread::panicking() {
            return;
        }
        self.eng.finish_frame();
    }
}

/// Contact-shadow quad. `+Y` is the historical ground corners, built with the
/// same subtractions. Any other unit normal gets a quad in its plane, wound
/// so the geometric normal equals `normal` (`u × v = -normal`).
fn shadow_corners(center: Vec3, normal: Vec3, radius: f32) -> [[f32; 3]; 4] {
    if normal == Vec3::Y {
        let (cx, cy, cz, r) = (center.x, center.y, center.z, radius);
        return [
            [cx - r, cy, cz - r],
            [cx - r, cy, cz + r],
            [cx + r, cy, cz + r],
            [cx + r, cy, cz - r],
        ];
    }
    let helper = if normal.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = normal.cross(helper).normalize();
    let v = u.cross(normal);
    let corner = |a: f32, b: f32| {
        let p = center + (u * a + v * b) * radius;
        [p.x, p.y, p.z]
    };
    [
        corner(-1.0, -1.0),
        corner(-1.0, 1.0),
        corner(1.0, 1.0),
        corner(1.0, -1.0),
    ]
}

pub struct Frame3D<'f, 'e> {
    frame: &'f mut Frame<'e>,
}

impl Frame3D<'_, '_> {
    /// Sets the chunk→LOD coverage box. Must equal the streamed full-res volume.
    pub fn set_lod_clip(&mut self, v: CoverageVolume) {
        let half = v.half.max(Vec3::ZERO);
        if self.frame.eng.lists.lod_half == half {
            return;
        }
        self.frame.eng.lists.lod_half = half;
    }

    /// Per-detail-level morph bands, index = detail level k (0..16; extra entries
    /// ignored), measured around `eye` in the same frame as mesh `block`s (world
    /// for flat meshes, the caller's frontier frame for `caged_at` meshes). A
    /// level whose band has a non-positive half component does not morph.
    /// An empty slice turns morphing off (the default).
    pub fn set_lod_morph(&mut self, eye: DVec3, bands: &[LodMorph]) {
        let next = LodMorphFrame::from_api(eye, bands);
        if self.frame.eng.lists.lod_morph == next {
            return;
        }
        self.frame.eng.lists.lod_morph = next;
    }

    /// Sets the procedural sky drawn behind this frame's geometry. The
    /// background pass shades only pixels the terrain did not cover (a
    /// reversed-Z depth trick), so it is near-free. Call once inside the
    /// `begin_3d` scope; leaving it unset shows the flat clear colour.
    pub fn set_sky(&mut self, desc: SkyDesc) {
        if self.frame.eng.lists.sky == Some(desc) {
            return;
        }
        self.frame.eng.lists.sky = Some(desc);
    }

    /// Local up and altitude above the local surface datum for sky, fog, and
    /// clouds. The engine derives the tangent basis into the frame-uniform tail.
    /// Call inside the `begin_3d` scope; leaving it unset keeps `up = +Y` and
    /// altitude `0` (the `+Y` sky matches the old world-Y gradient).
    pub fn set_local_frame(&mut self, up: Vec3, altitude: f32) {
        let next = LocalFrame { up, altitude };
        if self.frame.eng.lists.local == Some(next) {
            return;
        }
        self.frame.eng.lists.local = Some(next);
    }

    /// Far bodies for this frame's sky pass (planets, moons, the home cube).
    /// At most [`MAX_FAR_BODIES`] are kept; the engine sorts them far to near.
    /// Same lifetime as [`set_sky`](Self::set_sky).
    pub fn set_far_bodies(&mut self, bodies: &[FarBody]) {
        let n = crate::far_body::store(bodies, &mut self.frame.eng.lists.far_bodies);
        self.frame.eng.lists.far_count = n;
    }

    /// Light this frame from `over` instead of the composed sun. `None` restores
    /// the sun `begin_3d` was given. Call inside the `begin_3d` scope.
    pub fn set_sun_override(&mut self, over: Option<SunOverride>) {
        self.frame.eng.lists.sun_override = over;
    }

    /// Debug-flat override (`DebugView::TerrainKey`): `Some(key)`
    /// makes every 3D mesh fragment output `key` while still writing depth, so
    /// occlusion/silhouette stay exact and the sky-hole detector distinguishes
    /// real terrain coverage from the magenta clear. `None` restores normal
    /// shading. Rides the per-frame UBO's `extras` lane (no push-constant or
    /// pipeline change).
    pub fn set_debug_flat(&mut self, color: Option<Color>) {
        self.frame.eng.lists.debug_flat = color;
    }

    pub fn draw_cube(&mut self, center: Vec3, size: Vec3, color: Color) {
        let min = center - size * 0.5;
        let max = center + size * 0.5;
        let c = [color.r, color.g, color.b, color.a];
        let verts = &mut self.frame.eng.lists.cube_verts;
        for face in cube_faces(min, max) {
            for idx in [0usize, 1, 2, 0, 2, 3] {
                verts.push(DebugVertex {
                    pos: face[idx],
                    color: c,
                });
            }
        }
    }

    /// Box centred at `center`, half-extents `half`, rotated by `rot`
    /// (columns = local axes in world space). Each face is shaded by the frame's
    /// sky key light (the same source terrain uses, see [`KeyLight`]) baked into
    /// the vertex colour, so limbs read as 3D and track the time of day.
    pub fn draw_box(&mut self, center: Vec3, half: Vec3, rot: Mat3, color: Color) {
        let lists = &self.frame.eng.lists;
        let key = if lists.sun_override.is_none() {
            lists
                .scene
                .as_ref()
                .expect("draw_box is inside a begin_3d scope")
                .key_light
        } else {
            KeyLight::from_uniforms(lists.lit_uniforms())
        };
        debug_assert!(
            (key.dir.length_squared() - 1.0).abs() < 1e-4,
            "KeyLight::dir is unit length (normalized once in begin_3d)"
        );
        // Local-space corner layout and per-face normals share the cube ordering.
        let faces = cube_faces(-half, half);
        const NORMALS: [Vec3; 6] = [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let verts = &mut self.frame.eng.lists.cube_verts;
        for (face, local_n) in faces.iter().zip(NORMALS) {
            let n = rot * local_n;
            let lit = key.ambient + key.sun * n.dot(key.dir).max(0.0);
            let shaded = |v: u8, chan: f32| (v as f32 * chan).round().clamp(0.0, 255.0) as u8;
            let c = [
                shaded(color.r, lit.x),
                shaded(color.g, lit.y),
                shaded(color.b, lit.z),
                color.a,
            ];
            for idx in [0usize, 1, 2, 0, 2, 3] {
                let l = face[idx];
                let world = center + rot * Vec3::new(l[0], l[1], l[2]);
                verts.push(DebugVertex {
                    pos: [world.x, world.y, world.z],
                    color: c,
                });
            }
        }
    }

    /// A flat, translucent decal centred at `center` in the plane perpendicular
    /// to `normal` (a contact shadow). `radius` is the half-width of the square;
    /// `color`'s alpha controls darkness. `normal` must be a unit vector. `+Y`
    /// is the historical ground quad. Drawn with the blended, depth-read-only
    /// debug pipeline, so it blends over terrain without occluding geometry
    /// behind it. No sun offset: a contact/AO blob sits on its owner.
    pub fn draw_shadow(&mut self, center: Vec3, normal: Vec3, radius: f32, color: Color) {
        let c = [color.r, color.g, color.b, color.a];
        let corners = shadow_corners(center, normal, radius);
        let verts = &mut self.frame.eng.lists.shadow_verts;
        for idx in [0usize, 1, 2, 0, 2, 3] {
            verts.push(DebugVertex {
                pos: corners[idx],
                color: c,
            });
        }
    }

    pub fn draw_cube_wires(&mut self, center: Vec3, size: Vec3, color: Color) {
        let min = center - size * 0.5;
        let max = center + size * 0.5;
        let c = [color.r, color.g, color.b, color.a];
        let corners = [
            [min.x, min.y, min.z],
            [max.x, min.y, min.z],
            [max.x, min.y, max.z],
            [min.x, min.y, max.z],
            [min.x, max.y, min.z],
            [max.x, max.y, min.z],
            [max.x, max.y, max.z],
            [min.x, max.y, max.z],
        ];
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        let verts = &mut self.frame.eng.lists.line_verts;
        for (a, b) in EDGES {
            verts.push(DebugVertex {
                pos: corners[a],
                color: c,
            });
            verts.push(DebugVertex {
                pos: corners[b],
                color: c,
            });
        }
    }
}

fn push_quad_2d(
    verts: &mut Vec<Vertex2D>,
    top_left: [f32; 2],
    bottom_right: [f32; 2],
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    color: Color,
) {
    let c = [color.r, color.g, color.b, color.a];
    let (x0, y0) = (top_left[0], top_left[1]);
    let (x1, y1) = (bottom_right[0], bottom_right[1]);
    let (u0, v0) = (uv_min[0], uv_min[1]);
    let (u1, v1) = (uv_max[0], uv_max[1]);
    let quad = [
        Vertex2D {
            pos: [x0, y0],
            uv: [u0, v0],
            color: c,
        },
        Vertex2D {
            pos: [x0, y1],
            uv: [u0, v1],
            color: c,
        },
        Vertex2D {
            pos: [x1, y1],
            uv: [u1, v1],
            color: c,
        },
    ];
    verts.extend_from_slice(&quad);
    let quad = [
        Vertex2D {
            pos: [x0, y0],
            uv: [u0, v0],
            color: c,
        },
        Vertex2D {
            pos: [x1, y1],
            uv: [u1, v1],
            color: c,
        },
        Vertex2D {
            pos: [x1, y0],
            uv: [u1, v0],
            color: c,
        },
    ];
    verts.extend_from_slice(&quad);
}

/// Corner lists per face, wound CCW as seen from outside the cube.
fn cube_faces(min: Vec3, max: Vec3) -> [[[f32; 3]; 4]; 6] {
    [
        // +Y (top)
        [
            [min.x, max.y, min.z],
            [min.x, max.y, max.z],
            [max.x, max.y, max.z],
            [max.x, max.y, min.z],
        ],
        // -Y (bottom)
        [
            [min.x, min.y, min.z],
            [max.x, min.y, min.z],
            [max.x, min.y, max.z],
            [min.x, min.y, max.z],
        ],
        // +X
        [
            [max.x, min.y, min.z],
            [max.x, max.y, min.z],
            [max.x, max.y, max.z],
            [max.x, min.y, max.z],
        ],
        // -X
        [
            [min.x, min.y, min.z],
            [min.x, min.y, max.z],
            [min.x, max.y, max.z],
            [min.x, max.y, min.z],
        ],
        // +Z
        [
            [min.x, min.y, max.z],
            [max.x, min.y, max.z],
            [max.x, max.y, max.z],
            [min.x, max.y, max.z],
        ],
        // -Z
        [
            [min.x, min.y, min.z],
            [min.x, max.y, min.z],
            [max.x, max.y, min.z],
            [max.x, min.y, min.z],
        ],
    ]
}

/// The directional key light used to shade oriented boxes ([`Frame3D::draw_box`]).
/// It is the single typed source of avatar shading: derived from the per-frame
/// UBO (`frame_uniforms`) — the SAME lighting truth the terrain reads — so
/// a peer and the terrain around it can never be lit inconsistently. `sun`/
/// `ambient` are per-channel RGB multipliers; `dir` points toward the light.
#[derive(Clone, Copy)]
pub(crate) struct KeyLight {
    dir: Vec3,
    sun: Vec3,
    ambient: Vec3,
}

impl KeyLight {
    /// Look with no sky set (e.g. `bin/demo.rs`): a fixed overhead key with an
    /// ambient floor, matching the box shading before sky-matching landed.
    const DEFAULT: KeyLight = KeyLight {
        dir: Vec3::new(0.35, 0.85, 0.38),
        sun: Vec3::splat(0.55),
        ambient: Vec3::splat(0.55),
    };

    /// Derive the key from this frame's composed lighting UBO. Formula mirrors
    /// `frame_snapshot::compose`/`legacy_env` exactly: `sun` is the linear `light`
    /// lane; `ambient` is `zenith` re-scaled to the `ambient_floor` luma (the
    /// `candle.w` lane). Clamped to [0,1] to reproduce the old
    /// `Rgb::to_srgb8_legacy` exit, which truncated linear values to 8-bit with
    /// NO sRGB curve (so the retarget is pixel-identical up to ±1/255). With no
    /// uniforms set (e.g. `bin/demo.rs`) fall back to [`KeyLight::DEFAULT`].
    fn from_uniforms(u: FrameUniformsGpu) -> Self {
        let sun = Vec3::new(u.light[0], u.light[1], u.light[2]).clamp(Vec3::ZERO, Vec3::ONE);
        let zenith = Vec3::new(u.zenith[0], u.zenith[1], u.zenith[2]);
        let ambient_floor = u.candle[3];
        let luma = 0.2126 * zenith.x + 0.7152 * zenith.y + 0.0722 * zenith.z;
        let ambient = if luma > 0.0 {
            zenith * (ambient_floor / luma)
        } else {
            zenith
        };
        let dir = Vec3::new(u.sun_dir_elev[0], u.sun_dir_elev[1], u.sun_dir_elev[2]);
        KeyLight {
            dir: dir
                .try_normalize()
                .or_else(|| Self::DEFAULT.dir.try_normalize())
                .unwrap_or(Vec3::Y),
            sun,
            ambient: ambient.clamp(Vec3::ZERO, Vec3::ONE),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stars gain rides `extras.x`; the flag gate must zero exactly that
    /// channel. `extras.yz` are engine-derived glow terms (filled by
    /// `prepare_derived`); `w` stays reserved zero.
    #[test]
    fn stars_gate_zeroes_extras_x() {
        let mut u = FrameUniformsGpu::full_bright();
        u.extras = [1.0, 0.0, 0.0, 0.0];
        let on = gate_uniforms(&crate::engine::RenderFlags::default(), u);
        assert_eq!(on.extras[0], 1.0);
        assert!((on.extras[1] - crate::genconst::GLOW_POW_DAY).abs() < 1e-5);
        assert!((on.extras[2] - 0.5).abs() < 1e-5);
        assert_eq!(on.extras[3], 0.0);
        let flags = crate::engine::RenderFlags {
            stars: false,
            ..Default::default()
        };
        let off = gate_uniforms(&flags, u);
        assert_eq!(off.extras[0], 0.0);
        assert!((off.extras[1] - crate::genconst::GLOW_POW_DAY).abs() < 1e-5);
    }

    #[test]
    fn shadow_quad_on_plus_y_matches_the_ground_corners() {
        let c = Vec3::new(1.0, 2.0, 3.0);
        let r = 4.0;
        assert_eq!(
            shadow_corners(c, Vec3::Y, r),
            [
                [c.x - r, c.y, c.z - r],
                [c.x - r, c.y, c.z + r],
                [c.x + r, c.y, c.z + r],
                [c.x + r, c.y, c.z - r],
            ]
        );
    }

    #[test]
    fn shadow_quad_lies_in_the_plane_of_its_normal() {
        let center = Vec3::new(3.0, -1.0, 2.0);
        let radius = 1.5;
        let normals = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec3::new(0.2, -0.9, 0.3).normalize(),
        ];
        for normal in normals {
            let corners = shadow_corners(center, normal, radius);
            let p = |i: usize| Vec3::from_array(corners[i]);
            let n = (p(1) - p(0)).cross(p(2) - p(0)).normalize();
            assert!(
                (n - normal).length() < 1e-5,
                "normal {normal:?} wound to {n:?}"
            );
            for corner in corners {
                let d = (Vec3::from_array(corner) - center).dot(normal);
                assert!(d.abs() < 1e-5, "normal {normal:?} corner off plane by {d}");
            }
        }
    }

    #[test]
    fn sun_override_replaces_direction_and_colour_and_hides_the_disc() {
        fn same_gpu(a: FrameUniformsGpu, b: FrameUniformsGpu) {
            assert_eq!(bytemuck::bytes_of(&a), bytemuck::bytes_of(&b));
        }
        fn same_sky(a: SkyDesc, b: SkyDesc) {
            assert_eq!(a.sun_dir, b.sun_dir);
            assert_eq!(a.sun_tint, b.sun_tint);
            assert_eq!(a.sun_angular_radius.to_bits(), b.sun_angular_radius.to_bits());
        }

        let mut base = FrameUniformsGpu::full_bright();
        // Night, so an override that forgot `light.w = 1` would flip the direction.
        base.sun_dir_elev = [0.0, -1.0, 0.0, -1.0];
        base.light = [0.05, 0.06, 0.1, 0.0];
        base.zenith = [0.2, 0.3, 0.5, 0.1];
        base.horizon = [0.4, 0.45, 0.5, 0.02];
        base.candle = [1.0, 0.8, 0.4, 0.25];
        base.exposure_dither = [1.2, 0.0, 0.1, -0.2];
        base.extras = [0.7, 0.0, 0.0, 0.0];
        base.anim = [3.0, 4.0, 5.0, 6.0];
        base.prepare_derived();

        let over = SunOverride {
            dir: Vec3::new(0.0, 2.0, 0.0),
            color: LinearRgb([1.15, 0.40, 0.07]),
            show_disc: false,
        };
        let lit = apply_sun_override(base, over, Vec3::Y);
        assert_eq!(lit.sun_dir_elev[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(lit.sun_dir_elev[1].to_bits(), 1.0f32.to_bits());
        assert_eq!(lit.sun_dir_elev[2].to_bits(), 0.0f32.to_bits());
        assert_eq!(lit.sun_dir_elev[3].to_bits(), 1.0f32.to_bits());
        assert_eq!(lit.light[0].to_bits(), 1.15f32.to_bits());
        assert_eq!(lit.light[1].to_bits(), 0.40f32.to_bits());
        assert_eq!(lit.light[2].to_bits(), 0.07f32.to_bits());
        assert_eq!(lit.light[3].to_bits(), 1.0f32.to_bits());
        assert_eq!(lit.zenith, base.zenith);
        assert_eq!(lit.horizon, base.horizon);
        assert_eq!(lit.candle, base.candle);
        assert_eq!(lit.exposure_dither, base.exposure_dither);
        assert_eq!(lit.anim, base.anim);
        assert_eq!(lit.extras[0].to_bits(), base.extras[0].to_bits());
        assert_eq!(lit.extras[2].to_bits(), base.extras[2].to_bits());
        assert_eq!(lit.extras[3].to_bits(), base.extras[3].to_bits());
        assert_eq!(lit.extras[1].to_bits(), crate::genconst::GLOW_POW_DAY.to_bits());

        let nan_dir = SunOverride {
            dir: Vec3::new(f32::NAN, 0.0, 0.0),
            ..over
        };
        same_gpu(apply_sun_override(base, nan_dir, Vec3::Y), base);
        same_gpu(
            apply_sun_override(base, SunOverride { dir: Vec3::ZERO, ..over }, Vec3::Y),
            base,
        );
        let nan_color = SunOverride {
            color: LinearRgb([f32::NAN, 0.0, 0.0]),
            ..over
        };
        same_gpu(apply_sun_override(base, nan_color, Vec3::Y), base);

        let side = apply_sun_override(
            base,
            SunOverride {
                dir: Vec3::new(4.0, 0.0, 0.0),
                ..over
            },
            Vec3::Y,
        );
        assert_eq!(side.sun_dir_elev[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(side.sun_dir_elev[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(side.sun_dir_elev[2].to_bits(), 0.0f32.to_bits());
        assert_eq!(side.sun_dir_elev[3].to_bits(), 0.0f32.to_bits());

        let desc = SkyDesc {
            sun_dir: Vec3::new(0.0, 2.0, 0.0),
            sun_tint: LinearRgb([0.9, 0.8, 0.7]),
            sun_angular_radius: 0.03,
        };
        same_sky(sky_with_override(desc, None), desc);
        let hidden = sky_with_override(desc, Some(over));
        assert_eq!(hidden.sun_dir, Vec3::Y);
        assert_eq!(hidden.sun_tint, LinearRgb([0.0, 0.0, 0.0]));
        assert_eq!(
            hidden.sun_angular_radius.to_bits(),
            desc.sun_angular_radius.to_bits()
        );
        let shown = sky_with_override(
            desc,
            Some(SunOverride {
                show_disc: true,
                ..over
            }),
        );
        assert_eq!(shown.sun_dir, Vec3::Y);
        assert_eq!(shown.sun_tint, over.color);
        assert_eq!(
            shown.sun_angular_radius.to_bits(),
            desc.sun_angular_radius.to_bits()
        );
        same_sky(sky_with_override(desc, Some(nan_dir)), desc);

        let mut lists = DrawLists::new();
        assert!(lists.sun_override.is_none());
        lists.sun_override = Some(over);
        lists.reset();
        assert!(lists.sun_override.is_none());
        // No scene: the override is not consulted, and the filler stays full-bright.
        lists.sun_override = Some(over);
        same_gpu(lists.lit_uniforms(), FrameUniformsGpu::full_bright());
    }

    #[test]
    fn lod_morph_packs_the_eye_split_and_reset_clears_it() {
        let mut lists = DrawLists::new();
        assert_eq!(lists.lod_morph.count, 0);
        assert_eq!(
            LodMorphFrame::from_api(DVec3::new(1.0, 2.0, 3.0), &[]),
            LodMorphFrame::off()
        );
        let eye = DVec3::new(1_200_000_000.25, -8.5, 3.0);
        let mut bands = vec![
            LodMorph {
                half: Vec3::new(10.0, 20.0, 30.0),
                start: 0.25,
            };
            17
        ];
        bands[1].half = Vec3::new(0.0, 1.0, 1.0);
        bands[15].start = 0.1;
        bands[16].start = 0.9;
        lists.lod_morph = LodMorphFrame::from_api(eye, &bands);
        assert_eq!(lists.lod_morph.count, 16);
        assert_eq!(lists.lod_morph.bands[15].start, 0.1);
        let gpu = lists.lod_morph.to_gpu();
        let split = EyeSplit::of(eye);
        assert_eq!(
            gpu.eye_block,
            [split.block[0], split.block[1], split.block[2], LOD_MORPH_FLAG]
        );
        assert_eq!(gpu.eye_frac[0].to_bits(), split.frac[0].to_bits());
        assert_eq!(gpu.eye_frac[1].to_bits(), split.frac[1].to_bits());
        assert_eq!(gpu.eye_frac[2].to_bits(), split.frac[2].to_bits());
        assert_eq!(gpu.eye_frac[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(gpu.bands[0], [10.0, 20.0, 30.0, 0.25]);
        assert_eq!(gpu.bands[1][0].to_bits(), 0.0f32.to_bits());
        assert_eq!(gpu.bands[15][3].to_bits(), 0.1f32.to_bits());
        lists.lod_morph = LodMorphFrame::from_api(eye, &[]);
        assert_eq!(lists.lod_morph.to_gpu(), LodMorphGpu::default());
        lists.lod_morph = LodMorphFrame::from_api(eye, &bands);
        lists.reset();
        assert_eq!(lists.lod_morph, LodMorphFrame::off());
        assert_eq!(lists.lod_morph.to_gpu().eye_block[3], 0);
    }
}
