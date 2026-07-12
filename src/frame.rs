/// Per-frame draw recording: `Frame` (2D overlay + frame lifecycle) and
/// `Frame3D` (world rendering inside a `begin_3d` scope). Everything records
/// into reused CPU lists; submission happens when the `Frame` drops.
use glam::{DVec3, Mat3, Mat4, Vec2, Vec3};

use std::num::NonZeroU32;

use crate::camera::{Aspect, Camera3D, Frustum, WarpMap};
use crate::color::{Color, LinearRgb};
use crate::skeleton::{FrameUniformsGpu, JitterOffset};
use crate::engine::Engine;
use crate::font;
use crate::mesh::{DebugVertex, MeshHandle, Pass};
use crate::vk::pipeline::Vertex2D;

/// Sky-pass-private state, set by the app inside a `begin_3d` scope: sun
/// geometry plus the disc tint. The sky COLOURS (zenith/horizon gradient, fog
/// glow) are NOT here — the sky fragment reads them from the per-frame
/// `FrameUniforms` UBO, the SAME linear source the terrain fog reads, so the two
/// can never diverge (one source of truth for sky data). The engine adds the inverse
/// view-projection at record time, so the app never touches a matrix.
#[derive(Clone, Copy)]
pub struct SkyDesc {
    pub sun_dir: Vec3,
    /// Linear disc/glow tint (no OETF on this path); the analytic sun disc adds
    /// it around the sun direction.
    pub sun_tint: LinearRgb,
    /// Angular radius of the sun disc, in radians; the shader derives the disc's
    /// core/rim edge cosines from it (generated `SUN_DISC_*` cone constants).
    pub sun_angular_radius: f32,
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
    u
}

/// A fully-resolved mesh draw: identity + culling metadata + placement, with NO
/// Vulkan handle. Recorded on the main thread (culling reads the main-owned
/// [`HandleAllocator`](crate::vk::buffers::HandleAllocator) metadata) and
/// consumed by the render path, which resolves `slot`/`generation` against its
/// residency mirror. `Send` POD so the whole [`DrawLists`] crosses threads.
#[derive(Clone, Copy)]
pub(crate) struct MeshDraw {
    /// Validated against the render-side generation mirror at resolve time.
    pub slot: u32,
    pub generation: NonZeroU32,
    /// World-space (local) AABB, for the render-side six-way face cull.
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub bounds: [u32; 7],
    pub vertex_offset: i32,
    pub pass: Pass,
    pub offset: Vec3,
    pub scale: f32,
}

/// CPU-side draw lists for one frame. Vec capacities persist across frames.
///
/// `Clone` exists solely for deterministic capture ([`crate::screenshot_to`]):
/// the last submitted lists are retained so the blocking capture can re-present
/// the same scene until the readback PNG lands, rather than a blank frame.
#[derive(Clone)]
pub(crate) struct DrawLists {
    pub clear: LinearRgb,
    pub view_proj: Mat4,
    /// The current 3D scope's camera, retained so the render thread can fit the
    /// shadow cascades around this frame's frustum. `None`
    /// until `begin_3d`; `has_3d` gates its use.
    pub camera: Option<Camera3D>,
    /// This frame's lighting uniforms. Resolved from the required [`Lighting`]
    /// argument to [`Frame::begin_3d`], so it is `Some` for every 3D frame
    /// (`has_3d` implies this is set); `None` only on pure-2D frames, where the
    /// renderer writes a full-bright filler the mesh shaders never sample.
    /// Written into the per-frame UBO (set 0, binding 2) each frame; the single
    /// source of truth for shader-side sky, fog, candle/ambient, and shadow
    /// evaluation, and for avatar key lighting (see [`KeyLight`]).
    pub frame_uniforms: Option<FrameUniformsGpu>,
    /// Camera world position for the current 3D scope; feeds six-way face
    /// culling (which needs the camera in each mesh's local frame).
    pub cam_pos: Vec3,
    /// World-space position of the RENDER-SPACE ORIGIN (the rebase point), in
    /// f64 — the input TAA's reprojection translation derives from
    /// `camera_delta = prev_eye − eye`, f64 subtract, narrowed only inside
    /// `Reprojection::pack`). A camera-at-origin app passes its true eye here
    /// (its `cam_pos` is zero and carries NO translation — the "whole game
    /// jitters while moving" bug); an app drawing in absolute
    /// coordinates passes ZERO (its translation already lives in the view
    /// matrix; a nonzero value here would double-count it).
    pub eye: DVec3,
    /// `tan(fovy/2)` for the current 3D camera; the renderer derives the
    /// vertical focal length (`0.5·height/tan_half`) for VRS depth thresholding.
    pub fovy_tan_half: f32,
    /// Wide-FOV lens for the current 3D scope. `Identity` in rectilinear mode. The
    /// projection is already widened to the source frustum for `Active`; this
    /// carries the coefficients the tonemap resample uses to compress the periphery.
    pub warp_map: WarpMap,
    /// Sub-pixel camera jitter (PIXELS, ±0.5) for this 3D scope, injected here —
    /// the sole injection point. It rides to the renderer, which applies
    /// it to the mesh view-proj *only* at push-constant packing (a local matrix
    /// that never escapes, `vk::jittered_clip`); `view_proj` above stays CLEAN so
    /// culling, VRS fingerprinting, and TAA reprojection never see the jitter.
    /// `JitterOffset::ZERO` outside a 3D scope.
    pub jitter: JitterOffset,
    pub has_3d: bool,
    /// Procedural sky palette for this frame's background pass, or `None` to
    /// leave the flat clear colour showing. Set via [`Frame3D::set_sky`].
    pub sky: Option<SkyDesc>,
    /// Mesh draws (chunks and LOD); full-res wins the near ground via dither,
    /// not depth bias.
    pub mesh_draws: Vec<MeshDraw>,
    /// Chunk→LOD dither radius; full-res chunks own the near ground by default
    /// (0.0 disables).
    pub lod_clip: f32,
    pub cube_verts: Vec<DebugVertex>,
    pub line_verts: Vec<DebugVertex>,
    /// Translucent ground decals (contact shadows), drawn with the blended,
    /// depth-read-only debug pipeline after the opaque cubes.
    pub shadow_verts: Vec<DebugVertex>,
    pub verts_2d: Vec<Vertex2D>,
    /// Minimap verts (drawn by `tris2d_tex` pipeline).
    pub tex_verts_2d: Vec<Vertex2D>,
    /// Debug-flat override (`DebugView::TerrainKey`): when set,
    /// every 3D mesh fragment outputs this flat key colour while still writing
    /// depth, so the sky-hole detector sees real terrain coverage (key) versus
    /// the magenta clear. `None` renders normally. Rides the per-frame UBO's
    /// `reserved` lane. Set via [`Frame3D::set_debug_flat`].
    pub debug_flat: Option<Color>,
}

impl DrawLists {
    pub fn new() -> Self {
        Self {
            clear: LinearRgb([0.0, 0.0, 0.0]),
            view_proj: Mat4::IDENTITY,
            camera: None,
            frame_uniforms: None,
            cam_pos: Vec3::ZERO,
            eye: DVec3::ZERO,
            fovy_tan_half: 1.0,
            warp_map: WarpMap::Identity,
            jitter: JitterOffset::ZERO,
            has_3d: false,
            sky: None,
            mesh_draws: Vec::new(),
            lod_clip: 0.0,
            cube_verts: Vec::new(),
            line_verts: Vec::new(),
            shadow_verts: Vec::new(),
            verts_2d: Vec::new(),
            tex_verts_2d: Vec::new(),
            debug_flat: None,
        }
    }

    pub fn reset(&mut self) {
        self.has_3d = false;
        self.camera = None;
        self.sky = None;
        self.warp_map = WarpMap::Identity;
        self.jitter = JitterOffset::ZERO;
        self.frame_uniforms = None;
        self.mesh_draws.clear();
        self.lod_clip = 0.0;
        self.cube_verts.clear();
        self.line_verts.clear();
        self.shadow_verts.clear();
        self.verts_2d.clear();
        self.tex_verts_2d.clear();
        self.debug_flat = None;
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
    /// (see [`DrawLists::eye`]): a camera-at-origin app passes its true eye
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
        let frustum = Frustum::from_view_proj(&view_proj);
        self.eng.lists.view_proj = view_proj;
        self.eng.lists.camera = Some(*cam);
        self.eng.lists.cam_pos = cam.position;
        self.eng.lists.eye = eye;
        self.eng.lists.fovy_tan_half = (cam.fovy.to_radians() * 0.5).tan();
        self.eng.lists.warp_map = warp_map;
        // Advance the temporal jitter sequence once per 3D scope. The
        // renderer's own `frame_index` drives reprojection/dither on the render
        // thread; here on the record thread we keep an independent 1:1 sequence
        // counter (each recorded frame submits exactly one `draw_frame`), so the
        // mesh view-proj sees a fresh Halton offset every frame. The value only
        // travels to `jittered_clip`; it never perturbs the clean `view_proj`.
        let seq = JITTER_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // taa off: no jitter AND no resolve, one coupled state — the same
        // `flags.taa` gates the resolve pass (vk/mod.rs); never toggle one
        // without the other (jitter without resolve is permanent shimmer).
        self.eng.lists.jitter = if self.eng.flags.taa {
            crate::skeleton::jitter_at(seq)
        } else {
            JitterOffset::ZERO
        };
        self.eng.lists.has_3d = true;
        // Resolve lighting to a concrete UBO here, at the single 3D entry point,
        // so `has_3d` and `frame_uniforms.is_some()` are established together —
        // a 3D frame always carries lighting. `Composed` runs through the env
        // feature gates; `FullBright` is the fixed lit neutral.
        self.eng.lists.frame_uniforms = Some(match light {
            Lighting::Composed(u) => gate_uniforms(&self.eng.flags, u),
            Lighting::FullBright => FrameUniformsGpu::full_bright(),
        });
        Frame3D {
            frame: self,
            frustum,
        }
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
        self.eng.lists.tex_verts_2d.extend_from_slice(&[tl, bl, br, tl, br, tr]);
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

pub struct Frame3D<'f, 'e> {
    frame: &'f mut Frame<'e>,
    frustum: Frustum,
}

impl Frame3D<'_, '_> {
    /// Draws an uploaded mesh scaled by `scale` about its local origin then
    /// translated by `offset` (both applied GPU-side, so meshes stay in small
    /// local coordinates for camera-relative rendering). `scale` is `1.0` for a
    /// near chunk and `2^k` for a coarser LOD tile. Skipped automatically when
    /// its scaled, translated AABB is outside the view frustum.
    pub fn draw_mesh(&mut self, handle: MeshHandle, offset: Vec3, scale: f32) {
        self.push_mesh(handle, offset, scale);
    }

    /// Sets the chunk→LOD dither radius so full-res chunks own the near ground.
    /// Call once per `begin_3d` with render distance; 0.0 disables.
    pub fn set_lod_clip(&mut self, radius: f32) {
        self.frame.eng.lists.lod_clip = radius.max(0.0);
    }

    fn push_mesh(&mut self, handle: MeshHandle, offset: Vec3, scale: f32) {
        let Some(meta) = self.frame.eng.client.mesh_meta(handle) else {
            return;
        };
        let (aabb_min, aabb_max) = meta.aabb();
        // The scale must thread into the frustum test too: a scaled tile's
        // world AABB is `local * scale + offset`, and using the unscaled box
        // would mis-cull it near the view edges.
        if !self
            .frustum
            .intersects_aabb(aabb_min * scale + offset, aabb_max * scale + offset)
        {
            return;
        }
        self.frame.eng.lists.mesh_draws.push(MeshDraw {
            slot: handle.slot,
            generation: handle.generation,
            aabb_min,
            aabb_max,
            bounds: meta.bounds,
            vertex_offset: meta.vertex_offset,
            pass: meta.pass,
            offset,
            scale,
        });
    }

    /// Sets the procedural sky drawn behind this frame's geometry. The
    /// background pass shades only pixels the terrain did not cover (a
    /// reversed-Z depth trick), so it is near-free. Call once inside the
    /// `begin_3d` scope; leaving it unset shows the flat clear colour.
    pub fn set_sky(&mut self, desc: SkyDesc) {
        self.frame.eng.lists.sky = Some(desc);
    }

    /// Debug-flat override (`DebugView::TerrainKey`): `Some(key)`
    /// makes every 3D mesh fragment output `key` while still writing depth, so
    /// occlusion/silhouette stay exact and the sky-hole detector distinguishes
    /// real terrain coverage from the magenta clear. `None` restores normal
    /// shading. Rides the per-frame UBO's `reserved` lane (no push-constant or
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
        let key = KeyLight::from_lists(&self.frame.eng.lists);
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
            let n = (rot * local_n).normalize_or_zero();
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

    /// A flat, translucent ground decal centred at `center` (a contact shadow).
    /// `radius` is the half-width of the square blob; `color`'s alpha controls
    /// darkness. Drawn with the blended, depth-read-only debug pipeline, so it
    /// blends over terrain without occluding geometry behind it. No sun offset:
    /// a contact/AO blob sits directly under its owner.
    pub fn draw_shadow(&mut self, center: Vec3, radius: f32, color: Color) {
        let c = [color.r, color.g, color.b, color.a];
        // A single ground quad in the XZ plane at `center.y`, wound CCW from above.
        let corners = [
            [center.x - radius, center.y, center.z - radius],
            [center.x - radius, center.y, center.z + radius],
            [center.x + radius, center.y, center.z + radius],
            [center.x + radius, center.y, center.z - radius],
        ];
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
struct KeyLight {
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
    fn from_lists(lists: &DrawLists) -> Self {
        match lists.frame_uniforms {
            Some(u) => {
                let sun = Vec3::new(u.light[0], u.light[1], u.light[2]).clamp(Vec3::ZERO, Vec3::ONE);
                let zenith = Vec3::new(u.zenith[0], u.zenith[1], u.zenith[2]);
                let ambient_floor = u.candle[3];
                let luma = 0.2126 * zenith.x + 0.7152 * zenith.y + 0.0722 * zenith.z;
                let ambient = if luma > 0.0 { zenith * (ambient_floor / luma) } else { zenith };
                let dir = Vec3::new(u.sun_dir_elev[0], u.sun_dir_elev[1], u.sun_dir_elev[2]);
                KeyLight {
                    dir: dir.normalize_or(Self::DEFAULT.dir),
                    sun,
                    ambient: ambient.clamp(Vec3::ZERO, Vec3::ONE),
                }
            }
            None => Self::DEFAULT,
        }
    }
}
