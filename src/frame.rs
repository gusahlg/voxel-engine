/// Per-frame draw recording: `Frame` (2D overlay + frame lifecycle) and
/// `Frame3D` (world rendering inside a `begin_3d` scope). Everything records
/// into reused CPU lists; submission happens when the `Frame` drops.
use glam::{Mat3, Mat4, Vec2, Vec3};

use std::num::NonZeroU32;

use crate::camera::{Aspect, Camera3D, Frustum, WarpMap};
use crate::color::Color;
use crate::engine::Engine;
use crate::font;
use crate::mesh::{DebugVertex, MeshHandle, Pass};
use crate::surface::SurfaceHandle;
use crate::vk::pipeline::{SkyLight, Vertex2D};

/// Palette-only description of the procedural sky, set by the app inside a
/// `begin_3d` scope. The engine composes the GPU push constant (adding the
/// inverse view-projection and deriving the sun-disc cosines) at record time,
/// so the app never touches a matrix. Colours are the two anchors the fragment
/// shader interpolates by ray elevation — sampled from the app's single
/// radiance source of truth, not re-derived on the GPU.
#[derive(Clone, Copy)]
pub struct SkyDesc {
    pub sun_dir: Vec3,
    /// Sky colour looking straight up.
    pub zenith: Color,
    /// Sky colour at the horizon (matches terrain fog for a seamless edge).
    pub horizon: Color,
    /// Warm glow colour smeared along the horizon toward the sun's azimuth.
    pub sun_tint: Color,
    /// Linear exposure multiplier applied to the whole sky.
    pub exposure: f32,
    /// Angular radius of the sun disc, in radians; the engine derives the
    /// inner/outer edge cosines for the analytic disc from it.
    pub sun_angular_radius: f32,
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
    /// Depth-biased opaque pipeline (far-LOD tiles).
    pub biased: bool,
}

/// A fully-resolved colored-surface draw (Zone-3 far skin). Surface analogue of
/// [`MeshDraw`]; frustum-culled at record time so the render path only resolves
/// the buffer and issues the draw.
#[derive(Clone, Copy)]
pub(crate) struct SurfaceDraw {
    pub slot: u32,
    pub generation: NonZeroU32,
    pub index_first: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
    pub offset: Vec3,
    pub scale: f32,
}

/// CPU-side draw lists for one frame. Vec capacities persist across frames.
pub(crate) struct DrawLists {
    pub clear: Color,
    pub view_proj: Mat4,
    /// Sky lighting/fog for the mesh pipeline. The default [`SkyLight::IDENTITY`]
    /// (white sun, black ambient, fog density 0) renders exactly as an unlit
    /// scene (`light = shade·ao·max(sky,block)`, no fog).
    pub sky_light: SkyLight,
    /// Camera world position for the current 3D scope; feeds six-way face
    /// culling (which needs the camera in each mesh's local frame).
    pub cam_pos: Vec3,
    /// `tan(fovy/2)` for the current 3D camera; the renderer derives the
    /// vertical focal length (`0.5·height/tan_half`) for VRS depth thresholding.
    pub fovy_tan_half: f32,
    /// Wide-FOV lens for the current 3D scope. `Identity` in rectilinear mode. The
    /// projection is already widened to the source frustum for `Active`; this
    /// carries the coefficients the tonemap resample uses to compress the periphery.
    pub warp_map: WarpMap,
    pub has_3d: bool,
    /// Procedural sky palette for this frame's background pass, or `None` to
    /// leave the flat clear colour showing. Set via [`Frame3D::set_sky`].
    pub sky: Option<SkyDesc>,
    /// Each draw: mesh handle, camera-relative offset, and uniform scale
    /// (`1.0` for near chunks, `2^k` for LOD tiles). The `bool` selects the
    /// depth-biased opaque pipeline (far-LOD tiles), so full-res chunks win at
    /// coincident depth without z-fighting.
    pub mesh_draws: Vec<MeshDraw>,
    /// Retained colored-surface draws (Zone-3 far skin): resolved snapshot.
    /// Recorded after the opaque mesh runs, before sky.
    pub surface_draws: Vec<SurfaceDraw>,
    /// Horizontal radius (metres) within which the Zone-3 skin fragments are
    /// discarded, so the skin renders only BEYOND the near zones (chunks + LOD
    /// tiles) instead of poking through them. `0.0` (default) clips nothing.
    pub skin_clip: f32,
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
            clear: Color::BLACK,
            view_proj: Mat4::IDENTITY,
            sky_light: SkyLight::IDENTITY,
            cam_pos: Vec3::ZERO,
            fovy_tan_half: 1.0,
            warp_map: WarpMap::Identity,
            has_3d: false,
            sky: None,
            mesh_draws: Vec::new(),
            surface_draws: Vec::new(),
            skin_clip: 0.0,
            cube_verts: Vec::new(),
            line_verts: Vec::new(),
            shadow_verts: Vec::new(),
            verts_2d: Vec::new(),
            tex_verts_2d: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.has_3d = false;
        self.sky = None;
        self.warp_map = WarpMap::Identity;
        self.sky_light = SkyLight::IDENTITY;
        self.mesh_draws.clear();
        self.surface_draws.clear();
        self.skin_clip = 0.0;
        self.cube_verts.clear();
        self.line_verts.clear();
        self.shadow_verts.clear();
        self.verts_2d.clear();
        self.tex_verts_2d.clear();
    }
}

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
    pub fn begin_3d(&mut self, cam: &Camera3D) -> Frame3D<'_, 'e> {
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
        self.eng.lists.cam_pos = cam.position;
        self.eng.lists.fovy_tan_half = (cam.fovy.to_radians() * 0.5).tan();
        self.eng.lists.warp_map = warp_map;
        self.eng.lists.has_3d = true;
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

    pub fn draw_fps(&mut self, x: i32, y: i32) {
        let fps = self.eng.fps();
        let text = format!("{fps:2} FPS");
        self.draw_text(&text, x, y, 20, Color::LIME);
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
        self.push_mesh(handle, offset, scale, false);
    }

    /// Like [`draw_mesh`](Self::draw_mesh) but draws through the depth-biased
    /// opaque pipeline: fragments are pushed slightly toward the reversed-Z far
    /// plane so normal opaque meshes win at coincident depth. Used for far-LOD
    /// terrain tiles that sit behind full-res chunks over the same ground.
    pub fn draw_mesh_biased(&mut self, handle: MeshHandle, offset: Vec3, scale: f32) {
        self.push_mesh(handle, offset, scale, true);
    }

    /// Records a retained colored-surface draw (Zone-3 far skin) at `offset`
    /// with uniform `scale`, applied GPU-side so the surface stays in small
    /// camera-relative coordinates. Skipped when its scaled, translated AABB is
    /// outside the view frustum.
    /// Sets the horizontal radius (metres, camera-relative) within which the
    /// Zone-3 skin is discarded in the fragment shader, so the far grey skin
    /// renders only BEYOND the near zones (full-res chunks + LOD tiles) instead
    /// of poking through them. Call once inside the `begin_3d` scope with the
    /// tile ring's outer radius; `0.0` (default) draws the whole skin.
    pub fn set_skin_clip(&mut self, radius: f32) {
        self.frame.eng.lists.skin_clip = radius.max(0.0);
    }

    pub fn draw_surface(&mut self, handle: SurfaceHandle, offset: Vec3, scale: f32) {
        let Some(meta) = self.frame.eng.client.surface_meta(handle) else {
            return;
        };
        let (aabb_min, aabb_max) = meta.aabb();
        // Cull using the scaled AABB to avoid false culls at view edges.
        if !self
            .frustum
            .intersects_aabb(aabb_min * scale + offset, aabb_max * scale + offset)
        {
            return;
        }
        self.frame.eng.lists.surface_draws.push(SurfaceDraw {
            slot: handle.slot,
            generation: handle.generation,
            index_first: meta.index_first,
            index_count: meta.index_count,
            vertex_offset: meta.vertex_offset,
            offset,
            scale,
        });
    }

    fn push_mesh(&mut self, handle: MeshHandle, offset: Vec3, scale: f32, biased: bool) {
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
            biased,
        });
    }

    /// Sets the sky lighting and fog applied to every mesh drawn this frame.
    ///
    /// `sun_light` tints the sky-lit portion of each face, `ambient` is the
    /// unlit floor (block-lit torches are unaffected by either), and geometry
    /// fades toward `fog` with `exp(-distance · fog_density)`. Call once inside
    /// the `begin_3d` scope; the defaults (white sun, black ambient, zero
    /// density) reproduce the unlit look when never called.
    pub fn set_sky_light(&mut self, sun_light: Color, ambient: Color, fog: Color, fog_density: f32) {
        let n = |c: Color| [c.r, c.g, c.b].map(|v| v as f32 / 255.0);
        let [sr, sg, sb] = n(sun_light);
        let [ar, ag, ab] = n(ambient);
        let [fr, fg, fb] = n(fog);
        self.frame.eng.lists.sky_light = SkyLight {
            sun_light: [sr, sg, sb, 1.0],
            ambient: [ar, ag, ab, 1.0],
            fog: [fr, fg, fb, fog_density],
        };
    }

    /// Sets the procedural sky drawn behind this frame's geometry. The
    /// background pass shades only pixels the terrain did not cover (a
    /// reversed-Z depth trick), so it is near-free. Call once inside the
    /// `begin_3d` scope; leaving it unset shows the flat clear colour.
    pub fn set_sky(&mut self, desc: SkyDesc) {
        self.frame.eng.lists.sky = Some(desc);
    }

    pub fn draw_cube(&mut self, center: Vec3, size: Vec3, color: Color) {
        let min = center - size * 0.5;
        let max = center + size * 0.5;
        let c = [color.r, color.g, color.b, color.a];
        let verts = &mut self.frame.eng.lists.cube_verts;
        // 6 faces, each two triangles, corners wound CCW seen from outside.
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
/// It is the single typed source of avatar shading: derived from the frame's sky
/// state (the same `sky_light`/`sky` terrain reads), so a peer and the terrain
/// around it can never be lit inconsistently. `sun`/`ambient` are per-channel RGB
/// multipliers; `dir` points toward the light.
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

    /// Derive the key from the frame's sky. When a procedural sky is set, take its
    /// sun direction and the sky lighting colours the mesh pipeline uses; otherwise
    /// fall back to [`KeyLight::DEFAULT`].
    fn from_lists(lists: &DrawLists) -> Self {
        match lists.sky {
            Some(desc) => {
                let [sr, sg, sb, _] = lists.sky_light.sun_light;
                let [ar, ag, ab, _] = lists.sky_light.ambient;
                KeyLight {
                    dir: desc.sun_dir.normalize_or(Self::DEFAULT.dir),
                    sun: Vec3::new(sr, sg, sb),
                    ambient: Vec3::new(ar, ag, ab),
                }
            }
            None => Self::DEFAULT,
        }
    }
}
