/// Per-frame draw recording: `Frame` (2D overlay + frame lifecycle) and
/// `Frame3D` (world rendering inside a `begin_3d` scope). Everything records
/// into reused CPU lists; submission happens when the `Frame` drops.
use glam::{Mat4, Vec2, Vec3};

use crate::camera::{Camera3D, Frustum};
use crate::color::Color;
use crate::engine::Engine;
use crate::font;
use crate::mesh::{MeshHandle, Vertex};
use crate::vk::pipeline::Vertex2D;

/// CPU-side draw lists for one frame. Vec capacities persist across frames.
pub(crate) struct DrawLists {
    pub clear: Color,
    pub view_proj: Mat4,
    pub has_3d: bool,
    pub mesh_draws: Vec<MeshHandle>,
    pub cube_verts: Vec<Vertex>,
    pub line_verts: Vec<Vertex>,
    pub verts_2d: Vec<Vertex2D>,
}

impl DrawLists {
    pub fn new() -> Self {
        Self {
            clear: Color::BLACK,
            view_proj: Mat4::IDENTITY,
            has_3d: false,
            mesh_draws: Vec::new(),
            cube_verts: Vec::new(),
            line_verts: Vec::new(),
            verts_2d: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.has_3d = false;
        self.mesh_draws.clear();
        self.cube_verts.clear();
        self.line_verts.clear();
        self.verts_2d.clear();
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
        let extent = self.eng.renderer.extent();
        let aspect = extent.width.max(1) as f32 / extent.height.max(1) as f32;
        let view_proj = cam.view_proj(aspect);
        let frustum = Frustum::from_view_proj(&view_proj);
        self.eng.lists.view_proj = view_proj;
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
    /// Draws an uploaded mesh; skipped automatically when its AABB is
    /// outside the view frustum.
    pub fn draw_mesh(&mut self, handle: MeshHandle) {
        let Some((aabb_min, aabb_max)) = self.frame.eng.renderer.mesh_aabb(handle) else {
            return;
        };
        if !self.frustum.intersects_aabb(aabb_min, aabb_max) {
            return;
        }
        self.frame.eng.lists.mesh_draws.push(handle);
    }

    pub fn draw_cube(&mut self, center: Vec3, size: Vec3, color: Color) {
        let min = center - size * 0.5;
        let max = center + size * 0.5;
        let c = [color.r, color.g, color.b, color.a];
        let verts = &mut self.frame.eng.lists.cube_verts;
        // 6 faces, each two triangles, corners wound CCW seen from outside.
        for face in cube_faces(min, max) {
            for idx in [0usize, 1, 2, 0, 2, 3] {
                verts.push(Vertex {
                    pos: face[idx],
                    color: c,
                });
            }
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
            verts.push(Vertex {
                pos: corners[a],
                color: c,
            });
            verts.push(Vertex {
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
        Vertex2D { pos: [x0, y0], uv: [u0, v0], color: c },
        Vertex2D { pos: [x0, y1], uv: [u0, v1], color: c },
        Vertex2D { pos: [x1, y1], uv: [u1, v1], color: c },
    ];
    verts.extend_from_slice(&quad);
    let quad = [
        Vertex2D { pos: [x0, y0], uv: [u0, v0], color: c },
        Vertex2D { pos: [x1, y1], uv: [u1, v1], color: c },
        Vertex2D { pos: [x1, y0], uv: [u1, v0], color: c },
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
