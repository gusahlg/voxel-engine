/// CPU-side mesh data and the opaque handle to its GPU copy.
///
/// The engine is unlit: a vertex is a world-space position plus an RGBA8
/// color (16 bytes). Lighting/shading is baked into vertex colors by the
/// caller (the game bakes per-face directional shade).
use bytemuck::{Pod, Zeroable};

use crate::color::Color;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [u8; 4],
}

impl Vertex {
    pub fn new(pos: [f32; 3], color: Color) -> Self {
        Self {
            pos,
            color: [color.r, color.g, color.b, color.a],
        }
    }
}

/// Triangle mesh with u32 indices. Reusable as a scratch buffer: `clear`
/// keeps the allocations.
#[derive(Default)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Generational handle to a GPU mesh. Cheap to copy; freeing is explicit via
/// `Engine::free_mesh` (deferred internally until the GPU is done with it).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshHandle {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}
