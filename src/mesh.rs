/// CPU-side mesh data and the opaque handle to its GPU copy.
///
/// The engine is unlit but textured: a vertex is a world-space position, a
/// texture UV, and an RGBA8 color (24 bytes). `color.rgb` multiplies the
/// sampled texel (the game bakes per-face directional shade into it);
/// `color.a` is NOT alpha — it is the block-texture-array LAYER index
/// (the 3D pipeline never blends). Layer 0 is guaranteed all-white, so
/// `uv == [0,0]` + `color.a == 0` renders plain flat vertex color.
use bytemuck::{Pod, Zeroable};

use crate::color::Color;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    /// rgb = color multiplier, a = texture array layer (not alpha).
    pub color: [u8; 4],
}

impl Vertex {
    /// Flat-colored vertex: uv `[0,0]`, `color.a` passed through as the
    /// texture layer. For pure flat color the caller must supply
    /// `color.a == 0` (layer 0 is always white).
    pub fn new(pos: [f32; 3], color: Color) -> Self {
        Self {
            pos,
            uv: [0.0, 0.0],
            color: [color.r, color.g, color.b, color.a],
        }
    }

    /// Textured vertex: `rgb_shade` multiplies the sampled texel, `layer`
    /// selects the block-texture-array layer.
    pub fn textured(pos: [f32; 3], uv: [f32; 2], rgb_shade: [u8; 3], layer: u8) -> Self {
        Self {
            pos,
            uv,
            color: [rgb_shade[0], rgb_shade[1], rgb_shade[2], layer],
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

#[cfg(test)]
mod tests {
    use super::Vertex;

    #[test]
    fn vertex_is_24_bytes_no_padding() {
        assert_eq!(std::mem::size_of::<Vertex>(), 24);
        assert_eq!(std::mem::offset_of!(Vertex, pos), 0);
        assert_eq!(std::mem::offset_of!(Vertex, uv), 12);
        assert_eq!(std::mem::offset_of!(Vertex, color), 20);
    }

    #[test]
    fn textured_packs_layer_in_alpha() {
        let v = Vertex::textured([1.0, 2.0, 3.0], [0.5, 0.25], [10, 20, 30], 7);
        assert_eq!(v.color, [10, 20, 30, 7]);
        assert_eq!(v.uv, [0.5, 0.25]);
    }

    #[test]
    fn new_zeroes_uv() {
        let v = Vertex::new([0.0; 3], crate::color::Color::new(1, 2, 3, 0));
        assert_eq!(v.uv, [0.0, 0.0]);
        assert_eq!(v.color, [1, 2, 3, 0]);
    }
}
