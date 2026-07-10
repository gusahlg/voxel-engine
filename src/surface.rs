/// CPU-side colored-surface data and the opaque handle to its GPU copy.
///
/// A [`SurfaceVertex`] is 16 bytes UNPACKED: a world-space (camera-relative)
/// position and an RGBA8 colour. Unlike the 8-byte packed voxel
/// [`MeshVertex`](crate::MeshVertex) — 5-bit local coords, no RGB, 6 fixed
/// normals — a surface vertex can express an arbitrary world-Y and a free
/// colour, which is exactly what a continuous grey height surface (the Zone-3
/// far backdrop) needs. Drawn camera-relative through the `surface3d` pipeline,
/// reusing the per-draw offset SSBO + indirect machinery of the voxel path.
use crate::vk::vertex_input::vertex_struct;

vertex_struct! {
    /// 16-byte unpacked surface vertex: position (f32×3) + RGBA8 colour. The
    /// `vertex_struct!` macro derives `#[repr(C)]`, `Pod`/`Zeroable`, and the
    /// GPU `binding()`/`ATTRIBUTES`; 12 + 4 = 16 bytes, no padding.
    pub struct SurfaceVertex {
        pub pos: [f32; 3],
        pub color: [u8; 4],
    }
}

/// A retained colored triangle mesh: one vertex list and one index list (no
/// six-way face buckets — a surface is opaque and single-pass). Built one quad
/// at a time via [`Self::quad`], mirroring [`MeshData::quad`](crate::MeshData).
pub struct SurfaceData {
    pub(crate) verts: Vec<SurfaceVertex>,
    pub(crate) indices: Vec<u32>,
}

impl SurfaceData {
    /// An empty surface.
    pub fn new() -> Self {
        Self {
            verts: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Appends one quad: four corners wound CCW as seen from outside → four
    /// vertices plus six indices (`0,1,2` + `0,2,3`). The `surface3d` pipeline
    /// is double-sided (`cull: NONE`), so winding decides only the shade normal,
    /// not visibility.
    pub fn quad(&mut self, corners: [SurfaceVertex; 4]) {
        let base = self.verts.len() as u32;
        self.verts.extend_from_slice(&corners);
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn is_empty(&self) -> bool {
        self.verts.is_empty()
    }

    /// The vertices, for dependent-crate tests that assert on emitted geometry
    /// (crack-free shared edges, outward winding). `#[doc(hidden)]` and
    /// read-only — build geometry with [`Self::quad`].
    #[doc(hidden)]
    pub fn verts(&self) -> &[SurfaceVertex] {
        &self.verts
    }

    /// The index list, paired with [`Self::verts`] for dependent-crate tests.
    #[doc(hidden)]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
}

impl Default for SurfaceData {
    fn default() -> Self {
        Self::new()
    }
}

/// Generational handle to a GPU surface mesh. Distinct type from
/// [`MeshHandle`](crate::MeshHandle) so the two registries can't be crossed;
/// fields are minted only by the surface registry. Freeing is explicit via
/// `Engine::free_surface` (deferred internally until the GPU is done with it).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SurfaceHandle {
    pub(crate) slot: u32,
    pub(crate) generation: std::num::NonZeroU32,
}

impl SurfaceHandle {
    /// Construct a handle from raw parts. For dependent-crate tests that need to
    /// populate handle-carrying state without a live GPU — the fields are
    /// otherwise crate-private. NOT for production use (mirrors
    /// [`MeshHandle::from_raw_parts`](crate::MeshHandle::from_raw_parts)).
    ///
    /// `generation` is 1-based (0 is the reserved niche); passing 0 panics.
    #[doc(hidden)]
    pub fn from_raw_parts(index: u32, generation: u32) -> Self {
        Self {
            slot: index,
            generation: std::num::NonZeroU32::new(generation)
                .expect("generation is 1-based"),
        }
    }
}
