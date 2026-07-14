/// CPU-side mesh data and the opaque handle to its GPU copy.
///
/// A [`MeshVertex`] is 8 bytes: two `u32`s packing a chunk-local integer
/// position, a face normal, a texture-array layer, and reserved AO/light bits.
/// The vertex stores NO uv and NO color — the shader derives uv from the
/// position + normal (the sampler is REPEAT, so greedy quads tile) and derives
/// per-face directional shade from the normal. See the bit table below; the
/// `SHIFT_*`/`MASK_*` consts are mirrored in `shaders/mesh3d.vert.slang`.
///
/// Word 0: x[0..5] y[5..10] z[10..15] normal[15..18] layer[18..32]
/// Word 1: ao[0..2] skylight[2..6] blocklight[6..10]
///
/// Immediate debug geometry uses the separate unpacked [`DebugVertex`].
use crate::vk::vertex_input::vertex_struct;

// Bit shifts, mirrored by the Slang unpack in mesh3d.vert.slang.
pub const SHIFT_X: u32 = 0;
pub const SHIFT_Y: u32 = 5;
pub const SHIFT_Z: u32 = 10;
pub const SHIFT_NORMAL: u32 = 15;
pub const SHIFT_LAYER: u32 = 18;
pub const SHIFT_AO: u32 = 0;
pub const SHIFT_SKY: u32 = 2;
pub const SHIFT_BLOCK: u32 = 6;
/// Water material bit in `w1` (bit 10). Set by mesher for liquid blocks;
/// read by mesh3d.frag to select animated water shading in transparent pass.
pub const SHIFT_WATER: u32 = 10;
/// Per-axis micro-offsets in `w1` (bits 11-17): -2..=1 values that nudge vertices
/// on double-covered LOD borders to break z-fighting. Default zero (no offset).
pub const SHIFT_MICRO_X: u32 = 11;
pub const SHIFT_MICRO_Y: u32 = 13;
pub const SHIFT_MICRO_Z: u32 = 15;
/// Two-bit mask for one micro-offset axis.
pub const MASK_MICRO: u32 = 0x3;

// Field masks (applied on pack so a debug-only out-of-range value can never
// corrupt an adjacent field; the typed API keeps values in range anyway).
const MASK_COORD: u32 = 0x1F; // 5 bits, holds 0..=16
const MASK_NORMAL: u32 = 0x7; // 3 bits
const MASK_LAYER: u32 = 0x3FFF; // 14 bits — word 0's remaining span, 16384 layers
const MASK_AO: u32 = 0x3; // 2 bits
const MASK_LIGHT: u32 = 0xF; // 4 bits

/// Face normal. The discriminant IS the 3-bit index stored in the vertex and
/// decoded by the shader: `0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z`.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Normal {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Normal {
    /// Unit direction as integer components.
    pub const fn direction(self) -> [i8; 3] {
        match self {
            Normal::PosX => [1, 0, 0],
            Normal::NegX => [-1, 0, 0],
            Normal::PosY => [0, 1, 0],
            Normal::NegY => [0, -1, 0],
            Normal::PosZ => [0, 0, 1],
            Normal::NegZ => [0, 0, -1],
        }
    }

    /// Inverse of the discriminant index. The 3-bit field only ever holds a
    /// value written from a [`Normal`], so the panic arm is unreachable in
    /// practice; it exists so an out-of-range decode fails loudly.
    pub(crate) const fn from_index(i: u8) -> Normal {
        match i {
            0 => Normal::PosX,
            1 => Normal::NegX,
            2 => Normal::PosY,
            3 => Normal::NegY,
            4 => Normal::PosZ,
            5 => Normal::NegZ,
            _ => panic!("normal index out of range"),
        }
    }
}

/// Ambient-occlusion level, `0..=3`. `NONE` (3) means no occlusion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ao(u8);

impl Ao {
    pub const NONE: Ao = Ao(3);

    pub fn new(v: u8) -> Ao {
        debug_assert!(v <= 3, "AO out of range 0..=3");
        Ao(v)
    }
}

/// Skylight + blocklight, each `0..=15`. `FULL` is full-bright.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Light {
    sky: u8,
    block: u8,
}

impl Light {
    /// Full-bright: a torch on every vertex. Rarely what terrain wants — see [`Self::DAY`].
    pub const FULL: Light = Light { sky: 15, block: 15 };
    /// Outdoor daylight: full sky, no blocklight, so the shader keeps the sun/ambient
    /// tint instead of clamping to white. The correct "unlit surface" light.
    pub const DAY: Light = Light { sky: 15, block: 0 };

    pub fn new(sky: u8, block: u8) -> Light {
        debug_assert!(sky <= 15 && block <= 15, "light out of range 0..=15");
        Light { sky, block }
    }
}

vertex_struct! {
    /// 8-byte packed world-mesh vertex. Build via [`MeshVertex::new`] — the sole
    /// constructor, which demands AO and light so no path can leave them unstated.
    pub struct MeshVertex {
        packed: [u32; 2],
    }
}

impl MeshVertex {
    /// The sole vertex constructor: AO and light are non-optional, so no mesher
    /// can silently default them (the bug that washed out far LOD tiles).
    pub fn new(pos: [u8; 3], normal: Normal, layer: u16, ao: Ao, light: Light, water: bool) -> Self {
        Self::pack(pos, normal, layer, ao.0, light.sky, light.block, water)
    }

    const fn pack(pos: [u8; 3], normal: Normal, layer: u16, ao: u8, sky: u8, block: u8, water: bool) -> Self {
        // Chunk-local coords must be 0..=16. The 5-bit field also holds
        // 17..=31, so an out-of-range coord stores silently at the wrong
        // position rather than corrupting a neighbor — caught in debug only.
        debug_assert!(
            pos[0] <= 16 && pos[1] <= 16 && pos[2] <= 16,
            "coord out of range 0..=16"
        );
        let w0 = (pos[0] as u32 & MASK_COORD) << SHIFT_X
            | (pos[1] as u32 & MASK_COORD) << SHIFT_Y
            | (pos[2] as u32 & MASK_COORD) << SHIFT_Z
            | ((normal as u32) & MASK_NORMAL) << SHIFT_NORMAL
            | ((layer as u32) & MASK_LAYER) << SHIFT_LAYER;
        let w1 = ((ao as u32) & MASK_AO) << SHIFT_AO
            | ((sky as u32) & MASK_LIGHT) << SHIFT_SKY
            | ((block as u32) & MASK_LIGHT) << SHIFT_BLOCK
            | (water as u32) << SHIFT_WATER;
        Self { packed: [w0, w1] }
    }

    /// Decodes the water material bit — the mesher's per-face liquid flag,
    /// selecting animated water shading in the transparent fragment shader.
    pub fn is_water(&self) -> bool {
        (self.packed[1] >> SHIFT_WATER) & 1 == 1
    }

    /// Sets per-axis micro-offsets (-2..=1) that nudge vertices on LOD borders
    /// to break z-fighting. Chains after [`Self::new`] without widening its signature.
    pub fn with_micro(mut self, micro: [i8; 3]) -> Self {
        debug_assert!(
            micro.iter().all(|&m| (-2..=1).contains(&m)),
            "micro offset out of range -2..=1"
        );
        for (shift, &m) in [SHIFT_MICRO_X, SHIFT_MICRO_Y, SHIFT_MICRO_Z].iter().zip(&micro) {
            self.packed[1] |= ((m as u32) & MASK_MICRO) << shift;
        }
        self
    }

    /// Decodes the per-axis micro-offsets — the two's-complement inverse of
    /// [`Self::with_micro`]; `[0, 0, 0]` for any vertex built without it.
    pub fn micro(&self) -> [i8; 3] {
        let w1 = self.packed[1];
        [SHIFT_MICRO_X, SHIFT_MICRO_Y, SHIFT_MICRO_Z].map(|shift| {
            let raw = (w1 >> shift) & MASK_MICRO;
            if raw >= 2 { raw as i8 - 4 } else { raw as i8 }
        })
    }

    /// Decodes the chunk-local integer position as floats (for CPU-side AABBs).
    pub fn local_pos(&self) -> [f32; 3] {
        let ([x, y, z], ..) = unpack(self.packed);
        [x as f32, y as f32, z as f32]
    }

    /// Decodes the face normal — how [`MeshData::quad`] routes a quad into its
    /// direction bucket without the caller passing the normal twice.
    pub fn normal(&self) -> Normal {
        let (_, ni, ..) = unpack(self.packed);
        Normal::from_index(ni)
    }

    /// Decodes the texture-array layer.
    pub fn layer(&self) -> u16 {
        let (.., layer, _, _, _) = unpack(self.packed);
        layer
    }

    /// Decodes the ambient-occlusion level (`0..=3`).
    pub fn ao(&self) -> Ao {
        let (.., ao, _, _) = unpack(self.packed);
        Ao(ao)
    }

    /// Decodes the baked skylight + blocklight.
    pub fn light(&self) -> Light {
        let (.., sky, block) = unpack(self.packed);
        Light { sky, block }
    }
}

/// The sole CPU-side mirror of the Slang unpack in `mesh3d.vert.slang`. Returns
/// raw field integers (`pos`, normal index, layer, ao, sky, block); typed
/// callers map from there. Keeping one decoder means the shift/mask consts are
/// applied in exactly one place per direction (pack/unpack).
const fn unpack(packed: [u32; 2]) -> ([u32; 3], u8, u16, u8, u8, u8) {
    let [w0, w1] = packed;
    let pos = [
        (w0 >> SHIFT_X) & MASK_COORD,
        (w0 >> SHIFT_Y) & MASK_COORD,
        (w0 >> SHIFT_Z) & MASK_COORD,
    ];
    let normal = ((w0 >> SHIFT_NORMAL) & MASK_NORMAL) as u8;
    let layer = ((w0 >> SHIFT_LAYER) & MASK_LAYER) as u16;
    let ao = ((w1 >> SHIFT_AO) & MASK_AO) as u8;
    let sky = ((w1 >> SHIFT_SKY) & MASK_LIGHT) as u8;
    let block = ((w1 >> SHIFT_BLOCK) & MASK_LIGHT) as u8;
    (pos, normal, layer, ao, sky, block)
}

vertex_struct! {
    /// Immediate debug vertex: world-space position + RGBA8 color. Used by
    /// `draw_cube`/`draw_cube_wires`; rendered by the debug pipelines.
    pub struct DebugVertex {
        pub pos: [f32; 3],
        pub color: [u8; 4],
    }
}

/// Which draw *technique* a whole mesh belongs to — a property OF a mesh, not a
/// sub-range within one: the world meshes each technique into a separate
/// [`MeshData`], so a mesh is uniformly one pass. The set is closed (there are
/// only three ways a face ever reaches the GPU), so keying on technique — not on
/// material identity — scales with a fixed 3, never with block count. Discriminant
/// order is also draw order (opaque → cutout → blend last):
///
/// - [`Pass::Opaque`] — depth read/write, no blend, cull back. ~99% of geometry.
/// - [`Pass::Cutout`] — depth read/write, alpha *test* (`discard`), cull back.
///   Binary see-through (atlas holes: leaves/grates); writes depth so it needs no
///   sort. Reserved: no block sources per-texel alpha yet, so nothing emits it.
/// - [`Pass::Blend`] — depth read-only, alpha *over*, back-to-front. Tinted
///   see-through you look *through* (water/stained glass/ice).
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Pass {
    Opaque = 0,
    Cutout = 1,
    Blend = 2,
}

impl Pass {
    /// Every pass in discriminant (= draw) order. The single source of both the
    /// pass set and its width: containers key on `Pass::COUNT` so they follow the
    /// enum by construction and never widen by hand.
    pub const ALL: [Pass; 3] = [Pass::Opaque, Pass::Cutout, Pass::Blend];
    /// Number of passes — the width of any per-pass container.
    pub const COUNT: usize = Self::ALL.len();
}

/// Triangle mesh built one quad at a time, indices bucketed by face direction.
///
/// Six [`Normal`]-indexed index buckets share one vertex array; [`Self::quad`]
/// routes each quad by its normal and emits correctly wound indices, so
/// direction mis-sorting and winding bugs are unrepresentable at the call site.
/// Upload concatenates the buckets in `Normal` order into one index blob and
/// records the per-direction boundaries, which the renderer uses for optional
/// six-way face culling. Reusable as scratch: [`Self::clear`] keeps capacity.
pub struct MeshData {
    pub(crate) vertices: Vec<MeshVertex>,
    /// Indices per face direction, indexed by `Normal as usize`. Concatenated
    /// in that order at upload. Each entry references the shared `vertices`.
    pub(crate) buckets: [Vec<u32>; 6],
    pub(crate) pass: Pass,
}

impl MeshData {
    /// An empty mesh tagged with its draw pass.
    pub fn new(pass: Pass) -> Self {
        Self {
            vertices: Vec::new(),
            buckets: std::array::from_fn(|_| Vec::new()),
            pass,
        }
    }

    /// Appends one quad: four corners wound CCW as seen from outside → four
    /// vertices plus six indices (two triangles, `0,1,2` + `0,2,3`) routed into
    /// the bucket for `corners[0]`'s face normal. All four corners are expected
    /// to share that normal (greedy quads do).
    pub fn quad(&mut self, corners: [MeshVertex; 4]) {
        let dir = corners[0].normal() as usize;
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&corners);
        self.buckets[dir].extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// The mesh's draw pass.
    pub fn pass(&self) -> Pass {
        self.pass
    }

    /// Clears all geometry but keeps every allocation (vertices and buckets)
    /// and the pass tag, for reuse as a scratch buffer.
    pub fn clear(&mut self) {
        self.vertices.clear();
        for bucket in &mut self.buckets {
            bucket.clear();
        }
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// The packed vertices, for tests in dependent crates that assert on emitted
    /// geometry (greedy-mesh area vs a reference sweep, byte-identical far
    /// chunks). `#[doc(hidden)]` and read-only — mirrors the
    /// [`MeshHandle::from_raw_parts`] "for dependent-crate tests" precedent; NOT
    /// a production surface (build geometry with [`Self::quad`]).
    #[doc(hidden)]
    pub fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
    }

    /// The six per-direction index buckets (concatenated in [`Normal`] order at
    /// upload). `#[doc(hidden)]` — dependent-crate tests only, as [`Self::vertices`].
    #[doc(hidden)]
    pub fn buckets(&self) -> &[Vec<u32>; 6] {
        &self.buckets
    }
}

/// LOD detail level: `FULL` (0) is full-res; each step doubles the cell size.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Detail(u8);

impl Detail {
    /// Full-resolution chunks.
    pub const FULL: Detail = Detail(0);

    /// The coarsest representable level: `2^15` metre cells — far beyond any
    /// real pyramid, but small enough that [`scale`](Self::scale) can never
    /// overflow its shift. Public inputs are clamped here rather than trusted.
    pub const MAX_LEVEL: u8 = 15;

    /// A detail level, clamped to [`MAX_LEVEL`](Self::MAX_LEVEL): this is a
    /// public draw input, and an arbitrary `u8` must yield a bounded scale,
    /// not an overflowing `1 << level`.
    pub fn new(level: u8) -> Detail {
        Detail(level.min(Self::MAX_LEVEL))
    }

    /// Per-draw uniform scale: `2^level` metres per cell.
    pub fn scale(self) -> f32 {
        (1u32 << self.0) as f32
    }
}

/// Generational handle to a GPU mesh. Cheap to copy; freeing is explicit via
/// `Engine::free_mesh` (deferred internally until the GPU is done with it).
///
/// `generation` is a [`NonZeroU32`] so `Option<MeshHandle>` occupies the niche
/// and stays 8 bytes (the streaming lane stores millions of them). Generations
/// are therefore 1-based and never reach 0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshHandle {
    pub(crate) slot: u32,
    pub(crate) generation: std::num::NonZeroU32,
}

impl MeshHandle {
    /// Construct a handle from raw parts. For tests in dependent crates that
    /// need to populate handle-carrying state (e.g. mesh-lifecycle transitions)
    /// without a live GPU — the fields are otherwise crate-private. NOT for
    /// production use: a fabricated handle indexes no real GPU mesh.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Types the shared raw [`unpack`] back into the API surface — the one
    /// check no validation layer can do: that pack/unpack (and thus the
    /// const-fn shifts and the documented table) agree.
    fn typed_unpack(v: MeshVertex) -> ([u8; 3], Normal, u16, u8, u8, u8) {
        let ([x, y, z], ni, layer, ao, sky, block) = unpack(v.packed);
        (
            [x as u8, y as u8, z as u8],
            Normal::from_index(ni),
            layer,
            ao,
            sky,
            block,
        )
    }

    #[test]
    fn round_trip_over_representative_sweep() {
        let normals = [
            Normal::PosX,
            Normal::NegX,
            Normal::PosY,
            Normal::NegY,
            Normal::PosZ,
            Normal::NegZ,
        ];
        // The vertex must stay 8 bytes — the 14-bit layer lives in word 0's
        // spare span, never in a wider struct.
        assert_eq!(std::mem::size_of::<MeshVertex>(), 8);
        for pos in [[0, 0, 0], [16, 16, 16], [1, 7, 15], [16, 0, 9]] {
            for normal in normals {
                for layer in [0u16, 1, 127, 255, 256, 4095, 16383] {
                    for ao in [0u8, 1, 3] {
                        for (sky, block) in [(0u8, 15u8), (15, 0), (7, 8), (15, 15)] {
                            for water in [false, true] {
                                let v = MeshVertex::new(
                                    pos,
                                    normal,
                                    layer,
                                    Ao::new(ao),
                                    Light::new(sky, block),
                                    water,
                                );
                                assert_eq!(typed_unpack(v), (pos, normal, layer, ao, sky, block));
                                assert_eq!(v.is_water(), water, "water bit round-trips");
                                assert_eq!(v.micro(), [0, 0, 0], "new() leaves micro zero");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn micro_offsets_round_trip_two_complement() {
        // Every encodable value including both negatives and the zero default;
        // with_micro must not disturb the position/normal/layer/ao/light/water
        // fields it is layered on top of.
        for mx in [-2i8, -1, 0, 1] {
            for my in [-2i8, -1, 0, 1] {
                for mz in [-2i8, -1, 0, 1] {
                    let base = MeshVertex::new(
                        [3, 5, 7],
                        Normal::NegY,
                        42,
                        Ao::new(2),
                        Light::new(11, 4),
                        true,
                    );
                    let v = base.with_micro([mx, my, mz]);
                    assert_eq!(v.micro(), [mx, my, mz]);
                    // Layered bits leave everything else exactly as `base`.
                    assert_eq!(typed_unpack(v), typed_unpack(base));
                    assert!(v.is_water());
                }
            }
        }
        // The default constructor decodes to no offset.
        let plain = MeshVertex::new([0, 0, 0], Normal::PosX, 0, Ao::NONE, Light::DAY, false);
        assert_eq!(plain.micro(), [0, 0, 0]);
    }

    #[test]
    fn day_light_keeps_sky_drops_block() {
        let v = MeshVertex::new([2, 3, 4], Normal::PosY, 5, Ao::NONE, Light::DAY, false);
        assert_eq!(typed_unpack(v), ([2, 3, 4], Normal::PosY, 5, 3, 15, 0));
        assert!(!v.is_water(), "non-water vertex clears the water bit");
    }

    /// A quad wound `[0,1,2,0,2,3]` seen from outside, layer irrelevant.
    fn quad_for(normal: Normal) -> [MeshVertex; 4] {
        std::array::from_fn(|i| MeshVertex::new([i as u8, 0, 0], normal, 0, Ao::NONE, Light::FULL, false))
    }

    #[test]
    fn quad_routes_by_normal_with_correct_winding() {
        let mut data = MeshData::new(Pass::Opaque);
        // Two +X quads, one -Z quad: buckets fill by Normal index; others empty.
        data.quad(quad_for(Normal::PosX));
        data.quad(quad_for(Normal::PosX));
        data.quad(quad_for(Normal::NegZ));

        assert_eq!(data.vertices.len(), 12);
        assert_eq!(data.buckets[Normal::PosX as usize].len(), 12); // 2 quads × 6
        assert_eq!(data.buckets[Normal::NegZ as usize].len(), 6);
        for empty in [Normal::NegX, Normal::PosY, Normal::NegY, Normal::PosZ] {
            assert!(data.buckets[empty as usize].is_empty());
        }
        // First quad: base 0, second quad: base 4 — winding preserved per quad.
        assert_eq!(
            &data.buckets[Normal::PosX as usize][..6],
            &[0, 1, 2, 0, 2, 3]
        );
        assert_eq!(
            &data.buckets[Normal::PosX as usize][6..],
            &[4, 5, 6, 4, 6, 7]
        );
        assert_eq!(
            &data.buckets[Normal::NegZ as usize][..],
            &[8, 9, 10, 8, 10, 11]
        );
    }

    #[test]
    fn clear_keeps_pass_and_empties_buckets() {
        let mut data = MeshData::new(Pass::Blend);
        data.quad(quad_for(Normal::PosY));
        assert!(!data.is_empty());
        data.clear();
        assert!(data.is_empty());
        assert_eq!(data.pass(), Pass::Blend);
        assert!(data.buckets.iter().all(|b| b.is_empty()));
    }
}
