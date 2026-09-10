//! Per-layer material descriptors, indexed by the 14-bit [`crate::MeshVertex`]
//! `layer` field.
//!
//! The GPU table is a fixed 16384-entry SSBO (256 KiB). The default entry is
//! [`MaterialDesc::ARRAY_LAYER`]: sample the block texture array, full alpha,
//! no emissive. Existing games that never call
//! [`crate::Engine::set_material_descs`] keep bit-identical shading.

use bytemuck::{Pod, Zeroable};

/// Entries in the device-local material table. Matches the 14-bit layer field
/// (`0x3FFF + 1`) packed in [`crate::MeshVertex`].
pub const MATERIAL_DESC_CAPACITY: usize = 16384;

/// `flags` bit 0: procedural two-colour pattern instead of the texture-array layer.
pub const MATERIAL_FLAG_PROCEDURAL: u16 = 1 << 0;
/// `flags` bit 1: reserved (emissive only at night).
pub const MATERIAL_FLAG_EMISSIVE_ONLY_NIGHT: u16 = 1 << 1;

const _: () = assert!(MATERIAL_DESC_CAPACITY == (1 << 14));
const _: () = assert!(size_of::<MaterialDesc>() == 16);
const _: () = assert!(align_of::<MaterialDesc>() == 2);

/// 16-byte per-layer material record. Index = the vertex `layer` id.
///
/// Packed little-endian; the fragment shader unpacks the same 16 bytes as a
/// `uint4`. `flags` bit 0 selects [`MATERIAL_FLAG_PROCEDURAL`] vs the texture
/// array layer; bit 1 is reserved as [`MATERIAL_FLAG_EMISSIVE_ONLY_NIGHT`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct MaterialDesc {
    pub rgb: [u8; 3],
    pub rgb2: [u8; 3],
    pub frequency: u8,
    pub roughness: u8,
    pub alpha: u8,
    pub glow: u8,
    pub flags: u16,
    pub _pad: [u8; 4],
}

impl MaterialDesc {
    /// Sample the block texture array at this layer; alpha 255, glow 0.
    /// The default table is filled with this so unset layers match the
    /// pre-descriptor texture path.
    pub const ARRAY_LAYER: Self = Self {
        rgb: [0; 3],
        rgb2: [0; 3],
        frequency: 0,
        roughness: 0,
        alpha: 255,
        glow: 0,
        flags: 0,
        _pad: [0; 4],
    };

    /// `flags` bit 0 clear, `alpha == 255`, `glow == 0`: the fragment takes
    /// the existing array-sample (or flat-colour) path with no multiply and
    /// no emissive, so output stays bit-identical to a table that never
    /// existed.
    pub const fn is_array_layer_identity(self) -> bool {
        (self.flags & MATERIAL_FLAG_PROCEDURAL) == 0 && self.alpha == 255 && self.glow == 0
    }
}

impl Default for MaterialDesc {
    fn default() -> Self {
        Self::ARRAY_LAYER
    }
}

/// A full table of [`MaterialDesc::ARRAY_LAYER`] (the GPU default).
pub fn default_material_table() -> Box<[MaterialDesc]> {
    vec![MaterialDesc::ARRAY_LAYER; MATERIAL_DESC_CAPACITY].into_boxed_slice()
}

/// Replace the table with `descs` (index = layer id) and reset the unused
/// tail to [`MaterialDesc::ARRAY_LAYER`]. Returns the used prefix length.
/// Truncates to [`MATERIAL_DESC_CAPACITY`].
pub fn write_material_set(table: &mut [MaterialDesc], descs: &[MaterialDesc]) -> usize {
    debug_assert_eq!(table.len(), MATERIAL_DESC_CAPACITY);
    let n = descs.len().min(MATERIAL_DESC_CAPACITY);
    table[..n].copy_from_slice(&descs[..n]);
    table[n..].fill(MaterialDesc::ARRAY_LAYER);
    n
}

/// Append `descs` at `used`. Returns the new used count, clamped to capacity.
pub fn write_material_append(
    table: &mut [MaterialDesc],
    used: usize,
    descs: &[MaterialDesc],
) -> usize {
    debug_assert_eq!(table.len(), MATERIAL_DESC_CAPACITY);
    let used = used.min(MATERIAL_DESC_CAPACITY);
    let room = MATERIAL_DESC_CAPACITY - used;
    let n = descs.len().min(room);
    if n > 0 {
        table[used..used + n].copy_from_slice(&descs[..n]);
    }
    used + n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_table_is_array_layer_identity() {
        let table = default_material_table();
        assert_eq!(table.len(), MATERIAL_DESC_CAPACITY);
        for (i, d) in table.iter().enumerate() {
            assert_eq!(*d, MaterialDesc::ARRAY_LAYER, "layer {i}");
            assert!(d.is_array_layer_identity(), "layer {i}");
            assert_eq!(d.flags & MATERIAL_FLAG_PROCEDURAL, 0, "layer {i}");
            assert_eq!(d.alpha, 255, "layer {i}");
            assert_eq!(d.glow, 0, "layer {i}");
        }
    }

    #[test]
    fn array_layer_default_is_not_zeroed() {
        let z = MaterialDesc::zeroed();
        assert_eq!(z.alpha, 0);
        assert!(!z.is_array_layer_identity());
        assert_ne!(z, MaterialDesc::ARRAY_LAYER);
        assert_eq!(MaterialDesc::default(), MaterialDesc::ARRAY_LAYER);
    }

    #[test]
    fn packed_layout_is_16_bytes_little_endian() {
        let d = MaterialDesc {
            rgb: [1, 2, 3],
            rgb2: [4, 5, 6],
            frequency: 7,
            roughness: 8,
            alpha: 9,
            glow: 10,
            flags: 0x1122,
            _pad: [0xAA, 0xBB, 0xCC, 0xDD],
        };
        let bytes: &[u8] = bytemuck::bytes_of(&d);
        assert_eq!(
            bytes,
            &[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0x22, 0x11, 0xAA, 0xBB, 0xCC, 0xDD
            ]
        );
        // Shader unpacks this as uint4: rgb in v.x low 24, rgb2.r in v.x high 8.
        let words: [u32; 4] = bytemuck::cast(d);
        assert_eq!(words[0], 0x0403_0201);
        assert_eq!(words[1], 0x0807_0605);
        assert_eq!(words[2], 0x1122_0A09);
        assert_eq!(words[3], 0xDDCC_BBAA);
    }

    #[test]
    fn write_set_replaces_prefix_and_resets_tail() {
        let mut table = default_material_table();
        let a = MaterialDesc {
            rgb: [9, 0, 0],
            alpha: 200,
            flags: MATERIAL_FLAG_PROCEDURAL,
            ..MaterialDesc::ARRAY_LAYER
        };
        let b = MaterialDesc {
            rgb: [0, 9, 0],
            glow: 1,
            ..MaterialDesc::ARRAY_LAYER
        };
        assert_eq!(write_material_set(&mut table, &[a, b]), 2);
        assert_eq!(table[0], a);
        assert_eq!(table[1], b);
        assert_eq!(table[2], MaterialDesc::ARRAY_LAYER);
        assert_eq!(table[MATERIAL_DESC_CAPACITY - 1], MaterialDesc::ARRAY_LAYER);
        assert!(!table[0].is_array_layer_identity());
        assert!(!table[1].is_array_layer_identity());
        assert!(table[2].is_array_layer_identity());

        assert_eq!(write_material_set(&mut table, &[a]), 1);
        assert_eq!(table[0], a);
        assert_eq!(table[1], MaterialDesc::ARRAY_LAYER);
    }

    #[test]
    fn write_append_grows_until_capacity() {
        let mut table = default_material_table();
        let a = MaterialDesc {
            frequency: 4,
            ..MaterialDesc::ARRAY_LAYER
        };
        let used = write_material_append(&mut table, 0, &[a, a]);
        assert_eq!(used, 2);
        let used = write_material_append(&mut table, used, &[a]);
        assert_eq!(used, 3);
        assert_eq!(table[0].frequency, 4);
        assert_eq!(table[2].frequency, 4);
        assert_eq!(table[3], MaterialDesc::ARRAY_LAYER);

        let overflow = vec![a; 8];
        let used = write_material_append(&mut table, MATERIAL_DESC_CAPACITY - 2, &overflow);
        assert_eq!(used, MATERIAL_DESC_CAPACITY);
    }

    #[test]
    fn reserved_emissive_night_flag_does_not_break_identity() {
        let d = MaterialDesc {
            flags: MATERIAL_FLAG_EMISSIVE_ONLY_NIGHT,
            ..MaterialDesc::ARRAY_LAYER
        };
        // Bit 1 is reserved and ignored by the identity guard (bit 0 + alpha + glow).
        assert!(d.is_array_layer_identity());
        let p = MaterialDesc {
            flags: MATERIAL_FLAG_PROCEDURAL | MATERIAL_FLAG_EMISSIVE_ONLY_NIGHT,
            ..MaterialDesc::ARRAY_LAYER
        };
        assert!(!p.is_array_layer_identity());
    }
}
