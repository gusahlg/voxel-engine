//! Build-time generated constants shared verbatim with the Slang shaders.
//!
//! These `pub const` items are emitted by `build.rs` (see its `build_table`)
//! into `$OUT_DIR/gen_constants.rs` and, under the same names, into
//! `shaders/generated/shader_constants.slang` which `common.slang` includes.
//! Editing the values here has no effect — change the table in `build.rs`.
//!
//! Purpose: kill CPU↔GPU drift in sky/lighting math by defining every
//! cross-boundary constant exactly once.

include!(concat!(env!("OUT_DIR"), "/gen_constants.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_table_endpoints_and_monotone() {
        assert_eq!(SRGB8_TO_LINEAR[0], 0.0);
        assert_eq!(SRGB8_TO_LINEAR[255], 1.0);
        for w in SRGB8_TO_LINEAR.windows(2) {
            assert!(
                w[1] > w[0],
                "sRGB decode must be strictly monotone: {} !> {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn halton_bounded_and_zero_mean() {
        let mut sum = [0.0f64; 2];
        for p in HALTON_23 {
            for axis in 0..2 {
                assert!(
                    p[axis] >= -0.5 && p[axis] < 0.5,
                    "halton offset {} out of [-0.5, 0.5)",
                    p[axis]
                );
                sum[axis] += p[axis] as f64;
            }
        }
        let n = HALTON_23.len() as f64;
        for axis in 0..2 {
            let mean = sum[axis] / n;
            assert!(mean.abs() < 0.1, "halton axis {axis} mean {mean} not ~= 0");
        }
    }

    #[test]
    fn scalars_present() {
        assert_eq!(CANDLE_CLAMP, 4.0);
        assert!(CANDLE_HIGH_MUL > 1.0);
        assert_eq!(BLOOM_MAX_MIPS, BLOOM_SPIRAL_LOD as u32 + 1);
        assert_eq!(SPILL_FACTOR, 4);
        assert_eq!(SPILL_WG, 8);
        assert_eq!(TAA_TILE, 16);
        assert_eq!(EXPOSURE_TILE % 8, 0);
        // FarShape::Rounded: 2 is the sphere; the sky caps p at 16384.
        assert_eq!(FAR_ROUNDED_P_MIN, 2.0);
        assert_eq!(FAR_ROUNDED_P_MAX, 16384.0);
    }

    /// Fields of one packed vertex word as `(shift, mask)`: no two overlap
    /// and together they cover `0..bits` without a gap.
    fn assert_packed_word(fields: &[(u32, u32)], bits: u32) {
        let mut used = 0u64;
        for &(shift, mask) in fields {
            assert!(
                mask != 0 && mask & (mask + 1) == 0,
                "mask {mask:#x} is not a low run of ones"
            );
            let field = (mask as u64) << shift;
            assert_eq!(used & field, 0, "field at shift {shift} overlaps");
            used |= field;
        }
        assert_eq!(used, (1u64 << bits) - 1, "word fields leave a gap");
    }

    #[test]
    fn packed_vertex_layout_tiles_both_words() {
        assert_packed_word(
            &[
                (SHIFT_X, MASK_COORD),
                (SHIFT_Y, MASK_COORD),
                (SHIFT_Z, MASK_COORD),
                (SHIFT_NORMAL, MASK_NORMAL),
                (SHIFT_LAYER, MASK_LAYER),
            ],
            32,
        );
        assert_packed_word(
            &[
                (SHIFT_AO, MASK_AO),
                (SHIFT_SKY, MASK_LIGHT),
                (SHIFT_BLOCK, MASK_LIGHT),
                (SHIFT_WATER, 1),
                (SHIFT_MICRO_X, MASK_MICRO),
                (SHIFT_MICRO_Y, MASK_MICRO),
                (SHIFT_MICRO_Z, MASK_MICRO),
                (SHIFT_MORPH, MASK_MORPH),
            ],
            SHIFT_MORPH + MASK_MORPH.count_ones(),
        );
        // Morph is word 1's top field and must end inside the 32-bit word.
        assert!(SHIFT_MORPH + MASK_MORPH.count_ones() <= 32);
        // The shaders sign-extend micro with `<< (30 - shift) >> 30` and morph
        // with `<< (24 - SHIFT_MORPH) >> 24`: field widths 2 and 8.
        assert_eq!(MASK_MICRO.count_ones(), 2);
        assert_eq!(MASK_MORPH.count_ones(), 8);
        // mesh3d.vert reads the AO level as `w1 & MASK_AO`, with no shift.
        assert_eq!(SHIFT_AO, 0);
        // AO levels 0..=3 map to 0.4..=1.0; level 3 is unoccluded.
        assert_eq!(AO_MIN, 0.4);
        assert_eq!(AO_MIN + MASK_AO as f32 * AO_STEP, 1.0);
        // Light levels are 0..=15 (mesh3d.vert divides by 15).
        assert_eq!(MASK_LIGHT, 15);
    }

    #[test]
    fn sky_cloud_lut_in_quality_band() {
        assert!(
            (256..=512).contains(&SKY_CLOUD_LUT_SIZE),
            "SKY_CLOUD_LUT_SIZE={SKY_CLOUD_LUT_SIZE} outside 256..=512"
        );
        assert!(
            SKY_CLOUD_LUT_SIZE.is_multiple_of(SKY_CLOUD_LUT_WG),
            "LUT size must divide evenly by the workgroup edge"
        );
    }
}
