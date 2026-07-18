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
    }
}
