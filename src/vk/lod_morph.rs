//! CPU mirror of `lod_morph_cells` in `shaders/lod_morph.slang`.
//!
//! `k` is the detail level the vertex shader already derives:
//! `(detail_bits & mask) - DETAIL_GPU_BIAS`. `q` is the unmorphed position in
//! block units, before the cage and before the placement offset.
//!
//! The shader is the production caller. Host tests are the Rust caller, so the
//! mirror is allowed to look unused to the library build.

#![allow(dead_code)]

/// Cells along local +Y (`a * dy`). A level with no positive band, `dy == 0`,
/// or morphing off contributes nothing.
///
/// `bands[k]` is `(half.xyz, start)`. A short slice is a missing band; `k`
/// outside `0..16` is too, matching the 16-wide GPU array.
#[allow(clippy::too_many_arguments)]
pub(crate) fn morph_offset(
    morph_on: bool,
    eye_block: [i32; 3],
    eye_frac: [f32; 3],
    bands: &[[f32; 4]],
    block: [i32; 3],
    k: i32,
    q: [f32; 3],
    dy: i32,
) -> f32 {
    if !morph_on || dy == 0 || k < 0 || k >= 16 {
        return 0.0;
    }
    let Some(band) = bands.get(k as usize) else {
        return 0.0;
    };
    if !(band[0] > 0.0 && band[1] > 0.0 && band[2] > 0.0) {
        return 0.0;
    }
    let p = morph_position(block, eye_block, eye_frac, q);
    let d = (p[0].abs() / band[0])
        .max(p[1].abs() / band[1])
        .max(p[2].abs() / band[2]);
    smoothstep(band[3], 1.0, d) * dy as f32
}

/// `float3(block - eye_block) - eye_frac + q`, with wrapping integer subtraction
/// so a large eye stays exact. Twin of the shader's `p`.
pub(crate) fn morph_position(
    block: [i32; 3],
    eye_block: [i32; 3],
    eye_frac: [f32; 3],
    q: [f32; 3],
) -> [f32; 3] {
    [
        block[0].wrapping_sub(eye_block[0]) as f32 - eye_frac[0] + q[0],
        block[1].wrapping_sub(eye_block[1]) as f32 - eye_frac[1] + q[1],
        block[2].wrapping_sub(eye_block[2]) as f32 - eye_frac[2] + q[2],
    ]
}

/// HLSL `smoothstep`: saturate, then the Hermite cubic.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk::pipeline::EyeSplit;

    fn band(half: [f32; 3], start: f32) -> [f32; 4] {
        [half[0], half[1], half[2], start]
    }

    fn along_x(q_x: f32, start: f32, dy: i32) -> f32 {
        morph_offset(
            true,
            [0; 3],
            [0.0; 3],
            &[band([1.0, 1.0, 1.0], start)],
            [0; 3],
            0,
            [q_x, 0.0, 0.0],
            dy,
        )
    }

    #[test]
    fn blend_is_zero_inside_the_start_and_one_at_the_edge() {
        assert_eq!(along_x(0.1, 0.5, 1), 0.0);
        assert_eq!(along_x(0.5, 0.5, 1), 0.0);
        assert_eq!(along_x(1.0, 0.5, 1), 1.0);
        assert_eq!(along_x(2.0, 0.5, 1), 1.0);
        assert_eq!(along_x(2.0, 0.5, -4), -4.0);
        // Midpoint of smoothstep(0.25, 1, 0.625) is t = 0.5, so a = 0.5.
        assert_eq!(along_x(0.625, 0.25, 1).to_bits(), 0.5f32.to_bits());
    }

    #[test]
    fn blend_is_monotonic_between_the_edges() {
        let mut prev = along_x(0.25, 0.25, 1);
        assert_eq!(prev, 0.0);
        for i in 1..=8 {
            let d = 0.25 + (i as f32) * (0.75 / 8.0);
            let next = along_x(d, 0.25, 1);
            assert!(next > prev, "d {d}: {next} <= {prev}");
            prev = next;
        }
        assert_eq!(prev.to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn distance_is_chebyshev_not_a_sum() {
        let half = band([10.0, 10.0, 10.0], 0.5);
        let at = |q: [f32; 3]| morph_offset(true, [0; 3], [0.0; 3], &[half], [0; 3], 0, q, 1);
        // Each axis is 0.4 of the box. A sum of ratios would be 0.8 and already blend.
        assert_eq!(at([4.0, 4.0, 0.0]), 0.0);
        // A corner where every ratio equals d blends the same as one axis at that d.
        assert_eq!(at([8.0, 8.0, 8.0]).to_bits(), at([8.0, 0.0, 0.0]).to_bits());
    }

    #[test]
    fn fractional_eye_cancels_a_matching_q() {
        // p.x = 0 - 0.5 + 0.5 = 0, so a = 0. Dropping the frac term would yield 0.5.
        let offset = morph_offset(
            true,
            [10, 0, 0],
            [0.5, 0.0, 0.0],
            &[band([1.0, 1.0, 1.0], 0.0)],
            [10, 0, 0],
            0,
            [0.5, 0.0, 0.0],
            1,
        );
        assert_eq!(offset, 0.0);
    }

    #[test]
    fn a_level_without_a_band_does_not_move() {
        let live = band([10.0, 10.0, 10.0], 0.0);
        let outside = [100.0, 0.0, 0.0];
        let go = |bands: &[[f32; 4]], k: i32, on: bool, dy: i32| {
            morph_offset(on, [0; 3], [0.0; 3], bands, [0; 3], k, outside, dy)
        };
        assert_eq!(go(&[live], 1, true, 4), 0.0);
        assert_eq!(go(&[], 0, true, 4), 0.0);
        assert_eq!(go(&[live], -1, true, 4), 0.0);
        assert_eq!(go(&[live; 16], 16, true, 4), 0.0);
        assert_eq!(go(&[band([0.0, 10.0, 10.0], 0.0)], 0, true, 4), 0.0);
        assert_eq!(go(&[band([10.0, -1.0, 10.0], 0.0)], 0, true, 4), 0.0);
        assert_eq!(go(&[band([10.0, 10.0, 0.0], 0.0)], 0, true, 4), 0.0);
        assert_eq!(go(&[band([f32::NAN, 10.0, 10.0], 0.0)], 0, true, 4), 0.0);
        assert_eq!(go(&[live], 0, false, 4), 0.0);
        assert_eq!(go(&[live], 0, true, 0), 0.0);
        assert_eq!(go(&[live], 0, true, -4), -4.0);
    }

    #[test]
    fn a_large_eye_keeps_p_exact_and_i32_wraps() {
        let eye = glam::DVec3::new(1_200_000_000.25, -1_200_000_000.75, 1_200_000_000.125);
        let split = EyeSplit::of(eye);
        assert_eq!(split.block[0], 1_200_000_000);
        assert_eq!(split.block[1], -1_200_000_001);
        assert_eq!(split.block[2], 1_200_000_000);
        assert_eq!(split.frac[0].to_bits(), 0.25f32.to_bits());
        assert_eq!(split.frac[1].to_bits(), 0.25f32.to_bits());
        assert_eq!(split.frac[2].to_bits(), 0.125f32.to_bits());

        let delta = [3i32, -5, 7];
        let block = [
            split.block[0].wrapping_add(delta[0]),
            split.block[1].wrapping_add(delta[1]),
            split.block[2].wrapping_add(delta[2]),
        ];
        let q = [0.25f32, -0.5, 0.125];
        let p = morph_position(block, split.block, split.frac, q);
        for i in 0..3 {
            let truth = delta[i] as f64 + q[i] as f64 - split.frac[i] as f64;
            assert!(
                (p[i] as f64 - truth).abs() < 1e-3,
                "axis {i}: p {} truth {truth}",
                p[i]
            );
            let naive = block[i] as f32 - eye[i] as f32 + q[i];
            let geometric = block[i] as f64 + q[i] as f64 - eye[i];
            assert!(
                (naive as f64 - geometric).abs() > 1.0,
                "naive f32 subtraction should miss at |eye| ~ 1.2e9"
            );
        }

        let wrapped = morph_position([i32::MIN, 0, 0], [i32::MAX, 0, 0], [0.0; 3], [0.0; 3]);
        assert_eq!(wrapped[0], 1.0);
        assert_eq!(i32::MIN.wrapping_sub(i32::MAX), 1);
    }

    fn spirv_words(name: &str) -> Vec<u32> {
        let path = std::path::Path::new(env!("OUT_DIR")).join(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert_eq!(bytes.len() % 4, 0, "{name} is not a word stream");
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn walk_spirv(words: &[u32], mut f: impl FnMut(u16, &[u32])) {
        assert_eq!(words[0], 0x07230203, "SPIR-V magic");
        let mut i = 5;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            let op = (words[i] & 0xFFFF) as u16;
            assert!(count >= 1 && i + count <= words.len());
            f(op, &words[i + 1..i + count]);
            i += count;
        }
    }

    fn spirv_string(words: &[u32]) -> String {
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .take_while(|&b| b != 0)
            .collect();
        String::from_utf8(bytes).expect("spirv string")
    }

    fn member_offset(words: &[u32], name: &str) -> Option<u32> {
        let mut found = None;
        walk_spirv(words, |op, ops| {
            if op == 6 && spirv_string(&ops[2..]) == name {
                found = Some((ops[0], ops[1]));
            }
        });
        let (ty, member) = found?;
        let mut offset = None;
        walk_spirv(words, |op, ops| {
            if op == 72 && ops.len() >= 4 && ops[0] == ty && ops[1] == member && ops[2] == 35 {
                offset = Some(ops[3]);
            }
        });
        offset
    }

    /// Slang lowers `float4 morph_band[16]` to a std140 wrapper struct whose
    /// member is the array. The stride lives on that array, not on the wrapper.
    fn array_stride(words: &[u32], type_id: u32) -> Option<u32> {
        let mut stride = None;
        let mut children = Vec::new();
        walk_spirv(words, |op, ops| {
            if op == 71 && ops.len() >= 3 && ops[0] == type_id && ops[1] == 6 {
                stride = Some(ops[2]);
            }
            if op == 30 && ops.first() == Some(&type_id) {
                children.extend_from_slice(&ops[1..]);
            }
        });
        if let Some(stride) = stride {
            return Some(stride);
        }
        children
            .into_iter()
            .find_map(|child| array_stride(words, child))
    }

    fn member_array_stride(words: &[u32], name: &str) -> Option<u32> {
        let mut found = None;
        walk_spirv(words, |op, ops| {
            if op == 6 && spirv_string(&ops[2..]) == name {
                found = Some((ops[0], ops[1]));
            }
        });
        let (ty, member) = found?;
        let mut member_ty = None;
        walk_spirv(words, |op, ops| {
            if op == 30 && ops[0] == ty {
                member_ty = Some(ops[1 + member as usize]);
            }
        });
        array_stride(words, member_ty?)
    }

    fn descriptor_bindings(words: &[u32]) -> Vec<(u32, u32)> {
        let mut set = std::collections::HashMap::<u32, u32>::new();
        let mut binding = std::collections::HashMap::<u32, u32>::new();
        walk_spirv(words, |op, ops| {
            if op == 71 && ops.len() >= 3 {
                match ops[1] {
                    33 => {
                        binding.insert(ops[0], ops[2]);
                    }
                    34 => {
                        set.insert(ops[0], ops[2]);
                    }
                    _ => {}
                }
            }
        });
        let mut pairs: Vec<(u32, u32)> = binding
            .iter()
            .filter_map(|(id, b)| set.get(id).map(|s| (*s, *b)))
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    fn assert_morph_tail(name: &str) {
        let words = spirv_words(name);
        assert_eq!(member_offset(&words, "sky_bitangent"), Some(208), "{name}");
        assert_eq!(
            member_offset(&words, "morph_eye_block"),
            Some(224),
            "{name}"
        );
        assert_eq!(member_offset(&words, "morph_eye_frac"), Some(240), "{name}");
        assert_eq!(member_offset(&words, "morph_band"), Some(256), "{name}");
        assert_eq!(
            member_array_stride(&words, "morph_band"),
            Some(16),
            "{name}"
        );
    }

    #[test]
    fn shader_ubo_tail_matches_the_cpu_struct() {
        assert_morph_tail("mesh3d.vert.spv");
        assert_morph_tail("mesh3d_caged.vert.spv");
        assert_morph_tail("shadow_depth.vert.spv");
        let lean = spirv_words("mesh3d_opaque_lean.frag.spv");
        assert_eq!(member_offset(&lean, "sky_bitangent"), Some(208));
        if member_offset(&lean, "morph_eye_block").is_some() {
            assert_eq!(member_offset(&lean, "morph_eye_block"), Some(224));
            assert_eq!(member_offset(&lean, "morph_eye_frac"), Some(240));
            assert_eq!(member_offset(&lean, "morph_band"), Some(256));
        }
        let shadow = spirv_words("shadow_depth.vert.spv");
        let bindings = descriptor_bindings(&shadow);
        assert!(
            bindings.contains(&(0, 2)),
            "shadow vertex shader must bind the frame UBO at set 0 binding 2, got {bindings:?}"
        );
    }
}
