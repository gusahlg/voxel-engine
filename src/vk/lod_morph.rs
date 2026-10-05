//! CPU mirror of `lod_morph_a` in `shaders/lod_morph.slang`.
//!
//! `k` is the detail level the vertex shader already derives:
//! `(detail_bits & mask) - DETAIL_GPU_BIAS`. `d` is the Euclidean length of the
//! unmorphed camera-relative world position. A caged slide is
//! `(a * dy / 16) * cage_dty` — the trilinear map is linear in `t.y`, so that
//! equals a second sample at the moved `t`.
//!
//! The shader is the production caller. Host tests are the Rust caller, so
//! most of the mirror is allowed to look unused to the library build.
//! [`band_valid`] is also how the frame packs the GPU flag.

#![allow(dead_code)]

/// A band produces a blend when `end > start` and both ends are finite and
/// non-negative. Twin of `lod_morph_band_ok`: `!(end > start)` already rejects
/// NaN, and `is_finite` rejects ±infinity the way the shader's `isinf` does.
pub(crate) fn band_valid(start: f32, end: f32) -> bool {
    start >= 0.0 && end > start && start.is_finite() && end.is_finite()
}

/// `a = smoothstep(start, end, d)`, or 0 when the band is not [`band_valid`].
pub(crate) fn morph_a(start: f32, end: f32, d: f32) -> f32 {
    if !band_valid(start, end) {
        return 0.0;
    }
    smoothstep(start, end, d)
}

/// Level `k`'s blend from the packed `morph_band[8]` tail. A short slice, a
/// level outside `0..16`, or morphing off is a missing band (`a = 0`).
/// Slot `i` holds level `2i` in xy and level `2i+1` in zw.
pub(crate) fn level_a(morph_on: bool, bands: &[[f32; 4]], k: i32, d: f32) -> f32 {
    if !morph_on || !(0..16).contains(&k) {
        return 0.0;
    }
    let Some(slot) = bands.get((k as usize) / 2) else {
        return 0.0;
    };
    let (start, end) = if k % 2 == 0 {
        (slot[0], slot[1])
    } else {
        (slot[2], slot[3])
    };
    morph_a(start, end, d)
}

/// Flat morph: `world.y += a * dy * scale`. `dy == 0` or `a == 0` does not move.
pub(crate) fn flat_world(world: [f32; 3], a: f32, dy: i32, scale: f32) -> [f32; 3] {
    let mut out = world;
    if dy != 0 {
        out[1] += a * dy as f32 * scale;
    }
    out
}

/// ∂/∂t.y of the trilinear cage map. Twin of `cage_dty` in `shaders/cage.slang`.
pub(crate) fn cage_dty(corners: [[f32; 3]; 8], t: [f32; 3]) -> [f32; 3] {
    let c0 = lerp3(corners[0], corners[1], t[0]);
    let c1 = lerp3(corners[2], corners[3], t[0]);
    let c2 = lerp3(corners[4], corners[5], t[0]);
    let c3 = lerp3(corners[6], corners[7], t[0]);
    lerp3(sub3(c1, c0), sub3(c3, c2), t[2])
}

/// World delta of a caged morph: `(a * dy / 16) * dty`.
pub(crate) fn caged_delta(a: f32, dy: i32, dty: [f32; 3]) -> [f32; 3] {
    let s = a * dy as f32 / 16.0;
    [dty[0] * s, dty[1] * s, dty[2] * s]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// HLSL `smoothstep`: saturate, then the Hermite cubic.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Slang lowers `float4 morph_band[8]` to a std140 wrapper struct whose
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
        assert_eq!(member_offset(&words, "morph_band"), Some(224), "{name}");
        assert_eq!(member_offset(&words, "morph_flag"), Some(352), "{name}");
        assert!(
            member_offset(&words, "morph_eye_block").is_none(),
            "{name} still has the old eye"
        );
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
        if member_offset(&lean, "morph_band").is_some() {
            assert_eq!(member_offset(&lean, "morph_band"), Some(224));
            assert_eq!(member_offset(&lean, "morph_flag"), Some(352));
        }
        let shadow = spirv_words("shadow_depth.vert.spv");
        let bindings = descriptor_bindings(&shadow);
        assert!(
            bindings.contains(&(0, 2)),
            "shadow vertex shader must bind the frame UBO at set 0 binding 2, got {bindings:?}"
        );
    }

    #[test]
    fn blend_is_zero_inside_start_and_one_at_and_past_end() {
        let start = 4.0;
        let end = 10.0;
        assert_eq!(morph_a(start, end, 0.0), 0.0);
        assert_eq!(morph_a(start, end, start), 0.0);
        assert_eq!(morph_a(start, end, end).to_bits(), 1.0f32.to_bits());
        assert_eq!(morph_a(start, end, end + 25.0).to_bits(), 1.0f32.to_bits());
        // Midpoint: t = 0.5, so a = 0.5.
        assert_eq!(morph_a(start, end, 7.0).to_bits(), 0.5f32.to_bits());
        let moved = flat_world([1.0, 2.0, 3.0], 1.0, -4, 2.0);
        assert_eq!(moved, [1.0, 2.0 + -4.0 * 2.0, 3.0]);
        let still = flat_world([1.0, 2.0, 3.0], 0.0, -4, 2.0);
        assert_eq!(still[1].to_bits(), 2.0f32.to_bits());
        assert_eq!(flat_world([1.0, 2.0, 3.0], 1.0, 0, 2.0), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn blend_is_monotonic_between_the_edges() {
        let start = 4.0;
        let end = 10.0;
        let mut prev = morph_a(start, end, start);
        assert_eq!(prev, 0.0);
        for i in 1..=8 {
            let d = start + (end - start) * (i as f32) / 8.0;
            let next = morph_a(start, end, d);
            assert!(next > prev, "d {d}: {next} <= {prev}");
            prev = next;
        }
        assert_eq!(prev.to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn invalid_or_missing_bands_do_not_move() {
        assert!(!band_valid(5.0, 1.0), "inverted");
        assert!(!band_valid(2.0, 2.0), "equal");
        assert!(!band_valid(f32::NAN, 4.0));
        assert!(!band_valid(1.0, f32::NAN));
        assert!(!band_valid(-1.0, 8.0));
        assert!(!band_valid(-4.0, -1.0));
        assert!(!band_valid(0.0, f32::INFINITY));
        assert!(!band_valid(f32::INFINITY, f32::INFINITY));
        assert!(!band_valid(f32::NEG_INFINITY, 1.0));
        assert!(band_valid(0.0, 1.0));
        for (start, end) in [
            (5.0, 1.0),
            (2.0, 2.0),
            (f32::NAN, 4.0),
            (1.0, f32::NAN),
            (-1.0, 8.0),
            (-4.0, -1.0),
            (0.0, f32::INFINITY),
            (f32::NEG_INFINITY, 4.0),
        ] {
            assert_eq!(morph_a(start, end, 100.0), 0.0, "{start}..{end}");
        }
        // Empty tail, morphing off, and a level with no slot.
        let live = [[0.0, 8.0, 0.0, 0.0]];
        assert_eq!(level_a(true, &[], 0, 100.0), 0.0);
        assert_eq!(level_a(false, &live, 0, 100.0), 0.0);
        assert_eq!(level_a(true, &live, 1, 100.0), 0.0);
        assert_eq!(level_a(true, &live, -1, 100.0), 0.0);
        assert_eq!(level_a(true, &live, 16, 100.0), 0.0);
        assert_eq!(level_a(true, &live, 0, 0.0), 0.0);
        assert_eq!(level_a(true, &live, 0, 8.0).to_bits(), 1.0f32.to_bits());
        // Level 3 lives in zw of slot 1.
        let odd = [[0.0, 0.0, 0.0, 0.0], [1.0, 3.0, 4.0, 6.0]];
        assert_eq!(level_a(true, &odd, 2, 3.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(level_a(true, &odd, 3, 4.0), 0.0);
        assert_eq!(level_a(true, &odd, 3, 6.0).to_bits(), 1.0f32.to_bits());
    }

    fn xorshift(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    fn rand_f(state: &mut u32, lo: f32, hi: f32) -> f32 {
        let u = (xorshift(state) >> 8) as f32 / (1u32 << 24) as f32;
        lo + (hi - lo) * u
    }

    #[test]
    fn caged_dty_shortcut_matches_a_second_trilinear() {
        let id = crate::cage::identity_corners().map(|v| v.to_array());
        let t = [0.25, 0.5, 0.75];
        let dy = 6;
        let a = 0.5;
        let shortcut = caged_delta(a, dy, cage_dty(id, t));
        let moved = [t[0], t[1] + a * dy as f32 / 16.0, t[2]];
        let full = sub3(
            crate::cage::trilinear(id, moved),
            crate::cage::trilinear(id, t),
        );
        for i in 0..3 {
            assert!(
                (shortcut[i] - full[i]).abs() < 1e-5,
                "identity axis {i}: shortcut {} full {}",
                shortcut[i],
                full[i]
            );
        }
        // The identity box is 16 tall, so the slide in cells is a * dy.
        assert!((shortcut[1] - a * dy as f32).abs() < 1e-5);

        // Chunk-sized random cages. The map is exactly linear in t.y; the gap is
        // f32 rounding of the two samples. |dy| stays inside ±16 so the slide
        // stays under a few tens of cells: a full i8 dy on this cage reaches
        // ~100, where one f32 ulp is already 1.5e-5.
        let mut state = 0xA5A5_1234u32;
        for n in 0..64 {
            let corners = std::array::from_fn(|corner| {
                let on = |bit: u32, span: f32| {
                    if corner & (1 << bit) != 0 { span } else { 0.0 }
                };
                [
                    on(0, 16.0) + rand_f(&mut state, -4.0, 4.0),
                    on(1, 16.0) + rand_f(&mut state, -4.0, 4.0),
                    on(2, 16.0) + rand_f(&mut state, -4.0, 4.0),
                ]
            });
            let t = [
                rand_f(&mut state, -0.25, 1.25),
                rand_f(&mut state, -0.25, 1.25),
                rand_f(&mut state, -0.25, 1.25),
            ];
            let a = rand_f(&mut state, 0.0, 1.0);
            let dy = (xorshift(&mut state) % 33) as i32 - 16;
            let shortcut = caged_delta(a, dy, cage_dty(corners, t));
            let moved = [t[0], t[1] + a * dy as f32 / 16.0, t[2]];
            let full = sub3(
                crate::cage::trilinear(corners, moved),
                crate::cage::trilinear(corners, t),
            );
            for i in 0..3 {
                let err = (shortcut[i] - full[i]).abs();
                assert!(
                    err < 1e-5,
                    "cage {n} axis {i}: shortcut {} full {} err {err} dy {dy} a {a}",
                    shortcut[i],
                    full[i],
                );
            }
        }
    }
}
