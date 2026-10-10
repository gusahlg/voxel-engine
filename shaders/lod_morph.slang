// LOD geomorph. Included only by the mesh vertex shaders (color + shadow),
// after `ConstantBuffer<FrameUniforms> frame` is declared. Twin of
// `morph_a` / `band_valid` in src/vk/lod_morph.rs.
//
// dy is a signed cell offset along the mesh's local +Y (w1 bits [17:25)).
// a = smoothstep over that detail level's Euclidean camera-distance band.
// The caller has already formed the unmorphed camera-relative world position;
// d = length(world). Flat adds a*dy*scale to world.y. Caged adds
// (a*dy/16) * cage_dty, because the trilinear map is linear in t.y.

#ifndef VOXEL_LOD_MORPH_SLANG
#define VOXEL_LOD_MORPH_SLANG

// Bits [17:25) of w1, 8-bit two's complement (SHIFT_MORPH, MASK_MORPH).
// Twin of MeshVertex::morph: `<< (24 - SHIFT_MORPH)` parks bit 24 on the
// sign, `>> 24` brings bit 17 down to bit 0.
int lod_morph_dy(uint w1)
{
    return int(w1 << (24 - SHIFT_MORPH)) >> 24;
}

// True when the band can produce a non-zero blend. !(end > start) covers
// inverted, equal, and NaN. Negative and +inf are rejected on their own:
// +inf can still satisfy end > start.
bool lod_morph_band_ok(float start, float end)
{
    return start >= 0.0 && end > start && !isinf(start) && !isinf(end);
}

// Blend factor for detail level k. 0 when k is outside 0..16 or the band is
// invalid. Does not look at dy; the caller skips the add when dy is 0.
float lod_morph_a(uint detail, float d)
{
    int k = int(detail & ((1u << DETAIL_GPU_BITS) - 1u)) - int(DETAIL_GPU_BIAS);
    if (k < 0 || k >= 16)
        return 0.0;
    // Level 2i in xy, level 2i+1 in zw.
    float4 packed = frame.morph_band[uint(k) >> 1];
    float start = ((k & 1) == 0) ? packed.x : packed.z;
    float end = ((k & 1) == 0) ? packed.y : packed.w;
    if (!lod_morph_band_ok(start, end))
        return 0.0;
    return smoothstep(start, end, d);
}

#endif
