// LOD geomorph. Included only by the mesh vertex shaders (color + shadow),
// after `ConstantBuffer<FrameUniforms> frame` is declared. Twin of
// `morph_offset` in src/vk/lod_morph.rs.
//
// `q` is the unmorphed local position in block units, before the cage and
// before the placement offset. Returns `a * dy` cells along local +Y.

#ifndef VOXEL_LOD_MORPH_SLANG
#define VOXEL_LOD_MORPH_SLANG

float lod_morph_cells(uint w1, uint detail, int3 block, float3 q)
{
    // Bits [17:23) of w1, 6-bit two's complement (SHIFT_MORPH, MASK_MORPH).
    // Twin of MeshVertex::morph: `<< (26 - SHIFT_MORPH)` parks bit 22 on the
    // sign, `>> 26` brings bit 17 down to bit 0.
    int dy = int(w1 << (26 - SHIFT_MORPH)) >> 26;
    if (dy == 0)
        return 0.0;
    int k = int(detail & ((1u << DETAIL_GPU_BITS) - 1u)) - int(DETAIL_GPU_BIAS);
    if (k < 0 || k >= 16)
        return 0.0;
    float4 band = frame.morph_band[uint(k)];
    // A non-positive half (and NaN) disables the level.
    if (!(band.x > 0.0 && band.y > 0.0 && band.z > 0.0))
        return 0.0;
    // Integer subtraction wraps, matching the CPU `wrapping_sub`.
    float3 p = float3(block - frame.morph_eye_block.xyz) - frame.morph_eye_frac.xyz + q;
    float d = max(abs(p.x) / band.x, max(abs(p.y) / band.y, abs(p.z) / band.z));
    float a = smoothstep(band.w, 1.0, d);
    return a * float(dy);
}

#endif
