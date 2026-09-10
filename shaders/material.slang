// Per-layer material table (set 0 binding 7). 16384 × 16 B, index = the
// 14-bit vertex `layer` id. Default entries are ARRAY_LAYER (flags bit 0
// clear, alpha 255, glow 0): the fragment takes the existing texture-array
// sample so goldens stay bit-identical unless a descriptor says otherwise.

#ifndef VOXEL_MATERIAL_SLANG
#define VOXEL_MATERIAL_SLANG

#include "common.slang"

// Mirrors src/vk/mesh3d_desc.rs::MESH3D_BINDING_MATERIALS.
[[vk::binding(7, 0)]]
StructuredBuffer<uint4> materials;

struct MaterialDesc
{
    uint3 rgb;
    uint3 rgb2;
    uint frequency;
    uint roughness;
    uint alpha;
    uint glow;
    uint flags;
};

MaterialDesc load_material(uint layer)
{
    // Little-endian pack of src/material.rs::MaterialDesc (16 bytes / uint4).
    uint4 v = materials[layer];
    MaterialDesc d;
    d.rgb = uint3(v.x & 0xffu, (v.x >> 8) & 0xffu, (v.x >> 16) & 0xffu);
    d.rgb2 = uint3((v.x >> 24) & 0xffu, v.y & 0xffu, (v.y >> 8) & 0xffu);
    d.frequency = (v.y >> 16) & 0xffu;
    d.roughness = (v.y >> 24) & 0xffu;
    d.alpha = v.z & 0xffu;
    d.glow = (v.z >> 8) & 0xffu;
    d.flags = (v.z >> 16) & 0xffffu;
    return d;
}

// Shader-side guard: flags bit 0 clear + alpha 255 + glow 0 is the default
// table. The caller must take the existing array-sample / FLAT_COLOR path
// with no extra multiply and no emissive.
bool material_is_identity(MaterialDesc d)
{
    return ((d.flags & 1u) == 0u) && (d.alpha == 255u) && (d.glow == 0u);
}

bool material_is_procedural(MaterialDesc d)
{
    return (d.flags & 1u) != 0u;
}

uint material_hash(int2 c, uint layer)
{
    uint n = (uint(c.x) * 0x8da6b343u) ^ (uint(c.y) * 0xd8163841u) ^ (layer * 0xcb1ab31fu);
    n ^= n >> 16;
    n *= 0x7feb352du;
    n ^= n >> 15;
    return n;
}

float material_hash01(int2 c, uint layer)
{
    return float(material_hash(c, layer) >> 8) * (1.0 / 16777215.0);
}

// Deterministic 2-D value noise hashed from (cell, layer), in [0, 1].
float material_value_noise(float2 p, uint layer)
{
    int2 i = int2(floor(p));
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = material_hash01(i, layer);
    float b = material_hash01(i + int2(1, 0), layer);
    float c = material_hash01(i + int2(0, 1), layer);
    float d = material_hash01(i + int2(1, 1), layer);
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

// t = fbm(uv_block * (1 + frequency/32), octaves = 1 + roughness/64 (1..=4),
// gain 0.5). Octaves above the first fade to 0.5 by the screen-space footprint
// (fwidth of block uv × that octave's frequency). `dist` is camera-relative
// length (agreed signature); minification is fwidth-driven.
float material_fbm(float2 uv_block, uint frequency, uint roughness, uint layer, float dist)
{
    float fw = max(length(fwidth(uv_block)), dist * 0.0);
    float freq = 1.0 + float(frequency) * (1.0 / 32.0);
    uint octaves = 1u + roughness / 64u;
    float2 p = uv_block * freq;
    float sum = 0.0;
    float amp = 1.0;
    float norm = 0.0;
    for (uint i = 0u; i < 4u; i++)
    {
        if (i >= octaves)
            break;
        float n = material_value_noise(p, layer);
        if (i > 0u)
            n = lerp(0.5, n, saturate(1.0 - fw * freq));
        sum += n * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
        p *= 2.0;
    }
    return sum / max(norm, 1e-6);
}

float3 procedural_albedo(MaterialDesc d, float2 uv_block, float dist, uint layer)
{
    float t = material_fbm(uv_block, d.frequency, d.roughness, layer, dist);
    float w = smoothstep(0.35, 0.65, t);
    float3 c0 = float3(d.rgb) * (1.0 / 255.0);
    float3 c1 = float3(d.rgb2) * (1.0 / 255.0);
    float3 srgb = lerp(c0, c1, w);
    // Fine grain on luminance; fades to 0 (its mean) under minification.
    // Skipped at roughness 0 so the 1-octave path stays inside the ALU budget.
    if (d.roughness != 0u)
    {
        float fw = max(length(fwidth(uv_block)), dist * 0.0);
        float2 gp = uv_block * 16.0;
        float g = (material_hash01(int2(floor(gp)), layer) - 0.5)
                * float(d.roughness) * (1.0 / 512.0)
                * saturate(1.0 - fw * 16.0);
        srgb = saturate(srgb + g);
    }
    return srgb_to_linear(srgb);
}

#endif
