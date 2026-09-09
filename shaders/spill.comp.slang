// Quarter-resolution spill: bloom golden-angle gather + godray depth march.
//
// One compute dispatch per presented frame, recorded on the bloom chain's
// command buffer. Each thread writes one RGBA16F texel at 1/SPILL_FACTOR of
// the render extent:
//   rgb = bloomSpill(warped uv) * BLOOM_STRENGTH  +  godrays(unwarped uv)
//   a   = 1 (unused)
// Tonemap then takes one bilinear tap of this image at the UNwarped output uv
// (so the two agree under the wide-FOV remap) and adds exposure * rgb.
//
// Bloom-off / godrays-off: the corresponding term is zero (push scalar); when
// BOTH are off the host skips this dispatch and binds a 1x1 black image.
//
// No subgroup ops: portable core 1.3. Storage image carries an explicit
// `rgba16f` format so no shaderStorageImageWriteWithoutFormat is needed.

#include "common.slang"

struct Push
{
    uint2 dst_dim;        // spill image size in texels
    float s;              // cylindrical warp strength (<=0 rectilinear)
    float atan_s;         // precomputed atan(s)
    float4 godray0;       // sun uv (xy), strength gate (z), jitter.x (w)
    float4 godray1;       // sun tint (rgb), jitter.y (w)
    float bloom_strength; // BLOOM_STRENGTH, or 0 when the bloom lane is off
};
[[vk::push_constant]] Push pc;

[[vk::binding(0, 0)]] Sampler2D bloomTex;
[[vk::binding(1, 0)]] Sampler2D depthTex;
[[vk::image_format("rgba16f")]] [[vk::binding(2, 0)]] RWTexture2D<float4> spillOut;

// Golden-angle spiral gather; spatial rotation only (no temporal jitter).
// Evaluated once per spill texel; `pix` is the spill texel's pixel coord so
// the per-pixel hash still breaks the tap pattern up.
float3 bloomSpill(float2 uv, float2 pix)
{
    const float GOLDEN_ANGLE = 2.399963229728653; // 2π·(1 − 1/φ)
    float rot = frac(sin(dot(pix, float2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
    int n = int(BLOOM_SPIRAL_SAMPLES);
    float3 acc = float3(0.0);
    for (int i = 0; i < n; ++i)
    {
        float t = (float(i) + 0.5) / BLOOM_SPIRAL_SAMPLES;
        float a = rot + float(i) * GOLDEN_ANGLE;
        float2 off = float2(cos(a), sin(a)) * (t * BLOOM_SPIRAL_RADIUS);
        // Chain base is quarter-res; BLOOM_SPIRAL_LOD (1) is the eighth-res blur.
        acc += bloomTex.SampleLevel(uv + off, BLOOM_SPIRAL_LOD).rgb;
    }
    return acc / BLOOM_SPIRAL_SAMPLES;
}

// Screen-space godrays: march toward sun in unwarped output uv, accumulate
// sky pixels (reversed-Z depth ~0) after warping each tap to match the HDR
// sample space. Returns pre-exposure light (tonemap scales by exposure).
float3 godrays(float2 outUv)
{
    if (pc.godray0.z <= 0.0)
        return float3(0.0);
    float2 toSun = pc.godray0.xy - outUv;
    float sunDist = length(toSun);
    float2 dir = toSun / max(sunDist, 1.0e-6);
    float2 step = dir * (min(sunDist, GODRAY_MAX_LEN) / GODRAY_SAMPLES);
    float jitter = 0.5;
    int n = int(GODRAY_SAMPLES);
    float occ = 0.0;
    for (int i = 0; i < n; ++i)
    {
        float2 p = outUv + step * (float(i) + jitter);
        float2 duv = warpSampleUv(p, pc.s, pc.atan_s) + float2(pc.godray0.w, pc.godray1.w);
        float d = depthTex.SampleLevel(duv, 0.0).r;
        occ += (d < 1.0e-5) ? 1.0 : 0.0;
    }
    occ /= GODRAY_SAMPLES;
    float sunward = saturate(1.0 - sunDist / GODRAY_MAX_LEN);
    float glow = pow(sunward, GODRAY_GLOW_EXP);
    float3 term = pc.godray1.rgb * (occ * glow * GODRAY_STRENGTH);
    return min(term, float3(GODRAY_STRENGTH));
}

[shader("compute")]
[numthreads(SPILL_WG, SPILL_WG, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dst_dim.x || gid.y >= pc.dst_dim.y)
        return;
    // Unwarped output uv (texel centre). Tonemap samples this image at the
    // same unwarped uv, so the bilinear upsample agrees with this write.
    float2 outUv = (float2(gid.xy) + 0.5) / float2(pc.dst_dim);
    float3 rgb = float3(0.0);
    if (pc.bloom_strength > 0.0)
    {
        // Bloom is gathered in the same warped source uv the tonemap HDR tap uses.
        float2 uv = warpSampleUv(outUv, pc.s, pc.atan_s);
        rgb += bloomSpill(uv, float2(gid.xy) + 0.5) * pc.bloom_strength;
    }
    rgb += godrays(outUv);
    spillOut[gid.xy] = float4(rgb, 1.0);
}
