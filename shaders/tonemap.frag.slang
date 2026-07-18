// Tonemap: display-space sigmoid from linear HDR (sigmoid IS the OETF).
// End-to-end linear rendering (sRGB texture decode, linear mips, linear fog/sky).

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;
    // Pixel coordinates for bloom spiral rotation (rasterizer system value).
    float4 frag_coord : SV_Position;
};

struct PushConstants
{
    float exposure;   // linear multiplier before sigmoid
    float s;          // cylindrical warp strength (<=0 rectilinear)
    float atan_s;     // precomputed atan(s)
    float _pad0;      // float4 alignment
    // Godrays: sun screen pos, strength gate.
    float4 godray0;
    float4 godray1;
    float vignette; // radial darkening; 0 = off
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Set 0, binding 0: the HDR offscreen, pushed at record time.
[[vk::binding(0, 0)]]
Sampler2D hdrTex;

// Bloom mip pyramid; sampled via golden-angle spiral, added pre-sigmoid.
[[vk::binding(1, 0)]]
Sampler2D bloomTex;

// Scene depth; point-sampled for sky mask (reversed-Z ⇒ depth ~0).
[[vk::binding(2, 0)]]
Sampler2D depthTex;

// Golden-angle spiral gather; spatial rotation only (no temporal jitter).
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
        acc += bloomTex.SampleLevel(uv + off, BLOOM_SPIRAL_LOD).rgb;
    }
    return acc / BLOOM_SPIRAL_SAMPLES;
}

// Display-space sigmoid: linear HDR to encoded output (curve + OETF fused).
float3 tonemapSigmoid(float3 x)
{
    float3 c = 1.4 * x;
    c = c / pow(pow(c, 2.5) + 1.0, 0.4);
    return pow(c, 1.15);
}

// Remap output x-coord to source uv, compressing wide periphery. Horizontal only.
float2 warpSampleUv(float2 uv, float s, float atan_s)
{
    if (s <= 0.0)
        return uv;
    float px = uv.x * 2.0 - 1.0;
    float qx = tan(px * atan_s) / s;
    return float2(qx * 0.5 + 0.5, uv.y);
}

// Screen-space godrays: march toward sun, accumulate sky pixels (depth ~0),
// radial falloff. Returns pre-exposure light (scaled by exposure by caller).
float3 godrays(float2 outUv, float2 frag_coord)
{
    if (pc.godray0.z <= 0.0)
        return float3(0.0);
    float2 toSun = pc.godray0.xy - outUv;
    float sunDist = length(toSun);
    float2 dir = toSun / max(sunDist, 1.0e-6);
    // March toward sun with fixed half-step offset for first sample.
    float2 step = dir * (min(sunDist, GODRAY_MAX_LEN) / GODRAY_SAMPLES);
    float jitter = 0.5;
    int n = int(GODRAY_SAMPLES);
    float occ = 0.0;
    for (int i = 0; i < n; ++i)
    {
        float2 p = outUv + step * (float(i) + jitter);
        // Sample depth through same warp as colour to match perspective.
        // Depth is jittered; correct sample to avoid TAA shimmer.
        float2 duv = warpSampleUv(p, pc.s, pc.atan_s) + float2(pc.godray0.w, pc.godray1.w);
        float d = depthTex.SampleLevel(duv, 0.0).r;
        occ += (d < 1.0e-5) ? 1.0 : 0.0;
    }
    occ /= GODRAY_SAMPLES;
    float sunward = saturate(1.0 - sunDist / GODRAY_MAX_LEN);
    float glow = pow(sunward, GODRAY_GLOW_EXP);
    // Clamp godray term to prevent bright sun tint blowout.
    float3 term = pc.godray1.rgb * (occ * glow * GODRAY_STRENGTH);
    return min(term, float3(GODRAY_STRENGTH));
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float2 uv = warpSampleUv(input.uv, pc.s, pc.atan_s);
    // Exposure multiplies the linear HDR before the tone curve.
    float3 hdr = hdrTex.Sample(uv).rgb * pc.exposure;
    // Bloom pre-sigmoid (scale by exposure).
    hdr += bloomSpill(uv, input.frag_coord.xy) * (pc.exposure * BLOOM_STRENGTH);
    // Godrays also pre-sigmoid, independent additive term (scale by exposure).
    hdr += godrays(input.uv, input.frag_coord.xy) * pc.exposure;
    float3 ldr = tonemapSigmoid(hdr);

    // Radial vignette.
    if (pc.vignette > 0.0)
    {
        float d = length(input.uv - 0.5) * 1.41421356;
        ldr *= 1.0 - pc.vignette * 0.28 * smoothstep(0.5, 1.0, d);
    }

    return float4(ldr, 1.0);
}
