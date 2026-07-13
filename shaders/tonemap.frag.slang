// tonemap.frag.slang — samples the linear-HDR offscreen, applies exposure and
// a display-space sigmoid, which maps linear HDR straight to
// display-encoded output (it IS the OETF — no separate sRGB encode). The scene
// is rendered in linear light end-to-end (sRGB texture decode, linear mips,
// linear fog/sky). Chosen by the L10/L1 verdict (2026-07-11) over Khronos PBR
// Neutral + sRGB OETF: the sigmoid gave better day contrast, true cave-black,
// and dark-but-legible night. The swapchain is a UNORM surface with an
// sRGB-nonlinear color space, so this shader owns the OETF via the sigmoid.

// common.slang owns the ONE `dither()` (ordered Bayer, unit amplitude) and the
// generated `POST_TONEMAP_DITHER_GAIN`. Post-tonemap dither is done here
// in OUTPUT space so its amplitude is a fixed fraction of an 8-bit LSB,
// independent of exposure and the tone curve. Dithering in HDR before the curve
// would let exposure scale the noise, causing visible banding.
#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;
    // Output-pixel coordinate for the ordered dither cell, supplied by
    // the fullscreen vertex stage as the rasteriser system value.
    float4 frag_coord : SV_Position;
};

struct PushConstants
{
    float exposure;   // linear multiplier before the curve
    float s;          // cylindrical warp strength (<=0 is rectilinear)
    float atan_s;     // precomputed atan(s)
    float dither_phase; // temporal dither phase
    // Godrays: sun screen pos (xy), strength gate (z), tint in godray1.
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

// Display-space sigmoid: linear HDR → display-encoded output in one step
// (curve + OETF fused). The 1.4 pre-gain, 2.5/0.4 sigmoid shoulder, and 1.15
// output gamma are from the L10/L1 verdict.
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
    // March covers up to GODRAY_MAX_LEN toward the sun; the dither offsets the
    // first sample so the low tap count doesn't band.
    float2 step = dir * (min(sunDist, GODRAY_MAX_LEN) / GODRAY_SAMPLES);
    float jitter = dither(pc.dither_phase, frag_coord) + 0.5; // [0,1)
    int n = int(GODRAY_SAMPLES);
    float occ = 0.0;
    for (int i = 0; i < n; ++i)
    {
        float2 p = outUv + step * (float(i) + jitter);
        // Depth lives in offscreen (unwarped) space, so sample through the same
        // periphery remap as the colour. Reversed-Z: sky is the cleared 0.0.
        float d = depthTex.SampleLevel(warpSampleUv(p, pc.s, pc.atan_s), 0.0).r;
        occ += (d < 1.0e-5) ? 1.0 : 0.0;
    }
    occ /= GODRAY_SAMPLES;
    float sunward = saturate(1.0 - sunDist / GODRAY_MAX_LEN);
    float glow = pow(sunward, GODRAY_GLOW_EXP);
    return pc.godray1.rgb * (occ * glow * GODRAY_STRENGTH);
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

    // Post-tonemap dither (output space, after tone curve). Amplitude is
    // independent of exposure to avoid scaling quantization noise.
    ldr += dither(pc.dither_phase, input.frag_coord.xy) * POST_TONEMAP_DITHER_GAIN;
    return float4(ldr, 1.0);
}
