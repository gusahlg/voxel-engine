// tonemap.frag.slang — samples the linear-HDR offscreen, applies exposure and
// the Khronos PBR Neutral tone map, then encodes to sRGB. The scene is now
// rendered in linear light end-to-end (sRGB texture decode, linear mips, linear
// fog/sky), so no input decode is needed here. PBR Neutral compresses highlights
// while preserving hue and saturation — the right fit for a colorful voxel world
// (unlike AgX, which desaturates highlights filmically). The swapchain is a
// UNORM surface with an sRGB-nonlinear color space, so this shader owns the OETF.

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
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Set 0, binding 0: the HDR offscreen, pushed at record time.
[[vk::binding(0, 0)]]
Sampler2D hdrTex;

// Khronos PBR Neutral tone mapping (KhronosGroup/ToneMapping). Input and output
// are linear; highlights roll off toward white with a controlled desaturation.
float3 pbrNeutral(float3 color)
{
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.05;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression)
        return color;

    float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return lerp(color, newPeak.xxx, g);
}

// Shader display-space sigmoid (replaces pbrNeutral + linearToSrgb).
float3 makeupSigmoid(float3 x)
{
    float3 c = 1.4 * x;
    c = c / pow(pow(c, 2.5) + 1.0, 0.4);
    return pow(c, 1.15);
}

float3 linearToSrgb(float3 c)
{
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(max(c, 0.0), float3(1.0 / 2.4)) - 0.055;
    return select(c <= 0.0031308, lo, hi);
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

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float2 uv = warpSampleUv(input.uv, pc.s, pc.atan_s);
    // Exposure multiplies the linear HDR before the tone curve.
    float3 hdr = hdrTex.Sample(uv).rgb * pc.exposure;
#ifdef MAKEUP_SIGMOID
    float3 ldr = makeupSigmoid(hdr);
#else
    float3 ldr = linearToSrgb(pbrNeutral(hdr));
#endif

    // Post-tonemap dither (output space, after tone curve). Amplitude is
    // independent of exposure to avoid scaling quantization noise.
    ldr += dither(pc.dither_phase, input.frag_coord.xy) * POST_TONEMAP_DITHER_GAIN;
    return float4(ldr, 1.0);
}
