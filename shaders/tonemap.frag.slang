// tonemap.frag.slang — samples the linear-HDR offscreen, applies exposure and
// the Khronos PBR Neutral tone map, then encodes to sRGB. The scene is now
// rendered in linear light end-to-end (sRGB texture decode, linear mips, linear
// fog/sky), so no input decode is needed here. PBR Neutral compresses highlights
// while preserving hue and saturation — the right fit for a colorful voxel world
// (unlike AgX, which desaturates highlights filmically). The swapchain is a
// UNORM surface with an sRGB-nonlinear color space, so this shader owns the OETF.

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;
};

struct PushConstants
{
    float exposure; // linear multiplier applied before the curve
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

float3 linearToSrgb(float3 c)
{
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(max(c, 0.0), float3(1.0 / 2.4)) - 0.055;
    return select(c <= 0.0031308, lo, hi);
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float3 hdr = hdrTex.Sample(input.uv).rgb * pc.exposure;
    return float4(linearToSrgb(pbrNeutral(hdr)), 1.0);
}
