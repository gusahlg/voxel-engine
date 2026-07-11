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
    // Wide-FOV periphery remap (camera::WarpPush). The scene is rendered into the
    // offscreen at a WIDER rectilinear horizontal fov; here each output pixel maps
    // its presented-NDC.x back to the source-NDC.x it should sample, compressing the
    // periphery. `s <= 0` is rectilinear: the sample uv is untouched (no-op).
    float s;      // cylindrical strength
    float atan_s; // precomputed atan(s)
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

// Presented-NDC.x -> source-NDC.x. Inverse of the forward map atan(qx*s)/atan(s):
// qx = tan(px * atan(s)) / s. Mirrors camera::WarpMap::unwarp_ndc. Horizontal only.
float2 warpSampleUv(float2 uv, float s, float atan_s)
{
    if (s <= 0.0)
        return uv;
    float px = uv.x * 2.0 - 1.0;         // uv -> presented NDC x
    float qx = tan(px * atan_s) / s;     // -> source NDC x (stays in [-1,1])
    return float2(qx * 0.5 + 0.5, uv.y); // -> source uv; vertical unchanged
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float2 uv = warpSampleUv(input.uv, pc.s, pc.atan_s);
    float3 hdr = hdrTex.Sample(uv).rgb * pc.exposure;
    return float4(linearToSrgb(pbrNeutral(hdr)), 1.0);
}
