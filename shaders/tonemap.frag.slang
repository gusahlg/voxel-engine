// Tonemap: display-space sigmoid from linear HDR (sigmoid IS the OETF).
// End-to-end linear rendering (sRGB texture decode, linear mips, linear fog/sky).
//
// The expensive bloom spiral and godray depth march live in the quarter-res
// spill pass (`spill.comp.slang`); this fragment is HDR + one bilinear spill
// tap + sigmoid + vignette.

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;
};

struct PushConstants
{
    float exposure; // linear multiplier before sigmoid
    float s;        // cylindrical warp strength (<=0 rectilinear)
    float atan_s;   // precomputed atan(s)
    float vignette; // radial darkening; 0 = off
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Set 0, binding 0: the HDR offscreen, pushed at record time.
[[vk::binding(0, 0)]]
Sampler2D hdrTex;

// Quarter-res bloom+godray spill; one bilinear tap at unwarped output uv.
[[vk::binding(1, 0)]]
Sampler2D spillTex;

// Display-space sigmoid: linear HDR to encoded output (curve + OETF fused).
//
// Algebraic rewrite of the original
//   c = 1.4 x
//   y = c / (c^2.5 + 1)^0.4
//   return y^1.15
// as 2 log2 + 2 exp2 per channel:
//   c^2.5 = exp2(2.5 log2 c)
//   log y = log2 c - 0.4 log2(c^2.5 + 1)
//   y^1.15 = exp2(1.15 log y)
// Guard c <= 0 → 0 (log2 of 0/neg is undefined). Host test in color.rs.
float3 tonemapSigmoid(float3 x)
{
    float3 c = 1.4 * x;
    bool3 ok = c > 0.0;
    float3 logc = log2(select(ok, c, float3(1.0)));
    float3 t = logc - 0.4 * log2(exp2(2.5 * logc) + 1.0);
    return select(ok, exp2(1.15 * t), float3(0.0));
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float2 uv = warpSampleUv(input.uv, pc.s, pc.atan_s);
    // Exposure multiplies the linear HDR before the tone curve.
    float3 hdr = hdrTex.Sample(uv).rgb * pc.exposure;
    // Spill is pre-exposure (BLOOM_STRENGTH already applied to the bloom part);
    // sample at UNwarped output uv so this tap matches the spill texel's outUv.
    hdr += spillTex.Sample(input.uv).rgb * pc.exposure;
    float3 ldr = tonemapSigmoid(hdr);

    // Radial vignette.
    if (pc.vignette > 0.0)
    {
        float d = length(input.uv - 0.5) * 1.41421356;
        ldr *= 1.0 - pc.vignette * 0.28 * smoothstep(0.5, 1.0, d);
    }

    return float4(ldr, 1.0);
}
