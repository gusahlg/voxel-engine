// Tonemap: display-space sigmoid from linear HDR (sigmoid IS the OETF).
// End-to-end linear rendering (sRGB texture decode, linear mips, linear fog/sky).
//
// The expensive bloom spiral and godray depth march live in the quarter-res
// spill pass (`spill.comp.slang`); this fragment is HDR + one bilinear spill
// tap + sigmoid + vignette.
//
// `-DTAA_FUSED` compiles the present-time TAA resolve into this same pass:
// attachment 0 is the LDR swapchain, attachment 1 is the swapchain-sized
// RGBA16F history. The off variant has no second attachment and no TAA ALU.

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;
};

#ifdef TAA_FUSED
struct PushConstants
{
    float4x4 reproj;      // f64-composed prev_view_proj * inv(cur_view_proj)
    float2 render_extent; // HDR/depth size in texels (may differ from swapchain)
    float2 jitter_px;     // raster jitter this presented frame
    float exposure;
    float s;
    float atan_s;
    float vignette;
    float blend;
    uint history_valid; // 0 first present, after resize, or after TAA toggle
    uint depth_valid;
    uint _pad;
};
#else
struct PushConstants
{
    float exposure; // linear multiplier before sigmoid
    float s;        // cylindrical warp strength (<=0 rectilinear)
    float atan_s;   // precomputed atan(s)
    float vignette; // radial darkening; 0 = off
};
#endif

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Set 0, binding 0: the HDR offscreen, pushed at record time.
[[vk::binding(0, 0)]]
Sampler2D hdrTex;

// Quarter-res bloom+godray spill; one bilinear tap at unwarped output uv.
[[vk::binding(1, 0)]]
Sampler2D spillTex;

#ifdef TAA_FUSED
// Last presented resolved HDR, in presented (swapchain) uv space.
[[vk::binding(2, 0)]]
Sampler2D historyTex;

// This frame's sampleable depth at render resolution (nearest sampler).
[[vk::binding(3, 0)]]
Sampler2D depthTex;
#endif

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

float3 applyVignette(float3 ldr, float2 uv_out)
{
    if (pc.vignette > 0.0)
    {
        float d = length(uv_out - 0.5) * 1.41421356;
        ldr *= 1.0 - pc.vignette * 0.28 * smoothstep(0.5, 1.0, d);
    }
    return ldr;
}

#ifdef TAA_FUSED

// Texel-centre uv ⇄ NDC, matching camera::world_to_screen (GL-style y-up viewport:
// pixel-y-down ↔ ndc-y-up, so ndc.y = 1 − 2·uv.y).
float2 uv_to_ndc(float2 uv) { return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); }
float2 ndc_to_uv(float2 ndc) { return float2(ndc.x * 0.5 + 0.5, (1.0 - ndc.y) * 0.5); }

float3 rgb_to_ycocg(float3 c) { return float3(0.25*c.r + 0.5*c.g + 0.25*c.b, 0.5*(c.r - c.b), -0.25*c.r + 0.5*c.g - 0.25*c.b); }
float3 ycocg_to_rgb(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

// 5-tap YCoCg variance clamp + depth-aware reprojection. `source_uv` is in
// render/HDR space (after warpSampleUv); history is in presented space.
float3 taaResolve(float2 source_uv, float3 current)
{
    float2 texel = 1.0 / pc.render_extent;
    float3 c0 = rgb_to_ycocg(current);
    float3 cn = rgb_to_ycocg(hdrTex.Sample(source_uv + float2(0.0, -texel.y)).rgb);
    float3 cs = rgb_to_ycocg(hdrTex.Sample(source_uv + float2(0.0, texel.y)).rgb);
    float3 cw = rgb_to_ycocg(hdrTex.Sample(source_uv + float2(-texel.x, 0.0)).rgb);
    float3 ce = rgb_to_ycocg(hdrTex.Sample(source_uv + float2(texel.x, 0.0)).rgb);
    float3 m1 = c0 + cn + cs + cw + ce;
    float3 m2 = c0 * c0 + cn * cn + cs * cs + cw * cw + ce * ce;
    float3 mean = m1 / 5.0;
    float3 sigma = sqrt(max(m2 / 5.0 - mean * mean, 0.0));
    float3 nmin = mean - VARIANCE_GAMMA * sigma;
    float3 nmax = mean + VARIANCE_GAMMA * sigma;

    float3 resolved = current;
    if (pc.history_valid != 0)
    {
        float2 ndc = uv_to_ndc(source_uv);
        // Point depth at the jittered pixel (clean ray is offset by jitter/dim).
        // Same rule as the retired compute resolve: `dp = floor(px + jitter + 0.5)`
        // with `px` the render-res pixel whose centre is `source_uv`.
        float2 px = source_uv * pc.render_extent - 0.5;
        int2 dp = int2(floor(px + pc.jitter_px + 0.5));
        dp = clamp(dp, int2(0, 0), int2(pc.render_extent) - 1);
        float2 depth_uv = (float2(dp) + 0.5) / pc.render_extent;
        float d = pc.depth_valid != 0 ? depthTex.Sample(depth_uv).x : 0.0;
        float4 prevClip = mul(pc.reproj, float4(ndc, d, 1.0));
        if (prevClip.w > 0.0)
        {
            float2 prevSourceUv = ndc_to_uv(prevClip.xy / prevClip.w);
            float2 prevPresentedUv = unwarpSampleUv(prevSourceUv, pc.s, pc.atan_s);
            if (all(prevPresentedUv >= 0.0) && all(prevPresentedUv <= 1.0))
            {
                float3 hist = rgb_to_ycocg(historyTex.Sample(prevPresentedUv).rgb);
                hist = clamp(hist, nmin, nmax);
                resolved = lerp(current, ycocg_to_rgb(hist), pc.blend);
            }
        }
    }
    return resolved;
}

struct TaaFragmentOutput
{
    [[vk::location(0)]] float4 ldr;
    [[vk::location(1)]] float4 history;
};

[shader("fragment")]
TaaFragmentOutput fragmentMain(FragmentInput input)
{
    float2 source_uv = warpSampleUv(input.uv, pc.s, pc.atan_s);
    float3 current = hdrTex.Sample(source_uv).rgb;
    float3 resolved = taaResolve(source_uv, current);
    // Spill is pre-exposure; sample at UNwarped output uv so this tap matches
    // the spill texel's outUv.
    float3 hdr = (resolved + spillTex.Sample(input.uv).rgb) * pc.exposure;
    float3 ldr = applyVignette(tonemapSigmoid(hdr), input.uv);

    TaaFragmentOutput o;
    o.ldr = float4(ldr, 1.0);
    o.history = float4(resolved, 1.0);
    return o;
}

#else

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float2 uv = warpSampleUv(input.uv, pc.s, pc.atan_s);
    // Exposure multiplies the linear HDR before the tone curve.
    float3 hdr = hdrTex.Sample(uv).rgb * pc.exposure;
    // Spill is pre-exposure (BLOOM_STRENGTH already applied to the bloom part);
    // sample at UNwarped output uv so this tap matches the spill texel's outUv.
    hdr += spillTex.Sample(input.uv).rgb * pc.exposure;
    float3 ldr = applyVignette(tonemapSigmoid(hdr), input.uv);
    return float4(ldr, 1.0);
}

#endif
