// Tonemap: display-space sigmoid from linear HDR (sigmoid IS the OETF).
// End-to-end linear rendering (sRGB texture decode, linear mips, linear fog/sky).
//
// The expensive bloom spiral and godray depth march live in the quarter-res
// spill pass (`spill.comp.slang`); this fragment is HDR + one bilinear spill
// tap + sigmoid + vignette.
//
// `-DTAA_FUSED` compiles the present-time TAAU resolve into this same pass:
// attachment 0 is the LDR swapchain, attachment 1 is the swapchain-sized
// RGBA16F history. Raster jitter means render texel t (centre t+0.5) holds
// the unjittered scene at s_t = t + 0.5 - jitter_px; each output pixel is
// reconstructed from a 3×3 of those texels with a Gaussian in OUTPUT pixels
// (UE σ = 0.47, w = exp(-2.29 |d_out|²), d_out = (s_t - p) * (output/render)
// per axis). At ratio 1 this matches the previous 0.47-render-px kernel; when
// upsampling the kernel stays 0.47 output px so `cur` is not a 2×-wide
// low-pass (which also kept w_max ≈ 1 and overwrote history). History is a
// 5-tap Catmull-Rom (Jimenez 2016 Filmic SMAA). Current-frame weight is
// (1-blend)*w_max; a Karis 2014 velocity boost is a runtime push-constant
// (default boost 1.0 = off) so camera rotation does not re-inject the
// low-res `cur` at ~30%/present. The off variant has no second attachment
// and no TAA ALU.

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;
    float4 frag_pos : SV_Position;
};

#ifdef TAA_FUSED
struct PushConstants
{
    float4x4 reproj;      // f64-composed prev_view_proj * inv(cur_view_proj)
    float2 render_extent; // HDR/depth size in texels (may differ from swapchain)
    float2 jitter_px;     // raster jitter this presented frame
    float2 output_extent; // swapchain / history size in texels
    float exposure;
    float s;
    float atan_s;
    float vignette;
    float blend;
    uint history_valid; // 0 first present, after resize, or after TAA toggle
    uint depth_valid;
    float motion_boost; // 1 = off (Karis velocity feedback disabled)
    float motion_px;    // saturate(v / motion_px) lerp factor
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

// 5-tap Catmull-Rom (Jimenez 2016 Filmic SMAA): 16-tap cubic → 9 bilinear
// taps at texel-centre offsets; drop the four corners and renormalise.
// History is swapchain-sized (`pc.output_extent`); linear sampler. Negative
// lobes can overshoot — the caller neighbourhood-clamps in YCoCg.
float3 historyCatmullRom(float2 uv)
{
    float2 texel = 1.0 / pc.output_extent;
    float2 samplePos = uv * pc.output_extent;
    float2 texPos1 = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - texPos1;

    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / w12;

    float2 texPos0 = (texPos1 - 1.0) * texel;
    float2 texPos3 = (texPos1 + 2.0) * texel;
    float2 texPos12 = (texPos1 + offset12) * texel;

    float wT = w12.x * w0.y;
    float wL = w0.x * w12.y;
    float wC = w12.x * w12.y;
    float wR = w3.x * w12.y;
    float wB = w12.x * w3.y;

    float3 result = 0.0;
    result += historyTex.SampleLevel(float2(texPos12.x, texPos0.y), 0).rgb * wT;
    result += historyTex.SampleLevel(float2(texPos0.x, texPos12.y), 0).rgb * wL;
    result += historyTex.SampleLevel(float2(texPos12.x, texPos12.y), 0).rgb * wC;
    result += historyTex.SampleLevel(float2(texPos3.x, texPos12.y), 0).rgb * wR;
    result += historyTex.SampleLevel(float2(texPos12.x, texPos3.y), 0).rgb * wB;
    return result / (wT + wL + wC + wR + wB);
}

// 3×3 Gaussian reconstruction in output pixels + unweighted YCoCg variance
// clamp + depth-aware reprojection. `source_uv` is in render/HDR space (after
// warpSampleUv); `presented_uv` is swapchain uv (input.uv). History is
// Catmull-Rom sampled in presented space. Current-frame weight is
// (1-blend)*w_max, optionally raised with pixel-space velocity when
// `pc.motion_boost` > 1 (default 1 = off).
float3 taaResolve(float2 source_uv, float2 presented_uv)
{
    float2 p = source_uv * pc.render_extent;
    float2 to_output = pc.output_extent / pc.render_extent;

    // Nearest texel whose unjittered sample s_t ≈ p. Identical to the old
    // depth rule `dp = floor(px + jitter + 0.5)` with `px = p - 0.5`.
    int2 tc = int2(floor(p + pc.jitter_px));
    int2 tmin = int2(0, 0);
    int2 tmax = int2(pc.render_extent) - 1;

    float3 acc = 0.0;
    float w_sum = 0.0;
    float w_max = 0.0;
    float3 m1 = 0.0;
    float3 m2 = 0.0;
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int2 t = clamp(tc + int2(dx, dy), tmin, tmax);
            // Texel-centre sample: one texel, not a bilinear mix of neighbours.
            float2 uv = (float2(t) + 0.5) / pc.render_extent;
            float3 c = rgb_to_ycocg(hdrTex.SampleLevel(uv, 0).rgb);
            float2 d_out = ((float2(t) + 0.5 - pc.jitter_px) - p) * to_output;
            float w = exp(-2.29 * dot(d_out, d_out));
            acc += w * c;
            w_sum += w;
            w_max = max(w_max, w);
            m1 += c;
            m2 += c * c;
        }
    }
    float3 cur = acc / w_sum;
    float3 mean = m1 / 9.0;
    float3 nsigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    float3 nmin = mean - VARIANCE_GAMMA * nsigma;
    float3 nmax = mean + VARIANCE_GAMMA * nsigma;

    float3 resolved = cur;
    if (pc.history_valid != 0)
    {
        float2 ndc = uv_to_ndc(source_uv);
        int2 dp = clamp(tc, tmin, tmax);
        float2 depth_uv = (float2(dp) + 0.5) / pc.render_extent;
        float z = pc.depth_valid != 0 ? depthTex.Sample(depth_uv).x : 0.0;
        float4 prevClip = mul(pc.reproj, float4(ndc, z, 1.0));
        if (prevClip.w > 0.0)
        {
            float2 prevSourceUv = ndc_to_uv(prevClip.xy / prevClip.w);
            float2 prevPresentedUv = unwarpSampleUv(prevSourceUv, pc.s, pc.atan_s);
            if (all(prevPresentedUv >= 0.0) && all(prevPresentedUv <= 1.0))
            {
                float3 hist = rgb_to_ycocg(historyCatmullRom(prevPresentedUv));
                hist = clamp(hist, nmin, nmax);
                float2 hp = prevPresentedUv * pc.output_extent;
                float v = length(hp - presented_uv * pc.output_extent);
                float cur_w = (1.0 - pc.blend) * w_max;
                cur_w = lerp(cur_w, min(1.0, cur_w * pc.motion_boost), saturate(v / pc.motion_px));
                resolved = lerp(hist, cur, cur_w);
            }
        }
    }
    return ycocg_to_rgb(resolved);
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
    float3 resolved = taaResolve(source_uv, input.uv);
    // Spill is pre-exposure; sample at UNwarped output uv so this tap matches
    // the spill texel's outUv.
    float3 hdr = (resolved + spillTex.Sample(input.uv).rgb) * pc.exposure;
    float3 ldr = applyVignette(tonemapSigmoid(hdr), input.uv);
    // Output-space dither after the OETF, swapchain only (not TAA history).
    ldr += (hash12(input.frag_pos.xy * 0.937) - 0.5) / 255.0;

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
    // Output-space dither after the OETF, immediately before the 8-bit write.
    ldr += (hash12(input.frag_pos.xy * 0.937) - 0.5) / 255.0;
    return float4(ldr, 1.0);
}

#endif
