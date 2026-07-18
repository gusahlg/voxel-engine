// taa_resolve.comp.slang — temporal anti-aliasing resolve.
// Reproject previous frame, neighbourhood-clamp, blend. Writes history and resolved output.
// All matrices are CLEAN (jitter only at rasterization). Reprojection is depth-aware
// so camera translation reprojects correctly at all distances.

#include "generated/shader_constants.slang" // VARIANCE_GAMMA

// Descriptor set layout: current=0, history=1, reproj UBO=2, output=3.
// The resolved-history output rides binding 3 of the same set.
[[vk::binding(0, 0)]] Sampler2D currentTex;   // this frame's HDR offscreen
[[vk::binding(1, 0)]] Sampler2D historyTex;   // last frame's resolved output

struct Reproj
{
    float4x4 prev_view_proj; // maps CURRENT camera-relative world → PREVIOUS clip
    float4 camera_delta;     // xyz = prev_origin − cur_origin (informational)
};
[[vk::binding(2, 0)]] ConstantBuffer<Reproj> reproj;

[[vk::binding(3, 0)]] RWTexture2D<float4> outHistory; // resolved integrator (write)

// This frame's depth (reversed-Z, 0 = far/sky), single-sampled only.
// Under MSAA the host binds a dummy and clears `depth_valid` instead.
[[vk::binding(4, 0)]] Sampler2D depthTex;

struct Push
{
    float4x4 inv_cur_view_proj; // current inverse view-proj
    uint2 dim;                  // output resolution in texels
    float2 jitter_px;           // raster jitter this frame
    float blend;                // history feedback weight
    uint history_valid;         // 0 first frame or after resize
    uint depth_valid;           // 0 under MSAA: use far-plane unproject
};
[[vk::push_constant]] Push pc;

// Texel-centre uv ⇄ NDC, matching camera::world_to_screen (GL-style y-up viewport:
// pixel-y-down ↔ ndc-y-up, so ndc.y = 1 − 2·uv.y).
float2 uv_to_ndc(float2 uv) { return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); }
float2 ndc_to_uv(float2 ndc) { return float2(ndc.x * 0.5 + 0.5, (1.0 - ndc.y) * 0.5); }

// YCoCg ⇄ RGB. Variance clamp in YCoCg bounds luma independently of chroma.
float3 rgb_to_ycocg(float3 c) { return float3(0.25*c.r + 0.5*c.g + 0.25*c.b, 0.5*(c.r - c.b), -0.25*c.r + 0.5*c.g - 0.25*c.b); }
float3 ycocg_to_rgb(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dim.x || gid.y >= pc.dim.y)
        return;

    float2 texel = float2(pc.dim);
    float2 uv = (float2(gid.xy) + 0.5) / texel;
    float3 current = currentTex.SampleLevel(uv, 0).rgb;

    // Neighbourhood statistics (3×3) in YCoCg — avoid jitter re-injection crawl
    // by using mean±stddev bounds instead of raw min/max from jittered samples.
    float3 m1 = float3(0.0);
    float3 m2 = float3(0.0);
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            float2 nuv = uv + float2(dx, dy) / texel;
            float3 c = rgb_to_ycocg(currentTex.SampleLevel(nuv, 0).rgb);
            m1 += c;
            m2 += c * c;
        }
    }
    float3 mean = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    float3 nmin = mean - VARIANCE_GAMMA * sigma;
    float3 nmax = mean + VARIANCE_GAMMA * sigma;

    float3 resolved = current;
    if (pc.history_valid != 0)
    {
        // Unproject current texel, then reproject to previous frame clip space.
        float2 ndc = uv_to_ndc(uv);
        // Depth sampled at jittered position (clean ray passes jitter/dim offset).
        float2 depth_uv = uv + pc.jitter_px / float2(pc.dim);
        float d = pc.depth_valid != 0 ? depthTex.SampleLevel(depth_uv, 0).x : 0.0;
        // Reproject in homogeneous space (avoids inf on far plane).
        float4 worldH = mul(pc.inv_cur_view_proj, float4(ndc, d, 1.0));
        float4 prevClip = mul(reproj.prev_view_proj, worldH);
        if (prevClip.w > 0.0)
        {
            float2 prevUv = ndc_to_uv(prevClip.xy / prevClip.w);
            if (all(prevUv >= 0.0) && all(prevUv <= 1.0))
            {
                float3 hist = rgb_to_ycocg(historyTex.SampleLevel(prevUv, 0).rgb);
                hist = clamp(hist, nmin, nmax);
                resolved = lerp(current, ycocg_to_rgb(hist), pc.blend);
            }
        }
    }

    outHistory[gid.xy] = float4(resolved, 1.0);
}
