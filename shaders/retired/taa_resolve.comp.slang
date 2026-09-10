// Retired: TAA now resolves in the present-time tonemap fragment
// (`shaders/tonemap.frag.slang`, `-DTAA_FUSED`). Quarantined under
// `shaders/retired/`; not in build.rs shipping jobs. Kept as the original
// 5-tap YCoCg / depth-aware reprojection reference.
//
// taa_resolve.comp.slang — temporal anti-aliasing resolve.
// 16×16 groupshared tile of the current HDR (1-pixel apron), 5-tap YCoCg
// variance clamp, depth-aware reprojection. All matrices are CLEAN (jitter only
// at rasterization). The host f64-composes prev*inv(cur) into one push matrix.

#include "generated/shader_constants.slang" // VARIANCE_GAMMA, TAA_TILE

// Descriptor set layout: current=0, history=1, output=2, depth=3.
[[vk::binding(0, 0)]] Sampler2D currentTex;   // this frame's HDR offscreen
[[vk::binding(1, 0)]] Sampler2D historyTex;   // last frame's resolved output

[[vk::binding(2, 0)]] RWTexture2D<float4> outHistory; // resolved integrator (write)

// This frame's depth (reversed-Z, 0 = far/sky). Point-sampled: the host binds a
// nearest sampler so reversed-Z is not interpolated.
[[vk::binding(3, 0)]] Sampler2D depthTex;

struct Push
{
    float4x4 reproj;    // f64-composed prev_view_proj * inv(cur_view_proj)
    uint2 dim;          // output resolution in texels
    float2 jitter_px;   // raster jitter this frame
    float blend;        // history feedback weight
    uint history_valid; // 0 first frame or after resize
    uint depth_valid;   // 0: use far-plane unproject
};
[[vk::push_constant]] Push pc;

// Texel-centre uv ⇄ NDC, matching camera::world_to_screen (GL-style y-up viewport:
// pixel-y-down ↔ ndc-y-up, so ndc.y = 1 − 2·uv.y).
float2 uv_to_ndc(float2 uv) { return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); }
float2 ndc_to_uv(float2 ndc) { return float2(ndc.x * 0.5 + 0.5, (1.0 - ndc.y) * 0.5); }

// YCoCg ⇄ RGB. Variance clamp in YCoCg bounds luma independently of chroma.
float3 rgb_to_ycocg(float3 c) { return float3(0.25*c.r + 0.5*c.g + 0.25*c.b, 0.5*(c.r - c.b), -0.25*c.r + 0.5*c.g - 0.25*c.b); }
float3 ycocg_to_rgb(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

// 16×16 tile + 1-pixel apron.
groupshared float4 tile[(TAA_TILE + 2) * (TAA_TILE + 2)];

[shader("compute")]
[numthreads(TAA_TILE, TAA_TILE, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uint3 wid: SV_GroupID)
{
    const uint apron = TAA_TILE + 2;
    int2 origin = int2(wid.xy * TAA_TILE) - 1;
    uint linear = lid.y * TAA_TILE + lid.x;
    uint apron_n = apron * apron;
    uint tile_n = TAA_TILE * TAA_TILE;
    // Cooperative load of the 18×18 neighbourhood (all threads must participate).
    for (uint i = linear; i < apron_n; i += tile_n)
    {
        uint ty = i / apron;
        uint tx = i - ty * apron;
        int2 px = origin + int2(int(tx), int(ty));
        px = clamp(px, int2(0, 0), int2(pc.dim) - 1);
        float2 uv = (float2(px) + 0.5) / float2(pc.dim);
        tile[i] = float4(currentTex.SampleLevel(uv, 0).rgb, 1.0);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gid.x >= pc.dim.x || gid.y >= pc.dim.y)
        return;

    uint2 t = lid.xy + 1;
    float3 current = tile[t.y * apron + t.x].rgb;

    // 5-tap cross neighbourhood in YCoCg (centre + N/S/E/W).
    float3 c0 = rgb_to_ycocg(current);
    float3 cn = rgb_to_ycocg(tile[(t.y - 1) * apron + t.x].rgb);
    float3 cs = rgb_to_ycocg(tile[(t.y + 1) * apron + t.x].rgb);
    float3 cw = rgb_to_ycocg(tile[t.y * apron + (t.x - 1)].rgb);
    float3 ce = rgb_to_ycocg(tile[t.y * apron + (t.x + 1)].rgb);
    float3 m1 = c0 + cn + cs + cw + ce;
    float3 m2 = c0 * c0 + cn * cn + cs * cs + cw * cw + ce * ce;
    float3 mean = m1 / 5.0;
    float3 sigma = sqrt(max(m2 / 5.0 - mean * mean, 0.0));
    float3 nmin = mean - VARIANCE_GAMMA * sigma;
    float3 nmax = mean + VARIANCE_GAMMA * sigma;

    float2 texel = float2(pc.dim);
    float2 uv = (float2(gid.xy) + 0.5) / texel;
    float3 resolved = current;
    if (pc.history_valid != 0)
    {
        float2 ndc = uv_to_ndc(uv);
        // Point depth at the jittered pixel (clean ray is offset by jitter/dim).
        int2 dp = int2(floor(float2(gid.xy) + pc.jitter_px + 0.5));
        dp = clamp(dp, int2(0, 0), int2(pc.dim) - 1);
        float2 depth_uv = (float2(dp) + 0.5) / texel;
        float d = pc.depth_valid != 0 ? depthTex.SampleLevel(depth_uv, 0).x : 0.0;
        float4 prevClip = mul(pc.reproj, float4(ndc, d, 1.0));
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
