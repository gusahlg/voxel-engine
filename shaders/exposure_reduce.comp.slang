// exposure_reduce.comp.slang — first stage of scene metering.
// One 8×8 workgroup per 16×16-texel tile. Each thread sums a 2×2 of the tile,
// then a groupshared tree reduces to one mean log2(luma). CPU does the final
// reduction over tiles. log2 space (not linear) to avoid firefly domination.

#include "common.slang"

// Tile size from genconst; CPU divides the same way.
static const uint TILE = EXPOSURE_TILE;
// LUMA_FLOOR from genconst (keeps log2 finite on black tiles).

[[vk::binding(0, 0)]] Sampler2D hdrTex;                 // the linear-HDR offscreen
[[vk::binding(1, 0)]] RWStructuredBuffer<float> tileMeans; // one entry per tile

struct Push
{
    uint2 hdr_dim; // HDR target size in texels
    uint2 tiles;   // tile-grid dimensions = ceil(hdr_dim / TILE)
};
[[vk::push_constant]] Push pc;

groupshared float gs_sum[64];
groupshared uint gs_cnt[64];

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 lid: SV_GroupThreadID, uint3 wid: SV_GroupID)
{
    uint2 origin = wid.xy * TILE;
    uint sub = TILE / 8; // 2 when TILE=16: each thread covers a 2×2 of the tile
    float sum_log2 = 0.0;
    uint counted = 0;
    for (uint ty = 0; ty < sub; ++ty)
    {
        for (uint tx = 0; tx < sub; ++tx)
        {
            uint2 px = origin + lid.xy * sub + uint2(tx, ty);
            if (px.x >= pc.hdr_dim.x || px.y >= pc.hdr_dim.y)
                continue;
            float2 uv = (float2(px) + 0.5) / float2(pc.hdr_dim);
            float3 hdr = hdrTex.SampleLevel(uv, 0).rgb;
            float luma = max(luma709(hdr), LUMA_FLOOR);
            sum_log2 += log2(luma);
            counted += 1;
        }
    }

    uint idx = lid.y * 8 + lid.x;
    gs_sum[idx] = sum_log2;
    gs_cnt[idx] = counted;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 32; s > 0; s >>= 1)
    {
        if (idx < s)
        {
            gs_sum[idx] += gs_sum[idx + s];
            gs_cnt[idx] += gs_cnt[idx + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (idx == 0 && wid.x < pc.tiles.x && wid.y < pc.tiles.y)
    {
        float mean_log2 = gs_cnt[0] > 0 ? gs_sum[0] / float(gs_cnt[0]) : log2(LUMA_FLOOR);
        tileMeans[wid.y * pc.tiles.x + wid.x] = mean_log2;
    }
}
