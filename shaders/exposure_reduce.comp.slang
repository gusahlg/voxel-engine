// exposure_reduce.comp.slang — first stage of scene metering.
// One thread per 16x16-texel tile. Averages log2(luma) over tile, writes to buffer.
// CPU does final reduction. log2 space (not linear) to avoid firefly domination.

// Rec. 709 luma weights (linear-light). Matches the tonemap's working space.
static const float3 LUMA_709 = float3(0.2126, 0.7152, 0.0722);
static const uint TILE = 16;
static const float LUMA_FLOOR = 1e-4; // keeps log2 finite on black tiles

[[vk::binding(0, 0)]] Sampler2D hdrTex;                 // the linear-HDR offscreen
[[vk::binding(1, 0)]] RWStructuredBuffer<float> tileMeans; // one entry per tile

struct Push
{
    uint2 hdr_dim; // HDR target size in texels
    uint2 tiles;   // tile-grid dimensions = ceil(hdr_dim / TILE)
};
[[vk::push_constant]] Push pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.tiles.x || gid.y >= pc.tiles.y)
        return;

    uint2 origin = gid.xy * TILE;
    float sum_log2 = 0.0;
    uint counted = 0;
    for (uint ty = 0; ty < TILE; ++ty)
    {
        uint py = origin.y + ty;
        if (py >= pc.hdr_dim.y)
            break;
        for (uint tx = 0; tx < TILE; ++tx)
        {
            uint px = origin.x + tx;
            if (px >= pc.hdr_dim.x)
                break;
            float2 uv = (float2(px, py) + 0.5) / float2(pc.hdr_dim);
            float3 hdr = hdrTex.SampleLevel(uv, 0).rgb;
            float luma = max(dot(hdr, LUMA_709), LUMA_FLOOR);
            sum_log2 += log2(luma);
            counted += 1;
        }
    }

    float mean_log2 = counted > 0 ? sum_log2 / float(counted) : log2(LUMA_FLOOR);
    tileMeans[gid.y * pc.tiles.x + gid.x] = mean_log2;
}
