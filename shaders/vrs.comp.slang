// Variable-rate-shading classifier: one thread per shading-rate tile.
//
// Reads the previous-cycle depth buffer (reversed-Z, infinite far: d = near/z,
// so d→0 is far and the cleared sky is exactly d = 0) and writes an R8_UINT
// rate code. Encoding: (log2(w) << 2) | log2(h), i.e. 0 = 1x1, 0b0101 = 2x2,
// 0b1010 = 4x4.
//
// Sky (depth == 0 everywhere) is 2x2, or 4x4 when the device advertises it.
// Far + flat terrain stays 2x2. Near or discontinuous tiles stay 1x1.
// One-frame-stale depth is OK because the output is conservative: full-rate
// tiles are dilated by one, and a tile that was near last frame is never
// coarsened. Tile/depth extents come from push constants so GetDimensions
// stays out of the sample loop.

[[vk::binding(0, 0)]] Sampler2D depthTex;
[[vk::image_format("r8ui")]] [[vk::binding(1, 0)]] RWTexture2D<uint> rateOut;
[[vk::image_format("r8ui")]] [[vk::binding(2, 0)]] RWTexture2D<uint> history;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> mixCounts; // [1x1, 2x2, 4x4]

struct Push {
    float dThreshold; // far/flat iff tile depth < this
    uint texelW;      // framebuffer pixels per tile, x
    uint texelH;      // framebuffer pixels per tile, y
    uint tilesX;
    uint tilesY;
    uint depthW;
    uint depthH;
    uint flags; // bit0 = allow 4x4, bit1 = use history
};
[[vk::push_constant]] Push pc;

float sampleDepth(float2 pixel, float2 depthDim) {
    return depthTex.SampleLevel((pixel + 0.5) / depthDim, 0).x;
}

void tileMinMax(int2 tile, float2 depthDim, out float dmin, out float dmax) {
    float2 base = float2(tile) * float2(pc.texelW, pc.texelH);
    float2 span = float2(pc.texelW - 1, pc.texelH - 1);
    dmax = 0.0;
    dmin = 1.0;
    for (int oy = 0; oy <= 1; ++oy) {
        for (int ox = 0; ox <= 1; ++ox) {
            float d = sampleDepth(base + float2(ox, oy) * span, depthDim);
            dmax = max(dmax, d);
            dmin = min(dmin, d);
        }
    }
    float dc = sampleDepth(base + span * 0.5, depthDim);
    dmax = max(dmax, dc);
    dmin = min(dmin, dc);
}

uint classify(float dmin, float dmax) {
    if (dmax < 1.0e-6) {
        return ((pc.flags & 1u) != 0) ? uint((2u << 2) | 2u) : uint((1u << 2) | 1u);
    }
    bool farTile = dmax < pc.dThreshold;
    bool flat = (dmax - dmin) < (pc.dThreshold * 0.5);
    return (farTile && flat) ? uint((1u << 2) | 1u) : 0u;
}

bool isFullRate(int2 tile, float2 depthDim) {
    if (tile.x < 0 || tile.y < 0 || tile.x >= int(pc.tilesX) || tile.y >= int(pc.tilesY)) {
        return false;
    }
    float dmin, dmax;
    tileMinMax(tile, depthDim, dmin, dmax);
    return classify(dmin, dmax) == 0u;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID) {
    if (gid.x >= pc.tilesX || gid.y >= pc.tilesY) return;

    float2 depthDim = float2(pc.depthW, pc.depthH);
    int2 tile = int2(gid.xy);

    float dmin, dmax;
    tileMinMax(tile, depthDim, dmin, dmax);
    uint raw = classify(dmin, dmax);

    bool neighborFull = false;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (isFullRate(tile + int2(dx, dy), depthDim)) {
                neighborFull = true;
            }
        }
    }

    uint prev = uint((1u << 2) | 1u);
    if ((pc.flags & 2u) != 0) {
        prev = history[gid.xy];
    }

    uint rate = raw;
    if (raw == 0u || prev == 0u || neighborFull) {
        rate = 0u;
    }

    rateOut[gid.xy] = rate;
    history[gid.xy] = raw;

    uint mixIdx = rate == 0u ? 0u : (rate == uint((1u << 2) | 1u) ? 1u : 2u);
    uint unused;
    InterlockedAdd(mixCounts[mixIdx], 1, unused);
}
