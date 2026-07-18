// Variable-rate-shading classifier: one thread per shading-rate tile.
//
// Reads the previous-cycle depth buffer (reversed-Z, infinite far: d = near/z,
// so d→0 is far and the cleared sky is exactly d = 0) and writes an R8_UINT
// rate code. A tile shades coarse (2x2) when it is entirely far/sky AND holds
// no strong depth discontinuity — a silhouette crossing the tile forces full
// rate so block edges stay crisp. Encoding matches the CPU side:
// (log2(w) << 2) | log2(h), i.e. 0 = 1x1, 0b0101 = 2x2.

[[vk::binding(0, 0)]] Sampler2D depthTex;
[[vk::binding(1, 0)]] RWTexture2D<uint> rateOut;

struct Push {
    float dThreshold; // coarse iff tile depth < this (far / sky)
    uint texelW;      // framebuffer pixels per tile, x
    uint texelH;      // framebuffer pixels per tile, y
};
[[vk::push_constant]] Push pc;

float sampleDepth(float2 pixel, float2 depthDim) {
    return depthTex.SampleLevel((pixel + 0.5) / depthDim, 0).x;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID) {
    uint tw, th;
    rateOut.GetDimensions(tw, th);
    if (gid.x >= tw || gid.y >= th) return;

    uint dw, dh;
    depthTex.GetDimensions(dw, dh);
    float2 depthDim = float2(dw, dh);

    float2 base = float2(gid.xy) * float2(pc.texelW, pc.texelH);
    float2 span = float2(pc.texelW - 1, pc.texelH - 1);

    // Four corners + center of the tile.
    float dmax = 0.0;
    float dmin = 1.0;
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

    bool farTile = dmax < pc.dThreshold;
    bool flat = (dmax - dmin) < (pc.dThreshold * 0.5);
    // Empty tiles forced to full rate (genuine far has dmax > 0).
    bool hole = dmax < 1.0e-6;
    rateOut[gid.xy] = (farTile && flat && !hole) ? uint((1u << 2) | 1u) : 0u;
}
