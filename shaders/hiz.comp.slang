// Hi-Z depth pyramid: one dispatch per mip, each dest texel is the MIN of
// the 2x2 (3x3 when a source axis is odd) source texels. Reversed-Z: MIN is
// the FARTHEST value, so a texel is a conservative occluder for its footprint.
//
// Two entry points from one source (same idea as bloom.comp.slang):
//   * reduce_depth: mip 0 from the sampleable scene depth (MSAA-resolved when
//     MSAA is on). Depth already rests in SAMPLEABLE_DEPTH_REST_LAYOUT.
//   * reduce_mip: later mips from the previous (larger) pyramid level.
//
// No subgroup ops: a plain per-level chain is simpler and portable. Storage
// images carry an explicit `r32f` format so no write-without-format feature
// is needed.

#include "common.slang"

struct Push
{
    uint2 dst_dim;
    uint2 src_dim;
};
[[vk::push_constant]] Push pc;

[[vk::binding(0, 0)]] Sampler2D depthTex;
[[vk::image_format("r32f")]] [[vk::binding(1, 0)]] RWTexture2D<float> srcMip;
[[vk::image_format("r32f")]] [[vk::binding(2, 0)]] RWTexture2D<float> dstMip;

float sampleDepth(uint2 p, uint2 hi)
{
    uint2 c = min(p, hi);
    float2 uv = (float2(c) + 0.5) / float2(pc.src_dim);
    return depthTex.SampleLevel(uv, 0).x;
}

float loadMip(uint2 p, uint2 hi)
{
    return srcMip[min(p, hi)];
}

// Conservative farthest (MIN) over the dest texel's source footprint.
// Even axes: 2x2. An odd source axis adds the leftover row/column (up to 3x3)
// so every source texel contributes to at least one dest and a dest that
// straddles the extra texel still sees it.
float reduceDepth(uint2 s, uint2 hi)
{
    float d = min(
        min(sampleDepth(s, hi), sampleDepth(s + uint2(1, 0), hi)),
        min(sampleDepth(s + uint2(0, 1), hi), sampleDepth(s + uint2(1, 1), hi)));
    bool odd_x = (pc.src_dim.x & 1u) != 0;
    bool odd_y = (pc.src_dim.y & 1u) != 0;
    if (odd_x)
        d = min(d, min(sampleDepth(s + uint2(2, 0), hi), sampleDepth(s + uint2(2, 1), hi)));
    if (odd_y)
        d = min(d, min(sampleDepth(s + uint2(0, 2), hi), sampleDepth(s + uint2(1, 2), hi)));
    if (odd_x && odd_y)
        d = min(d, sampleDepth(s + uint2(2, 2), hi));
    return d;
}

float reduceMip(uint2 s, uint2 hi)
{
    float d = min(
        min(loadMip(s, hi), loadMip(s + uint2(1, 0), hi)),
        min(loadMip(s + uint2(0, 1), hi), loadMip(s + uint2(1, 1), hi)));
    bool odd_x = (pc.src_dim.x & 1u) != 0;
    bool odd_y = (pc.src_dim.y & 1u) != 0;
    if (odd_x)
        d = min(d, min(loadMip(s + uint2(2, 0), hi), loadMip(s + uint2(2, 1), hi)));
    if (odd_y)
        d = min(d, min(loadMip(s + uint2(0, 2), hi), loadMip(s + uint2(1, 2), hi)));
    if (odd_x && odd_y)
        d = min(d, loadMip(s + uint2(2, 2), hi));
    return d;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void reduce_depth(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dst_dim.x || gid.y >= pc.dst_dim.y)
        return;
    dstMip[gid.xy] = reduceDepth(gid.xy * 2, pc.src_dim - 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void reduce_mip(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dst_dim.x || gid.y >= pc.dst_dim.y)
        return;
    dstMip[gid.xy] = reduceMip(gid.xy * 2, pc.src_dim - 1);
}
