// bloom.comp.slang — HDR bloom threshold + downsample pyramid.
//
// Two entry points, one dispatch each per mip level:
//   * threshold: bilinear-downsamples the full-res HDR offscreen to the half-res
//     mip 0 and keeps only the bright spill via a smoothstep soft knee on exposed
//     luma. One bilinear tap doubles as the 2×2 average.
//   * downsample: 2×2 box from the previous (larger) mip into this one.
//
// No subgroup ops: a plain per-level dispatch chain is simpler and
// subgroup-size-portable vs. a single-pass SPD. Storage images carry an
// explicit `rgba16f` format so no `shaderStorageImageRead/WriteWithoutFormat`
// device feature is needed.
//
// Determinism: a pure function of this frame's HDR — no temporal state, no clock.

#include "common.slang"

struct Push
{
    uint2 dst_dim;   // this dispatch's output mip size in texels
    uint2 src_dim;   // source mip size (downsample bound-clamp; unused by threshold)
    float exposure;  // CPU-metered multiplier (threshold knee only)
    float thr_lo;    // soft-knee low edge, on luma·exposure
    float thr_hi;    // soft-knee high edge
    float thr_scale; // spill scale (0.5)
};
[[vk::push_constant]] Push pc;

// binding 0: full-res HDR offscreen (combined image sampler, linear-clamp).
[[vk::binding(0, 0)]] Sampler2D hdrTex;
// binding 1: source mip (downsample read). binding 2: destination mip (write).
[[vk::image_format("rgba16f")]] [[vk::binding(1, 0)]] RWTexture2D<float4> srcMip;
[[vk::image_format("rgba16f")]] [[vk::binding(2, 0)]] RWTexture2D<float4> dstMip;

[shader("compute")]
[numthreads(8, 8, 1)]
void threshold(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dst_dim.x || gid.y >= pc.dst_dim.y)
        return;
    // Half-res texel centre → one bilinear tap averages the 2×2 full-res block.
    float2 uv = (float2(gid.xy) + 0.5) / float2(pc.dst_dim);
    float3 hdr = hdrTex.SampleLevel(uv, 0).rgb;
    float luma = luma709(hdr);
    float knee = smoothstep(pc.thr_lo, pc.thr_hi, luma * pc.exposure);
    dstMip[gid.xy] = float4(hdr * knee * pc.thr_scale, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void downsample(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dst_dim.x || gid.y >= pc.dst_dim.y)
        return;
    uint2 hi = pc.src_dim - 1; // clamp the far taps when the source dim is odd
    uint2 s = gid.xy * 2;
    float3 c = srcMip[min(s, hi)].rgb
             + srcMip[min(s + uint2(1, 0), hi)].rgb
             + srcMip[min(s + uint2(0, 1), hi)].rgb
             + srcMip[min(s + uint2(1, 1), hi)].rgb;
    dstMip[gid.xy] = float4(c * 0.25, 1.0);
}
