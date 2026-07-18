// STORAGE image write test: compute writes swapchain format (no fallback).
// Validates driver STORAGE support for swapchain pixel format.

[[vk::binding(0, 0)]] [format("rgba8")] RWTexture2D<float4> img;

struct Params { uint2 dim; };
[[vk::push_constant]] Params pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dim.x || gid.y >= pc.dim.y)
        return;
    // Deterministic gradient for host verification.
    float4 c = float4(
        float(gid.x) / float(pc.dim.x),
        float(gid.y) / float(pc.dim.y),
        0.25,
        1.0);
    img[gid.xy] = c;
}
