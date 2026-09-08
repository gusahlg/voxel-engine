// Direction-space cloud LUT: one thread per octahedral texel, one dispatch
// per sky frame. RGB = premultiplied lit colour, A = cover. Sampled bilinear
// by sky.frag; world-anchored via frame.anim (camera xz/y + time).

#include "sky_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<FrameUniforms> frame;

[[vk::image_format("rgba16f")]]
[[vk::binding(1, 0)]]
RWTexture2D<float4> lut;

[shader("compute")]
[numthreads(SKY_CLOUD_LUT_WG, SKY_CLOUD_LUT_WG, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID)
{
    uint n = SKY_CLOUD_LUT_SIZE;
    if (gid.x >= n || gid.y >= n)
        return;
    float2 uv = (float2(gid.xy) + 0.5) / float(n);
    float3 ray = sky_oct_decode(uv);
    float cover;
    float3 lit = clouds(ray, frame, cover);
    lut[gid.xy] = float4(lit, cover);
}
