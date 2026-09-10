// tris2d.frag.slang — samples the R8 font atlas as coverage. Solid shapes use
// UVs pointing at the atlas' all-white cell, so one pipeline draws everything.

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    float4 color;
};

[[vk::binding(0, 0)]]
Sampler2D atlas;

// The overlay is composited into the linear-HDR offscreen and tonemapped with
// the scene, so its sRGB vertex colour is linearised to blend in linear light.
[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float coverage = atlas.Sample(input.uv).r;
    return float4(srgb_to_linear(input.color.rgb), input.color.a * coverage);
}
