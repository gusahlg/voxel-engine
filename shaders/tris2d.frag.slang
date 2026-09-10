// tris2d.frag.slang — overlay fragment. Default samples the R8 font atlas as
// coverage (solid shapes use UVs pointing at the atlas' all-white cell, so one
// pipeline draws everything). `-D TRIS2D_TEX` samples an RGBA texture (the
// minimap) and tints it by the vertex colour. Same vertex layout and
// pixels_to_ndc push constant; only the sample/tint differs.

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    float4 color;
};

[[vk::binding(0, 0)]]
#ifdef TRIS2D_TEX
Sampler2D map;
#else
Sampler2D atlas;
#endif

// The overlay is composited into the linear-HDR offscreen and tonemapped with
// the scene, so sRGB vertex colour is linearised to blend in linear light.
[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
#ifdef TRIS2D_TEX
    float4 t = map.Sample(input.uv);
    return float4(srgb_to_linear(t.rgb) * srgb_to_linear(input.color.rgb), t.a * input.color.a);
#else
    float coverage = atlas.Sample(input.uv).r;
    return float4(srgb_to_linear(input.color.rgb), input.color.a * coverage);
#endif
}
