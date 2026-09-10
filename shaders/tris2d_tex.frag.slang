// tris2d_tex.frag.slang — clone of tris2d.frag, but samples an RGBA texture
// (the minimap) and tints it by the vertex colour. Same vertex layout and
// pixels_to_ndc push constant as tris2d; only the fragment module differs.

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    float4 color;
};

[[vk::binding(0, 0)]]
Sampler2D map;

// Match the atlas path: the overlay is composited into the linear-HDR
// offscreen and tonemapped with the scene, so sRGB inputs are linearised.
[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float4 t = map.Sample(input.uv);
    return float4(srgb_to_linear(t.rgb) * srgb_to_linear(input.color.rgb), t.a * input.color.a);
}
