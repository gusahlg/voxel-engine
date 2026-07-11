// debug.frag.slang — pass the interpolated vertex color through, linearized so
// it composites in the linear-HDR offscreen with the rest of the scene (the
// tonemap pass owns the OETF back to display).

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float4 color;
};

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    return float4(srgb_to_linear(input.color.rgb), input.color.a);
}
