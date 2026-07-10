// debug.frag.slang — pass the interpolated vertex color through, linearized so
// it composites in the linear-HDR offscreen with the rest of the scene (the
// tonemap pass owns the OETF back to display).

struct FragmentInput
{
    [[vk::location(0)]]
    float4 color;
};

float3 srgbToLinear(float3 c)
{
    float3 lo = c / 12.92;
    float3 hi = pow(max((c + 0.055) / 1.055, 0.0), float3(2.4, 2.4, 2.4));
    return select(c <= 0.04045, lo, hi);
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    return float4(srgbToLinear(input.color.rgb), input.color.a);
}
