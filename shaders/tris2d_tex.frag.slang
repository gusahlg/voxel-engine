// tris2d_tex.frag.slang — samples an RGBA overlay texture, tinted by vertex color.
//
// Used for the minimap. Texture RGB is stored as sRGB-like UI color data, then
// linearised because the overlay is composited into the linear-HDR offscreen.

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    float4 color;
};

[[vk::binding(0, 0)]]
Sampler2D tex;

float3 srgbToLinear(float3 c)
{
    float3 lo = c / 12.92;
    float3 hi = pow(max((c + 0.055) / 1.055, 0.0), float3(2.4, 2.4, 2.4));
    return select(c <= 0.04045, lo, hi);
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float4 texel = tex.Sample(input.uv);
    float4 tinted = texel * input.color;
    return float4(srgbToLinear(tinted.rgb), tinted.a);
}
