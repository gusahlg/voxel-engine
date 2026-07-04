// tris2d.frag.slang — samples the R8 font atlas as coverage. Solid shapes use
// UVs pointing at the atlas' all-white cell, so one pipeline draws everything.

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    float4 color;
};

[[vk::binding(0, 0)]]
Sampler2D atlas;

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float coverage = atlas.Sample(input.uv).r;
    return float4(input.color.rgb, input.color.a * coverage);
}
