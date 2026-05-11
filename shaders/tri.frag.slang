// tri.frag.slang

struct FragmentInput
{
    [[vk::location(0)]]
    float3 color;
};

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    return float4(input.color, 1.0);
}
