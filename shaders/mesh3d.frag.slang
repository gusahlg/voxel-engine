// mesh3d.frag.slang — unlit vertex color passthrough.

struct FragmentInput
{
    [[vk::location(0)]]
    float4 color;
};

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    return input.color;
}
