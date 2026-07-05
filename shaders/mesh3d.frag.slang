// mesh3d.frag.slang — unlit textured: vertex rgb multiplies a block-texture-
// array sample. color.a is the LAYER index (a * 255), not alpha — the 3D
// pipelines never blend. Layer 0 is guaranteed all-white, so flat-colored
// geometry sets uv = (0,0) and a = 0.

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    float4 color;
};

// Set 1: set 0 on the 3D layout is the vertex stage's per-draw offsets SSBO.
[[vk::binding(0, 1)]]
Sampler2DArray blockTex;

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float layer = round(input.color.a * 255.0);
    float3 tex = blockTex.Sample(float3(input.uv, layer)).rgb;
    return float4(input.color.rgb * tex, 1.0);
}
