// debug.vert.slang — immediate debug geometry (cubes, wire cubes, lines).
// World-space float position + RGBA8 color, transformed by the view_proj push
// constant. No descriptor set: nothing sampled, no per-draw offset SSBO.

struct VertexInput
{
    [[vk::location(0)]]
    float3 pos;

    [[vk::location(1)]]
    float4 color; // R8G8B8A8_UNORM, arrives as 0..1
};

struct VertexOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float4 color;
};

struct PushConstants
{
    float4x4 view_proj;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    // Plain rectilinear projection; wide-FOV compression is a post-process, so
    // debug geometry stays registered with the terrain through the same resample.
    output.position = mul(pc.view_proj, float4(input.pos, 1.0));
    output.color = input.color;
    return output;
}
