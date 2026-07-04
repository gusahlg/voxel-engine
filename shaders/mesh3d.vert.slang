// mesh3d.vert.slang — world-space vertices, single view_proj push constant.

struct VertexInput
{
    [[vk::location(0)]]
    float3 position;

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
    output.position = mul(pc.view_proj, float4(input.position, 1.0));
    output.color = input.color;
    return output;
}
