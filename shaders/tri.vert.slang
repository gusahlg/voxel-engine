// tri.vert.slang

struct VertexInput
{
    [[vk::location(0)]]
    float3 position;

    [[vk::location(1)]]
    float3 color;
};

struct VertexOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float3 color;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;

    output.position = float4(input.position, 1.0);
    output.color = input.color;

    return output;
}
