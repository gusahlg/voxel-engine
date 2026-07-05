// mesh3d.vert.slang — camera-relative vertices: a per-draw float4 offset is
// added to the input position before the view_proj multiply, so meshes can
// carry small local-space coordinates while the world spans huge distances.
// Passes uv + color through; color.a carries the texture-array layer index.

struct VertexInput
{
    [[vk::location(0)]]
    float3 position;

    [[vk::location(1)]]
    float2 uv;

    [[vk::location(2)]]
    float4 color; // R8G8B8A8_UNORM, arrives as 0..1; a = layer / 255
};

struct VertexOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    float4 color;
};

struct PushConstants
{
    float4x4 view_proj;
    float4 offset; // per-draw translation; .w unused
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = mul(pc.view_proj, float4(input.position + pc.offset.xyz, 1.0));
    output.uv = input.uv;
    output.color = input.color;
    return output;
}
