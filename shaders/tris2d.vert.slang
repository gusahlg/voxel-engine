// tris2d.vert.slang — pixel-space overlay quads.
//
// Positions arrive in framebuffer pixels (origin top-left, y down). The push
// constant holds (2/width, 2/height). The engine renders with a NEGATIVE
// viewport height (GL-style y-up NDC), so pixel y maps to ndc.y = 1 - y*scale.

struct VertexInput
{
    [[vk::location(0)]]
    float2 position;

    [[vk::location(1)]]
    float2 uv;

    [[vk::location(2)]]
    float4 color; // R8G8B8A8_UNORM
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
    float2 pixels_to_ndc;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = float4(
        input.position.x * pc.pixels_to_ndc.x - 1.0,
        1.0 - input.position.y * pc.pixels_to_ndc.y,
        0.0,
        1.0);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}
