// tonemap.vert.slang — vertex-less fullscreen triangle from SV_VertexID.
// Emits UVs in [0,1] for sampling the HDR offscreen; the oversized triangle
// (ids 0/1/2) covers the whole NDC square. z = 0 (no depth attachment here).

struct VertexOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 uv;
};

[shader("vertex")]
VertexOutput vertexMain(uint id: SV_VertexID)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    VertexOutput output;
    output.uv = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}
