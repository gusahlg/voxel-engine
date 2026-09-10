// fullscreen.vert.slang — vertex-less fullscreen triangle from SV_VertexID.
// Ids 0/1/2 map to a single oversized triangle covering the NDC square.
// Two entry points (same source, two compile jobs):
//   * vertexTonemap: UV in [0,1] for sampling the HDR offscreen. z = 0
//     (no depth attachment on the tonemap pass).
//   * vertexSky: NDC xy; position at z = 0.0 (reversed-Z far plane) so the
//     fragment survives GREATER_OR_EQUAL only where terrain has not written
//     closer depth.

float2 fullscreen_uv(uint id)
{
    return float2((id << 1) & 2, id & 2);
}

struct TonemapOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 uv;
};

[shader("vertex")]
TonemapOutput vertexTonemap(uint id: SV_VertexID)
{
    float2 uv = fullscreen_uv(id);
    TonemapOutput output;
    output.uv = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

struct SkyOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 ndc;
};

[shader("vertex")]
SkyOutput vertexSky(uint id: SV_VertexID)
{
    // (0,0) (2,0) (0,2) in UV → (-1,-1) (3,-1) (-1,3) in NDC: covers [-1,1]².
    float2 uv = fullscreen_uv(id);
    SkyOutput output;
    output.ndc = uv * 2.0 - 1.0;
    output.position = float4(output.ndc, 0.0, 1.0);
    return output;
}
