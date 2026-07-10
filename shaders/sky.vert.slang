// sky.vert.slang — vertex-less fullscreen triangle from SV_VertexID.
// Ids 0/1/2 map to a single oversized triangle covering the NDC square. The
// output position sits at z = 0.0, the reversed-Z far plane, so the fragment
// pass survives the GREATER_OR_EQUAL depth test only where terrain has not
// written closer depth (i.e. background pixels).

struct VertexOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 ndc;
};

[shader("vertex")]
VertexOutput vertexMain(uint id: SV_VertexID)
{
    // (0,0) (2,0) (0,2) in UV → (-1,-1) (3,-1) (-1,3) in NDC: covers [-1,1]².
    float2 uv = float2((id << 1) & 2, id & 2);
    VertexOutput output;
    output.ndc = uv * 2.0 - 1.0;
    output.position = float4(output.ndc, 0.0, 1.0);
    return output;
}
