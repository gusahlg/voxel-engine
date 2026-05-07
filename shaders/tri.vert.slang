struct VertexOutput {
    float4 position: SV_Position;
    [[vk::location(0)]]
    float3 color: COLOR0;
};

[shader("vertex")]
VertexOutput main(uint vertex_index: SV_VertexID)
{
    VertexOutput output;

    if (vertex_index == 0) {
        output.position = float4(0.0, -0.5, 0.0, 1.0);
        output.color = float3(1.0, 0.0, 0.0);
    } else if (vertex_index == 1) {
        output.position = float4(0.5, 0.5, 0.0, 1.0);
        output.color = float3(0.0, 1.0, 0.0);
    } else {
        output.position = float4(-0.5, 0.5, 0.0, 1.0);
        output.color = float3(0.0, 0.0, 1.0);
    }

    return output;
}
