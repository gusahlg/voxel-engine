// shadow_depth.vert.slang — depth-only occluder pass for cascaded shadow map.
// Reuses mesh3d layout (packed uint2) and same SSBO. Light-space view_proj in push budget.
// Position only; no fragment shader (rasterizer writes depth only).

[[vk::binding(0, 0)]]
StructuredBuffer<float4> draw_offsets;

struct VertexInput
{
    [[vk::location(0)]]
    uint2 packed;

    uint instance : SV_VulkanInstanceID; // raw SPIR-V InstanceIndex
};

struct PushConstants
{
    // One cascade's light-space CleanViewProj, pushed per cascade in the pass's
    // own push budget (64 B of the guaranteed 128 B minimum).
    float4x4 view_proj;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("vertex")]
float4 vertexMain(VertexInput input) : SV_Position
{
    // The w1 micro-offset bits (mesh.rs SHIFT_MICRO_*) are deliberately ignored
    // here: the occluder depth bias dwarfs the ±MICRO_STEP seam nudge.
    uint w0 = input.packed.x;
    float x = float((w0 >> 0) & 0x1F);
    float y = float((w0 >> 5) & 0x1F);
    float z = float((w0 >> 10) & 0x1F);

    float4 placement = draw_offsets[input.instance]; // xyz = offset, w = scale
    float3 world = float3(x, y, z) * placement.w + placement.xyz;

    return mul(pc.view_proj, float4(world, 1.0));
}
