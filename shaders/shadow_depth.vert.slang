// shadow_depth.vert.slang — depth-only occluder pass for cascaded shadow map.
// Reuses mesh3d layout (packed uint2) and same SSBO. Light-space view_proj in push budget.
// Position only; no fragment shader (rasterizer writes depth only).

// Per-draw placement. std430 layout mirrors vk::buffers::DrawOffset exactly —
// the SAME 32-byte record mesh3d.vert reads. Declaring this as float4 (as an
// early draft did) halves the stride: every odd instance then reads
// fade/mode/flat_rgba bits as its placement and every later even instance
// reads the WRONG draw's offset, scattering occluders and making shadows
// churn whenever the distance sort reorders the draw list (i.e. whenever the
// camera moves).
struct DrawOffset
{
    float3 offset;
    float scale;
    float fade;
    uint mode;
    uint flat_rgba;
    uint pad;
};

[[vk::binding(0, 0)]]
StructuredBuffer<DrawOffset> draw_offsets;

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

    DrawOffset placement = draw_offsets[input.instance];
    float3 world = float3(x, y, z) * placement.scale + placement.offset;

    return mul(pc.view_proj, float4(world, 1.0));
}
