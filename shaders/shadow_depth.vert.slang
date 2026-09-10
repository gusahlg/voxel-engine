// Depth-only shadow map occluder pass; reuses mesh3d vertex format and record layout.
#include "common.slang"
#include "mesh_record.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<MeshRecord> mesh_records;

struct VertexInput
{
    [[vk::location(0)]]
    uint2 packed;

    uint instance : SV_VulkanInstanceID; // raw SPIR-V InstanceIndex
};

struct PushConstants
{
    // Light-space view-proj per cascade (mirrors Mesh3dPush layout).
    float4x4 view_proj;
    float clip;
    float clip_v;
    float2 pad;
    int3 cam_block;
    int pad0;
    float3 cam_frac;
    float pad1;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("vertex")]
float4 vertexMain(VertexInput input) : SV_Position
{
    // The w1 micro-offset bits (SHIFT_MICRO_*) are deliberately ignored
    // here: the occluder depth bias dwarfs the ±MICRO_STEP seam nudge.
    uint w0 = input.packed.x;
    float x = float((w0 >> SHIFT_X) & MASK_COORD);
    float y = float((w0 >> SHIFT_Y) & MASK_COORD);
    float z = float((w0 >> SHIFT_Z) & MASK_COORD);

    MeshRecord rec = mesh_records[input.instance];
    float scale = detail_scale(rec.detail_pass);
    float3 offset = float3(rec.block - pc.cam_block) - pc.cam_frac + rec.local_off;
    float3 world = float3(x, y, z) * scale + offset;

    return mul(pc.view_proj, float4(world, 1.0));
}
