// Depth-only shadow map occluder pass; reuses mesh3d vertex format and record layout.
#include "common.slang"

// Per-mesh record: mirrors mesh3d's MeshRecord exactly (std430, 64 bytes).
// Changing the struct stride breaks instance offset calculation.
struct MeshRecord
{
    int3 block;       // world-space origin
    uint detail_pass; // [0:4] detail+bias, [4:6] pass
    float3 local_off; // mover offset; zero for terrain
    uint pad0;
    float3 aabb_min;
    uint index_count;
    float3 aabb_max;
    int vertex_offset;
};

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
    // The w1 micro-offset bits (mesh.rs SHIFT_MICRO_*) are deliberately ignored
    // here: the occluder depth bias dwarfs the ±MICRO_STEP seam nudge.
    uint w0 = input.packed.x;
    float x = float((w0 >> 0) & 0x1F);
    float y = float((w0 >> 5) & 0x1F);
    float z = float((w0 >> 10) & 0x1F);

    MeshRecord rec = mesh_records[input.instance];
    float scale = exp2(float(rec.detail_pass & 0xFu) - float(DETAIL_GPU_BIAS));
    float3 offset = float3(rec.block - pc.cam_block) - pc.cam_frac + rec.local_off;
    float3 world = float3(x, y, z) * scale + offset;

    return mul(pc.view_proj, float4(world, 1.0));
}
