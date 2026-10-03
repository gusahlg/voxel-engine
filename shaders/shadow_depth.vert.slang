// Depth-only shadow map occluder pass; reuses mesh3d vertex format and record layout.
#include "common.slang"
#include "cage.slang"

// Per-mesh record: mirrors mesh3d's MeshRecord exactly (std430, 80 bytes).
// Changing the struct stride breaks instance offset calculation.
struct MeshRecord
{
    int3 block;       // world-space origin
    uint detail_pass; // [0:5] detail+bias, [5:7] pass
    float3 local_off; // mover offset; zero for terrain
    uint cage;        // 0 = flat; else index into the cage table
    float3 aabb_min;
    uint index_count;
    float3 aabb_max;
    int vertex_offset;
    uint3 face_quads; // u16 pairs: upload slots 0|1, 2|3, 4|5
    uint flags; // bit 0 = packed face-runs valid (cull.comp)
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
    float half_x;
    float half_y;
    float2 pad;
    int3 cam_block;
    int pad0;
    float3 cam_frac;
    float half_z;
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
    float scale = detail_scale(rec.detail_pass);
    float3 q = float3(x, y, z) * scale;
    float3 world;
    if (rec.cage != 0u)
        world = cage_world(cages[rec.cage], q, scale, pc.cam_block, pc.cam_frac);
    else
    {
        float3 offset = float3(rec.block - pc.cam_block) - pc.cam_frac + rec.local_off;
        world = q + offset;
    }

    return mul(pc.view_proj, float4(world, 1.0));
}
