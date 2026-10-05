// Depth-only shadow map occluder pass; reuses mesh3d vertex format and record layout.
#include "common.slang"
#include "cage.slang"

[[vk::binding(2, 0)]]
ConstantBuffer<FrameUniforms> frame;

#include "lod_morph.slang"

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
    // half/centre are the LOD clip tail. This pass does not discard by the box;
    // the fields stay so cam_block/cam_frac keep the mesh3d offsets.
    float4x4 view_proj;
    float half_x;
    float half_y;
    float2 pad;
    int3 cam_block;
    int pad0;
    float3 cam_frac;
    float half_z;
    float3 centre;
    float centre_pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("vertex")]
float4 vertexMain(VertexInput input) : SV_Position
{
    // The w1 micro-offset bits (mesh.rs SHIFT_MICRO_*) are ignored in the base
    // position: the occluder depth bias dwarfs the ±MICRO_STEP seam nudge.
    // When morphing is on they feed only the blend metric, so `a` matches the
    // color pass; the base position stays micro-free.
    uint w0 = input.packed.x;
    float x = float((w0 >> 0) & 0x1F);
    float y = float((w0 >> 5) & 0x1F);
    float z = float((w0 >> 10) & 0x1F);

    MeshRecord rec = mesh_records[input.instance];
    float scale = detail_scale(rec.detail_pass);
    float3 q = float3(x, y, z) * scale;
    if ((frame.morph_eye_block.w & 1) != 0)
    {
        uint w1 = input.packed.y;
        int mx = int(w1 << 19) >> 30;
        int my = int(w1 << 17) >> 30;
        int mz = int(w1 << 15) >> 30;
        float3 micro = float3(float(mx), float(my), float(mz)) * MICRO_STEP;
        float3 q_metric = (float3(x, y, z) + micro) * scale;
        q.y += lod_morph_cells(w1, rec.detail_pass, rec.block, q_metric) * scale;
    }
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
