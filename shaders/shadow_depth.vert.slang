// Depth-only shadow map occluder pass; reuses mesh3d vertex format and record layout.
#include "common.slang"
#include "cage.slang"

[[vk::binding(2, 0)]]
ConstantBuffer<FrameUniforms> frame;

#include "lod_morph.slang"

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
    // Micro (the w1 SHIFT_MICRO_* bits) stays off the occluder while morphing
    // is off: the depth bias dwarfs ±MICRO_STEP, and that path must stay
    // bit-identical. While the flag is set, q includes micro the way mesh3d.vert
    // does, so the unmorphed world, `d`, and the morph delta match the color
    // pass (one placement; the caged slide is (a*dy/16)*cage_dty, not a second
    // trilinear).
    uint w0 = input.packed.x;
    float x = float((w0 >> SHIFT_X) & MASK_COORD);
    float y = float((w0 >> SHIFT_Y) & MASK_COORD);
    float z = float((w0 >> SHIFT_Z) & MASK_COORD);

    MeshRecord rec = mesh_records[input.instance];
    float scale = detail_scale(rec.detail_pass);
    float3 q = float3(x, y, z) * scale;
    int dy = 0;
    // One uniform branch. Off leaves `q` micro-free and `dy` at 0.
    [branch]
    if (frame.morph_flag.x != 0)
    {
        uint w1 = input.packed.y;
        int mx = int(w1 << (30 - SHIFT_MICRO_X)) >> 30;
        int my = int(w1 << (30 - SHIFT_MICRO_Y)) >> 30;
        int mz = int(w1 << (30 - SHIFT_MICRO_Z)) >> 30;
        float3 micro = float3(float(mx), float(my), float(mz)) * MICRO_STEP;
        q = (float3(x, y, z) + micro) * scale;
        dy = lod_morph_dy(w1);
    }
    float3 world;
    if (rec.cage != 0u)
        world = cage_world(cages[rec.cage], q, scale, pc.cam_block, pc.cam_frac);
    else
    {
        float3 offset = float3(rec.block - pc.cam_block) - pc.cam_frac + rec.local_off;
        world = q + offset;
    }
    if (dy != 0)
    {
        float a = lod_morph_a(rec.detail_pass, length(world));
        if (rec.cage != 0u)
        {
            float3 t = q / (16.0 * scale);
            world += (a * float(dy) / 16.0) * cage_dty(cages[rec.cage], t);
        }
        else
            world.y += a * float(dy) * scale;
    }

    return mul(pc.view_proj, float4(world, 1.0));
}
