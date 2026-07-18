// Camera-relative packed mesh vertices (uint2: position, normal, layer, baked light).
// Placement: world = local * 2^k + (block − cam_block) − cam_frac + local_off.
// Integer subtraction exact at any world coordinate; per-vertex light interpolated.
//
// SV_VulkanInstanceID (not SV_InstanceID) to read plain InstanceIndex.
//
// Shifts/masks mirror mesh.rs. UV and shade derive from position and normal.

#include "common.slang"

// Per-mesh placement record (std430, mirrors vk::buffers::MeshRecord).
struct MeshRecord
{
    int3 block;       // world-space origin
    uint detail_pass; // [0:4] detail+bias, [4:6] pass
    float3 local_off; // mover offset; zero for terrain
    uint _pad;        // std430 padding
    float3 aabb_min;
    uint index_count;
    float3 aabb_max;
    int vertex_offset;
};

// Per-mesh dynamic style (std430, mirrors vk::buffers::DrawDyn).
struct DrawDyn
{
    uint mode;      // bit 0: FLAT_COLOR
    uint flat_rgba; // sRGB RGBA8 encoded
};

[[vk::binding(0, 0)]]
StructuredBuffer<MeshRecord> mesh_records;

[[vk::binding(6, 0)]]
StructuredBuffer<DrawDyn> draw_dyn;

struct VertexInput
{
    [[vk::location(0)]]
    uint2 packed;

    uint instance : SV_VulkanInstanceID; // raw SPIR-V InstanceIndex
};

struct VertexOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    nointerpolation uint layer;

    // Camera-relative world position for fog distance and shadow projection.
    [[vk::location(2)]]
    float3 world;

    // Face normal for shadow slope bias.
    [[vk::location(3)]]
    nointerpolation float3 nrm;

    // Lighting: x = AO, y = skylight, z = blocklight (all normalized).
    [[vk::location(4)]]
    float3 lightparams;

    // Water material bit; selects animated water shading in transparent path.
    [[vk::location(5)]]
    nointerpolation uint water;

    // LOD scale from detail_pass.
    [[vk::location(6)]]
    nointerpolation float scale;

    // Per-draw style: flat-color mode and RGBA.
    [[vk::location(7)]]
    nointerpolation uint mode;
    [[vk::location(8)]]
    nointerpolation uint flat_rgba;
};

struct PushConstants
{
    float4x4 view_proj;
    // LOD slab extents (fragment pass only).
    float clip;
    float clip_v;
    float2 pad;
    // f64 camera eye split: eye = cam_block + cam_frac, frac in [0,1).
    int3 cam_block;
    int pad0;
    float3 cam_frac;
    float pad1;
    // Lighting from per-frame UBO (binding 2).
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Planet curvature: droop is visual only (CLIP position only, not world).
// Keeps culling conservative and shadows against flat silhouette.

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    uint w0 = input.packed.x;
    uint w1 = input.packed.y;

    float x = float((w0 >> 0) & 0x1F);
    float y = float((w0 >> 5) & 0x1F);
    float z = float((w0 >> 10) & 0x1F);
    uint normal = (w0 >> 15) & 0x7;   // 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z
    uint layer = (w0 >> 18) & 0x3FFF;
    // w1: AO [0:2], skylight [2:6], blocklight [6:10], water [10].
    uint water = (w1 >> 10) & 0x1;

    // Per-axis 2-bit two's-complement micro-offsets. Truth: mesh.rs
    // SHIFT_MICRO_{X,Y,Z}=11,13,15, MASK_MICRO=0x3 (raw 0,1,2,3 -> 0,+1,-2,-1).
    // The seam-overdraw anti-z-fight nudge; zero for ordinary vertices.
    int mx = int((w1 >> 11) & 0x3); if (mx >= 2) mx -= 4;
    int my = int((w1 >> 13) & 0x3); if (my >= 2) my -= 4;
    int mz = int((w1 >> 15) & 0x3); if (mz >= 2) mz -= 4;
    float3 micro = float3(float(mx), float(my), float(mz)) * MICRO_STEP;

    // UV mapping: X face (z,y), Y face (x,z), Z face (x,y).
    float2 uv;
    if (normal < 2)
        uv = float2(z, y);
    else if (normal < 4)
        uv = float2(x, z);
    else
        uv = float2(x, y);

    // Baked AO: level 0..3 → diffuse multiplier 0.4..1.0.
    uint ao_level = w1 & 0x3;
    float ao_factor = 0.4 + float(ao_level) * 0.2;
    // Baked light levels: skylight [2:6], blocklight [6:10], 4-bit each.
    uint sky = (w1 >> 2) & 0xF;
    uint block = (w1 >> 6) & 0xF;
    float sky_amount = float(sky) / 15.0;
    float block_amount = float(block) / 15.0;

    // Face normal for the receiver's slope bias (matches the `normal` index).
    static const float3 FACE_NORMAL[6] = {
        float3(1.0, 0.0, 0.0),  float3(-1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),  float3(0.0, -1.0, 0.0),
        float3(0.0, 0.0, 1.0),  float3(0.0, 0.0, -1.0),
    };
    float3 N = FACE_NORMAL[normal];

    // Micro-offset is added in LOCAL space so it scales with the draw (2^k LOD).
    MeshRecord rec = mesh_records[input.instance];
    DrawDyn style = draw_dyn[input.instance];
    float scale = exp2(float(rec.detail_pass & 0xFu) - float(DETAIL_GPU_BIAS));
    // Camera-relative placement: block delta first, then fractional parts.
    float3 offset = float3(rec.block - pc.cam_block) - pc.cam_frac + rec.local_off;
    float3 world = (float3(x, y, z) + micro) * scale + offset;

    VertexOutput output;
    // Round-world horizon: curve on-screen silhouette only, preserving world position for fog/shadows.
    float3 curved = world;
    curved.y -= min(dot(world.xz, world.xz) * CURVE_INV_2R, CURVE_MAX_DROP);
    // Plain rectilinear projection; wide-FOV compression is a post-process.
    output.position = mul(pc.view_proj, float4(curved, 1.0));
    // Resolve depth ties in coplanar LOD levels: push each farther proportionally to cell size.
    if (scale > 1.0)
        output.position.z *= 1.0 - scale / 4096.0;
    // Scale UV to maintain texel density across LOD levels.
    output.uv = uv * scale;
    output.layer = layer;
    output.world = world;
    output.nrm = N;
    output.lightparams = float3(ao_factor, sky_amount, block_amount);
    output.water = water;
    output.scale = scale;
    output.mode = style.mode;
    output.flat_rgba = style.flat_rgba;
    return output;
}
