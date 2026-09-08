// Camera-relative packed mesh vertices (uint2: position, normal, layer, baked light).
// Placement: world = local * 2^k + (block − cam_block) − cam_frac + local_off.
// Integer subtraction exact at any world coordinate; per-vertex light interpolated.
//
// SV_VulkanInstanceID (not SV_InstanceID) to read plain InstanceIndex.
//
// Shifts/masks mirror mesh.rs. UV derives from position and normal. Baked
// sky/block amounts become RGB here (× frame light/candle) so the fragment
// interpolates colours, not levels. Face index is packed — no float3 normal
// varying; the fragment looks `FACE_NORMAL` up.

#include "common.slang"

// Per-mesh placement record (std430, mirrors vk::buffers::MeshRecord, 80 bytes).
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
    uint3 face_quads; // u16 pairs: upload slots 0|1, 2|3, 4|5
    uint _pad2;
};

// Per-mesh dynamic style (std430, mirrors vk::buffers::DrawDyn).
struct DrawDyn
{
    uint mode;      // bit 0: FLAT_COLOR
    uint flat_rgba; // sRGB RGBA8 encoded
};

[[vk::binding(0, 0)]]
StructuredBuffer<MeshRecord> mesh_records;

[[vk::binding(2, 0)]]
ConstantBuffer<FrameUniforms> frame;

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

    // Camera-relative world position for fog distance and shadow projection.
    [[vk::location(1)]]
    float3 world;

    // rgb = frame.light.rgb * sky_amount; w = sky_amount (fallback / skip).
    [[vk::location(2)]]
    float4 sky;

    // rgb = frame.candle.rgb * block_amount; a = AO factor.
    [[vk::location(3)]]
    float4 block_ao;

    // x: layer[0:14] water[14] FLAT_COLOR[15] face[16:19] detail[20:24]
    // y: flat_rgba
    [[vk::location(4)]]
    nointerpolation uint2 ids;
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
    int mx = int(w1 << 19) >> 30;
    int my = int(w1 << 17) >> 30;
    int mz = int(w1 << 15) >> 30;
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

    // Micro-offset is added in LOCAL space so it scales with the draw (2^k LOD).
    MeshRecord rec = mesh_records[input.instance];
    DrawDyn style = draw_dyn[input.instance];
    uint detail = rec.detail_pass & 0xFu;
    float scale = detail_scale(detail);
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
    output.world = world;
    output.sky = float4(frame.light.rgb * sky_amount, sky_amount);
    output.block_ao = float4(frame.candle.rgb * block_amount, ao_factor);
    output.ids = uint2(
        (layer & 0x3FFFu)
            | (water << 14)
            | ((style.mode & 1u) << 15)
            | ((normal & 7u) << 16)
            | (detail << 20),
        style.flat_rgba);
    return output;
}
