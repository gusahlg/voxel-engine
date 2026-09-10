// Camera-relative packed mesh vertices (uint2: position, normal, layer, baked light).
// Placement: world = local * 2^k + (block − cam_block) − cam_frac + local_off.
// Integer subtraction exact at any world coordinate; per-vertex light interpolated.
//
// SV_VulkanInstanceID (not SV_InstanceID) to read plain InstanceIndex.
//
// Shifts/masks from genconst (build.rs `build_table()`). UV derives from
// position and normal. Baked sky/block/AO levels interpolate as scalars;
// the fragment multiplies by
// frame light/candle (uniform × interpolated scalar == interpolated product).
// Face index is packed — no float3 normal varying; the fragment looks
// `FACE_NORMAL` up.

#include "common.slang"
#include "mesh_record.slang"

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

    // x = sky_amount, y = block_amount, z = ao_factor.
    [[vk::location(2)]]
    float3 light_lv;

    // x: layer[0:14] water[14] FLAT_COLOR[15] face[16:19] detail[20:24]
    // y: flat_rgba
    [[vk::location(3)]]
    nointerpolation uint2 ids;
};

struct PushConstants
{
    float4x4 view_proj;
    // LOD slab extents (fragment pass only).
    float clip;
    float clip_v;
    float2 inv_render_extent; // fragment water absorb; unused here
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

    float x = float((w0 >> SHIFT_X) & MASK_COORD);
    float y = float((w0 >> SHIFT_Y) & MASK_COORD);
    float z = float((w0 >> SHIFT_Z) & MASK_COORD);
    uint normal = (w0 >> SHIFT_NORMAL) & MASK_NORMAL; // 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z
    uint layer = (w0 >> SHIFT_LAYER) & MASK_LAYER;
    // w1: AO [0:2], skylight [2:6], blocklight [6:10], water [10].
    uint water = (w1 >> SHIFT_WATER) & 1u;

    // Per-axis 2-bit two's-complement micro-offsets (SHIFT_MICRO_*, MASK_MICRO).
    // Left-shift 32-2-SHIFT_MICRO_* so arithmetic >> 30 sign-extends
    // (raw 0,1,2,3 -> 0,+1,-2,-1). Seam-overdraw anti-z-fight; zero otherwise.
    int mx = int(w1 << (32 - 2 - SHIFT_MICRO_X)) >> 30;
    int my = int(w1 << (32 - 2 - SHIFT_MICRO_Y)) >> 30;
    int mz = int(w1 << (32 - 2 - SHIFT_MICRO_Z)) >> 30;
    float3 micro = float3(float(mx), float(my), float(mz)) * MICRO_STEP;

    // UV mapping: X face (z,y), Y face (x,z), Z face (x,y).
    float2 uv;
    if (normal < 2)
        uv = float2(z, y);
    else if (normal < 4)
        uv = float2(x, z);
    else
        uv = float2(x, y);

    // Baked AO: level 0..3 → diffuse multiplier AO_MIN..AO_MIN+3*AO_STEP.
    uint ao_level = (w1 >> uint(SHIFT_AO)) & MASK_AO;
    float ao_factor = AO_MIN + float(ao_level) * AO_STEP;
    // Baked light levels: skylight [2:6], blocklight [6:10], 4-bit each.
    uint sky = (w1 >> SHIFT_SKY) & MASK_LIGHT;
    uint block = (w1 >> SHIFT_BLOCK) & MASK_LIGHT;
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
    output.light_lv = float3(sky_amount, block_amount, ao_factor);
    output.ids = uint2(
        (layer & MASK_LAYER)
            | (water << 14)
            | ((style.mode & 1u) << 15)
            | ((normal & MASK_NORMAL) << 16)
            | (detail << 20),
        style.flat_rgba);
    return output;
}
