// Camera-relative packed mesh vertices (uint2: position, normal, layer, baked light).
// Placement: world = local * 2^k + (block − cam_block) − cam_frac + local_off.
// Integer subtraction exact at any world coordinate; per-vertex light interpolated.
//
// SV_VulkanInstanceID (not SV_InstanceID) to read plain InstanceIndex.
//
// Shifts/masks mirror mesh.rs. UV derives from position and normal. A morphing
// side face (X or Z) uses the moved cell y so the texture does not stretch.
// Baked sky/block/AO levels interpolate as scalars; the fragment multiplies by
// frame light/candle (uniform × interpolated scalar == interpolated product).
// Face index is packed — no float3 normal varying; the fragment looks
// `FACE_NORMAL` up. The CAGED variant (build.rs `-DCAGED`) writes the bent
// face normal at location 4 instead, and is the only mesh3d vertex module
// that reads the cage table.

#include "common.slang"
#ifdef CAGED
#include "cage.slang"
#endif

// Per-mesh placement record (std430, mirrors vk::buffers::MeshRecord, 80 bytes).
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

#include "lod_morph.slang"

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

    // x: layer[0:14] water[14] FLAT_COLOR[15] face[16:19] detail[20:25]
    // y: flat_rgba
    [[vk::location(3)]]
    nointerpolation uint2 ids;

#ifdef CAGED
    // Bent face normal. Flat draws do not carry this varying.
    [[vk::location(4)]]
    nointerpolation float3 face_n;
#endif
};

struct PushConstants
{
    float4x4 view_proj;
    // LOD box half-extents. half_z is the eye-split tail pad (byte 108).
    float half_x;
    float half_y;
    float2 inv_render_extent; // fragment water absorb; unused here
    // f64 camera eye split: eye = cam_block + cam_frac, frac in [0,1).
    int3 cam_block;
    int pad0;
    float3 cam_frac;
    float half_z;
    // Box centre at byte 112 (16-byte aligned). With the trailing pad the
    // block is 128 bytes, the guaranteed push-constant limit. Fragment only.
    float3 centre;
    float centre_pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

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
    // w1: AO [0:2], skylight [2:6], blocklight [6:10], water [10],
    // micro [11:17), morph [17:25) (decoded in lod_morph.slang).
    uint water = (w1 >> 10) & 0x1;

    // Per-axis 2-bit two's-complement micro-offsets. Truth: mesh.rs
    // SHIFT_MICRO_{X,Y,Z}=11,13,15, MASK_MICRO=0x3 (raw 0,1,2,3 -> 0,+1,-2,-1).
    // The seam-overdraw anti-z-fight nudge; zero for ordinary vertices.
    int mx = int(w1 << 19) >> 30;
    int my = int(w1 << 17) >> 30;
    int mz = int(w1 << 15) >> 30;
    float3 micro = float3(float(mx), float(my), float(mz)) * MICRO_STEP;

    // UV mapping from the unmoved cell: X face (z, y), Y face (x, z), Z face (x, y).
    // While this vertex morphs, X and Z faces use (y + a*dy) instead of y, in
    // cells, before the `* scale` below. Y faces stay (x, z). The write is
    // skipped when a*dy == 0, so a clear morph flag leaves this path
    // bit-identical. The CPU morph mirror has no UV function; this comment is
    // that rule. The shadow vertex shader has no UVs and does not apply it.
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
    uint detail = rec.detail_pass & ((1u << DETAIL_GPU_BITS) - 1u);
    float scale = detail_scale(detail);
    // `q` is the local cell position times the detail scale, in [0, 16·scale]³
    // (micro can step slightly outside). Block UVs stay in this cell space,
    // not on the bent cage; a morph only adds a*dy to side-face V.
    float3 q = (float3(x, y, z) + micro) * scale;
#ifdef CAGED
    Cage cg = cages[rec.cage];
    float3 t = q / (16.0 * scale);
    float3 world = cage_trilinear(cg, t) + float3(cg.anchor - pc.cam_block) - pc.cam_frac;
#else
    // Camera-relative placement: block delta first, then fractional parts.
    // Integer subtraction wraps, matching the CPU `wrapping_sub`.
    float3 offset = float3(rec.block - pc.cam_block) - pc.cam_frac + rec.local_off;
    float3 world = q + offset;
#endif
    // One uniform branch. Off leaves `world`, `t`, and `uv` untouched, so the
    // bent normal below (taken at this unmoved `t`) and the texture stay
    // bit-identical. Flat world +Y is the terrain's up; off does not add +0.
    // `a` is computed once and reused for the position slide and the side-face V.
    [branch]
    if (frame.morph_flag.x != 0)
    {
        int dy = lod_morph_dy(w1);
        if (dy != 0)
        {
            float a = lod_morph_a(detail, length(world));
            // One cell slide, shared by the position and the side-face V.
            float ady = a * float(dy);
#ifdef CAGED
            // The trilinear map is linear in t.y for fixed t.x, t.z, so moving
            // q.y by a*dy*scale (t.y by a*dy/16) moves world by exactly
            // (a*dy/16) * cage_dty. No second trilinear sample.
            world += (ady / 16.0) * cage_dty(cg, t);
#else
            world.y += ady * scale;
#endif
            // X (normal < 2) and Z (normal >= 4) store y in uv.y. Y faces do not.
            // ady == 0 does not write `uv`.
            if (ady != 0.0 && (normal < 2u || normal >= 4u))
                uv.y = y + ady;
        }
    }

    VertexOutput output;
    // Plain rectilinear projection; wide-FOV compression is a post-process.
    output.position = mul(pc.view_proj, float4(world, 1.0));
    // Coarser cells lose depth ties to finer ones. For k <= 9 the factor is
    // exactly `1 - scale/4096`; beyond that it stays positive and decreasing.
    if (scale > 1.0)
    {
        float k = float(int(detail) - int(DETAIL_GPU_BIAS));
        float bias = 1.0 - min(scale, 512.0) / 4096.0 - max(k - 9.0, 0.0) * (1.0 / 512.0);
        output.position.z *= bias;
    }
    // Scale UV to maintain texel density across LOD levels.
    output.uv = uv * scale;
    output.world = world;
#ifdef CAGED
    // Bent face normal stays at the unmoved `t`. The morph slides the vertex
    // along ∂/∂t.y; a second tangent pair at t.y + a*dy/16 is a sub-cell
    // change and is not worth the extra derivatives.
    output.face_n = cage_face_normal(cg, t, normal);
#endif
    output.light_lv = float3(sky_amount, block_amount, ao_factor);
    output.ids = uint2(
        (layer & 0x3FFFu)
            | (water << 14)
            | ((style.mode & 1u) << 15)
            | ((normal & 7u) << 16)
            | (detail << 20),
        style.flat_rgba);
    return output;
}
