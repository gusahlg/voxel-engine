// Emit indirect draw commands per frustum test: one thread per mesh slot.
// Camera partitions are (group, arena, distance-bucket); shadows are
// per-cascade and unbucketed. Camera groups (CULL_CAMERA_GROUPS): 0 full-res
// Opaque, 1 Cutout, 2 coarse-LOD Opaque (scale > 1, the only camera draws
// that pay for the slab-clip `discard`). Group 2 meshes whose AABB is fully
// inside that slab are not emitted (every fragment would be discarded).

#include "common.slang"

// Mirrors vk::buffers::MeshRecord exactly (std430, 80 bytes).
struct MeshRecord
{
    int3 block;
    uint detail_pass; // [0:4] detail+bias, [4:6] pass
    float3 local_off;
    uint pad0;
    float3 aabb_min;
    uint index_count;
    float3 aabb_max;
    int vertex_offset;
    uint3 face_quads; // u16 pairs: upload slots 0|1, 2|3, 4|5
    uint flags; // bit 0 = packed face-runs valid; bit 1 = uploaded this frame
};

// Mirrors vk::cull::PartitionGpu (std430).
struct Partition
{
    uint offset;   // first command index
    uint capacity; // live draw count
};

// Mirrors VkDrawIndexedIndirectCommand.
struct DrawCmd
{
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance; // mesh slot index
};

// Mirrors vk::cull::CullParamsGpu (std140).
struct CullParams
{
    float4 cam_planes[5];     // camera frustum (5 planes)
    float4 shadow_planes[10]; // 2 cascades x 5 planes each
    int3 cam_block;
    uint slot_count;
    float3 cam_frac;
    uint arena_count;
    uint shadow_enabled;
    uint flags; // bit 0 = per-direction face-run culling
    float clip;   // full-res slab radius; 0 = disabled (same as mesh3d push)
    float clip_v; // full-res slab half-height; 0 = disabled
    float4x4 prev_view_proj; // previous frame's view-proj, origin-shifted to this eye
    uint hiz_mips;
    uint hiz_w;  // pyramid level-0 width (half render-res)
    uint hiz_h;
    uint occ_flags; // bit 0 = test against the previous pyramid
};

[[vk::binding(0, 0)]]
StructuredBuffer<MeshRecord> records;

// 0 = dead slot, else arena index + 1.
[[vk::binding(1, 0)]]
StructuredBuffer<uint> slot_arena;

[[vk::binding(2, 0)]]
StructuredBuffer<Partition> partitions;

[[vk::binding(3, 0)]]
RWStructuredBuffer<DrawCmd> commands;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> counts; // one per partition, zero-filled pre-dispatch

[[vk::binding(5, 0)]]
ConstantBuffer<CullParams> params;

// Per-slot camera visibility (app-computed LOD+occlusion); gates camera draws only.
// Shadows gate on cascade AABB test (off-frustum camera meshes can cast shadows).
[[vk::binding(6, 0)]]
StructuredBuffer<uint> visible;

// Per camera group (0 full-res opaque, 1 cutout, 2 coarse LOD):
// [draws0, index_count0, draws1, index_count1, draws2, index_count2].
// Written only when the stats push flag is set (profiling).
[[vk::binding(7, 0)]]
RWStructuredBuffer<uint> stats;

// Previous frame's Hi-Z pyramid (SHADER_READ_ONLY). Unused when occ_flags bit 0 is clear.
[[vk::binding(8, 0)]] Sampler2D hizTex;

struct Push
{
    uint flags; // bit0 = accumulate geometry stats
};
[[vk::push_constant]] Push pc;

// P-vertex test: same as camera.rs Frustum::intersects_aabb.
bool outside_plane(float4 plane, float3 mn, float3 mx)
{
    float3 corner = float3(
        plane.x >= 0.0 ? mx.x : mn.x,
        plane.y >= 0.0 ? mx.y : mn.y,
        plane.z >= 0.0 ? mx.z : mn.z);
    return dot(plane.xyz, corner) + plane.w < 0.0;
}

// Near-to-far bucket of camera distance of the AABB centre. Splits are genconst.
uint distance_bucket(float dist)
{
    uint b = 0;
    if (dist >= CULL_BUCKET_SPLIT_0)
        b = 1;
    if (dist >= CULL_BUCKET_SPLIT_1)
        b = 2;
    if (dist >= CULL_BUCKET_SPLIT_2)
        b = 3;
    return min(b, CULL_DISTANCE_BUCKETS - 1u);
}

// True when every fragment of a camera-relative AABB would be discarded by the
// coarse-LOD slab clip (`mesh3d.frag.slang`). Both extents must be active:
// `clip_v == 0` makes the fragment `inside_v` false, so nothing is discarded.
// Horizontal test uses the farthest xz corner (max |x|, max |z|) so a skip is
// exact for the per-fragment `length(world.xz) < clip` test. Host mirror:
// `lod_aabb_inside_slab` in vk/cull.rs.
// Host twin: `occ_mip_for_rect` in vk/cull.rs. Mip where the UV rect spans
// at most 2x2 texels of the pyramid (ceil so we go coarser on a split).
uint occ_mip_for_rect(float2 span_uv, uint2 level0, uint mip_count)
{
    float span_tex = max(span_uv.x * float(level0.x), span_uv.y * float(level0.y));
    float mip_f = ceil(log2(max(span_tex * 0.5, 1.0)));
    return min(uint(mip_f), mip_count - 1u);
}

// Host twin: `occ_hidden` in vk/cull.rs.
bool occ_hidden_z(float nearest_z, float occ_min)
{
    return nearest_z < occ_min - CULL_OCC_EPS;
}

// Project the 8 camera-relative AABB corners with the previous view-proj.
// Any corner with w <= 0 (crosses the previous near plane) → not occluded.
// Degenerate screen rect → not occluded. Else gather the 4 pyramid texels
// covering the clamped UV rect at a mip where the rect is at most 2x2, take
// their MIN (farthest reversed-Z occluder), and hide iff nearest_z is farther.
bool occlusion_hidden(float3 mn, float3 mx)
{
    if ((params.occ_flags & 1u) == 0u)
        return false;
    float nearest_z = 0.0;
    float2 uv_min = float2(1.0, 1.0);
    float2 uv_max = float2(0.0, 0.0);
    [unroll]
    for (int z = 0; z < 2; z++)
    {
        [unroll]
        for (int y = 0; y < 2; y++)
        {
            [unroll]
            for (int x = 0; x < 2; x++)
            {
                float3 p = float3(
                    x != 0 ? mx.x : mn.x,
                    y != 0 ? mx.y : mn.y,
                    z != 0 ? mx.z : mn.z);
                float4 clip = mul(params.prev_view_proj, float4(p, 1.0));
                if (clip.w <= 0.0)
                    return false;
                float3 ndc = clip.xyz / clip.w;
                float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                uv_min = min(uv_min, uv);
                uv_max = max(uv_max, uv);
                nearest_z = max(nearest_z, ndc.z);
            }
        }
    }
    uv_min = saturate(uv_min);
    uv_max = saturate(uv_max);
    float2 span = uv_max - uv_min;
    if (span.x <= 0.0 || span.y <= 0.0)
        return false;

    uint mip = occ_mip_for_rect(span, uint2(params.hiz_w, params.hiz_h), params.hiz_mips);
    float2 mip_size = max(float2(params.hiz_w, params.hiz_h) / exp2(float(mip)), float2(1.0, 1.0));
    int2 i0 = int2(floor(uv_min * mip_size));
    int2 i1 = int2(ceil(uv_max * mip_size) - 1.0);
    i1 = max(i1, i0);
    int2 dim = int2(mip_size);
    i0 = clamp(i0, int2(0, 0), dim - 1);
    i1 = clamp(i1, int2(0, 0), dim - 1);

    float2 inv = 1.0 / mip_size;
    float occ = min(
        min(hizTex.SampleLevel((float2(i0) + 0.5) * inv, mip).x,
            hizTex.SampleLevel((float2(i1.x, i0.y) + 0.5) * inv, mip).x),
        min(hizTex.SampleLevel((float2(i0.x, i1.y) + 0.5) * inv, mip).x,
            hizTex.SampleLevel((float2(i1) + 0.5) * inv, mip).x));
    return occ_hidden_z(nearest_z, occ);
}

bool lod_aabb_inside_slab(float3 mn, float3 mx, float clip, float clip_v)
{
    if (!(clip > 0.0 && clip_v > 0.0))
        return false;
    float2 far_xz = max(abs(mn.xz), abs(mx.xz));
    bool inside_h = length(far_xz) < clip;
    bool inside_v = max(abs(mn.y), abs(mx.y)) < clip_v;
    return inside_h && inside_v;
}

void emit(uint part, DrawCmd cmd)
{
#ifdef USE_WAVE_ATOMICS
    // Compact-by-key: one InterlockedAdd per unique partition in the wave,
    // then each lane writes at base + WavePrefixCountBits. Lanes that did not
    // call emit() are inactive here, so the ballot is exactly the emitters.
    while (true)
    {
        uint first = WaveReadLaneFirst(part);
        if (part == first)
        {
            uint n = WaveActiveCountBits(true);
            uint prefix = WavePrefixCountBits(true);
            uint base = 0;
            if (WaveIsFirstLane())
                InterlockedAdd(counts[first], n, base);
            base = WaveReadLaneFirst(base);
            uint i = base + prefix;
            if (i < partitions[first].capacity)
                commands[partitions[first].offset + i] = cmd;
            break;
        }
    }
#else
    uint i;
    InterlockedAdd(counts[part], 1, i);
    if (i < partitions[part].capacity)
        commands[partitions[part].offset + i] = cmd;
#endif
}

[shader("compute")]
[numthreads(CULL_WORKGROUP, 1, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= params.slot_count)
        return;
    uint word = slot_arena[slot];
    if (word == 0u)
        return; // dead slot
    bool cam_visible = (visible[slot >> 5] & (1u << (slot & 31u))) != 0u;
    // Invisible slots cannot contribute a camera draw; skip their record load
    // and frustum math unless this frame is actually regenerating shadows.
    if (!cam_visible && params.shadow_enabled == 0u)
        return;

    uint arena = word - 1u;
    MeshRecord rec = records[slot];
    uint pass = (rec.detail_pass >> 4) & 3u;
    if (pass > 1u)
        return; // Blend: CPU-sorted path

    float scale = detail_scale(rec.detail_pass);
    float3 offset = float3(rec.block - params.cam_block) - params.cam_frac + rec.local_off;
    float3 mn = rec.aabb_min * scale + offset;
    float3 mx = rec.aabb_max * scale + offset;

    DrawCmd cmd;
    cmd.index_count = rec.index_count;
    cmd.instance_count = 1u;
    cmd.first_index = 0u;
    cmd.vertex_offset = rec.vertex_offset;
    cmd.first_instance = slot;

    if (cam_visible)
    {
        bool in_frustum = true;
        [unroll]
        for (int p = 0; p < 5; p++)
            in_frustum = in_frustum && !outside_plane(params.cam_planes[p], mn, mx);
        if (in_frustum)
        {
            // Uploaded this frame: last frame has no depth for this mesh.
            bool skip_occ = (rec.flags & 2u) != 0u;
            if (!skip_occ && occlusion_hidden(mn, mx))
            {
                if ((pc.flags & 1u) != 0)
                {
                    uint unused;
                    InterlockedAdd(stats[6], 1u, unused);
                }
            }
            else
            {
            // Camera group: the pass index, except coarse-LOD opaque → group 2.
            uint group = (pass == 0u && scale > 1.0) ? 2u : pass;
            // Group 2 fully inside the slab would discard every fragment.
            if (group != 2u || !lod_aabb_inside_slab(mn, mx, params.clip, params.clip_v))
            {
                float dist = length(0.5 * (mn + mx));
                uint bucket = distance_bucket(dist);
                uint part = (group * params.arena_count + arena) * CULL_DISTANCE_BUCKETS + bucket;
                if ((params.flags & 1u) == 0u || (rec.flags & 1u) == 0u)
                {
                    emit(part, cmd);
                    if ((pc.flags & 1u) != 0)
                    {
                        uint unused;
                        InterlockedAdd(stats[group * 2u], 1u, unused);
                        InterlockedAdd(stats[group * 2u + 1u], cmd.index_count, unused);
                    }
                }
                else
                {
                    // Cumulative 6*quads from packed u16 pairs (upload slots 0|1, 2|3, 4|5).
                    uint b[7];
                    b[0] = 0u;
                    [unroll]
                    for (uint i = 0u; i < 3u; i++)
                    {
                        uint packed = rec.face_quads[i];
                        b[i * 2u + 1u] = b[i * 2u] + 6u * (packed & 0xFFFFu);
                        b[i * 2u + 2u] = b[i * 2u + 1u] + 6u * (packed >> 16);
                    }
                    // +axis vis iff mn[axis] < 0; −axis iff mx[axis] > 0. +Y uses the
                    // drooped AABB because mesh3d.vert lowers far vertices by
                    // min(|xz|²·CURVE_INV_2R, CURVE_MAX_DROP). −Y without droop is
                    // already conservative.
                    bool vis[6];
                    vis[0] = mn.x < 0.0;
                    float2 far = max(abs(mn.xz), abs(mx.xz));
                    float droop = min(dot(far, far) * CURVE_INV_2R, CURVE_MAX_DROP);
                    vis[1] = (mn.y - droop) < 0.0;
                    vis[2] = mn.z < 0.0;
                    vis[3] = mx.x > 0.0;
                    vis[4] = mx.y > 0.0;
                    vis[5] = mx.z > 0.0;
                    // Merge consecutive visible non-empty buckets. emit may be called
                    // ≤3 times per lane; the wave path already handles divergent callers.
                    uint start = ~0u;
                    for (uint k = 0u; k < 6u; k++)
                    {
                        bool occ = vis[k] && (b[k + 1u] != b[k]);
                        if (occ)
                        {
                            if (start == ~0u)
                                start = k;
                        }
                        else if (start != ~0u)
                        {
                            DrawCmd run = cmd;
                            run.first_index = b[start];
                            run.index_count = b[k] - b[start];
                            emit(part, run);
                            if ((pc.flags & 1u) != 0)
                            {
                                uint unused;
                                InterlockedAdd(stats[group * 2u], 1u, unused);
                                InterlockedAdd(stats[group * 2u + 1u], run.index_count, unused);
                            }
                            start = ~0u;
                        }
                    }
                    if (start != ~0u)
                    {
                        DrawCmd run = cmd;
                        run.first_index = b[start];
                        run.index_count = b[6] - b[start];
                        emit(part, run);
                        if ((pc.flags & 1u) != 0)
                        {
                            uint unused;
                            InterlockedAdd(stats[group * 2u], 1u, unused);
                            InterlockedAdd(stats[group * 2u + 1u], run.index_count, unused);
                        }
                    }
                }
            }
            }
        }
    }

    // Shadow casters: full-res opaque only. Coarse LOD excluded (harsh self-shadows).
    // Per-cascade partitions so the near cascade does not vertex-process far casters.
    if (pass == 0u && scale <= 1.0 && params.shadow_enabled != 0u)
    {
        uint shadow_base = CULL_CAMERA_GROUPS * params.arena_count * CULL_DISTANCE_BUCKETS;
        [unroll]
        for (int c = 0; c < 2; c++)
        {
            bool cast = true;
            [unroll]
            for (int p = 0; p < 5; p++)
                cast = cast && !outside_plane(params.shadow_planes[c * 5 + p], mn, mx);
            if (cast)
                emit(shadow_base + uint(c) * params.arena_count + arena, cmd);
        }
    }
}
