// Emit indirect draw commands per frustum test: one thread per mesh slot.
// Camera partitions are (group, arena, distance-bucket); shadows are
// per-cascade and unbucketed. Camera groups (CULL_CAMERA_GROUPS): 0 full-res
// Opaque, 1 Cutout, 2 coarse-LOD Opaque (scale > 1, the only camera draws
// that pay for the slab-clip `discard`).

#include "common.slang"

// Mirrors vk::buffers::MeshRecord exactly (std430).
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
    uint pad0;
    uint pad1;
    uint pad2;
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

    // Detail is biased; exp2 handles fractional scales for half-block levels.
    float scale = exp2(float(rec.detail_pass & 0xFu) - float(DETAIL_GPU_BIAS));
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
            // Camera group: the pass index, except coarse-LOD opaque → group 2.
            uint group = (pass == 0u && scale > 1.0) ? 2u : pass;
            float dist = length(0.5 * (mn + mx));
            uint bucket = distance_bucket(dist);
            uint part = (group * params.arena_count + arena) * CULL_DISTANCE_BUCKETS + bucket;
            emit(part, cmd);
            if ((pc.flags & 1u) != 0)
            {
                uint unused;
                InterlockedAdd(stats[group * 2u], 1u, unused);
                InterlockedAdd(stats[group * 2u + 1u], cmd.index_count, unused);
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
