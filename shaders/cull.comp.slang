// Emit indirect draw commands per frustum test: one thread per mesh slot.
// Camera partitions are (group, arena, distance-bucket); shadows are
// per-cascade and unbucketed. Camera groups (CULL_CAMERA_GROUPS): 0 full-res
// Opaque, 1 Cutout, 2 coarse-LOD Opaque (scale > 1), 3 full-res caged,
// 4 coarse-LOD caged. Groups 2 and 4 whose AABB is fully inside the LOD box
// are not emitted (every fragment would be discarded). A caged mesh is culled
// by its eight corners. Face-run culling keeps its direction buckets, tested
// in the cage's affine frame (`cage_direction_mask`); a singular frame draws
// the mesh whole. Shadows always draw whole.

#include "common.slang"
#include "cage.slang"

#include "mesh_record.slang"

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
    uint flags; // CULL_FLAG_FACE_RUNS = per-direction face-run culling
    float half_x; // full-res box half-extents; a non-positive component covers nothing
    float half_y;
    float half_z;
    // Centre is three scalars, not a float3: a float3 would align to 16.
    // lod_bucket_scale multiplies the distance splits for coarse-LOD groups
    // only (1 restores the full-res edges). The three pads keep this block
    // 320 bytes, matching vk::cull::CullParamsGpu; a float3 pad would align
    // to 16 and grow it.
    float centre_x;
    float centre_y;
    float centre_z;
    float lod_bucket_scale;
    float pad_b0;
    float pad_b1;
    float pad_b2;
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

// Per camera group, two uints each: [draws, index_count].
// 0 full-res opaque, 1 cutout, 2 coarse LOD, 3 caged, 4 caged LOD.
// Written only when the stats push flag is set (profiling).
[[vk::binding(7, 0)]]
RWStructuredBuffer<uint> stats;

struct Push
{
    uint flags; // CULL_FLAG_STATS = accumulate geometry stats
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

// Near-to-far bucket of camera distance of the AABB centre. Full-res groups
// pass scale 1 (splits 16/64/256). Coarse-LOD groups pass lod_bucket_scale
// (default 32: 512/2048/8192) so draws past 256 blocks still sort near-to-far.
uint distance_bucket(float dist, float scale)
{
    uint b = 0;
    if (dist >= CULL_BUCKET_SPLIT_0 * scale)
        b = 1;
    if (dist >= CULL_BUCKET_SPLIT_1 * scale)
        b = 2;
    if (dist >= CULL_BUCKET_SPLIT_2 * scale)
        b = 3;
    return min(b, CULL_DISTANCE_BUCKETS - 1u);
}

// True when every fragment of a camera-relative AABB would be discarded by the
// coarse-LOD box clip (`mesh3d.frag.slang`): abs(p - centre) < half on every
// axis. A non-positive half covers nothing. The skip is exact: the farthest
// corner from the centre on each axis is strictly inside. centre 0 matches
// the centred test. Host mirror: `lod_aabb_inside_box` in vk/cull_math.rs.
bool lod_aabb_inside_box(float3 mn, float3 mx, float3 centre, float half_x, float half_y, float half_z)
{
    if (!(half_x > 0.0 && half_y > 0.0 && half_z > 0.0))
        return false;
    float3 far = max(abs(mn - centre), abs(mx - centre));
    return far.x < half_x && far.y < half_y && far.z < half_z;
}

// Direction mask for a caged mesh, or 0 when the affine frame is singular
// (the caller draws the mesh whole). Bits 0..5 are upload order
// +X,+Y,+Z,−X,−Y,−Z. Bit 6 marks a usable mask so "every direction culled"
// cannot be confused with a singular frame.
//
// P_i = corners[i] + (anchor − cam_block) − cam_frac, the vertex shader's
// placement of corner i. E = [P1−P0, P2−P0, P4−P0], t_cam = E⁻¹(−P0).
// d is the max error of corners 3,5,6,7 against that frame, over the min
// edge; eps = 2d + CULL_CAGE_VIS_BIAS. Full cage range [0,1] (not the mesh's local
// bounds) is conservative. Host mirror: `cage_direction_vis` in
// vk/cull_math.rs. Shadows do not call this.
uint cage_direction_mask(Cage c)
{
    float3 o = float3(c.anchor - params.cam_block) - params.cam_frac;
    float3 P0 = c.corners[0].xyz + o;
    float3 P1 = c.corners[1].xyz + o;
    float3 P2 = c.corners[2].xyz + o;
    float3 P3 = c.corners[3].xyz + o;
    float3 P4 = c.corners[4].xyz + o;
    float3 P5 = c.corners[5].xyz + o;
    float3 P6 = c.corners[6].xyz + o;
    float3 P7 = c.corners[7].xyz + o;
    float3 e0 = P1 - P0;
    float3 e1 = P2 - P0;
    float3 e2 = P4 - P0;
    float3 c12 = cross(e1, e2);
    float3 c20 = cross(e2, e0);
    float3 c01 = cross(e0, e1);
    float det = dot(e0, c12);
    float l0 = length(e0);
    float l1 = length(e1);
    float l2 = length(e2);
    float vol = l0 * l1 * l2;
    if (!(abs(det) > CULL_CAGE_DET_REL * max(vol, 1.0)))
        return 0u;
    float3 b = -P0;
    float inv = 1.0 / det;
    float3 t_cam = float3(dot(b, c12), dot(b, c20), dot(b, c01)) * inv;
    float min_edge = min(l0, min(l1, l2));
    float dev = length(P3 - (P0 + e0 + e1));
    dev = max(dev, length(P5 - (P0 + e0 + e2)));
    dev = max(dev, length(P6 - (P0 + e1 + e2)));
    dev = max(dev, length(P7 - (P0 + e0 + e1 + e2)));
    float eps = 2.0 * (dev / min_edge) + CULL_CAGE_VIS_BIAS;
    uint mask = 0x40u;
    if (t_cam.x > -eps)
        mask |= 1u;
    if (t_cam.y > -eps)
        mask |= 2u;
    if (t_cam.z > -eps)
        mask |= 4u;
    if (t_cam.x < 1.0 + eps)
        mask |= 8u;
    if (t_cam.y < 1.0 + eps)
        mask |= 16u;
    if (t_cam.z < 1.0 + eps)
        mask |= 32u;
    return mask;
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
    uint pass = (rec.detail_pass >> DETAIL_GPU_BITS) & 3u;
    if (pass > 1u)
        return; // Blend: CPU-sorted path

    float scale = detail_scale(rec.detail_pass);
    float3 mn;
    float3 mx;
    // cage == 0 must not load the cage table (slot 0 is the zero cage).
    bool caged = rec.cage != 0u;
    if (caged)
    {
        Cage cg = cages[rec.cage];
        cage_corners_aabb(cg, mn, mx);
        float3 o = float3(cg.anchor - params.cam_block) - params.cam_frac;
        mn += o;
        mx += o;
    }
    else
    {
        float3 offset = float3(rec.block - params.cam_block) - params.cam_frac + rec.local_off;
        mn = rec.aabb_min * scale + offset;
        mx = rec.aabb_max * scale + offset;
    }

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
            // Flat: pass index, except coarse-LOD opaque → group 2.
            // Caged: group 3, or group 4 when scale > 1 (cutout included).
            uint group = caged
                ? ((scale > 1.0) ? CULL_CAGED_LOD_GROUP : CULL_CAGED_GROUP)
                : ((pass == 0u && scale > 1.0) ? CULL_OPAQUE_LOD_GROUP : pass);
            // LOD groups fully inside the slab would discard every fragment.
            bool lod_group = group == CULL_OPAQUE_LOD_GROUP || group == CULL_CAGED_LOD_GROUP;
            float3 lod_centre = float3(params.centre_x, params.centre_y, params.centre_z);
            if (!lod_group || !lod_aabb_inside_box(mn, mx, lod_centre, params.half_x, params.half_y, params.half_z))
            {
                float dist = length(0.5 * (mn + mx));
                float bucket_scale = lod_group ? params.lod_bucket_scale : 1.0;
                uint bucket = distance_bucket(dist, bucket_scale);
                uint part = (group * params.arena_count + arena) * CULL_DISTANCE_BUCKETS + bucket;
                bool face_runs = (params.flags & CULL_FLAG_FACE_RUNS) != 0u && (rec.flags & MESH_FLAG_FACE_RUNS) != 0u;
                bool vis[6];
                if (face_runs && caged)
                {
                    // Singular frame (mask 0) draws whole. Otherwise bits 0..5.
                    uint mask = cage_direction_mask(cages[rec.cage]);
                    if (mask == 0u)
                        face_runs = false;
                    else
                    {
                        [unroll]
                        for (uint k = 0u; k < 6u; k++)
                            vis[k] = (mask & (1u << k)) != 0u;
                    }
                }
                else if (face_runs)
                {
                    // +axis vis iff mn[axis] < 0; −axis iff mx[axis] > 0.
                    vis[0] = mn.x < 0.0;
                    vis[1] = mn.y < 0.0;
                    vis[2] = mn.z < 0.0;
                    vis[3] = mx.x > 0.0;
                    vis[4] = mx.y > 0.0;
                    vis[5] = mx.z > 0.0;
                }
                if (!face_runs)
                {
                    emit(part, cmd);
                    if ((pc.flags & CULL_FLAG_STATS) != 0)
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
                            if ((pc.flags & CULL_FLAG_STATS) != 0)
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
                        if ((pc.flags & CULL_FLAG_STATS) != 0)
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
