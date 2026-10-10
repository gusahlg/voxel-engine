// Per-mesh placement record (std430, 80 bytes). GPU twin of
// vk::records::MeshRecord, whose size and field offsets are pinned by the
// `const _: () = assert!(..)` lines in src/vk/records.rs. Included by
// cull.comp, mesh3d.vert and shadow_depth.vert; the instance index into
// `mesh_records` is the record slot, so the stride must match exactly.

#ifndef VOXEL_MESH_RECORD_SLANG
#define VOXEL_MESH_RECORD_SLANG

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
    uint flags;       // bit 0 = packed face-runs valid; clear → whole-mesh draw
};

#endif
