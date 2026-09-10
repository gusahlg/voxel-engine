// Per-mesh placement record (std430, 80 bytes).
// Mirrors vk::records::MeshRecord exactly. Host stride is pinned by
// `const _: () = assert!(size_of::<MeshRecord>() == 80)` in src/vk/records.rs;
// changing this layout without that assert (and the three include sites) is a
// silent GPU bug.

#ifndef VOXEL_MESH_RECORD_SLANG
#define VOXEL_MESH_RECORD_SLANG

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
    uint flags;       // bit 0 = packed face-runs valid; clear → whole-mesh draw
};

#endif
