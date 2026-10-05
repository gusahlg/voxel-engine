// Far-body table layout. Shared by far_body.slang and sky_tile.vert so the
// vertex and fragment stages declare the same storage-buffer block.
#ifndef VOXEL_FAR_TABLE_SLANG
#define VOXEL_FAR_TABLE_SLANG

// Ten float4 lanes, 160 bytes. atmosphere.a is the shape: 0 cube, 1 sphere,
// 2 inner sphere, 3 rounded, 4 mapped.
// seed.x is the noise seed (unused for mapped). seed.y is the rounded
// exponent as float bits (0 otherwise). seed.z is the map id + 1 (0 = not
// mapped). seed.w is 0.
// albedo2.w is the body's world distance for every shape. Each body is
// ray-tested with its centre at distance 1, so a normalised `t` or `facing`
// is not comparable across bodies; the loop orders hits by `t * distance`
// and rims and point blobs by `facing * distance`. Mapped spare lanes:
// albedo0.w = horizon, albedo1.w = air thickness in radius units. The other
// .w lanes stay 0. dir_rho.w stays the reference radius/distance, not the
// hi radius.
struct FarGpu
{
    float4 dir_rho;
    float4 rot;
    float4 albedo0;
    float4 albedo1;
    float4 albedo2;
    float4 albedo3;
    float4 albedo4;
    float4 albedo5;
    float4 atmosphere;
    uint4 seed;
};

struct FarTable
{
    // x = count, y = tile size in pixels, z = tiles_x, w = tiles_y.
    // yzw are 0 when the host has no view; the shader then walks every body.
    uint4 header;
    float4 cone[32]; // xyz = dir, w = sine bound (-1 = never reject)
    FarGpu body[32];
    // Bit k is kept body k. One word per tile, row-major. Host cap is 8192.
    uint tile_mask[8192];
    // Tile-quad draws. Not read by far_bodies.
    // x = n_base (mask == 0), y = n_full (mask != 0), z = width, w = height.
    // tile_index is three ascending row-major runs: mask == 0, then tiles
    // whose bits are only spheres or inner spheres, then the rest. n_full is
    // the last two runs. The vertex shader reads Vulkan InstanceIndex
    // (includes firstInstance), not D3D SV_InstanceID. The CPU's firstInstance
    // selects the run.
    uint4 list_header;
    uint tile_index[8192];
};

#endif
