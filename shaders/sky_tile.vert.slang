// Instanced sky tile quad. One instance is one 64 px screen tile (or the
// coarser size in the table header). Vertex 0..5 are two triangles covering
// the tile's pixel rect at the reversed-Z far plane (z = 0), the same depth
// as the fullscreen triangle in sky.vert.
//
// The negative-height viewport maps NDC y-up to framebuffer y-down. Corner
// ndc uses the same inverse as the CPU (`framebuffer_ndc`): position.xy is
// that ndc and w is 1, so the interpolated ndc equals the sample's clip xy,
// which is the fullscreen triangle's per-pixel ndc.
//
// `SV_VulkanInstanceID` is Vulkan's InstanceIndex, which includes
// firstInstance. Slang's `SV_InstanceID` subtracts BaseInstance (D3D), so the
// full draw's `firstInstance = n_base` would read the base list.

#include "far_table.slang"

// Set 0, binding 2: the same far-body table the fragment reads.
[[vk::binding(2, 0)]]
StructuredBuffer<FarTable> farTable;

struct VertexOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 ndc;
};

[shader("vertex")]
VertexOutput vertexMain(uint vid: SV_VertexID, uint iid: SV_VulkanInstanceID)
{
    uint tilePx = farTable[0].header.y;
    uint tilesX = farTable[0].header.z;
    uint width = farTable[0].list_header.z;
    uint height = farTable[0].list_header.w;
    uint tile = farTable[0].tile_index[iid];
    uint tx = tile % tilesX;
    uint ty = tile / tilesX;
    uint x0 = tx * tilePx;
    uint y0 = ty * tilePx;
    // Right and bottom tiles stop at the render extent (not the next grid line).
    uint x1 = min(x0 + tilePx, width);
    uint y1 = min(y0 + tilePx, height);

    // 0 TL, 1 BL, 2 BR, 3 TL, 4 BR, 5 TR in framebuffer pixels (y down).
    // CCW in y-up NDC. Cull is off; the order only has to cover the rect.
    bool right = vid == 2u || vid >= 4u;
    bool bottom = vid == 1u || vid == 2u || vid == 4u;
    float xf = right ? float(x1) : float(x0);
    float yf = bottom ? float(y1) : float(y0);

    float ndc_x = (xf / float(width)) * 2.0 - 1.0;
    float ndc_y = 1.0 - (yf / float(height)) * 2.0;

    VertexOutput output;
    output.ndc = float2(ndc_x, ndc_y);
    output.position = float4(output.ndc, 0.0, 1.0);
    return output;
}
