// surface3d.vert.slang — camera-relative colored-surface vertices for Zone-3 far skin.
// Input: float3 position, RGBA8 color. Per-draw float4: offset (xyz) + scale (w).
// SV_VulkanInstanceID reads plain InstanceIndex for SSBO indexing.


[[vk::binding(0, 0)]]
StructuredBuffer<float4> draw_offsets;

struct VertexInput
{
    [[vk::location(0)]]
    float3 pos;

    [[vk::location(1)]]
    float4 color; // RGBA8_UNORM attribute → 0..1 float4

    uint instance : SV_VulkanInstanceID; // raw SPIR-V InstanceIndex
};

struct VertexOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float4 color;

    [[vk::location(1)]]
    float dist; // camera-relative distance, for fragment fog

    [[vk::location(2)]]
    // Camera-relative box distance on the ground plane, for the near-zone clip.
    // Box, not radius: the tile ring it clips against is axis-aligned, so clipping
    // with a box matches exactly — no corner overlap where skin and tiles conflict.
    float boxdist;
};

// Mirrors mesh3d's push layout (Mesh3dPush = view_proj + a tail `clip` scalar).
// The vertex stage uses only view_proj; the fragment stage reads `clip`.
struct PushConstants
{
    float4x4 view_proj;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    float4 placement = draw_offsets[input.instance]; // xyz = offset, w = scale
    float3 world = input.pos * placement.w + placement.xyz;

    VertexOutput output;
    // Plain rectilinear projection; wide-FOV compression is a post-process.
    output.position = mul(pc.view_proj, float4(world, 1.0));
    output.color = input.color;
    output.dist = length(world); // camera at the origin, so depth = distance to world
    output.boxdist = max(abs(world.x), abs(world.z)); // ground-plane box distance
    return output;
}
