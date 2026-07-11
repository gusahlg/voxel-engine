// surface3d.vert.slang — camera-relative unpacked colored-surface vertices for
// the Zone-3 far grey skin. Input is a float3 world-local position plus an
// RGBA8 colour (declared as an R8G8B8A8_UNORM attribute, so it arrives as a
// float4 in 0..1). A per-draw float4 (indexed by the raw Vulkan InstanceIndex)
// carries an offset in xyz and a uniform scale in w: `world = local * scale +
// offset` before the view_proj multiply, exactly like mesh3d.vert — the skin is
// drawn camera-relative so it never jitters at Zone-3 distances.
//
// SV_VulkanInstanceID (not SV_InstanceID): reads the plain SPIR-V InstanceIndex,
// so `first_instance` doubles as the draw's SSBO slot (see mesh3d.vert).


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

// Mirrors mesh3d's push constant layout exactly (Mesh3dPush = view_proj + the
// SkyLight tail); the vertex stage uses view_proj, the fragment stage the fog.
struct PushConstants
{
    float4x4 view_proj;
    float4 sun_light;
    float4 ambient;
    float4 fog; // rgb colour, a = density
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
