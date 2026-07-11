// mesh3d.vert.slang — camera-relative packed world-mesh vertices. Input is a
// single uint2 (see mesh.rs for the bit table) with position, normal, layer, AO/light.
// Per-draw float4 carries offset (xyz) and scale (w): world = local * scale + offset.
// Meshes use small local coords; scale = 2^k for coarse LOD.
//
// SV_VulkanInstanceID (not SV_InstanceID) to read plain InstanceIndex.
//
// Shifts/masks mirror mesh.rs. UV and shade derive from position and normal.

#include "common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<float4> draw_offsets;

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

    [[vk::location(1)]]
    nointerpolation uint layer;

    // Camera-relative world position (camera sits at the origin). The fragment
    // derives fog distance (`length(world)`), the fog/reflection view ray
    // (`normalize(world)`), and the shadow-map light-space projection from it.
    [[vk::location(2)]]
    float3 world;

    // Face normal (unit), used by the fragment stage for the shadow slope bias.
    [[vk::location(3)]]
    nointerpolation float3 nrm;

    // Fragment-side lighting inputs: x = per-face directional shade, y = AO
    // factor, z = skylight amount [0,1] (also the sky-visibility term), w =
    // blocklight amount [0,1]. The lighting term itself is evaluated in the
    // fragment (candle curve, ambient floor, shadow) off the per-frame UBO.
    [[vk::location(4)]]
    float4 lightparams;
};

struct PushConstants
{
    float4x4 view_proj;
    // All sky/fog/candle lighting rides the per-frame UBO (set 0, binding 2);
    // this stage needs only the camera. The push range's tail `clip` lane
    // (surface pass only) is not declared here — a stage may declare a prefix.
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
    uint layer = (w0 >> 18) & 0xFF;
    uint ao = (w1 >> 0) & 0x3;
    uint sky = (w1 >> 2) & 0xF;
    uint block = (w1 >> 6) & 0xF;

    // UV mapping: X face (z,y), Y face (x,z), Z face (x,y).
    float2 uv;
    if (normal < 2)
        uv = float2(z, y);
    else if (normal < 4)
        uv = float2(x, z);
    else
        uv = float2(x, y);

    // Minecraft-style per-face base shade.
    float shade;
    if (normal == 2)
        shade = 1.0; // +Y top
    else if (normal == 3)
        shade = 0.5; // -Y bottom
    else if (normal < 2)
        shade = 0.8; // ±X
    else
        shade = 0.6; // ±Z

    float ao_factor = 0.4 + 0.2 * float(ao); // 3 → 1.0, 0 → 0.4
    float sky_amount = float(sky) / 15.0;
    float block_amount = float(block) / 15.0;

    // Face normal for the receiver's slope bias (matches the `normal` index).
    float3 N;
    if (normal == 0)      N = float3(1.0, 0.0, 0.0);
    else if (normal == 1) N = float3(-1.0, 0.0, 0.0);
    else if (normal == 2) N = float3(0.0, 1.0, 0.0);
    else if (normal == 3) N = float3(0.0, -1.0, 0.0);
    else if (normal == 4) N = float3(0.0, 0.0, 1.0);
    else                  N = float3(0.0, 0.0, -1.0);

    float4 placement = draw_offsets[input.instance]; // xyz = offset, w = scale
    float3 world = float3(x, y, z) * placement.w + placement.xyz;

    VertexOutput output;
    // Plain rectilinear projection; wide-FOV compression is a post-process.
    output.position = mul(pc.view_proj, float4(world, 1.0));
    // Scale UV to maintain texel density across LOD levels.
    output.uv = uv * placement.w;
    output.layer = layer;
    output.world = world;
    output.nrm = N;
    output.lightparams = float4(shade, ao_factor, sky_amount, block_amount);
    return output;
}
