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

    // Fragment-side lighting inputs: x = AO factor, y = skylight amount [0,1]
    // (also the sky-visibility term), z = blocklight amount [0,1]. Directional
    // response is now the fragment's omni N·L term, so the
    // old per-face `shade` lane is gone. The lighting term itself is evaluated
    // in the fragment (candle curve, ambient floor, shadow) off the per-frame UBO.
    [[vk::location(4)]]
    float3 lightparams;

    // Water material bit (w1 bit 10). Liquid faces select the animated water
    // shading in mesh3d.frag's transparent path; opaque geometry never sets it.
    [[vk::location(5)]]
    nointerpolation uint water;

    // LOD scale: packed in offsets SSBO (no extra binding)
    [[vk::location(6)]]
    nointerpolation float scale;
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
    uint water = (w1 >> 10) & 0x1;

    // Per-axis 2-bit two's-complement micro-offsets. Truth: mesh.rs
    // SHIFT_MICRO_{X,Y,Z}=11,13,15, MASK_MICRO=0x3 (raw 0,1,2,3 -> 0,+1,-2,-1).
    // The seam-overdraw anti-z-fight nudge; zero for ordinary vertices.
    int mx = int((w1 >> 11) & 0x3); if (mx >= 2) mx -= 4;
    int my = int((w1 >> 13) & 0x3); if (my >= 2) my -= 4;
    int mz = int((w1 >> 15) & 0x3); if (mz >= 2) mz -= 4;
    float3 micro = float3(float(mx), float(my), float(mz)) * MICRO_STEP;

    // UV mapping: X face (z,y), Y face (x,z), Z face (x,y).
    float2 uv;
    if (normal < 2)
        uv = float2(z, y);
    else if (normal < 4)
        uv = float2(x, z);
    else
        uv = float2(x, y);

    float ao_factor = 0.4 + 0.2 * float(ao); // 3 → 1.0, 0 → 0.4
    float sky_amount = float(sky) / 15.0;
    float block_amount = float(block) / 15.0;

    // Face normal for the receiver's slope bias (matches the `normal` index).
    static const float3 FACE_NORMAL[6] = {
        float3(1.0, 0.0, 0.0),  float3(-1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),  float3(0.0, -1.0, 0.0),
        float3(0.0, 0.0, 1.0),  float3(0.0, 0.0, -1.0),
    };
    float3 N = FACE_NORMAL[normal];

    // Micro-offset is added in LOCAL space so it scales with the draw (2^k LOD).
    float4 placement = draw_offsets[input.instance]; // xyz = offset, w = scale
    float3 world = (float3(x, y, z) + micro) * placement.w + placement.xyz;

    VertexOutput output;
    // Plain rectilinear projection; wide-FOV compression is a post-process.
    output.position = mul(pc.view_proj, float4(world, 1.0));
    // Scale UV to maintain texel density across LOD levels.
    output.uv = uv * placement.w;
    output.layer = layer;
    output.world = world;
    output.nrm = N;
    output.lightparams = float3(ao_factor, sky_amount, block_amount);
    output.water = water;
    output.scale = placement.w;
    return output;
}
