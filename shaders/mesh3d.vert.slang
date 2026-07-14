// mesh3d.vert.slang — camera-relative packed world-mesh vertices. Input is a
// single uint2 (see mesh.rs for the bit table) with position, normal, layer, AO/light.
// Per-draw float4 carries offset (xyz) and scale (w): world = local * scale + offset.
// Meshes use small local coords; scale = 2^k for coarse LOD.
//
// SV_VulkanInstanceID (not SV_InstanceID) to read plain InstanceIndex.
//
// Shifts/masks mirror mesh.rs. UV and shade derive from position and normal.

#include "common.slang"

// Per-draw placement. std430 layout mirrors vk::buffers::DrawOffset exactly.
struct DrawOffset
{
    float3 offset;
    float scale;
    float fade;
    uint mode;      // bit0 FLAT_COLOR, bit1 FADE_OUT
    uint flat_rgba; // sRGB RGBA8: R | G<<8 | B<<16 | A<<24
    uint pad;
};

[[vk::binding(0, 0)]]
StructuredBuffer<DrawOffset> draw_offsets;

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

    // Per-draw style: fade, mode, flat_rgba.
    [[vk::location(7)]]
    nointerpolation float fade;
    [[vk::location(8)]]
    nointerpolation uint mode;
    [[vk::location(9)]]
    nointerpolation uint flat_rgba;
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

// Gameplay-tuned planet curvature: distant terrain droops below the horizon,
// hidden by fog. PRESENTATION-ONLY by contract (E-06): the droop is applied to
// the CLIP position alone, never to `output.world` — so fog, lighting, shadow
// evaluation, and every CPU consumer (culling, world outputs, shadow depth)
// agree on the flat world. The bounded artifacts that buys: frustum culling is
// conservative near the bottom edge (a drooped tile may be drawn just
// off-screen, never wrongly culled away), and shadows are computed for the
// flat silhouette. The droop itself is clamped (CURVE_MAX_DROP) so an
// arbitrarily distant vertex can neither overflow nor fold the horizon.
static const float CURVE_INV_2R = 1.0 / (2.0 * 300000.0); // ~300 km visual radius
static const float CURVE_MAX_DROP = 4096.0; // metres; reached ~50 km out

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    uint w0 = input.packed.x;
    uint w1 = input.packed.y;

    float x = float((w0 >> 0) & 0x1F);
    float y = float((w0 >> 5) & 0x1F);
    float z = float((w0 >> 10) & 0x1F);
    uint normal = (w0 >> 15) & 0x7;   // 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z
    uint layer = (w0 >> 18) & 0x3FFF;
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
    DrawOffset placement = draw_offsets[input.instance];
    float3 world = (float3(x, y, z) + micro) * placement.scale + placement.offset;

    VertexOutput output;
    // Round-world horizon: curve on-screen silhouette only, preserving world position for fog/shadows.
    float3 curved = world;
    curved.y -= min(dot(world.xz, world.xz) * CURVE_INV_2R, CURVE_MAX_DROP);
    // Plain rectilinear projection; wide-FOV compression is a post-process.
    output.position = mul(pc.view_proj, float4(curved, 1.0));
    // Resolve depth ties in coplanar LOD levels: push each farther proportionally to cell size.
    if (placement.scale > 1.0)
        output.position.z *= 1.0 - placement.scale / 4096.0;
    // Scale UV to maintain texel density across LOD levels.
    output.uv = uv * placement.scale;
    output.layer = layer;
    output.world = world;
    output.nrm = N;
    output.lightparams = float3(ao_factor, sky_amount, block_amount);
    output.water = water;
    output.scale = placement.scale;
    output.fade = placement.fade;
    output.mode = placement.mode;
    output.flat_rgba = placement.flat_rgba;
    return output;
}
