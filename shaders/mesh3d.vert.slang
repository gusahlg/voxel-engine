// mesh3d.vert.slang — camera-relative packed world-mesh vertices. Input is a
// single uint2 (see mesh.rs for the bit table): a chunk-local integer position,
// a face-normal index, a texture-array layer, and reserved AO/light bits. A
// per-draw float4 (indexed by the raw Vulkan InstanceIndex) carries an offset
// in xyz and a uniform scale in w: the local position is scaled then offset
// (`world = local * scale + offset`) before the view_proj multiply, so meshes
// carry small local coordinates while the world spans huge distances and a
// single 16³-local mesh can stand in for a coarse LOD tile (scale = 2^k). One
// element per indirect command; immediate debug geometry uses a separate
// (descriptor-set-free) pipeline and never indexes this buffer.
//
// SV_VulkanInstanceID, NOT SV_InstanceID: Slang lowers SV_InstanceID with D3D
// semantics (InstanceIndex - BaseInstance), which would always read slot 0
// under the first_instance-as-index scheme. SV_VulkanInstanceID lowers to the
// plain InstanceIndex builtin.
//
// The shifts/masks below mirror the SHIFT_*/MASK_* consts in mesh.rs. UV is
// derived from position + normal (the sampler is REPEAT, so greedy quads tile);
// per-face directional shade is derived from the normal.

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

    [[vk::location(2)]]
    float3 light;

    [[vk::location(3)]]
    float dist; // camera-relative distance, for fragment fog
};

struct PushConstants
{
    float4x4 view_proj;
    float4 sun_light; // rgb tints the sky-lit term
    float4 ambient;   // rgb unlit floor
    float4 fog;       // rgb fog colour, a = density
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// CPU light colours arrive sRGB-encoded; lighting composites in linear.
float3 srgbToLinear(float3 c)
{
    float3 lo = c / 12.92;
    float3 hi = pow(max((c + 0.055) / 1.055, 0.0), float3(2.4, 2.4, 2.4));
    return select(c <= 0.04045, lo, hi);
}

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

    // Skylight is tinted by the sun over an ambient floor; blocklight (torches)
    // stays white and is unaffected by the sky. Defaults (sun=white, ambient=0)
    // collapse this to the old grayscale `max(sky, block)/15`.
    float3 lit = max(
        srgbToLinear(pc.ambient.rgb) + srgbToLinear(pc.sun_light.rgb) * sky_amount,
        block_amount.xxx);

    float4 placement = draw_offsets[input.instance]; // xyz = offset, w = scale
    float3 world = float3(x, y, z) * placement.w + placement.xyz;

    VertexOutput output;
    // Plain rectilinear projection; wide-FOV compression is a post-process
    // (tonemap resample), so the vertex path stays linear.
    output.position = mul(pc.view_proj, float4(world, 1.0));
    // Counter-scale UV by the per-draw scale so texel density stays locked to
    // world metres: a full-res chunk (scale 1) tiles once per block, and a
    // coarse LOD tile (scale 2^k) tiles 2^k times per cell — still once per
    // metre — instead of stretching one block texel across the whole cell. Keeps
    // far LOD terrain texturally faithful to the full-res band it abuts.
    output.uv = uv * placement.w;
    output.layer = layer;
    output.light = shade * ao_factor * lit;
    output.dist = length(world); // camera sits at the origin, so depth = distance to world
    return output;
}
