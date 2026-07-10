// mesh3d.frag.slang — unlit textured: sample the block-texture array at the
// derived (uv, layer), multiply rgb by the vertex lighting term (base face
// shade × AO × skylight/blocklight), and pass the sampled alpha through. The
// opaque pipeline draws with alpha-1 textures so the alpha is a no-op there;
// the transparent pipeline alpha-blends, so a translucent texture (e.g. water)
// shows through. Shared by both mesh pipelines.

struct FragmentInput
{
    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    nointerpolation uint layer;

    [[vk::location(2)]]
    float3 light;

    [[vk::location(3)]]
    float dist;
};

// Mirrors the vertex stage's push constant (same layout); the fragment reads
// only the fog tail. Declaring the full struct keeps `fog` at its offset (96).
struct PushConstants
{
    float4x4 view_proj;
    float4 sun_light;
    float4 ambient;
    float4 fog; // rgb colour, a = density
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Set 0, binding 1: shares the single 3D push-descriptor set with the vertex
// stage's per-draw offsets SSBO (binding 0). Vulkan permits only one push
// descriptor set per pipeline layout, so both bindings live in set 0.
[[vk::binding(1, 0)]]
Sampler2DArray blockTex;

// CPU colours arrive sRGB-encoded; the scene composites in linear light.
float3 srgbToLinear(float3 c)
{
    float3 lo = c / 12.92;
    float3 hi = pow(max((c + 0.055) / 1.055, 0.0), float3(2.4, 2.4, 2.4));
    return select(c <= 0.04045, lo, hi);
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    // Texture is sampled through an sRGB view, so tex.rgb is already linear.
    float4 tex = blockTex.Sample(float3(input.uv, float(input.layer)));
    float3 rgb = tex.rgb * input.light;
    // Exponential distance fog toward the (linearised) horizon colour.
    float fog = 1.0 - exp(-input.dist * pc.fog.a);
    rgb = lerp(rgb, srgbToLinear(pc.fog.rgb), fog);
    return float4(rgb, tex.a);
}
