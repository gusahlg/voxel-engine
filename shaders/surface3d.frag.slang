// surface3d.frag.slang — the Zone-3 far skin. Three things happen here:
//  1. NEAR-ZONE CLIP: fragments within `sun_light.w` horizontal metres of the
//     camera are discarded, so the skin renders only BEYOND the full-res chunks
//     and LOD tiles — it never pokes through the near zones (a coarse 64 m
//     height surface can't match fine terrain, so it is clipped, not depth-hid).
//  2. TIME-OF-DAY TINT: the grey is multiplied by the frame's sky light
//     (ambient + sun), matching how mesh3d shades a full-sky top face, so the
//     skin darkens at night and warms at sunset with the rest of the world.
//  3. FOG PARITY: fogged toward the horizon EXACTLY as mesh3d.frag fogs the
//     tiles, so the skin blends into the tiles/horizon seamlessly.
// No texture sampling: the surface has no layer and never samples the block array.

struct FragmentInput
{
    [[vk::location(0)]]
    float4 color;

    [[vk::location(1)]]
    float dist;

    [[vk::location(2)]]
    float boxdist; // camera-relative ground-plane box distance (near-zone clip)
};

// Mirrors the vertex stage's push constant (same layout). `sun_light.w` is
// repurposed by the surface pass to carry the near-zone clip radius (metres);
// `sun_light.rgb`/`ambient.rgb` drive the time-of-day tint and `fog` the fog.
struct PushConstants
{
    float4x4 view_proj;
    float4 sun_light; // rgb sky tint, w = near-zone clip radius (metres)
    float4 ambient;
    float4 fog; // rgb colour, a = density
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

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
    // Near-zone clip: discard fragments inside the tile ring's box so the skin
    // shows only beyond the chunks + LOD tiles. Box test (not radial) matches
    // the tile ring's axis-aligned shape exactly. (0 ⇒ no clip.)
    if (input.boxdist < pc.sun_light.w)
        discard;

    // Time-of-day tint: the baked grey (with its per-cell lambert shade) lit by
    // the sky, matching mesh3d's full-sky top face `ambient + sun_light`.
    float3 lit = srgbToLinear(pc.ambient.rgb) + srgbToLinear(pc.sun_light.rgb);
    float3 rgb = srgbToLinear(input.color.rgb) * lit;
    // Exponential distance fog toward the (linearised) horizon colour — copied
    // verbatim from mesh3d.frag so the skin fogs identically to the tiles.
    float fog = 1.0 - exp(-input.dist * pc.fog.a);
    rgb = lerp(rgb, srgbToLinear(pc.fog.rgb), fog);
    return float4(rgb, 1.0);
}
