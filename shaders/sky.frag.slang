// Procedural sky for background pixels; handles view-ray reconstruction and sun disc.
// Clouds are a single bilinear tap of the per-frame direction LUT (sky_cloud.comp).
// Shares FrameUniforms (binding 1) with fog to prevent divergence.

#include "sky_common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 ndc;
    float4 frag_pos : SV_Position;
};

// Set 0, binding 0: octahedral cloud LUT (RGBA16F, linear-clamp).
[[vk::binding(0, 0)]]
Sampler2D cloudLut;

// Set 0, binding 1: the shared per-frame uniforms — the sky's colours (linear
// zenith/horizon/light/turbidity) come from here, identical to the terrain fog.
[[vk::binding(1, 0)]]
ConstantBuffer<FrameUniforms> frame;

// Hash-grid stars; static in world directions (deterministic). Fades by night/moon.
float3 stars(float3 ray, float3 moon_dir, float night)
{
    // Project the ray to a 2D field; denom stays positive for any sky ray.
    float2 sc = ray.xz / (ray.y + length(ray.xz));
    float2 g = sc * STAR_DENSITY;
    float2 cell = floor(g);

    if (hash12(cell) < STAR_THRESHOLD)
        return float3(0.0);

    // Jitter the point off the cell centre so the grid regularity is invisible.
    float2 jit = float2(hash12(cell + 1.7), hash12(cell + 4.3)) - 0.5;
    float2 d = frac(g) - 0.5 - jit * 0.8;
    float point = exp(-STAR_SHARP * dot(d, d));

    // Invisible by day, out of the horizon band, and clear of the moon disc.
    float horizon = saturate(ray.y * STAR_HORIZON_FADE);
    float moon = 1.0 - smoothstep(STAR_MOON_FADE, 1.0, dot(ray, moon_dir));
    float3 star_col = float3(STAR_COLOR[0], STAR_COLOR[1], STAR_COLOR[2]);
    return star_col * (point * night * horizon * moon);
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    // Reconstruct the view ray from the (wide, rectilinear) source projection.
    // The tonemap resample compresses the sky's periphery in lockstep with the
    // terrain, so no per-pixel warp is needed here.
    float4 far = mul(frame.inv_view_proj, float4(input.ndc, 0.0, 1.0));
    float3 ray = normalize(far.xyz);

    // Fog IS the sky: the same analytic radiance the terrain fades toward, plus
    // the additive HDR sun disc. glow_pow / sun dir / disc cosines are hoisted.
    float3 col = sky_radiance(ray, frame)
               + sun_disc_cos(ray, frame.sun.xyz, frame.sun_tint.rgb, frame.sun.w, frame.sun_tint.w);

    // Moon opposite sun, stars and moon gated by night (1 - light.w).
    float night = 1.0 - frame.light.w;
    float3 moon_dir = -frame.sun.xyz;
    float3 moon_tint = float3(MOON_TINT[0], MOON_TINT[1], MOON_TINT[2]);
    col += sun_disc_cos(ray, moon_dir, moon_tint, frame.moon.x, frame.moon.y) * night;
    // Uniform branch (both factors are per-frame constants): skips the whole
    // hash-grid star evaluation by day and when `RenderFlags::stars` is off —
    // the multiplied-out zero was still being computed for every sky pixel.
    if (night * frame.extras.x > 0.0)
        col += stars(ray, moon_dir, night) * frame.extras.x;

    // Slab clouds composite LAST so opaque cloud occludes the sun/moon/stars it
    // covers. Hemisphere octahedral is injective only for y >= 0; skip the tap
    // below the cloud horizon so downward rays don't alias zenith texels.
    float cover = 0.0;
    float3 cloud_lit = float3(0.0);
    if (ray.y > CLOUD_HORIZON_OFFSET)
    {
        float4 t = cloudLut.SampleLevel(sky_oct_encode(ray), 0);
        cloud_lit = t.rgb;
        cover = t.a;
    }
    col = col * (1.0 - cover) + cloud_lit;

    col += (hash12(input.ndc * 937.0) - 0.5) / 255.0;
    return float4(col, 1.0);
}
