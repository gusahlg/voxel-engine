// Procedural sky for background pixels; handles view-ray reconstruction and sun disc.
// Clouds are a single bilinear tap of the per-frame direction LUT (sky_cloud.comp).
// Shares FrameUniforms (binding 1) with fog to prevent divergence.

#include "sky_common.slang"

// SKY_BASE compiles the far-body call out. `far_bodies` returns `col`
// unchanged when the tile mask is 0, and `fwidth` only feeds that call, so
// the shared radiance / disc / moon / stars / cloud sequence is the whole
// result for a body-free pixel.

struct FragmentInput
{
    [[vk::location(0)]]
    float2 ndc;
    float4 frag_pos : SV_Position;
};

struct PushConstants
{
    float4x4 inv_view_proj; // camera rebased to origin: (ndc,0,1) → view ray
    float4 sun;             // xyz unit sun dir, w = cos(radius * SUN_DISC_CORE)
    float4 sun_tint;        // rgb linear disc tint, w = cos(radius * SUN_DISC_RIM)
    float4 moon;            // x = moon cos_core, y = moon cos_rim
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Set 0, binding 0: octahedral cloud LUT (RGBA16F, linear-clamp).
[[vk::binding(0, 0)]]
Sampler2D cloudLut;

// Set 0, binding 1: the shared per-frame uniforms — the sky's colours (linear
// zenith/horizon/light/turbidity) come from here, identical to the terrain fog.
[[vk::binding(1, 0)]]
ConstantBuffer<FrameUniforms> frame;

#ifndef SKY_BASE
#include "far_body.slang"
#endif

// Hash-grid stars on a world-space cube map: dominant axis picks the face
// (ties X, then Y, then Z), then a 2D cell on that face. Fixed in the universe
// when the viewer turns or crosses a cube edge. Same cell population and point
// shape as the old stereographic grid. Horizon fade follows local up and opens
// to the whole sphere as the star floor (space) rises.
float3 stars(float3 ray, float3 moon_dir, float night)
{
    float3 a = abs(ray);
    float2 uv;
    float face;
    if (a.x >= a.y && a.x >= a.z)
    {
        uv = ray.yz / a.x;
        face = ray.x >= 0.0 ? 0.0 : 1.0;
    }
    else if (a.y >= a.z)
    {
        uv = ray.xz / a.y;
        face = ray.y >= 0.0 ? 2.0 : 3.0;
    }
    else
    {
        uv = ray.xy / a.z;
        face = ray.z >= 0.0 ? 4.0 : 5.0;
    }
    // uv is [-1, 1]. STAR_DENSITY matches the old zenith cells-per-radian.
    float2 g = uv * STAR_DENSITY;
    float2 cell = floor(g) + float2(face * 17.0, face * 41.0);

    if (hash12(cell) < STAR_THRESHOLD)
        return float3(0.0);

    // Jitter the point off the cell centre so the grid regularity is invisible.
    float2 jit = float2(hash12(cell + 1.7), hash12(cell + 4.3)) - 0.5;
    float2 d = frac(g) - 0.5 - jit * 0.8;
    float point = exp(-STAR_SHARP * dot(d, d));

    // Invisible by day, out of the local horizon band, and clear of the moon disc.
    // In space the star floor mixes the horizon gate toward 1, so stars fill the sphere.
    float horizon = saturate(dot(ray, frame.sky_up.xyz) * STAR_HORIZON_FADE);
    horizon = lerp(horizon, 1.0, frame.exposure_dither.y);
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
    float4 far = mul(pc.inv_view_proj, float4(input.ndc, 0.0, 1.0));
    float3 ray = normalize(far.xyz);

    // Fog IS the sky: the same analytic radiance the terrain fades toward, plus
    // the additive HDR sun disc. glow_pow / sun dir / disc cosines are hoisted.
    float3 inscatter = sky_radiance(ray, frame);
    float3 col = inscatter
               + sun_disc_cos(ray, pc.sun.xyz, pc.sun_tint.rgb, pc.sun.w, pc.sun_tint.w);

    // Moon opposite sun, stars and moon gated by night (1 - light.w).
    float night = 1.0 - frame.light.w;
    float3 moon_dir = -pc.sun.xyz;
    float3 moon_tint = float3(MOON_TINT[0], MOON_TINT[1], MOON_TINT[2]);
    col += sun_disc_cos(ray, moon_dir, moon_tint, pc.moon.x, pc.moon.y) * night;
    // Uniform branch (both factors are per-frame constants): skips the whole
    // hash-grid star evaluation by day and when `RenderFlags::stars` is off —
    // the multiplied-out zero was still being computed for every sky pixel.
    // Above the atmosphere the star floor (exposure_dither.y) keeps the field
    // up by day.
    float star_night = max(night, frame.exposure_dither.y);
    if (star_night * frame.extras.x > 0.0)
        col += stars(ray, moon_dir, star_night) * frame.extras.x;

    // Far bodies occlude stars, the sun disc and the moon behind them, and are
    // seen through the viewer's own air (`inscatter`, ~0 in space).
    // fwidth stays in this entry point, ahead of the body-count early-out and
    // the per-tile mask branch, so a uniform count of zero cannot flatten the
    // derivative. It does not write `col`: a mask of 0 returns `col` as it
    // stands. SKY_BASE leaves both out.
#ifndef SKY_BASE
    float px = max(length(fwidth(ray)), 1e-6);
    col = far_bodies(ray, col, px, input.frag_pos.xy);
#endif

    // Slab clouds composite LAST so opaque cloud occludes the sun/moon/stars it
    // covers. The octahedral map is the local upper hemisphere; skip the tap
    // below the local cloud horizon so downward rays don't alias zenith texels.
    float cover = 0.0;
    float3 cloud_lit = float3(0.0);
    if (sky_local(ray, frame).y > CLOUD_HORIZON_OFFSET)
    {
        float4 t = cloudLut.SampleLevel(sky_oct_encode(ray, frame), 0);
        cloud_lit = t.rgb;
        cover = t.a;
    }
    col = col * (1.0 - cover) + cloud_lit;

    // VOXEL_SKY_DEBUG. sky_tangent.w is 1 for the frame or 0. One uniform
    // branch: when it is 0 the colour above is what the pixel returns, and
    // far_sky_debug_rgb is not called. A non-zero lane replaces the colour
    // with the source key. R/G/B mark a mapped surface, a mapped air limb,
    // and a horizon skip; base sky (no far bodies) stays 0.25 grey. The steep
    // part of sky_radiance's elevation remap, -s <= y_local <= 0, adds 0.5.
    [branch]
    if (frame.sky_tangent.w != 0.0)
    {
        float3 dbg = float3(0.25, 0.25, 0.25);
#ifndef SKY_BASE
#ifndef FAR_NO_MAPPED
        dbg = far_sky_debug_rgb(ray, px, input.frag_pos.xy);
#endif
#endif
        float yLocal = sky_local(ray, frame).y;
        float sDip = frame.sky_bitangent.w;
        if (yLocal >= -sDip && yLocal <= 0.0)
            dbg += 0.5;
        return float4(dbg, 1.0);
    }
    return float4(col, 1.0);
}
