// Shared shader math: sRGB↔linear and atmosphere/lighting.
// Generated constants from voxel_engine::genconst.

#ifndef VOXEL_COMMON_SLANG
#define VOXEL_COMMON_SLANG

// Inside the guard so a stray double-include of common.slang cannot re-emit
// the generated `static const` tables (which have no guard of their own).
#include "generated/shader_constants.slang"

// sRGB to linear. CPU colours arrive sRGB-encoded; tonemap owns OETF back to display.
float3 srgb_to_linear(float3 c)
{
    float3 lo = c / 12.92;
    float3 hi = pow(max((c + 0.055) / 1.055, 0.0), float3(2.4, 2.4, 2.4));
    return select(c <= 0.04045, lo, hi);
}


// Per-frame uniform block. Generated from the lane_table in build.rs (the twin of
// voxel_engine::skeleton::FrameUniformsGpu) so a lane can never drift CPU↔GPU.
#include "generated/frame_uniforms.slang"

// Rec.709 relative luminance of a linear colour (the one weighting used for
// every luma below — no second copy).
float luma709(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// Cube-face unit normals indexed by the packed-vertex `normal` field
// (mesh.rs: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z). Shared so the fragment looks
// the face up instead of interpolating a float3 varying.
static const float3 FACE_NORMAL[6] = {
    float3(1.0, 0.0, 0.0),  float3(-1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),  float3(0.0, -1.0, 0.0),
    float3(0.0, 0.0, 1.0),  float3(0.0, 0.0, -1.0),
};

// Blocklight (torch/candle) response curve. `x` is a
// per-channel block-light level in [0,1] (`candle.rgb · level`); the curve lifts
// it into an HDR glow near full brightness. Constants match CPU values exactly,
// so any difference indicates a build or sync error.
float3 candle_curve(float3 x)
{
    float3 hi = pow(CANDLE_HIGH_MUL * x, float3(CANDLE_HIGH_POW));
    return min(x * sqrt(x) + hi, float3(CANDLE_CLAMP));
}

// Analytic sky radiance along `ray` — THE sky formula, single source of truth
// for the sky background, the distance fog (fog == sky evaluated at the
// fragment), and water reflection: an elevation gradient between the horizon
// and zenith anchors, plus a turbidity-scaled sun halo. `ray.y` below the
// horizon clamps to the horizon colour. Halo exponent and tint×scale are
// hoisted to `ambient_glow.w` / `glow_day.rgb` (CPU, once per frame).
float3 sky_radiance(float3 ray, FrameUniforms frame)
{
    float up = clamp(ray.y, 0.0, 1.0);
    float3 base = lerp(frame.horizon.rgb, frame.zenith.rgb, sqrt(up));
    float c = max(dot(ray, frame.sun_dir_elev.xyz), 0.0);
    float3 glow = frame.glow_day.rgb * pow(c, frame.ambient_glow.w);
    return base + glow;
}

// Analytic sun disc with precomputed cone cosines (sky push constants).
// `sun_dir` is unit. Full tint out to `cos_core`, smoothstepped to zero at `cos_rim`.
float3 sun_disc_cos(float3 ray, float3 sun_dir, float3 tint, float cos_core, float cos_rim)
{
    return tint * smoothstep(cos_rim, cos_core, dot(ray, sun_dir));
}

// Analytic sun disc — additive linear radiance in a small cone around
// `sun_dir`. `tint` is the disc's linear colour; `radius` its angular radius
// (rad). Full tint out to `radius·SUN_DISC_CORE`, smoothstepped to zero at
// `radius·SUN_DISC_RIM`. Additive, so it survives as an HDR highlight; the water glint
// reuses it later.
float3 sun_disc(float3 ray, float3 sun_dir, float3 tint, float radius)
{
    return sun_disc_cos(
        ray,
        normalize(sun_dir),
        tint,
        cos(radius * SUN_DISC_CORE),
        cos(radius * SUN_DISC_RIM));
}

// Unified lighting term. `sky` arrives occlusion-applied (direct sun already
// shaped by AO_DIRECT, dome fill already × full AO); `block` is linear
// blocklight colour pre-curve; `ao` the ambient-occlusion factor [0,1].
// Ambient and blocklight keep full AO; the max() is the sky-dome vs candle
// pick. Only caller is mesh3d.frag.
float3 light_eval(float3 sky, float3 block, float ao, FrameUniforms frame)
{
    return max(frame.ambient_glow.rgb * ao + sky, candle_curve(block) * ao);
}

// Far-field shadow fallback beyond shadow-map cascades. Uses baked sky-visibility
// term; day factor is `glow_day.w` = |2·day_night_mix − 1|, hoisted on the CPU.
float shadow_fallback(FrameUniforms frame, float visible_sky)
{
    return pow(visible_sky, SHADOW_FALLBACK_POW) * frame.glow_day.w;
}

#endif // VOXEL_COMMON_SLANG
