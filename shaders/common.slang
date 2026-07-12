// common.slang — shared shader math. Single source of truth for sRGB→linear,
// dither, and atmosphere/lighting evaluators. Generated constants come from
// build.rs (voxel_engine::genconst); generated file must exist before shader compile.

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

// 4x4 ordered Bayer dither. Returns signed threshold in [-0.5, +0.5).
// Caller scales by gain before quantization. Phase rotates pattern frame-to-frame.
float dither(float phase, float2 frag_coord)
{
    // Standard 4x4 Bayer matrix, values 0..15, row-major.
    const float bayer[16] = {
         0.0,  8.0,  2.0, 10.0,
        12.0,  4.0, 14.0,  6.0,
         3.0, 11.0,  1.0,  9.0,
        15.0,  7.0, 13.0,  5.0,
    };
    int2 p = int2(frag_coord) & 3;
    float b = bayer[p.y * 4 + p.x];
    // Normalize the cell to [0,1), add the temporal phase and wrap, then
    // recentre to [-0.5, +0.5) so the mean offset is ~0 (no DC brightening).
    return frac((b + 0.5) / 16.0 + phase) - 0.5;
}

// Per-frame uniform block. Generated from the lane_table in build.rs (the twin of
// voxel_engine::skeleton::FrameUniformsGpu) so a lane can never drift CPU↔GPU.
#include "generated/frame_uniforms.slang"

// Phase A/B evaluators (stubs declared here, bodies land later).

// Rec.709 relative luminance of a linear colour (the one weighting used for
// every luma below — no second copy).
float luma709(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// Blocklight (torch/candle) response curve. `x` is a
// per-channel block-light level in [0,1] (`candle.rgb · level`); the curve lifts
// it into an HDR glow near full brightness. Constants match CPU values exactly,
// so any difference indicates a build or sync error.
float3 candle_curve(float3 x)
{
    float3 hi = pow(CANDLE_HIGH_MUL * x, float3(CANDLE_HIGH_POW));
    return min(x * sqrt(x) + hi, float3(CANDLE_CLAMP));
}

// Omni ambient floor colour: the zenith tint re-scaled to `candle.w` luma
// (`ambient_floor`, composed by the game's `FrameSnapshot`), so shadowed faces
// and caves never read pure black yet keep the sky's hue.
float3 ambient_floor_color(FrameUniforms frame)
{
    float floor = frame.candle.w;
    float zl = luma709(frame.zenith.rgb);
    return zl > 1e-6 ? frame.zenith.rgb * (floor / zl) : float3(floor);
}

// Analytic sky radiance along `ray` — THE sky formula, single source of truth
// for the sky background, the distance fog (fog == sky evaluated at the
// fragment), and water reflection: an elevation gradient between the horizon
// and zenith anchors, plus a turbidity-scaled sun halo. `ray.y` below the
// horizon clamps to the horizon colour. (The CPU `Atmosphere::radiance_rgb`
// drifted from this formula and is scheduled for deletion; do not treat it
// as a mirror.)
float3 sky_radiance(float3 ray, FrameUniforms frame)
{
    float up = clamp(ray.y, 0.0, 1.0);
    float3 base = lerp(frame.horizon.rgb, frame.zenith.rgb, sqrt(up));
    float3 sun = normalize(frame.sun_dir_elev.xyz);
    float c = max(dot(ray, sun), 0.0);
    // Halo widens as the sun drops: low exponent at/below the horizon (wide golden-hour
    // halo), tightening to the noon look above GLOW_EDGE1. sun_dir_elev.w = elevation [-1,1].
    float glow_pow = lerp(GLOW_POW_SUNSET, GLOW_POW_DAY,
                          smoothstep(GLOW_EDGE0, GLOW_EDGE1, frame.sun_dir_elev.w));
    float3 glow = frame.light.rgb * (pow(c, glow_pow) * (0.5 + frame.zenith.w)); // zenith.w = turbidity
    return base + glow;
}

// Analytic sun disc — additive linear radiance in a small cone around
// `sun_dir`. `tint` is the disc's linear colour; `radius` its angular radius
// (rad). Full tint out to `radius·SUN_DISC_CORE`, smoothstepped to zero at
// `radius·SUN_DISC_RIM`. Additive, so it survives as an HDR highlight; the water glint
// reuses it later.
float3 sun_disc(float3 ray, float3 sun_dir, float3 tint, float radius)
{
    float d = dot(ray, normalize(sun_dir));
    float cos_core = cos(radius * SUN_DISC_CORE);
    float cos_rim = cos(radius * SUN_DISC_RIM);
    return tint * smoothstep(cos_rim, cos_core, d);
}

// Unified lighting term. `sky` = linear skylight colour (already shadow-
// attenuated AND omni-shaped by the caller — the directional N·L wrap folds into
// this direct term), `block` = linear blocklight colour
// pre-curve, `ao` the ambient-occlusion factor [0,1]. Folds in the candle curve
// and the omni ambient floor from `frame`. The ambient floor + AO stay global
// (omni-directional); only `sky` carries the directional shaping.
float3 light_eval(float3 sky, float3 block, float ao, FrameUniforms frame)
{
    float3 diffuse = max(ambient_floor_color(frame) + sky, candle_curve(block));
    return diffuse * ao;
}

// Far-field shadow fallback beyond shadow-map cascades. Uses baked sky-visibility
// term; scaled by |day_night·2−1| so strength varies with sun position.
float shadow_fallback(FrameUniforms frame, float visible_sky)
{
    return pow(visible_sky, SHADOW_FALLBACK_POW) * abs(frame.light.w * 2.0 - 1.0);
}

#endif // VOXEL_COMMON_SLANG
