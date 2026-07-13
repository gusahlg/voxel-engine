// sky.frag.slang — procedural sky for background pixels.
// Sky and fog read the same FrameUniforms (binding 2) so they can't diverge.
// Owns view-ray reconstruction, sun disc, and dither only.

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 ndc;
    // Framebuffer pixel position (system-generated) — the Bayer dither() for the
    // cloud march keys off it; no vertex-output change needed.
    float4 frag_pos : SV_Position;
};

struct PushConstants
{
    float4x4 inv_view_proj; // camera rebased to origin: (ndc,0,1) → view ray
    float4 sun;             // xyz sun dir, w = angular radius (rad)
    float4 sun_tint;        // rgb linear disc tint
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Set 0, binding 2: the shared per-frame uniforms — the sky's colours (linear
// zenith/horizon/light/turbidity) come from here, identical to the terrain fog.
[[vk::binding(2, 0)]]
ConstantBuffer<FrameUniforms> frame;

// Cheap hash in range 0–1; used only for sub-LSB dither, quality is irrelevant.
float hash12(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

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

// Smoothed 2D value noise from the same hash — a single octave in [0,1].
float cloud_value(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hash12(i);
    float b = hash12(i + float2(1.0, 0.0));
    float c = hash12(i + float2(0.0, 1.0));
    float d = hash12(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

// Cloud coverage: 2 noise octaves (rotated), drifted by time, thresholded.
float cloud_density(float2 world_xz, float time)
{
    float2 wind = float2(CLOUD_WIND[0], CLOUD_WIND[1]) * time;
    float2 base = world_xz * CLOUD_NOISE_SCALE;
    float n1 = cloud_value(base + wind);
    float n2 = cloud_value(base.yx * CLOUD_OCTAVE2_SCALE - wind * 0.5);
    // Weighted FBM blend: avoid 0.5/0.5 average (washes out texture detail).
    float n = smoothstep(0.05, 0.95, n1 * 0.65 + n2 * 0.35);
    return n > CLOUD_COVERAGE ? (n - CLOUD_COVERAGE) / (1.0 - CLOUD_COVERAGE) : 0.0;
}

// 2-plane slab clouds, marched camera-relative. Returns premultiplied radiance;
// `cover` is background occlusion (opacity × horizon fade) for one-step composite.
float3 clouds(float3 ray, FrameUniforms frame, float2 frag_coord, out float cover)
{
    cover = 0.0;
    // hide == 0 for ray.y <= CLOUD_HORIZON_OFFSET (> 0), so this also guards the
    // ray.y division below against a near-horizon blow-up.
    float hide = saturate((ray.y - CLOUD_HORIZON_OFFSET) * CLOUD_HORIZON_FADE);
    if (hide <= 0.0)
        return float3(0.0);

    // Slab planes in camera-relative space: world altitude minus camera world-y.
    float cam_y = frame.anim.w;
    float t0 = (CLOUD_BOTTOM - cam_y) / ray.y;
    float t1 = (CLOUD_BOTTOM + CLOUD_THICKNESS - cam_y) / ray.y;
    float t_enter = max(min(t0, t1), 0.0);
    float t_exit = max(t0, t1);
    if (t_exit <= t_enter)
        return float3(0.0); // camera at/above the slab — nothing ahead

    float time = frame.anim.x;
    float2 cam_xz = frame.anim.yz * ANIM_PERIOD; // wrapped camera world xz (metres)
    float3 sun = normalize(frame.sun_dir_elev.xyz);
    float bright = pow(max(dot(ray, sun), 0.0), 3.0);
    float3 lit_col = frame.light.rgb * CLOUD_LIGHT + frame.zenith.rgb * CLOUD_AMBIENT;

    int steps = int(CLOUD_STEPS);
    float dith = dither(frame.exposure_dither.y, frag_coord);
    float dt = (t_exit - t_enter) / float(steps);

    float trans = 1.0;
    float3 lit = float3(0.0);
    for (int i = 0; i < steps; i++)
    {
        float t = t_enter + (float(i) + 0.5 + dith) * dt;
        float density = cloud_density(cam_xz + (ray * t).xz, time);
        if (density <= 0.0)
            continue;
        float3 base = lit_col * lerp(1.0, CLOUD_DARK, sqrt(density));
        float silver = (1.0 - pow(density, 0.25)) * bright * bright;
        float3 step_col = lerp(base, base * CLOUD_SILVER, saturate(silver));
        float a = saturate(density * CLOUD_OPACITY);
        lit += trans * a * step_col;
        trans *= (1.0 - a);
        if (trans < 0.01)
            break;
    }
    cover = (1.0 - trans) * hide;
    return lit * hide;
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
    // the additive HDR sun disc.
    float3 col = sky_radiance(ray, frame)
               + sun_disc(ray, pc.sun.xyz, pc.sun_tint.rgb, pc.sun.w);

    // Moon opposite sun, stars and moon gated by night (1 - light.w).
    float night = 1.0 - frame.light.w;
    float3 moon_dir = -pc.sun.xyz;
    float3 moon_tint = float3(MOON_TINT[0], MOON_TINT[1], MOON_TINT[2]);
    col += sun_disc(ray, moon_dir, moon_tint, pc.sun.w * MOON_RADIUS_SCALE) * night;
    col += stars(ray, moon_dir, night);

    // Slab clouds composite LAST so opaque cloud occludes the sun/moon/stars it
    // covers. `cover` premultiplies the hidden background out in one lerp.
    float cover;
    float3 cloud_lit = clouds(ray, frame, input.frag_pos.xy, cover);
    col = col * (1.0 - cover) + cloud_lit;

    col += (hash12(input.ndc * 937.0) - 0.5) / 255.0;
    return float4(col, 1.0);
}
