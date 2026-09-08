// Shared sky helpers: hash, slab-cloud march, octahedral hemisphere encode.
// Included by sky.frag and sky_cloud.comp; not by fog/water (those only need
// sky_radiance from common.slang).

#ifndef VOXEL_SKY_COMMON_SLANG
#define VOXEL_SKY_COMMON_SLANG

#include "common.slang"

// Fast hash for procedural stars and cloud value-noise; quality doesn't matter.
float hash12(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

// Upper-hemisphere octahedral: injective for y >= 0. y < 0 aliases xz of -n
// and must not be sampled (sky.frag skips the LUT tap below CLOUD_HORIZON_OFFSET).
float2 sky_oct_encode(float3 d)
{
    float3 n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    return n.xz * 0.5 + 0.5;
}

float3 sky_oct_decode(float2 uv)
{
    float2 f = uv * 2.0 - 1.0;
    return normalize(float3(f.x, 1.0 - abs(f.x) - abs(f.y), f.y));
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
// Direction-only for a given frame: camera xz/y and time come from `frame.anim`.
float3 clouds(float3 ray, FrameUniforms frame, out float cover)
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
    float3 sun = frame.sun_dir_elev.xyz; // unit, prepared CPU-side
    float bright = pow(max(dot(ray, sun), 0.0), 3.0);
    float3 lit_col = frame.light.rgb * CLOUD_LIGHT + frame.zenith.rgb * CLOUD_AMBIENT;

    int steps = int(CLOUD_STEPS);
    float dt = (t_exit - t_enter) / float(steps);

    float trans = 1.0;
    float3 lit = float3(0.0);
    for (int i = 0; i < steps; i++)
    {
        float t = t_enter + (float(i) + 0.5) * dt;
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

#endif // VOXEL_SKY_COMMON_SLANG
