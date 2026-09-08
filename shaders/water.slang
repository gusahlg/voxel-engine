// water.slang — animated water surface shading for the transparent pass.
// Included by mesh3d.frag AFTER common.slang (uses FrameUniforms, sky_radiance,
// sun_disc, ANIM_PERIOD and the generated WATER_* constants). Selected per-vertex
// by the water material bit (mesh.rs SHIFT_WATER); glass/ice keep the plain
// transparent path. All math is camera-relative (camera at the origin); the wave
// phase is world-anchored and jitter-free via the `anim` lane.
//
// Waves use an analytic value-noise gradient from the same four lattice hashes
// as the height (hash only, no clocks — goldens stay bit-comparable at a pinned
// day ⇒ pinned anim phase). First pass: waves + fresnel + sky reflection + sun
// glint + an interim view-angle absorption tint. True depth-difference absorption
// is DEFERRED (it needs a sampled depth attachment, which requires splitting the
// single scene render scope); this file has no depth dependency.

#ifndef VOXEL_WATER_SLANG
#define VOXEL_WATER_SLANG

// Deterministic 2D hash in [0,1). No texture, no wall-clock — goldens stay
// bit-comparable at a pinned day (⇒ pinned anim phase).
float water_hash(float2 p)
{
    p = frac(p * float2(127.1, 311.7));
    p += dot(p, p + 33.33);
    return frac((p.x + p.y) * p.x);
}

// Value noise and its analytic gradient. Same four hashes as the old height-only
// lattice: v = lerp(lerp(a,b,u.x), lerp(c,d,u.x), u.y) with f = frac(p),
// u = f*f*(3-2f), du = 6*f*(1-f). Returns (value, d/dx, d/dy).
float3 water_vnoise_grad(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float2 du = 6.0 * f * (1.0 - f);
    float a = water_hash(i + float2(0.0, 0.0));
    float b = water_hash(i + float2(1.0, 0.0));
    float c = water_hash(i + float2(0.0, 1.0));
    float d = water_hash(i + float2(1.0, 1.0));
    float v = lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
    float k = a - b - c + d;
    float dvdx = du.x * ((b - a) + u.y * k);
    float dvdy = du.y * ((c - a) + u.x * k);
    return float3(v, dvdx, dvdy);
}

// Two-octave animated wave height + gradient at a world-anchored ground-plane
// coordinate. The two octaves scroll in different directions so the field never
// reads as a single translating pattern. Chain rule: first octave at p+(t,0)
// (gradient scale 1); second at p*2.03-(0, t*1.3) (scale 2.03), weight 0.5.
float3 water_height_grad(float2 p, float t)
{
    float3 n1 = water_vnoise_grad(p + float2(t, 0.0));
    float3 n2 = water_vnoise_grad(p * 2.03 - float2(0.0, t * 1.3));
    return float3(
        n1.x + 0.5 * n2.x,
        n1.y + 0.5 * n2.y * 2.03,
        n1.z + 0.5 * n2.z * 2.03);
}

// Perturbed surface normal from the analytic wave-height gradient, tilting the
// face normal within the ground plane. `base_n` is already a unit face normal.
// The analytic gradient is slightly sharper than the old finite difference at
// the old 0.15-unit step size.
float3 water_normal(float3 base_n, float2 p, float t)
{
    float3 g = water_height_grad(p, t);
    return normalize(base_n - float3(g.y, 0.0, g.z) * WATER_WAVE_AMP);
}

// Depth-difference absorption factor: how far the water body colour is pulled
// toward the deep tint, given the opaque/water forward distances (metres)
// reconstructed from the reversed-Z depth buffer. Distance is clamped >=0
// (opaque always behind the water it shows through). Compiled only for the
// depth-input-attachment variant; the interim path passes WATER_BODY_MIX.
float water_absorption(float opaque_d, float water_d)
{
    float dd = max(opaque_d - water_d, 0.0);
    return 1.0 - 1.0 / (dd * dd * WATER_ABS + WATER_ABS_BASE);
}

// Shade a water fragment. `world` is the camera-relative fragment position (view
// ray = normalize(world)); `base_n` the face normal; `lit_albedo` the block's
// already-lit colour; `alpha` its texel alpha. `absorb` is how far the body is
// mixed toward the deep tint (WATER_BODY_MIX interim, or water_absorption() when
// the depth input attachment is available). Returns colour + alpha for the
// alpha-blended transparent pass (fog is applied by the caller, as for all mesh).
float4 water_shade(float3 world, float3 base_n, float3 lit_albedo, float alpha, float absorb, FrameUniforms frame)
{
    float3 v = normalize(world); // camera → fragment (points into the surface)

    // World-anchored, jitter-free sample coord: fract(camera/T)*T + camera-relative
    // world, so the pattern is stable far from the origin.
    float2 p = (world.xz + frame.anim.yz * ANIM_PERIOD) * WATER_WAVE_FREQ;
    float t = frame.anim.x * WATER_WAVE_SPEED;
    float3 N = water_normal(base_n, p, t);

    // Reflection about the perturbed normal. For a +Y water surface and a
    // downward view ray this is the upper hemisphere where sky_radiance is the
    // valid reflection primitive.
    float3 r = reflect(v, N);
    float3 reflection = sky_radiance(r, frame);
    float3 glint = sun_disc(r, frame.sun_dir_elev.xyz, frame.light.rgb, WATER_GLINT_RADIUS) * WATER_GLINT;

    // Fresnel: reflection dominates at grazing angles.
    float grazing = clamp(1.0 + dot(N, v), 0.0, 1.0);
    float fresnel = WATER_FRESNEL_F0 + (1.0 - WATER_FRESNEL_F0) * grazing * grazing;

    // The water body is the lit block colour pulled toward a deep tint by
    // `absorb`, so it reads as a tinted volume rather than a flat blend. `absorb`
    // is the depth-difference factor when available (shallow shore transparent,
    // deep water saturated), else the flat WATER_BODY_MIX interim.
    float3 deep = float3(WATER_DEEP[0], WATER_DEEP[1], WATER_DEEP[2]);
    float3 body = lerp(lit_albedo, deep, absorb);

    float3 color = lerp(body, reflection, fresnel) + glint;
    // Grazing reflections read as a solid sheet; face-on water keeps its tint's
    // translucency so you still see into shallow water.
    float out_alpha = lerp(alpha, 1.0, fresnel);
    return float4(color, out_alpha);
}

#endif // VOXEL_WATER_SLANG
