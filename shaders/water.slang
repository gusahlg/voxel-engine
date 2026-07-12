// water.slang — animated water surface shading for the transparent pass.
// Included by mesh3d.frag AFTER common.slang (uses FrameUniforms, sky_radiance,
// sun_disc, ANIM_PERIOD and the generated WATER_* constants). Selected per-vertex
// by the water material bit (mesh.rs SHIFT_WATER); glass/ice keep the plain
// transparent path. All math is camera-relative (camera at the origin); the wave
// phase is world-anchored and jitter-free via the `anim` lane.
//
// First pass: waves + fresnel + sky reflection + sun glint + an interim
// view-angle absorption tint. True depth-difference absorption is DEFERRED (it
// needs a sampled depth attachment, which requires splitting the single scene
// render scope); this file has no depth dependency.

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

// Value noise: smoothstep-interpolated hash lattice.
float water_vnoise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = water_hash(i + float2(0.0, 0.0));
    float b = water_hash(i + float2(1.0, 0.0));
    float c = water_hash(i + float2(0.0, 1.0));
    float d = water_hash(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

// Two-octave animated wave height at a world-anchored ground-plane coordinate.
// The two octaves scroll in different directions so the field never reads as a
// single translating pattern.
float water_height(float2 p, float t)
{
    float h = water_vnoise(p + float2(t, 0.0));
    h += 0.5 * water_vnoise(p * 2.03 - float2(0.0, t * 1.3));
    return h;
}

// Perturbed surface normal from the wave-height gradient (finite differences),
// tilting the face normal within the ground plane.
float3 water_normal(float3 base_n, float2 p, float t)
{
    const float e = 0.15;
    float h = water_height(p, t);
    float hx = water_height(p + float2(e, 0.0), t);
    float hz = water_height(p + float2(0.0, e), t);
    float2 grad = float2(hx - h, hz - h) / e;
    return normalize(base_n - float3(grad.x, 0.0, grad.y) * WATER_WAVE_AMP);
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
//
// `reflectivity` scales the specular terms (Fresnel sky reflection + sun glint)
// down to zero: 1 for full-res chunk water, 0 for coarse LOD-tile water. LOD water
// is drawn with a per-pixel dither discard at the LOD→chunk transition
// (mesh3d.frag); a mirror-bright reflection toggling with that discard reads as
// flicker at ground level. Damping the specular terms on LOD water leaves the
// discard swinging only between near-identical body colours — invisible, exactly
// as for diffuse terrain — while near water keeps its full reflection.
float4 water_shade(float3 world, float3 base_n, float3 lit_albedo, float alpha, float absorb, float reflectivity, FrameUniforms frame)
{
    float3 v = normalize(world); // camera → fragment (points into the surface)

    // World-anchored, jitter-free sample coord: fract(camera/T)*T + camera-relative
    // world, so the pattern is stable far from the origin.
    float2 p = (world.xz + frame.anim.yz * ANIM_PERIOD) * WATER_WAVE_FREQ;
    float t = frame.anim.x * WATER_WAVE_SPEED;
    float3 N = water_normal(normalize(base_n), p, t);

    // Reflection about the perturbed normal. For a +Y water surface and a
    // downward view ray this is the upper hemisphere where sky_radiance is the
    // valid reflection primitive.
    float3 r = reflect(v, N);
    float3 reflection = sky_radiance(r, frame);
    float3 glint = sun_disc(r, frame.sun_dir_elev.xyz, frame.light.rgb, WATER_GLINT_RADIUS) * WATER_GLINT * reflectivity;

    // Fresnel: reflection dominates at grazing angles. `1 + dot(N,v)` is
    // `1 - cos(incidence)` since v points into the surface. Scaled by
    // `reflectivity` so LOD water collapses to its diffuse body (see header).
    float grazing = clamp(1.0 + dot(N, v), 0.0, 1.0);
    float fresnel = (WATER_FRESNEL_F0 + (1.0 - WATER_FRESNEL_F0) * grazing * grazing) * reflectivity;

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
