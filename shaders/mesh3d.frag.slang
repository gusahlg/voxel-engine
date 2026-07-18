// mesh3d.frag.slang — textured voxel shading. Lights from per-frame UBO with
// cascaded shadow attenuation. Fog == sky. Opaque and transparent pipelines.

#include "common.slang"
#include "water.slang"

struct FragmentInput
{
    float4 position : SV_Position; // .z for water depth reconstruction

    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    nointerpolation uint layer;

    [[vk::location(2)]]
    float3 world; // camera-relative (camera at origin)

    [[vk::location(3)]]
    nointerpolation float3 nrm;

    [[vk::location(4)]]
    float3 lightparams; // x ao, y sky_amount(=visible_sky), z block_amount

    [[vk::location(5)]]
    nointerpolation uint water; // liquid material bit → animated water shading

    [[vk::location(6)]]
    nointerpolation float scale; // LOD level cell size

    // Per-draw style: flat-color mode and RGBA.
    [[vk::location(7)]]
    nointerpolation uint mode;
    [[vk::location(8)]]
    nointerpolation uint flat_rgba;
};

// View-proj and LOD slab clip extents.
struct PushConstants
{
    float4x4 view_proj;
    float clip;
    float clip_v;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Set 0, binding 1: block texture array (shares the 3D push-descriptor set with
// the vertex stage's per-draw offsets SSBO at binding 0).
[[vk::binding(1, 0)]]
Sampler2DArray blockTex;

// Per-frame uniforms: lighting, sky, debug flags.
[[vk::binding(2, 0)]]
ConstantBuffer<FrameUniforms> frame;

// Set 0, binding 3: shadow sampling uniforms (CascadeUniformsGpu).
// Set by the shadow-plumbing pass; this shader consumes the frozen layout.
struct CascadeUniforms
{
    float4x4 view_proj[2]; // per-cascade light-space clip transform (CLEAN)
    float4 splits_fade;    // x,y split distances; z fade_band; w SHADOW_LIMIT
    float4 bias;           // x blur_texels, y slope_bias, z dist_bias, w texel_world(near)
};

[[vk::binding(3, 0)]]
ConstantBuffer<CascadeUniforms> cascade;

// Set 0, binding 4: the cascaded shadow map as ONE combined image + comparison
// sampler (reversed-Z GREATER_OR_EQUAL, matching the engine's depth policy). The
// producer (vk::shadow) exposes a single COMBINED_IMAGE_SAMPLER here, so the
// receiver samples it via `SampleCmp` with the sampler baked in (no separate
// binding-5 sampler).
[[vk::binding(4, 0)]]
Sampler2DArrayShadow shadowMap;

#ifdef WATER_DEPTH_ABSORPTION
// Scene depth input attachment for water depth absorption (absorption variant only).
// SPIR-V OpImageRead requires 4 components; depth in .x.
[[vk::input_attachment_index(0)]]
[[vk::binding(5, 0)]]
SubpassInput opaqueDepthInput;

// Near plane from genconst (single source with camera.rs).
static const float WATER_Z_NEAR = Z_NEAR;
#endif

// 3x3 PCF shadow sampling; returns lit fraction [0,1].
// Outside cascade footprint returns 1.0 (caller fades to fallback).
float shadow_map_term(float3 world, float3 N, float3 L, uint c)
{
    // Single source of truth: derive per-cascade scale directly from the light matrix.
    float4x4 vp = cascade.view_proj[c];
    float texel_world = 2.0 / (SHADOW_RESOLUTION * length(vp[0].xyz));
    float ndc_per_metre = length(vp[2].xyz);

    // Normal-offset bias (depth-range independent) to avoid acne without floating shadows.
    float ndl = clamp(dot(N, L), 0.0, 1.0);
    float tan_theta = sqrt(1.0 - ndl * ndl) / max(ndl, 0.1);
    float offset_texels = cascade.bias.y + cascade.bias.z * min(tan_theta, 4.0);
    float3 pos = world + N * (texel_world * offset_texels);

    float4 clip = mul(vp, float4(pos, 1.0));
    float3 ndc = clip.xyz / clip.w;
    if (any(abs(ndc.xy) > 1.0))
        return 1.0;
    // No Y-flip: viewport is y-down. A flip would mirror shadows as the camera turns.
    float2 uv = ndc.xy * 0.5 + 0.5;

    // Reversed-Z: positive slack biases toward lit (kills acne).
    float ref = ndc.z + texel_world * ndc_per_metre;

    // Map size is a compile-time truth (SHADOW_RESOLUTION, asserted against
    // vk/targets.rs), so no per-fragment GetDimensions query.
    float2 texel = float2(1.0 / SHADOW_RESOLUTION, 1.0 / SHADOW_RESOLUTION);

    // Fixed 3x3 PCF at the configured blur spread. The old rotated 4-tap rode
    // a per-pixel dither angle, which quantized every penumbra to quarters at
    // a random phase — reading as loud salt-and-pepper stipple on the ground
    // (no TAA runs by default to integrate it away). Nine fixed taps cost
    // five more SampleCmp and resolve to smooth ninths with zero noise.
    float spread = cascade.bias.x * 0.7071; // grid corner ~= old blur radius
    float2 taps[9] = {
        float2(0.0, 0.0),
        float2(spread, 0.0), float2(-spread, 0.0),
        float2(0.0, spread), float2(0.0, -spread),
        float2(spread, spread), float2(spread, -spread),
        float2(-spread, spread), float2(-spread, -spread),
    };

    float lit = 0.0;
    for (int i = 0; i < 9; i++)
        lit += shadowMap.SampleCmpLevelZero(float3(uv + taps[i] * texel, float(c)), ref);
    return lit * (1.0 / 9.0);
}

// Sun shadow factor [0,1]: the near/far cascade PCF blended into the far-field
// `shadow_fallback` across the `fade_band` at `SHADOW_LIMIT`, so the
// chunk→tile handoff shows no shading step.
float sun_shadow(float3 world, float3 N, float3 L, float dist, float visible_sky)
{
    float fallback = shadow_fallback(frame, visible_sky);
    float limit = cascade.splits_fade.w;
    float band = cascade.splits_fade.z;
    float fade = smoothstep(limit - band, limit, dist);
    // Fully past the fade the map term is multiplied out entirely — skip the
    // 4-tap PCF. Every fragment beyond SHADOW_LIMIT lands here (all far-LOD
    // terrain by construction), and the branch is spatially coherent.
    if (fade >= 1.0)
        return fallback;
    uint c = dist < cascade.splits_fade.x ? 0u : 1u;
    float map = shadow_map_term(world, N, L, c);
    return lerp(map, fallback, fade);
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    // Discard LOD tiles inside the full-res slab extent.
    if (input.scale > 1.0 && pc.clip > 0.0) {
        bool inside_h = length(input.world.xz) < pc.clip;
        bool inside_v = pc.clip_v > 0.0 && abs(input.world.y) < pc.clip_v;
        if (inside_h && inside_v)
            discard;
    }

    // Debug-flat (TerrainKey): output the flat key colour for every covered
    // fragment; depth still writes so the sky-hole detector sees real coverage.
    if (frame.extras.w > 0.5) {
        return float4(srgb_to_linear(frame.extras.rgb), 1.0);
    }

    // Base albedo: flat far material or texture-sampled.
    // block texture. Only the albedo changes — light_eval/fog stay identical.
    float3 base;
    float alpha;
    if ((input.mode & 1u) != 0u) { // FLAT_COLOR
        uint c = input.flat_rgba;
        float3 srgb = float3(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu) / 255.0;
        base = srgb_to_linear(srgb); // match the sRGB-view texture path
        alpha = 1.0;                 // coarse far material is opaque
    } else {
        // Texture sampled through an sRGB view, so tex.rgb is already linear.
        float4 tex = blockTex.Sample(float3(input.uv, float(input.layer)));
        base = tex.rgb;
        alpha = tex.a;
    }

    float ao = input.lightparams.x;
    float sky_amount = input.lightparams.y;
    float block_amount = input.lightparams.z;

    float dist = length(input.world);
    float3 N = normalize(input.nrm);
    float3 sun = normalize(frame.sun_dir_elev.xyz);
    float raw_ndl = dot(N, sun);           // signed, unclamped

    // Omnidirectional wrap: day/night cycle flips lit hemisphere to moon side.
    float ndl = lerp(-raw_ndl, raw_ndl, frame.light.w);
    float omni = ((ndl + 1.0) * 0.25) + 0.75;

    // Coarse LOD excluded from cascade (self-shadowing); use sky-based fallback.
    float shadow = input.scale > 1.0
        ? shadow_fallback(frame, sky_amount)
        : sun_shadow(input.world, N, sun, dist, sky_amount);

    // Skylight (sun-tinted, shadow-attenuated, omni-shaped) and pre-curve
    // blocklight, folded by the shared evaluator with the candle curve + omni
    // ambient floor. `omni` shapes the DIRECT term only; the floor stays global.
    // The max() is the sky-dome bounce: sun shadow removes the SUN, not the
    // sky, so the lit factor never drops below SHADOW_SKY_AMBIENT — shadowed
    // ground keeps readable, sky-proportional light instead of collapsing to
    // near-black. Fully lit surfaces are untouched (shadow*omni >= the floor),
    // and caves stay dark (the whole term still scales by sky_amount).
    float lit = max(shadow * omni, SHADOW_SKY_AMBIENT);
    float3 sky = frame.light.rgb * (sky_amount * lit);
    float3 block = frame.candle.rgb * block_amount;
    float3 rgb = base * light_eval(sky, block, ao, frame);

    // Water fragments (transparent pass) replace the flat blend with animated
    // waves + fresnel sky reflection + sun glint + interim absorption tint. The
    // material bit is set per-vertex only for liquid faces (mesh.rs SHIFT_WATER),
    // so opaque geometry never takes this branch. LOD-section water never sets
    // the bit (it meshes as an opaque solid — see world/section/mesh.rs), so only
    // full-res chunk water reaches here.
    if (input.water != 0u) {
#ifdef WATER_DEPTH_ABSORPTION
        // True depth-difference absorption: opaque forward distance behind the
        // water (input attachment) minus this fragment's own forward distance,
        // both reconstructed from reversed-Z depth.
        // Clamp cleared-depth holes away from 0 to avoid division by zero.
        float opaque_z = max(opaqueDepthInput.SubpassLoad().x, 1.0e-6);
        float opaque_d = WATER_Z_NEAR / opaque_z;
        float water_d = WATER_Z_NEAR / input.position.z;
        float absorb = water_absorption(opaque_d, water_d);
#else
        float absorb = WATER_BODY_MIX; // interim flat tint (no depth term)
#endif
        float4 w = water_shade(input.world, N, rgb, alpha, absorb, frame);
        rgb = w.rgb;
        alpha = w.a;
    }

    // Fog: fade toward sky radiance along the view ray.
    float3 fog_color = sky_radiance(normalize(input.world), frame);
    float fog = 1.0 - exp(-dist * frame.horizon.w); // horizon.w = fog density
    rgb = lerp(rgb, fog_color, fog);
    return float4(rgb, alpha);
}
