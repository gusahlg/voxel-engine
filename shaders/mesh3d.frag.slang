// mesh3d.frag.slang — textured voxel shading. Lights from per-frame UBO with
// cascaded shadow attenuation. Fog == sky. Opaque and transparent pipelines.

#include "common.slang"
#include "water.slang"

struct FragmentInput
{
    float4 position : SV_Position; // screen coords, for the PCF rotation dither

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

    // Per-draw style: fade, mode, flat_rgba.
    [[vk::location(7)]]
    nointerpolation float fade;
    [[vk::location(8)]]
    nointerpolation uint mode;
    [[vk::location(9)]]
    nointerpolation uint flat_rgba;
};

// Push constants: view_proj + LOD dither extents.
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

// Set 0, binding 2: per-frame uniforms (FrameUniforms, common.slang) — sky
// anchors, sun, candle/ambient, dither phase. `extras` doubles as the
// DebugView::TerrainKey channel ([r, g, b, enabled], sRGB).
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
// Set 0, binding 5: the scene depth as an input attachment (index 0 via the
// pipeline's VkRenderingInputAttachmentIndexInfo). Read same-scope by the water
// branch to reconstruct the opaque forward distance behind the water surface —
// dynamic_rendering_local_read, MSAA-off only. Compiled into the absorption
// variant only; the interim-tint build omits this and the depth term.
// Four-component by SPIR-V rule: `OpImageRead` (what `SubpassLoad` lowers to)
// must yield a 4-component result even for a depth image — a scalar
// `SubpassInput<float>` emits invalid SPIR-V (VUID-StandaloneSpirv-Result-04780,
// the E-08 module defect). Depth lands in `.x`.
[[vk::input_attachment_index(0)]]
[[vk::binding(5, 0)]]
SubpassInput opaqueDepthInput;

// Reversed-Z near plane: used to reconstruct forward view-space distance from depth.
// A drift here just rescales WATER_ABS.
static const float WATER_Z_NEAR = 0.05;
#endif

// Rotated 4-tap PCF against cascade `c`. Returns the lit fraction [0,1]
// (reversed-Z: SampleCmp passes where the fragment is at least as close to the
// light as the stored occluder). The rotation angle rides the ordered dither so
// the 4-tap kernel does not band; the dither is UN-shifted (jitter is zero until
// Phase E). Fragments outside the cascade footprint return 1.0 — the caller's
// fade into `shadow_fallback` covers that edge.
float shadow_map_term(float3 world, float3 N, float3 L, uint c, float2 frag_coord)
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
float sun_shadow(float3 world, float3 N, float3 L, float dist, float visible_sky, float2 frag_coord)
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
    float map = shadow_map_term(world, N, L, c, frag_coord);
    return lerp(map, fallback, fade);
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    // Dither LOD→chunk transition at slab extents to avoid holes showing sky.
    if (input.scale > 1.0 && pc.clip > 0.0) {
        float band_h = smoothstep(0.75 * pc.clip, pc.clip, length(input.world.xz));
        // Zero vertical coverage is an explicit branch: with clip_v == 0 the
        // full-res slab has no vertical extent, so every fragment is fully
        // outside it (band 1, LOD keeps) — never an equal-edge smoothstep,
        // whose result is undefined.
        float band_v = pc.clip_v > 0.0
            ? smoothstep(0.75 * pc.clip_v, pc.clip_v, abs(input.world.y))
            : 1.0;
        if (max(band_h, band_v) < dither(0.0, input.position.xy) + 0.5)
            discard;
    }

    // Temporal cross-fade with complementary dithering; CPU guarantees exact tiling.
    if (input.fade < 1.0) {
        float d = dither(0.0, input.position.xy) + 0.5;
        bool keep = (input.mode & 2u) != 0u ? (d >= 1.0 - input.fade) : (d < input.fade);
        if (!keep)
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

    float shadow = sun_shadow(input.world, N, sun, dist, sky_amount, input.position.xy);

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
        float opaque_d = WATER_Z_NEAR / opaqueDepthInput.SubpassLoad().x;
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
