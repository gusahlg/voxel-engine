// mesh3d.frag.slang — textured voxel shading. Lights from per-frame UBO with
// cascaded shadow attenuation. Fog == sky. Opaque and transparent pipelines.
//
// Variants (build.rs `-D`):
//   (none)                       full module: water/absorb branch + LOD-slab
//                                `discard`; the Blend pipelines.
//   MESH3D_OPAQUE                full-res opaque: no water code and NO `discard`
//                                (no OpKill ⇒ early depth write stays enabled).
//   MESH3D_OPAQUE + MESH3D_LOD   coarse-LOD opaque (scale > 1): keeps the slab
//                                clip `discard`, never samples the cascades.

#include "common.slang"
#ifndef MESH3D_OPAQUE
#include "water.slang"
#endif

struct FragmentInput
{
    float4 position : SV_Position; // .z for water depth reconstruction

    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    float3 world; // camera-relative (camera at origin)

    // x = sky_amount, y = block_amount, z = ao_factor (vertex scalars)
    [[vk::location(2)]]
    float3 light_lv;

    // x: layer[0:14] water[14] FLAT_COLOR[15] face[16:19] detail[20:24]
    // y: flat_rgba
    [[vk::location(3)]]
    nointerpolation uint2 ids;
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
    // Per-cascade receiver constants, hoisted off the fragment (vk/shadow.rs
    // derives them from the same fit): x texel_world, y ndc_per_metre (light
    // depth per world metre), z PCF tap spread in map UV units, w reversed-Z
    // reference slack (= texel_world * ndc_per_metre).
    float4 texel[2];
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

// PCF tap count: 4 = Castaño's optimized 4-tap/16-texel [1 2 1]² tent, 9 = the
// fixed 3x3 grid at the configured blur spread. Both are fixed-pattern (no
// per-pixel dither: that quantized every penumbra to quarters at a random
// phase and read as salt-and-pepper stipple with TAA off).
#ifndef PCF_TAPS
#define PCF_TAPS 4
#endif

// Map term assumed for a face turned away from the light (N·L <= 0), where the
// taps are skipped. 1.0 = lit: with the receiver's normal-offset bias at its
// 4-texel clamp, the map already reads such faces as lit almost everywhere
// (the offset point clears the face's own block), so this matches the shipped
// look; the omni wrap keeps them at 75% direct either way.
#ifndef BACKFACE_LIT
#define BACKFACE_LIT 1.0
#endif

// One hardware-bilinear comparison tap of cascade `c`.
float pcf_tap(float2 uv, uint c, float ref)
{
    return shadowMap.SampleCmpLevelZero(float3(uv, float(c)), ref);
}

// PCF shadow sampling; returns lit fraction [0,1]. `ndl` is the unflipped
// clamp(N·sun) the normal-offset bias scales by.
// Outside cascade footprint returns 1.0 (caller fades to fallback).
float shadow_map_term(float3 world, float3 N, float ndl, uint c)
{
    float4 t = cascade.texel[c];

    // Normal-offset bias (depth-range independent) to avoid acne without floating shadows.
    float tan_theta = sqrt(1.0 - ndl * ndl) / max(ndl, 0.1);
    float offset_texels = cascade.bias.y + cascade.bias.z * min(tan_theta, 4.0);
    float3 pos = world + N * (t.x * offset_texels);

    // Orthographic light: clip.w == 1, so clip IS ndc (no divide).
    float3 ndc = mul(cascade.view_proj[c], float4(pos, 1.0)).xyz;
    if (any(abs(ndc.xy) > 1.0))
        return 1.0;
    // No Y-flip: viewport is y-down. A flip would mirror shadows as the camera turns.
    float2 uv = ndc.xy * 0.5 + 0.5;

    // Reversed-Z: positive slack biases toward lit (kills acne).
    float ref = ndc.z + t.w;

#if PCF_TAPS == 9
    // Fixed 3x3 grid at the blur spread (uv units, hoisted); force-unrolled.
    float d = t.z;
    float lit = pcf_tap(uv, c, ref)
              + pcf_tap(uv + float2(d, 0.0), c, ref)
              + pcf_tap(uv + float2(-d, 0.0), c, ref)
              + pcf_tap(uv + float2(0.0, d), c, ref)
              + pcf_tap(uv + float2(0.0, -d), c, ref)
              + pcf_tap(uv + float2(d, d), c, ref)
              + pcf_tap(uv + float2(d, -d), c, ref)
              + pcf_tap(uv + float2(-d, d), c, ref)
              + pcf_tap(uv + float2(-d, -d), c, ref);
    return lit * (1.0 / 9.0);
#else
    // Four bilinear taps placed and weighted so their 2x2 footprints resolve
    // to an exact [1 2 1]⊗[1 2 1] tent over 16 texels (the classic optimized
    // PCF): 4x4-texel footprint, smooth for any sub-texel phase, 4 SampleCmp.
    // Map size is a compile-time truth (SHADOW_RESOLUTION, asserted against
    // vk/targets.rs), so no per-fragment GetDimensions query. Tap math is
    // explicit (force-unrolled; the 4 offsets depend on the sub-texel phase).
    const float inv = 1.0 / SHADOW_RESOLUTION;
    float2 st = uv * SHADOW_RESOLUTION + 0.5;
    float2 base = floor(st);
    float2 f = st - base;
    base = (base - 0.5) * inv;
    float2 uw = float2(3.0 - 2.0 * f.x, 1.0 + 2.0 * f.x);
    float2 vw = float2(3.0 - 2.0 * f.y, 1.0 + 2.0 * f.y);
    float2 u = float2((2.0 - f.x) / uw.x - 1.0, f.x / uw.y + 1.0) * inv;
    float2 v = float2((2.0 - f.y) / vw.x - 1.0, f.y / vw.y + 1.0) * inv;
    float lit = uw.x * vw.x * pcf_tap(base + float2(u.x, v.x), c, ref)
              + uw.y * vw.x * pcf_tap(base + float2(u.y, v.x), c, ref)
              + uw.x * vw.y * pcf_tap(base + float2(u.x, v.y), c, ref)
              + uw.y * vw.y * pcf_tap(base + float2(u.y, v.y), c, ref);
    return lit * (1.0 / 16.0);
#endif
}

// Sun shadow factor [0,1]: the near/far cascade PCF blended into the far-field
// `shadow_fallback` across the `fade_band` at `SHADOW_LIMIT`, so the
// chunk→tile handoff shows no shading step. `raw_ndl` = N·sun (unflipped);
// `ndl` = the day/night-flipped N·L the direct term is shaped by.
float sun_shadow(float3 world, float3 N, float raw_ndl, float ndl, float dist, float visible_sky)
{
    float fallback = shadow_fallback(frame, visible_sky);
    float limit = cascade.splits_fade.w;
    float band = cascade.splits_fade.z;
    float fade = smoothstep(limit - band, limit, dist);
    // Fully past the fade the map term is multiplied out entirely — skip the
    // PCF. Every fragment beyond SHADOW_LIMIT lands here (all far-LOD
    // terrain by construction), and the branch is spatially coherent.
    if (fade >= 1.0)
        return fallback;
    // A face turned away from the light source (N·L <= 0 after the day/night
    // flip) cannot receive direct light: skip the taps and treat the map as
    // BACKFACE_LIT (see the golden notes in vk/shadow.rs).
    if (ndl <= 0.0)
        return lerp(BACKFACE_LIT, fallback, fade);
    // Near/far PCF cross-fade over CASCADE_BLEND_FRAC of the near split.
    // Branch order guarantees no divide-by-zero at FRAC=0 (the band is empty
    // and the compiler folds it away). Near radius is split+4 so both
    // cascades cover the band; 8 taps only inside the ring.
    float s0 = cascade.splits_fade.x;
    float b0 = s0 * (1.0 - CASCADE_BLEND_FRAC);
    float ndl_c = clamp(raw_ndl, 0.0, 1.0);
    float map;
    if (dist < b0)       map = shadow_map_term(world, N, ndl_c, 0u);
    else if (dist >= s0) map = shadow_map_term(world, N, ndl_c, 1u);
    else map = lerp(shadow_map_term(world, N, ndl_c, 0u),
                    shadow_map_term(world, N, ndl_c, 1u), (dist - b0) / (s0 - b0));
    return lerp(map, fallback, fade);
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
#if defined(MESH3D_LOD)
    // Coarse-LOD partition is scale > 1 by construction; skip the scale test.
    if (pc.clip > 0.0) {
        bool inside_h = length(input.world.xz) < pc.clip;
        bool inside_v = pc.clip_v > 0.0 && abs(input.world.y) < pc.clip_v;
        if (inside_h && inside_v)
            discard;
    }
#elif !defined(MESH3D_OPAQUE)
    // Blend module: mixed full-res / LOD draws share this path.
    float scale = exp2(float((input.ids.x >> 20) & 0xFu) - float(DETAIL_GPU_BIAS));
    if (scale > 1.0 && pc.clip > 0.0) {
        bool inside_h = length(input.world.xz) < pc.clip;
        bool inside_v = pc.clip_v > 0.0 && abs(input.world.y) < pc.clip_v;
        if (inside_h && inside_v)
            discard;
    }
#endif

    // Debug-flat (TerrainKey): output the flat key colour for every covered
    // fragment; depth still writes so the sky-hole detector sees real coverage.
    // A uniform branch, never a discard.
    if (frame.extras.w > 0.5) {
        return float4(srgb_to_linear(frame.extras.rgb), 1.0);
    }

    uint packed = input.ids.x;
    uint layer = packed & 0x3FFFu;
    uint face = (packed >> 16) & 7u;

    // Base albedo: flat far material or texture-sampled.
    // block texture. Only the albedo changes — light_eval/fog stay identical.
    float3 base;
    float alpha;
    if ((packed & (1u << 15)) != 0u) { // FLAT_COLOR
        uint c = input.ids.y;
        float3 srgb = float3(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu) / 255.0;
        base = srgb_to_linear(srgb); // match the sRGB-view texture path
        alpha = 1.0;                 // coarse far material is opaque
    } else {
        // Texture sampled through an sRGB view, so tex.rgb is already linear.
        float4 tex = blockTex.Sample(float3(input.uv, float(layer)));
        base = tex.rgb;
        alpha = tex.a;
    }

    float sky_amount = input.light_lv.x;
    float block_amount = input.light_lv.y;
    float ao = input.light_lv.z;
    float3 N = FACE_NORMAL[face];
    float dist = length(input.world);
    float raw_ndl = dot(N, frame.sun_dir_elev.xyz);

    // Omnidirectional wrap: day/night cycle flips lit hemisphere to moon side.
    // lerp(-r, r, mix) = r * (2*mix - 1).
    float ndl = raw_ndl * (frame.light.w * 2.0 - 1.0);
    float omni = ((ndl + 1.0) * 0.25) + 0.75;

    // Coarse LOD excluded from cascade (self-shadowing); use sky-based fallback.
    // With sky_amount == 0 (caves) the whole skylight term below is zero, so
    // the shadow value is moot: skip the cascade taps there too.
#if defined(MESH3D_LOD)
    float shadow = shadow_fallback(frame, sky_amount);
#elif defined(MESH3D_OPAQUE)
    float shadow = sky_amount > 0.0
        ? sun_shadow(input.world, N, raw_ndl, ndl, dist, sky_amount)
        : 0.0;
#else
    float shadow = scale > 1.0 || sky_amount <= 0.0
        ? shadow_fallback(frame, sky_amount)
        : sun_shadow(input.world, N, raw_ndl, ndl, dist, sky_amount);
#endif

    // Skylight (sun-tinted, shadow-attenuated, omni-shaped) and pre-curve
    // blocklight, folded by the shared evaluator with the candle curve + omni
    // ambient floor. `omni` shapes the DIRECT term only; the floor stays global.
    // The max() is the sky-dome bounce: sun shadow removes the SUN, not the
    // sky, so the lit factor never drops below SHADOW_SKY_AMBIENT — shadowed
    // ground keeps readable, sky-proportional light instead of collapsing to
    // near-black. Fully lit surfaces are untouched (shadow*omni >= the floor),
    // and caves stay dark (the whole term still scales by sky_amount).
    // The floor now rides `frame.shadow_bounce` (SHADOW_SKY_AMBIENT × sun
    // colour, optionally tinted toward luma-matched zenith). AO_DIRECT lets
    // baked AO darken the dome/ambient/blocklight fully while only partially
    // darkening direct sun (legacy 1.0 is bit-identical to the old `ao * max`).
    float direct = max(shadow * omni - SHADOW_SKY_AMBIENT, 0.0);
    float ao_direct = 1.0 - AO_DIRECT * (1.0 - ao);
    float3 sky = frame.light.rgb * (sky_amount * direct * ao_direct)
               + frame.shadow_bounce.rgb * (sky_amount * ao);
    float3 block = frame.candle.rgb * block_amount;
    float3 rgb = base * light_eval(sky, block, ao, frame);

    // Water fragments (transparent pass) replace the flat blend with animated
    // waves + fresnel sky reflection + sun glint + interim absorption tint. The
    // material bit is set per-vertex only for liquid faces (mesh.rs SHIFT_WATER),
    // so opaque geometry never takes this branch. LOD-section water never sets
    // the bit (it meshes as an opaque solid — see world/section/mesh.rs), so only
    // full-res chunk water reaches here.
#ifndef MESH3D_OPAQUE
    if (((packed >> 14) & 1u) != 0u) {
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
#endif

    // Fog: fade toward sky radiance along the view ray. Uniform-density gate:
    // horizon.w == 0 (fog off / RenderFlags::fog) skips the sky eval and lerp.
    if (frame.horizon.w > 0.0) {
        float3 fog_color = sky_radiance(normalize(input.world), frame);
        float fog = 1.0 - exp(-dist * frame.horizon.w); // horizon.w = fog density
        rgb = lerp(rgb, fog_color, fog);
    }
    return float4(rgb, alpha);
}
