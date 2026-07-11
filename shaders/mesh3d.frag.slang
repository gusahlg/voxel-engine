// mesh3d.frag.slang — textured voxel shading. Lights from per-frame UBO with
// cascaded shadow attenuation. Fog == sky. Opaque and transparent pipelines.

#include "common.slang"

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
    float4 lightparams; // x shade, y ao, z sky_amount(=visible_sky), w block_amount
};

// Set 0, binding 1: block texture array (shares the 3D push-descriptor set with
// the vertex stage's per-draw offsets SSBO at binding 0).
[[vk::binding(1, 0)]]
Sampler2DArray blockTex;

// Set 0, binding 2: per-frame uniforms (FrameUniforms, common.slang) — sky
// anchors, sun, candle/ambient, dither phase. `reserved` doubles as the
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

// Rotated 4-tap PCF against cascade `c`. Returns the lit fraction [0,1]
// (reversed-Z: SampleCmp passes where the fragment is at least as close to the
// light as the stored occluder). The rotation angle rides the ordered dither so
// the 4-tap kernel does not band; the dither is UN-shifted (jitter is zero until
// Phase E). Fragments outside the cascade footprint return 1.0 — the caller's
// fade into `shadow_fallback` covers that edge.
float shadow_map_term(float3 world, float ndotl, uint c, float2 frag_coord)
{
    float4 clip = mul(cascade.view_proj[c], float4(world, 1.0));
    float3 ndc = clip.xyz / clip.w;
    if (any(abs(ndc.xy) > 1.0))
        return 1.0;
    float2 uv = ndc.xy * float2(0.5, -0.5) + 0.5;

    // Slope bias: more on grazing faces/far cascades (peter-panning preferred over acne).
    float slope = cascade.bias.y * clamp(1.0 - ndotl, 0.0, 1.0) + cascade.bias.z;
    float ref = ndc.z + slope;

    uint w, h, elems;
    shadowMap.GetDimensions(w, h, elems);
    float2 texel = 1.0 / float2(w, h);

    float ang = (dither(frame.exposure_dither.y, frag_coord) + 0.5) * 6.2831853;
    float2 r = float2(cos(ang), sin(ang)) * cascade.bias.x; // blur radius (texels)
    float2 t = float2(-r.y, r.x);
    float2 taps[4] = { r, -r, t, -t };

    float lit = 0.0;
    for (int i = 0; i < 4; i++)
        lit += shadowMap.SampleCmpLevelZero(float3(uv + taps[i] * texel, float(c)), ref);
    return lit * 0.25;
}

// Sun shadow factor [0,1]: the near/far cascade PCF blended into the far-field
// `shadow_fallback` across the `fade_band` at `SHADOW_LIMIT`, so the
// chunk→tile handoff shows no shading step.
float sun_shadow(float3 world, float dist, float ndotl, float visible_sky, float2 frag_coord)
{
    uint c = dist < cascade.splits_fade.x ? 0u : 1u;
    float map = shadow_map_term(world, ndotl, c, frag_coord);
    float fallback = shadow_fallback(frame, visible_sky);
    float limit = cascade.splits_fade.w;
    float band = cascade.splits_fade.z;
    return lerp(map, fallback, smoothstep(limit - band, limit, dist));
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    // Debug-flat (TerrainKey): output the flat key colour for every covered
    // fragment; depth still writes so the sky-hole detector sees real coverage.
    if (frame.reserved.w > 0.5) {
        return float4(srgb_to_linear(frame.reserved.rgb), 1.0);
    }

    // Texture sampled through an sRGB view, so tex.rgb is already linear.
    float4 tex = blockTex.Sample(float3(input.uv, float(input.layer)));

    float shade = input.lightparams.x;
    float ao = input.lightparams.y;
    float sky_amount = input.lightparams.z;
    float block_amount = input.lightparams.w;

    float dist = length(input.world);
    float3 N = normalize(input.nrm);
    float ndotl = max(dot(N, normalize(frame.sun_dir_elev.xyz)), 0.0);
    float shadow = sun_shadow(input.world, dist, ndotl, sky_amount, input.position.xy);

    // Skylight (sun-tinted, shadow-attenuated) and pre-curve blocklight, folded
    // by the shared evaluator with the candle curve + omni ambient floor.
    float3 sky = frame.light.rgb * (sky_amount * shadow);
    float3 block = frame.candle.rgb * block_amount;
    float3 rgb = tex.rgb * light_eval(sky, block, ao, shade, frame);

    // Fog IS the sky: fade toward the analytic sky radiance along the view ray.
    float3 fog_color = sky_radiance(normalize(input.world), frame);
    float fog = 1.0 - exp(-dist * frame.horizon.w); // horizon.w = fog density
    rgb = lerp(rgb, fog_color, fog);
    return float4(rgb, tex.a);
}
