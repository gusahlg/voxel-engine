// sky.frag.slang — procedural sky for background pixels.
//
// The CPU supplies the two anchor colours (zenith, horizon) sampled from the
// single Rust `Atmosphere::radiance` source of truth; this shader only does the
// high-frequency work: elevation interpolation, a horizon glow toward the sun,
// an analytic sun disc, exposure, and a hash dither to kill 8-bit banding.

struct FragmentInput
{
    [[vk::location(0)]]
    float2 ndc;
};

struct PushConstants
{
    float4x4 inv_view_proj; // camera rebased to origin: (ndc,0,1) → view ray
    float4 sun;             // xyz sun dir, w = exposure
    float4 zenith;          // rgb, w = daylight (reserved)
    float4 horizon;         // rgb, w = cos_inner (sun-disc inner edge)
    float4 sun_tint;        // rgb, w = cos_outer (sun-disc outer edge)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Cheap hash in range 0–1; used only for sub-LSB dither, quality is irrelevant.
float hash12(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float3 srgbToLinear(float3 c)
{
    float3 lo = c / 12.92;
    float3 hi = pow(max((c + 0.055) / 1.055, 0.0), float3(2.4, 2.4, 2.4));
    return select(c <= 0.04045, lo, hi);
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    // Reconstruct the view ray from the (wide, rectilinear) source projection.
    // The tonemap resample compresses the sky's periphery in lockstep with the
    // terrain, so no per-pixel warp is needed here.
    float4 far = mul(pc.inv_view_proj, float4(input.ndc, 0.0, 1.0));
    float3 ray = normalize(far.xyz);

    // CPU colours arrive sRGB-encoded; the sky composites in linear light
    // (the tonemap pass owns the OETF back to display).
    float3 horizon = srgbToLinear(pc.horizon.rgb);
    float3 zenith = srgbToLinear(pc.zenith.rgb);
    float3 sunTint = srgbToLinear(pc.sun_tint.rgb);

    float t = smoothstep(0.0, 1.0, ray.y);
    float3 col = lerp(horizon, zenith, t);

    // Horizon glow banded toward the sun's azimuth, strongest at the horizon.
    // The ray's horizontal component is deliberately NOT normalized: its magnitude
    // makes the glow fade toward the zenith AND the nadir,
    // and avoids normalize(0) when looking straight up/down. Normalizing it
    // kept the glow at full strength below the horizon, where a near-vertical
    // ray's azimuth swings wildly across neighbouring pixels — the smear
    // flickered and spun with small mouse movements.
    float toward = max(dot(float3(ray.x, 0.0, ray.z),
                           normalize(float3(pc.sun.x, 0.0, pc.sun.z))), 0.0);
    col = lerp(col, sunTint, pow(toward, 4.0));

    // Analytic sun disc: full white inside cos_inner, fading out to cos_outer.
    float d = dot(ray, pc.sun.xyz);
    col = lerp(col, float3(1.0, 1.0, 1.0), smoothstep(pc.sun_tint.w, pc.horizon.w, d));

    col *= pc.sun.w; // exposure
    col += (hash12(input.ndc * 937.0) - 0.5) / 255.0;
    return float4(col, 1.0);
}
