// sky.frag.slang — procedural sky for background pixels.
// Sky and fog read the same FrameUniforms (binding 2) so they can't diverge.
// Owns view-ray reconstruction, sun disc, and dither only.

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float2 ndc;
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

    col += (hash12(input.ndc * 937.0) - 0.5) / 255.0;
    return float4(col, 1.0);
}
