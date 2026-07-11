// taa_resolve.comp.slang — temporal anti-aliasing resolve.
// Reproject previous frame, neighbourhood-clamp, blend. Writes history and resolved output.
// All matrices are CLEAN (jitter only at rasterization). Reprojection is depth-aware
// so camera translation reprojects correctly at all distances.

// Descriptor set layout: current=0, history=1, reproj UBO=2, output=3.
// The resolved-history output rides binding 3 of the same set.
[[vk::binding(0, 0)]] Sampler2D currentTex;   // this frame's HDR offscreen
[[vk::binding(1, 0)]] Sampler2D historyTex;   // last frame's resolved output

struct Reproj
{
    float4x4 prev_view_proj; // maps CURRENT camera-relative world → PREVIOUS clip
    float4 camera_delta;     // xyz = prev_origin − cur_origin (informational)
};
[[vk::binding(2, 0)]] ConstantBuffer<Reproj> reproj;

[[vk::binding(3, 0)]] RWTexture2D<float4> outHistory; // resolved integrator (write)

// This frame's depth (reversed-Z, 0 = far/sky), single-sampled only.
// Under MSAA the host binds a dummy and clears `depth_valid` instead.
[[vk::binding(4, 0)]] Sampler2D depthTex;

struct Push
{
    float4x4 inv_cur_view_proj; // current inverse view-proj
    uint2 dim;                  // output size in texels
    float blend;                // history feedback weight (kept fraction)
    uint history_valid;         // 0 on the first frame / after a resize
    uint depth_valid;           // 0 under MSAA: fall back to far-plane unproject
};
[[vk::push_constant]] Push pc;

// Texel-centre uv ⇄ NDC, matching camera::world_to_screen (GL-style y-up viewport:
// pixel-y-down ↔ ndc-y-up, so ndc.y = 1 − 2·uv.y).
float2 uv_to_ndc(float2 uv) { return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); }
float2 ndc_to_uv(float2 ndc) { return float2(ndc.x * 0.5 + 0.5, (1.0 - ndc.y) * 0.5); }

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dim.x || gid.y >= pc.dim.y)
        return;

    float2 texel = float2(pc.dim);
    float2 uv = (float2(gid.xy) + 0.5) / texel;
    float3 current = currentTex.SampleLevel(uv, 0).rgb;

    // Neighbourhood colour AABB (3×3 of the current frame) — the ghosting guard.
    float3 nmin = current;
    float3 nmax = current;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            float2 nuv = uv + float2(dx, dy) / texel;
            float3 c = currentTex.SampleLevel(nuv, 0).rgb;
            nmin = min(nmin, c);
            nmax = max(nmax, c);
        }
    }

    float3 resolved = current;
    if (pc.history_valid != 0)
    {
        // Unproject the current texel at the far plane (reversed-Z far depth = 0),
        // giving a far camera-relative world point (direction), then reproject it
        // into the previous frame's clip space.
        float2 ndc = uv_to_ndc(uv);
        // Texel-centre sample returns the exact stored depth (no filtering
        // ambiguity at (gid+0.5)/dim). depth 0 = the far-plane fallback.
        float d = pc.depth_valid != 0 ? depthTex.SampleLevel(uv, 0).x : 0.0;
        float4 worldH = mul(pc.inv_cur_view_proj, float4(ndc, d, 1.0));
        float3 world = worldH.xyz / worldH.w;
        float4 prevClip = mul(reproj.prev_view_proj, float4(world, 1.0));
        if (prevClip.w > 0.0)
        {
            float2 prevUv = ndc_to_uv(prevClip.xy / prevClip.w);
            // Only integrate history that reprojects back on-screen.
            if (all(prevUv >= 0.0) && all(prevUv <= 1.0))
            {
                float3 hist = historyTex.SampleLevel(prevUv, 0).rgb;
                // Clamp the reprojected history into the current neighbourhood so
                // disoccluded / mismatched samples cannot ghost.
                hist = clamp(hist, nmin, nmax);
                resolved = lerp(current, hist, pc.blend);
            }
        }
    }

    outHistory[gid.xy] = float4(resolved, 1.0);
}
