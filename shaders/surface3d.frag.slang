// surface3d.frag.slang — Zone-3 far skin. Near-zone clip: discard fragments near
// camera so skin renders only beyond chunks/tiles. Time-of-day tint: match mesh3d
// top-face shading. Fog parity: identical fog-to-horizon so seams vanish.

#include "common.slang"

struct FragmentInput
{
    [[vk::location(0)]]
    float4 color;

    [[vk::location(1)]]
    float dist;

    [[vk::location(2)]]
    float boxdist; // camera-relative ground-plane box distance (near-zone clip)
};

// Mirrors the push layout (Mesh3dPush = view_proj + a tail `clip` scalar). The
// tint and fog come from the per-frame UBO (fog-is-sky); the only push datum the
// surface pass needs is `clip`, the near-zone clip radius (metres).
struct PushConstants
{
    float4x4 view_proj;
    float clip; // near-zone clip radius (metres)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Set 0, binding 2: per-frame uniforms — sky anchors, sun light, ambient floor,
// and the DebugView::TerrainKey `reserved` channel. The far skin
// keys off `reserved` exactly like mesh3d.frag so the sky-hole detector's input
// stays consistent; without this the skin's grey corrupts the key coverage.
[[vk::binding(2, 0)]]
ConstantBuffer<FrameUniforms> frame;

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    // Near-zone clip: discard fragments inside the tile ring's box so the skin
    // shows only beyond the chunks + LOD tiles. Box test (not radial) matches
    // the tile ring's axis-aligned shape exactly. clip == 0 disables clipping.
    if (input.boxdist < pc.clip)
        discard;

    // Debug-flat (TerrainKey): flat key colour with depth still written — same
    // branch shape as mesh3d.frag, so the far skin keys instead of leaking grey.
    if (frame.reserved.w > 0.5) {
        return float4(srgb_to_linear(frame.reserved.rgb), 1.0);
    }

    // Time-of-day tint: lit by the sky's full-face illumination.
    float3 lit = ambient_floor_color(frame) + frame.light.rgb;
    float3 rgb = srgb_to_linear(input.color.rgb) * lit;
    // Fog identically to the tiles.
    float fog = 1.0 - exp(-input.dist * frame.horizon.w); // horizon.w = fog density
    rgb = lerp(rgb, frame.horizon.rgb, fog);
    return float4(rgb, 1.0);
}
