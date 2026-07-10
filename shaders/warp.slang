// warp.slang — shared high-FOV nonlinear projection (De Carpentier-family
// adaptive-cylindrical). #included by every 3D vertex shader and by sky.frag so
// the exact same map is used everywhere; the CPU reference lives in
// camera::WarpParams (keep the two in lockstep).
//
// The strength `s` is the cylindrical_ratio (0 = pure perspective). It is smuggled
// per-frame in an unused push-constant .w lane (ambient.w for the mesh/surface
// pipelines, warp.x for debug, zenith.w for sky) so no struct grows.
//
// Forward warp (horizontal only) magnifies the center and compresses the
// periphery. See camera.rs for the full details and the properties the CPU tests
// lock down.

// Warp a clip-space position in place: divide out w for the NDC map, warp x,
// then re-multiply by w so depth (z,w) and perspective-correct interpolation are
// untouched — only the horizontal screen position moves. y is left rectilinear.
float4 applyWarp(float4 clip, float s)
{
    if (s <= 0.0)
        return clip;
    float nx = clip.x / clip.w;
    nx = atan(nx * s) / atan(s);
    return float4(nx * clip.w, clip.y, clip.z, clip.w);
}

// Inverse of the forward map on an NDC point (horizontal only). The sky pass
// applies this to each pixel's NDC before reconstructing its world ray, so the
// background lines up with the warped terrain.
float2 inverseWarpNdc(float2 ndc, float s)
{
    if (s <= 0.0)
        return ndc;
    ndc.x = tan(ndc.x * atan(s)) / s;
    return ndc;
}
