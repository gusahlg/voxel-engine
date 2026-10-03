// One bent-mesh cage (std430, 144 bytes) and the trilinear map through it.
// Included only by the cull shader, the shadow vertex shader, and the CAGED
// mesh vertex shader. Uncaged mesh3d does not include this file.

#ifndef VOXEL_CAGE_SLANG
#define VOXEL_CAGE_SLANG

struct Cage
{
    int3 anchor;
    uint pad0;
    float4 corners[8]; // xyz used; w is the std430 float3 pad
};

[[vk::binding(8, 0)]]
StructuredBuffer<Cage> cages;

float3 cage_lerp(float3 a, float3 b, float t)
{
    return a + (b - a) * t;
}

float3 cage_trilinear(Cage c, float3 t)
{
    float3 c0 = cage_lerp(c.corners[0].xyz, c.corners[1].xyz, t.x);
    float3 c1 = cage_lerp(c.corners[2].xyz, c.corners[3].xyz, t.x);
    float3 c2 = cage_lerp(c.corners[4].xyz, c.corners[5].xyz, t.x);
    float3 c3 = cage_lerp(c.corners[6].xyz, c.corners[7].xyz, t.x);
    return cage_lerp(cage_lerp(c0, c1, t.y), cage_lerp(c2, c3, t.y), t.z);
}

float3 cage_dtx(Cage c, float3 t)
{
    float3 e0 = c.corners[1].xyz - c.corners[0].xyz;
    float3 e1 = c.corners[3].xyz - c.corners[2].xyz;
    float3 e2 = c.corners[5].xyz - c.corners[4].xyz;
    float3 e3 = c.corners[7].xyz - c.corners[6].xyz;
    return cage_lerp(cage_lerp(e0, e1, t.y), cage_lerp(e2, e3, t.y), t.z);
}

float3 cage_dty(Cage c, float3 t)
{
    float3 c0 = cage_lerp(c.corners[0].xyz, c.corners[1].xyz, t.x);
    float3 c1 = cage_lerp(c.corners[2].xyz, c.corners[3].xyz, t.x);
    float3 c2 = cage_lerp(c.corners[4].xyz, c.corners[5].xyz, t.x);
    float3 c3 = cage_lerp(c.corners[6].xyz, c.corners[7].xyz, t.x);
    return cage_lerp(c1 - c0, c3 - c2, t.z);
}

float3 cage_dtz(Cage c, float3 t)
{
    float3 c0 = cage_lerp(c.corners[0].xyz, c.corners[1].xyz, t.x);
    float3 c1 = cage_lerp(c.corners[2].xyz, c.corners[3].xyz, t.x);
    float3 c2 = cage_lerp(c.corners[4].xyz, c.corners[5].xyz, t.x);
    float3 c3 = cage_lerp(c.corners[6].xyz, c.corners[7].xyz, t.x);
    return cage_lerp(c2, c3, t.y) - cage_lerp(c0, c1, t.y);
}

// Unit normal of `face` at parameter `t`. Odd faces flip. A zero tangent
// falls back to the flat face normal.
float3 cage_face_normal(Cage c, float3 t, uint face)
{
    float3 n;
    if (face < 2u)
        n = cross(cage_dty(c, t), cage_dtz(c, t));
    else if (face < 4u)
        n = cross(cage_dtz(c, t), cage_dtx(c, t));
    else
        n = cross(cage_dtx(c, t), cage_dty(c, t));
    if ((face & 1u) != 0u)
        n = -n;
    float len2 = dot(n, n);
    if (len2 == 0.0)
        return FACE_NORMAL[face];
    return n * rsqrt(len2);
}

void cage_corners_aabb(Cage c, out float3 mn, out float3 mx)
{
    mn = c.corners[0].xyz;
    mx = mn;
    [unroll]
    for (int i = 1; i < 8; i++)
    {
        mn = min(mn, c.corners[i].xyz);
        mx = max(mx, c.corners[i].xyz);
    }
}

// `q` is the local cell position times the detail scale.
float3 cage_world(Cage c, float3 q, float scale, int3 cam_block, float3 cam_frac)
{
    float3 t = q / (16.0 * scale);
    return cage_trilinear(c, t) + float3(c.anchor - cam_block) - cam_frac;
}

#endif
