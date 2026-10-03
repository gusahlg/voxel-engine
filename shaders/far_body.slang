// Far-body impostors for the sky pass. Ray tests match src/far_body.rs:
// the body is at distance 1 along dir, radius is radius/distance.
// Included from sky.frag after the frame uniform is declared.

#ifndef VOXEL_FAR_BODY_SLANG
#define VOXEL_FAR_BODY_SLANG

// Ten float4 lanes, 160 bytes. atmosphere.a is the shape: 0 cube, 1 sphere.
struct FarGpu
{
    float4 dir_rho;
    float4 rot;
    float4 albedo0;
    float4 albedo1;
    float4 albedo2;
    float4 albedo3;
    float4 albedo4;
    float4 albedo5;
    float4 atmosphere;
    uint4 seed;
};

struct FarTable
{
    uint4 header; // x = count
    FarGpu body[32];
};

// Set 0, binding 2: the host far-body table.
[[vk::binding(2, 0)]]
StructuredBuffer<FarTable> farTable;

float3 far_rotate(float4 q, float3 v)
{
    float3 b = q.xyz;
    float w = q.w;
    return v * (w * w - dot(b, b)) + b * (dot(v, b) * 2.0) + cross(b, v) * (w * 2.0);
}

float3 far_inv_rotate(float4 q, float3 v)
{
    return far_rotate(float4(-q.xyz, q.w), v);
}

float far_u01(int x, int y, int z, uint seed)
{
    uint h = uint(x) * 0x8DA6B343u ^ uint(y) * 0xD8163841u ^ uint(z) * 0xCB1AB31Fu ^ seed;
    h ^= h >> 16u;
    h *= 0x7FEB352Du;
    h ^= h >> 15u;
    h *= 0x846CA68Bu;
    h ^= h >> 16u;
    return float(h & 0x00FFFFFFu) * (1.0 / 16777215.0);
}

float far_value(float3 p, uint seed)
{
    float3 i = floor(p);
    float3 f = p - i;
    float3 u = f * f * (3.0 - 2.0 * f);
    int ix = int(i.x);
    int iy = int(i.y);
    int iz = int(i.z);
    float n000 = far_u01(ix, iy, iz, seed);
    float n100 = far_u01(ix + 1, iy, iz, seed);
    float n010 = far_u01(ix, iy + 1, iz, seed);
    float n110 = far_u01(ix + 1, iy + 1, iz, seed);
    float n001 = far_u01(ix, iy, iz + 1, seed);
    float n101 = far_u01(ix + 1, iy, iz + 1, seed);
    float n011 = far_u01(ix, iy + 1, iz + 1, seed);
    float n111 = far_u01(ix + 1, iy + 1, iz + 1, seed);
    float x00 = lerp(n000, n100, u.x);
    float x10 = lerp(n010, n110, u.x);
    float x01 = lerp(n001, n101, u.x);
    float x11 = lerp(n011, n111, u.x);
    return lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z);
}

float far_fbm(float3 p, uint seed)
{
    float sum = 0.0;
    float w = 0.5;
    [loop]
    for (uint o = 0u; o < 3u; o++)
    {
        sum += w * far_value(p, seed + o * 17u);
        p *= 2.0;
        w *= 0.5;
    }
    return sum * (1.0 / 0.875);
}

bool far_slab(float o, float d, float rho, inout float tMin, inout float tMax, inout uint face, uint axis)
{
    if (abs(d) < 1e-8)
        return abs(o) <= rho;
    float inv = 1.0 / d;
    float tNeg = (-rho - o) * inv;
    float tPos = (rho - o) * inv;
    float t1 = tNeg;
    float t2 = tPos;
    bool nearNeg = true;
    if (tNeg > tPos)
    {
        t1 = tPos;
        t2 = tNeg;
        nearNeg = false;
    }
    if (t1 > tMin)
    {
        tMin = t1;
        face = nearNeg ? axis * 2u + 1u : axis * 2u;
    }
    if (t2 < tMax)
        tMax = t2;
    return tMin <= tMax;
}

bool far_ray_sphere(float3 ray, float3 center, float rho, float facing, out float t, out float3 nrm)
{
    t = 0.0;
    nrm = float3(0.0, 0.0, 0.0);
    float disc = facing * facing - (1.0 - rho * rho);
    if (disc < 0.0)
        return false;
    t = facing - sqrt(disc);
    if (t <= 0.0)
        return false;
    nrm = (ray * t - center) / rho;
    return true;
}

bool far_ray_cube(float3 ray, float3 center, float rho, float4 rot, out float t, out float3 nrm, out uint face)
{
    t = 0.0;
    nrm = float3(0.0, 0.0, 0.0);
    face = 0u;
    float3 o = far_inv_rotate(rot, -center);
    float3 d = far_inv_rotate(rot, ray);
    float tMin = 0.0;
    float tMax = 1e30;
    if (!far_slab(o.x, d.x, rho, tMin, tMax, face, 0u))
        return false;
    if (!far_slab(o.y, d.y, rho, tMin, tMax, face, 1u))
        return false;
    if (!far_slab(o.z, d.z, rho, tMin, tMax, face, 2u))
        return false;
    if (tMin <= 0.0)
        return false;
    t = tMin;
    nrm = far_rotate(rot, FACE_NORMAL[face]);
    return true;
}

float3 far_face(FarGpu b, uint face)
{
    if (face == 0u) return b.albedo0.rgb;
    if (face == 1u) return b.albedo1.rgb;
    if (face == 2u) return b.albedo2.rgb;
    if (face == 3u) return b.albedo3.rgb;
    if (face == 4u) return b.albedo4.rgb;
    return b.albedo5.rgb;
}

float3 far_albedo(FarGpu b, float3 bodyP, uint face, bool sphere, float rho)
{
    uint seed = b.seed.x;
    if (sphere)
    {
        float h = far_fbm((bodyP / rho) * 3.0, seed);
        return lerp(b.albedo0.rgb, b.albedo1.rgb, smoothstep(0.38, 0.62, h));
    }
    float3 alb = far_face(b, face);
    float3 q = bodyP / rho;
    float2 uv = face < 2u ? q.yz : (face < 4u ? q.xz : q.xy);
    float region = far_u01(int(floor(uv.x * 4.0)), int(floor(uv.y * 4.0)), int(face), seed);
    alb *= 0.9 + 0.1 * region;
    float2 fr = frac(uv * 4.0);
    float edge = min(min(fr.x, 1.0 - fr.x), min(fr.y, 1.0 - fr.y));
    alb *= 0.84 + 0.16 * smoothstep(0.0, 0.08, edge);
    return alb;
}

float3 far_lit(float3 alb, float3 nrm, float3 ray, float3 air)
{
    // Same night flip as the terrain: light.w is 1 by day and 0 by night,
    // so the lit hemisphere follows the moon once the sun is down.
    float flip = frame.light.w * 2.0 - 1.0;
    float ndl = saturate(dot(nrm, frame.sun_dir_elev.xyz) * flip);
    float3 lit = alb * (frame.light.rgb * ndl + frame.ambient_glow.rgb);
    float fres = 1.0 - saturate(dot(nrm, -ray));
    fres = fres * fres * fres;
    return lit + air * fres;
}

// `px` is the pixel angle, from fwidth in the entry point (uniform flow).
float3 far_bodies(float3 ray, float3 col, float px)
{
    uint n = min(farTable[0].header.x, 32u);
    if (n == 0u)
        return col;

    float flip = frame.light.w * 2.0 - 1.0;
    float bestT = 1e30;
    [loop]
    for (uint i = 0u; i < n; i++)
    {
        FarGpu b = farTable[0].body[i];
        float rho = b.dir_rho.w;
        if (!(rho > 0.0 && rho < 1.0))
            continue;
        float3 center = b.dir_rho.xyz;
        float facing = dot(ray, center);
        if (facing <= 0.0)
            continue;

        float3 air = b.atmosphere.rgb;
        bool hasAir = dot(air, air) > 0.0;
        bool sphere = b.atmosphere.w > 0.5;
        // Angular diameter in pixels. Under ~0.7 px the disc is a soft point.
        float ang = (rho * 2.0) / px;
        if (ang < 0.7)
        {
            float tPoint = 1.0 - rho;
            if (tPoint < bestT)
            {
                float s = length(cross(ray, center));
                float blob = exp(-4.0 * (s * s) / (px * px));
                float phase = saturate(0.5 - 0.5 * dot(center, frame.sun_dir_elev.xyz) * flip);
                float gain = max(phase, 0.15);
                float tw = 0.75 + 0.25 * (float(b.seed.x & 255u) * (1.0 / 255.0));
                col += b.albedo0.rgb * (frame.light.rgb * gain + frame.ambient_glow.rgb) * (blob * tw);
            }
            continue;
        }

        float t = 0.0;
        float3 nrm = float3(0.0, 0.0, 0.0);
        uint face = 0u;
        bool hit = false;
        if (sphere)
        {
            float s = length(cross(ray, center));
            if (s <= rho)
            {
                hit = far_ray_sphere(ray, center, rho, facing, t, nrm);
                if (hit && t < bestT)
                {
                    bestT = t;
                    float3 bodyP = far_inv_rotate(b.rot, ray * t - center);
                    float3 alb = far_albedo(b, bodyP, 0u, true, rho);
                    float cov = saturate(0.5 + (rho - s) / px);
                    col = lerp(col, far_lit(alb, nrm, ray, air), cov);
                }
            }
            else if (hasAir)
            {
                float width = max(rho * 0.05, px * 1.25);
                if (s < rho + width && facing < bestT)
                {
                    float u = saturate((s - rho) / width);
                    float rim = 1.0 - u;
                    col += air * (rim * rim);
                }
            }
        }
        else
        {
            hit = far_ray_cube(ray, center, rho, b.rot, t, nrm, face);
            if (hit && t < bestT)
            {
                bestT = t;
                float3 bodyP = far_inv_rotate(b.rot, ray * t - center);
                float3 alb = far_albedo(b, bodyP, face, false, rho);
                col = lerp(col, far_lit(alb, nrm, ray, air), 1.0);
            }
            else if (!hit && hasAir)
            {
                float rhoRim = rho * 1.035 + px * 0.5;
                float tRim;
                float3 nRim;
                uint faceRim;
                if (rhoRim < 1.0 && far_ray_cube(ray, center, rhoRim, b.rot, tRim, nRim, faceRim) && tRim < bestT)
                {
                    float fres = 1.0 - saturate(dot(nRim, -ray));
                    fres = fres * fres * fres;
                    col += air * fres;
                }
            }
        }
    }
    return col;
}

#endif
