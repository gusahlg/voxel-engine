// Far-body impostors for the sky pass. Ray tests match src/far_body.rs.
// Outside, the body is at distance 1 along dir and radius is radius/distance.
// An inner sphere uses that space with radius/distance > 1 (the viewer is
// inside); the hit is the far root and the normal points at the centre.
// The table leads with a 16-byte cone per body so a pixel can reject the
// record before loading it. Included from sky.frag after the frame uniform.

#ifndef VOXEL_FAR_BODY_SLANG
#define VOXEL_FAR_BODY_SLANG

// Ten float4 lanes, 160 bytes. atmosphere.a is the shape: 0 cube, 1 sphere,
// 2 inner sphere, 3 rounded. seed.y is the rounded exponent as float bits.
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
    // x = count, y = tile size in pixels, z = tiles_x, w = tiles_y.
    // yzw are 0 when the host has no view; the shader then walks every body.
    uint4 header;
    float4 cone[32]; // xyz = dir, w = sine bound (-1 = never reject)
    FarGpu body[32];
    // Bit k is kept body k. One word per tile, row-major. Host cap is 8192.
    uint tile_mask[8192];
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

// Viewer inside (rho = radius/distance > 1). Centre at distance 1. The forward
// hit is the far root; the normal points back at the centre.
bool far_ray_inner(float3 ray, float3 center, float rho, float facing, out float t, out float3 nrm)
{
    t = 0.0;
    nrm = float3(0.0, 0.0, 0.0);
    float disc = facing * facing - (1.0 - rho * rho);
    if (disc < 0.0)
        return false;
    t = facing + sqrt(disc);
    if (t <= 0.0)
        return false;
    nrm = (center - ray * t) / rho;
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

// Dominant axis of a body-space normal. Ties break toward X, then Y, then Z.
uint far_dom_face(float3 n)
{
    float3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z)
        return n.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.z)
        return n.y >= 0.0 ? 2u : 3u;
    return n.z >= 0.0 ? 4u : 5u;
}

// s = ‖q‖_p and grad s. The origin reports zero.
void far_lp(float3 q, float p, out float s, out float3 g)
{
    float3 a = abs(q);
    float m = max(a.x, max(a.y, a.z));
    s = 0.0;
    g = float3(0.0, 0.0, 0.0);
    if (m == 0.0)
        return;
    float inv = 1.0 / m;
    float px = pow(a.x * inv, p);
    float py = pow(a.y * inv, p);
    float pz = pow(a.z * inv, p);
    float sum = px + py + pz;
    if (!(sum > 0.0))
        return;
    s = m * pow(sum, 1.0 / p);
    if (a.x > 0.0) g.x = px / sum * s / a.x * sign(q.x);
    if (a.y > 0.0) g.y = py / sum * s / a.y * sign(q.y);
    if (a.z > 0.0) g.z = pz / sum * s / a.z * sign(q.z);
}

// Sample at tHit when it is outside the solid. Normal from the gradient.
bool far_round_at(float4 rot, float3 o, float3 d, float tHit, float p, float rho, out float t, out float3 nrm, out uint face)
{
    t = 0.0;
    nrm = float3(0.0, 0.0, 0.0);
    face = 0u;
    if (!(tHit > 0.0))
        return false;
    float s;
    float3 g;
    far_lp(o + d * tHit, p, s, g);
    if (!(s >= rho))
        return false;
    float glen = length(g);
    if (!(glen > 1e-8))
        return false;
    t = tHit;
    nrm = far_rotate(rot, g / glen);
    face = far_dom_face(g);
    return true;
}

// Superellipsoid of face radius rho. Bound is rho * 3^(1/2 - 1/p), padded a hair
// so the entry stays outside the solid. A graze of that sphere, where it meets
// the body, falls back to the sphere point rather than opening a hole.
bool far_ray_rounded(float3 ray, float3 center, float rho, float4 rot, float exponent, float facing, out float t, out float3 nrm, out uint face)
{
    t = 0.0;
    nrm = float3(0.0, 0.0, 0.0);
    face = 0u;
    float p = clamp(exponent, 2.0, 32.0);
    float rhoB = rho * pow(3.0, 0.5 - 1.0 / p) * (1.0 + 2.0e-4);
    float disc = facing * facing - (1.0 - rhoB * rhoB);
    if (disc < 0.0)
        return false;
    float sd = sqrt(disc);
    float tNear = facing - sd;
    float tFar = facing + sd;
    if (!(tFar > 0.0))
        return false;

    float3 o = far_inv_rotate(rot, -center);
    float3 d = far_inv_rotate(rot, ray);
    float t0 = max(tNear, 0.0);
    float s0;
    float3 g0;
    far_lp(o, p, s0, g0);
    if (t0 == 0.0 && s0 <= rho)
        return false;

    float tC = clamp(facing, t0, tFar);
    float sC;
    float3 gC;
    far_lp(o + d * tC, p, sC, gC);
    if (sC > rho)
    {
        if (sC <= rho * 1.001 && tC > 0.0 && far_round_at(rot, o, d, tC, p, rho, t, nrm, face))
            return true;
        if (tNear > 0.0 && sd <= rhoB * 0.02)
        {
            float sB;
            float3 gB;
            far_lp(o + d * tNear, p, sB, gB);
            if (sB >= rho && sB <= rho * 1.01)
            {
                t = tNear;
                nrm = (ray * tNear - center) / rhoB;
                face = far_dom_face(far_inv_rotate(rot, nrm));
                return true;
            }
        }
        return false;
    }

    float lo = t0;
    float hi = tC;
    [loop]
    for (uint step = 0u; step < 6u; step++)
    {
        float sL;
        float3 gL;
        far_lp(o + d * lo, p, sL, gL);
        if (sL >= rho && sL <= rho * (1.0 + 2.0e-5))
            break;
        if (hi - lo < 1e-6)
            break;
        float slope = dot(gL, d);
        bool moved = false;
        if (slope < -1e-8)
        {
            float tN = lo - (sL - rho) / slope;
            if (tN > lo && tN < hi)
            {
                float sN;
                float3 gN;
                far_lp(o + d * tN, p, sN, gN);
                if (sN >= rho)
                {
                    lo = tN;
                    moved = true;
                }
                else
                    hi = tN;
            }
        }
        if (!moved)
        {
            float sH;
            float3 gH;
            far_lp(o + d * hi, p, sH, gH);
            float tF = 0.5 * (lo + hi);
            float span = sL - sH;
            if (abs(span) > 1e-12)
            {
                float guess = lo + (hi - lo) * (sL - rho) / span;
                if (guess > lo && guess < hi)
                    tF = guess;
            }
            float sF;
            float3 gF;
            far_lp(o + d * tF, p, sF, gF);
            if (sF >= rho)
                lo = tF;
            else
                hi = tF;
        }
    }
    if (far_round_at(rot, o, d, lo, p, rho, t, nrm, face))
        return true;
    if (tNear > 0.0)
    {
        float sB;
        float3 gB;
        far_lp(o + d * tNear, p, sB, gB);
        if (sB >= rho)
        {
            t = tNear;
            nrm = (ray * tNear - center) / rhoB;
            face = far_dom_face(far_inv_rotate(rot, nrm));
            return true;
        }
    }
    return false;
}

float3 far_face(uint i, uint face)
{
    if (face == 0u) return farTable[0].body[i].albedo0.rgb;
    if (face == 1u) return farTable[0].body[i].albedo1.rgb;
    if (face == 2u) return farTable[0].body[i].albedo2.rgb;
    if (face == 3u) return farTable[0].body[i].albedo3.rgb;
    if (face == 4u) return farTable[0].body[i].albedo4.rgb;
    return farTable[0].body[i].albedo5.rgb;
}

float3 far_albedo(uint i, float3 bodyP, uint face, bool sphere, float rho)
{
    uint seed = farTable[0].body[i].seed.x;
    if (sphere)
    {
        float h = far_fbm((bodyP / rho) * 3.0, seed);
        return lerp(
            farTable[0].body[i].albedo0.rgb,
            farTable[0].body[i].albedo1.rgb,
            smoothstep(0.38, 0.62, h));
    }
    float3 alb = far_face(i, face);
    float3 q = bodyP / rho;
    float2 uv = face < 2u ? q.yz : (face < 4u ? q.xz : q.xy);
    float region = far_u01(int(floor(uv.x * 4.0)), int(floor(uv.y * 4.0)), int(face), seed);
    alb *= 0.9 + 0.1 * region;
    float2 fr = frac(uv * 4.0);
    float edge = min(min(fr.x, 1.0 - fr.x), min(fr.y, 1.0 - fr.y));
    alb *= 0.84 + 0.16 * smoothstep(0.0, 0.08, edge);
    return alb;
}

// How much a disc of angular radius `rho` (sine of it) is the ground under the viewer: standing
// on a body it fills half the sky and continues the terrain at the horizon.
float far_near(float rho)
{
    return smoothstep(0.95, 0.9995, rho);
}

// A body beyond the viewer's air is seen through it: the air's in-scatter in front plus the body's
// light, so its night side is the sky colour by day (not a black hole in it). The body underfoot is
// the ground at the horizon and keeps its own colour.
float3 far_through_air(float3 lit, float3 inscatter, float rho)
{
    return lit + inscatter * (1.0 - far_near(rho));
}

float3 far_lit(float3 alb, float3 nrm, float3 ray, float3 air, float rho, bool core)
{
    // Same night flip as the terrain: light.w is 1 by day and 0 by night,
    // so the lit hemisphere follows the moon once the sun is down.
    // `core` is the inner wall: the light is at the centre and the normal points
    // there, so the whole wall faces it. No outdoor sky-dome fill.
    float flip = frame.light.w * 2.0 - 1.0;
    float ndl = core ? 1.0 : saturate(dot(nrm, frame.sun_dir_elev.xyz) * flip);
    // Standing on a body (it fills half the sky) its disc continues the terrain at the horizon, so
    // it takes the terrain's sky light (an omnidirectional share of the sun plus the bounce floor);
    // from afar it is a lit sphere with a dark night side.
    float near = core ? 0.0 : far_near(rho);
    float diffuse = lerp(ndl, 0.45 + 0.55 * ndl, near);
    float3 lit = alb * (frame.light.rgb * diffuse + frame.ambient_glow.rgb + frame.shadow_bounce.rgb * near);
    float fres = 1.0 - saturate(dot(nrm, -ray));
    fres = fres * fres * fres;
    return lit + air * fres;
}

// `px` is the pixel angle, from fwidth in the entry point (uniform flow).
// `pix` is SV_Position.xy (framebuffer pixels, origin top-left).
// `inscatter` is the viewer's sky radiance along `ray` (no sun disc, moon or stars).
float3 far_bodies(float3 ray, float3 col, float3 inscatter, float px, float2 pix)
{
    uint n = min(farTable[0].header.x, 32u);
    if (n == 0u)
        return col;

    // One mask for the whole tile. Bits are kept-index order, so walking the
    // lowest set bit keeps the original composite order.
    uint live = n == 32u ? 0xffffffffu : ((1u << n) - 1u);
    uint tilePx = farTable[0].header.y;
    uint tilesX = farTable[0].header.z;
    uint tilesY = farTable[0].header.w;
    uint mask;
    if (tilePx == 0u || tilesX == 0u || tilesY == 0u)
        mask = live;
    else
    {
        // Helpers at the quad edge can sit just outside the framebuffer.
        float2 p = max(pix, float2(0.0, 0.0));
        uint tx = min(uint(p.x) / tilePx, tilesX - 1u);
        uint ty = min(uint(p.y) / tilePx, tilesY - 1u);
        mask = farTable[0].tile_mask[ty * tilesX + tx] & live;
    }
    if (mask == 0u)
        return col;

    float flip = frame.light.w * 2.0 - 1.0;
    float bestT = 1e30;
    [loop]
    while (mask != 0u)
    {
        uint i = firstbitlow(mask);
        mask &= mask - 1u;
        // Reject before touching the 160-byte record. `f <= 0` also drops the
        // rounded air rim's ghost ring at the antipode (that rim has no facing test).
        float4 c = farTable[0].cone[i];
        if (c.w >= 0.0)
        {
            float f = dot(ray, c.xyz);
            float3 x = cross(ray, c.xyz);
            float lim = c.w + 3.0 * px; // 3 px covers the 1.25 px rim, the 0.5 px AA edge and the blob tail
            if (f <= 0.0 || dot(x, x) > lim * lim)
                continue;
        }

        float4 dirRho = farTable[0].body[i].dir_rho;
        float4 atmo = farTable[0].body[i].atmosphere;
        float rho = dirRho.w;
        float shape = atmo.w;
        bool inner = shape > 1.5 && shape < 2.5;
        bool rounded = shape > 2.5 && shape < 3.5;
        if (inner)
        {
            if (!(rho > 1.0))
                continue;
        }
        else if (!(rho > 0.0 && rho < 1.0))
            continue;
        float3 center = dirRho.xyz;
        float facing = dot(ray, center);
        // Inside the sphere every direction hits the far wall, including away from the centre.
        // A rounded body can contain the camera in its enclosing sphere while the camera is
        // still outside the solid, so a ray away from the centre may meet the bulge.
        if (!inner && !rounded && facing <= 0.0)
            continue;

        float3 air = atmo.rgb;
        bool hasAir = dot(air, air) > 0.0;
        if (inner)
        {
            float t = 0.0;
            float3 nrm = float3(0.0, 0.0, 0.0);
            if (far_ray_inner(ray, center, rho, facing, t, nrm) && t < bestT)
            {
                bestT = t;
                float4 rot = farTable[0].body[i].rot;
                float3 bodyP = far_inv_rotate(rot, ray * t - center);
                float3 alb = far_albedo(i, bodyP, 0u, true, rho);
                col = far_lit(alb, nrm, ray, air, rho, true);
            }
            continue;
        }
        bool sphere = shape > 0.5 && shape < 1.5;
        // Angular diameter in pixels. Under ~0.7 px the disc is a soft point.
        float ang = (rho * 2.0) / px;
        if (ang < 0.7 && facing > 0.0)
        {
            float tPoint = 1.0 - rho;
            if (tPoint < bestT)
            {
                float s = length(cross(ray, center));
                float blob = exp(-4.0 * (s * s) / (px * px));
                float phase = saturate(0.5 - 0.5 * dot(center, frame.sun_dir_elev.xyz) * flip);
                float gain = max(phase, 0.15);
                uint seed = farTable[0].body[i].seed.x;
                float tw = 0.75 + 0.25 * (float(seed & 255u) * (1.0 / 255.0));
                col += farTable[0].body[i].albedo0.rgb * (frame.light.rgb * gain + frame.ambient_glow.rgb) * (blob * tw);
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
                    float4 rot = farTable[0].body[i].rot;
                    float3 bodyP = far_inv_rotate(rot, ray * t - center);
                    float3 alb = far_albedo(i, bodyP, 0u, true, rho);
                    float cov = saturate(0.5 + (rho - s) / px);
                    col = lerp(col, far_through_air(far_lit(alb, nrm, ray, air, rho, false), inscatter, rho), cov);
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
        else if (rounded)
        {
            float pExp = asfloat(farTable[0].body[i].seed.y);
            float4 rot = farTable[0].body[i].rot;
            hit = far_ray_rounded(ray, center, rho, rot, pExp, facing, t, nrm, face);
            if (hit && t < bestT)
            {
                bestT = t;
                float3 bodyP = far_inv_rotate(rot, ray * t - center);
                float3 alb = far_albedo(i, bodyP, face, false, rho);
                col = far_through_air(far_lit(alb, nrm, ray, air, rho, false), inscatter, rho);
            }
            else if (!hit && hasAir)
            {
                float pRim = max(pExp, 2.0);
                float rhoB = rho * pow(3.0, 0.5 - 1.0 / pRim);
                float s = length(cross(ray, center));
                float width = max(rhoB * 0.05, px * 1.25);
                if (s > rhoB && s < rhoB + width && facing < bestT)
                {
                    float u = saturate((s - rhoB) / width);
                    float rim = 1.0 - u;
                    col += air * (rim * rim);
                }
            }
        }
        else
        {
            float4 rot = farTable[0].body[i].rot;
            hit = far_ray_cube(ray, center, rho, rot, t, nrm, face);
            if (hit && t < bestT)
            {
                bestT = t;
                float3 bodyP = far_inv_rotate(rot, ray * t - center);
                float3 alb = far_albedo(i, bodyP, face, false, rho);
                col = far_through_air(far_lit(alb, nrm, ray, air, rho, false), inscatter, rho);
            }
            else if (!hit && hasAir)
            {
                float rhoRim = rho * 1.035 + px * 0.5;
                float tRim;
                float3 nRim;
                uint faceRim;
                if (rhoRim < 1.0 && far_ray_cube(ray, center, rhoRim, rot, tRim, nRim, faceRim) && tRim < bestT)
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
