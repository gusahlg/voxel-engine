// Far-body impostors for the sky pass. Ray tests match src/far_body.rs.
// Outside, the body is at distance 1 along dir and radius is radius/distance.
// An inner sphere uses that space with radius/distance > 1 (the viewer is
// inside); the hit is the far root and the normal points at the centre.
// The table leads with a 16-byte cone per body so a pixel can reject the
// record before loading it. Included from sky.frag after the frame uniform.

#ifndef VOXEL_FAR_BODY_SLANG
#define VOXEL_FAR_BODY_SLANG

#include "far_table.slang"

// Set 0, binding 2: the host far-body table.
[[vk::binding(2, 0)]]
StructuredBuffer<FarTable> farTable;

// Set 0, binding 3. Eight headers of 8 uints, then the datum floats.
// Header of map m, at element m*8: g, float-offset into the datum array,
// min offset bits, max offset bits, 6-bit landed mask, albedo edge, 0, 0.
// Datum float i of map m is element 64 + offset + i. offset = m * 6 * 65 * 65.
[[vk::binding(3, 0)]]
StructuredBuffer<uint> farMaps;

// Set 0, binding 4: one cube per map. Empty slots are a 1x1 dummy.
// Sampled through far_cube so the index is a constant (no dynamic indexing).
[[vk::binding(4, 0)]]
SamplerCube farAlbedo[8];

// Odd degree-13 polynomial for atan on [-1, 1]. The equiangular chart only
// feeds that range, and the argument is clamped. Stepwise f32 Horner
// (p = p * u + c) stays within 6.61e-7 rad of atan, peaking near -0.972.
// A datum cell is pi / (4 * 32) ≈ 2.5e-2 rad, so this is far below a cell.
// Bits: c6..c0 = 3c04aa78 bd1aa180 3dad9679 be0aa033 3e4bb99e beaaa3a5 3f7ffffb.
float far_atan_approx(float x)
{
    float z = clamp(x, -1.0, 1.0);
    float u = z * z;
    float p = asfloat(0x3c04aa78u);
    p = p * u + asfloat(0xbd1aa180u);
    p = p * u + asfloat(0x3dad9679u);
    p = p * u + asfloat(0xbe0aa033u);
    p = p * u + asfloat(0x3e4bb99eu);
    p = p * u + asfloat(0xbeaaa3a5u);
    p = p * u + asfloat(0x3f7ffffbu);
    return z * p;
}

// Face basis matches far_map_basis: +X -X +Y -Y +Z -Z.
void far_face_basis(uint face, out float3 tu, out float3 n, out float3 tv)
{
    if (face == 0u)
    {
        tu = float3(0.0, -1.0, 0.0);
        n = float3(1.0, 0.0, 0.0);
        tv = float3(0.0, 0.0, 1.0);
    }
    else if (face == 1u)
    {
        tu = float3(0.0, 1.0, 0.0);
        n = float3(-1.0, 0.0, 0.0);
        tv = float3(0.0, 0.0, 1.0);
    }
    else if (face == 2u)
    {
        tu = float3(1.0, 0.0, 0.0);
        n = float3(0.0, 1.0, 0.0);
        tv = float3(0.0, 0.0, 1.0);
    }
    else if (face == 3u)
    {
        tu = float3(1.0, 0.0, 0.0);
        n = float3(0.0, -1.0, 0.0);
        tv = float3(0.0, 0.0, -1.0);
    }
    else if (face == 4u)
    {
        tu = float3(1.0, 0.0, 0.0);
        n = float3(0.0, 0.0, 1.0);
        tv = float3(0.0, -1.0, 0.0);
    }
    else
    {
        tu = float3(1.0, 0.0, 0.0);
        n = float3(0.0, 0.0, -1.0);
        tv = float3(0.0, 1.0, 0.0);
    }
}

// Equiangular bilinear of map's datum, plus the in-cell chart derivatives.
// Four loads; g < 2 or a tiny denominator is a flat zero. `ok` is false on
// that flat fallback.
bool far_datum_d(uint map, float3 d, out float value, out float drDxi, out float drDeta)
{
    value = 0.0;
    drDxi = 0.0;
    drDeta = 0.0;
    uint g = farMaps[map * 8u];
    if (g < 2u)
        return false;
    uint face = far_dom_face(d);
    float3 tu;
    float3 n;
    float3 tv;
    far_face_basis(face, tu, n, tv);
    float den = dot(d, n);
    if (abs(den) < 1e-8)
        return false;
    float xi = (4.0 / 3.14159265358979323846) * far_atan_approx(dot(d, tu) / den);
    float eta = (4.0 / 3.14159265358979323846) * far_atan_approx(dot(d, tv) / den);
    float scale = 0.5 * float(g - 1u);
    float u = clamp((xi + 1.0) * scale, 0.0, float(g - 1u));
    float v = clamp((eta + 1.0) * scale, 0.0, float(g - 1u));
    uint i0 = min(uint(floor(u)), g - 1u);
    uint j0 = min(uint(floor(v)), g - 1u);
    uint i1 = min(i0 + 1u, g - 1u);
    uint j1 = min(j0 + 1u, g - 1u);
    float fu = u - float(i0);
    float fv = v - float(j0);
    uint off = farMaps[map * 8u + 1u];
    uint base = 64u + off + face * g * g;
    float v00 = asfloat(farMaps[base + j0 * g + i0]);
    float v10 = asfloat(farMaps[base + j0 * g + i1]);
    float v01 = asfloat(farMaps[base + j1 * g + i0]);
    float v11 = asfloat(farMaps[base + j1 * g + i1]);
    float a = v00 + (v10 - v00) * fu;
    float b = v01 + (v11 - v01) * fu;
    float drDfu = (v10 - v00) + ((v11 - v01) - (v10 - v00)) * fv;
    value = a + (b - a) * fv;
    drDxi = drDfu * scale;
    drDeta = (b - a) * scale;
    return true;
}

// Equiangular bilinear of map's datum. Four loads; g < 2 is a flat zero.
float far_datum(uint map, float3 d)
{
    float value;
    float drDxi;
    float drDeta;
    far_datum_d(map, d, value, drDxi, drDeta);
    return value;
}

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

// Optical depth per air thickness. The mapped limb and the mapped surface
// chord share this. A vertical path equal to the shell thickness keeps
// exp(-0.35) of the surface; the old limb coefficient of 4 did not.
static const float FAR_AIR_K = 0.35;

float3 far_lit(float3 alb, float3 nrm, float3 ray, float3 air, float rho, bool core, bool rim)
{
    // Same night flip as the terrain: light.w is 1 by day and 0 by night,
    // so the lit hemisphere follows the moon once the sun is down.
    // `core` is the inner wall: the light is at the centre and the normal points
    // there, so the whole wall faces it. No outdoor sky-dome fill.
    // `rim` is the fresnel air edge. Mapped hits pass false and composite the
    // shell along the view chord instead; every other shape passes true.
    float flip = frame.light.w * 2.0 - 1.0;
    float ndl = core ? 1.0 : saturate(dot(nrm, frame.sun_dir_elev.xyz) * flip);
    // Standing on a body (it fills half the sky) its disc continues the terrain at the horizon, so
    // it takes the terrain's sky light (an omnidirectional share of the sun plus the bounce floor);
    // from afar it is a lit sphere with a dark night side.
    float near = core ? 0.0 : far_near(rho);
    float diffuse = lerp(ndl, 0.45 + 0.55 * ndl, near);
    float3 lit = alb * (frame.light.rgb * diffuse + frame.ambient_glow.rgb + frame.shadow_bounce.rgb * near);
    if (!rim)
        return lit;
    float fres = 1.0 - saturate(dot(nrm, -ray));
    fres = fres * fres * fres;
    return lit + air * fres;
}

// Constant-index sample. A dynamic index would need descriptor indexing.
float3 far_cube(uint map, float3 dir, float lod)
{
    if (map == 0u) return farAlbedo[0].SampleLevel(dir, lod).rgb;
    if (map == 1u) return farAlbedo[1].SampleLevel(dir, lod).rgb;
    if (map == 2u) return farAlbedo[2].SampleLevel(dir, lod).rgb;
    if (map == 3u) return farAlbedo[3].SampleLevel(dir, lod).rgb;
    if (map == 4u) return farAlbedo[4].SampleLevel(dir, lod).rgb;
    if (map == 5u) return farAlbedo[5].SampleLevel(dir, lod).rgb;
    if (map == 6u) return farAlbedo[6].SampleLevel(dir, lod).rgb;
    return farAlbedo[7].SampleLevel(dir, lod).rgb;
}

bool far_sphere_roots(float facing, float rho, out float tNear, out float tFar)
{
    tNear = 0.0;
    tFar = 0.0;
    if (!(rho > 0.0))
        return false;
    float disc = facing * facing - (1.0 - rho * rho);
    if (disc < 0.0)
        return false;
    float sd = sqrt(disc);
    tNear = facing - sd;
    tFar = facing + sd;
    return true;
}

// Datum samples the march may spend. The analytic normal is one more fetch
// at the hit, outside this cap. Matches src/far_body.rs.
static const int FAR_MAP_CAP = 5;
static const float FAR_MAP_FTOL = 1e-8;

struct FarMarch
{
    float3 ray;
    float3 center;
    float rho;
    float distance;
    float4 rot;
    uint map;
    int evals;
};

// d/dt (|p| - datum/distance). The chart term uses d(atan)/dx, which tracks
// the equiangular polynomial closely enough for the Newton correction.
float far_residual_slope(
    float3 ray, float3 center, float t, float distance, float4 rot,
    float3 uWorld, float3 uBody, float drDxi, float drDeta)
{
    float3 p = ray * t - center;
    float rad = length(p);
    float slopeRad = dot(uWorld, ray);
    if (rad < 1e-8 || !(distance > 0.0))
        return slopeRad;
    uint face = far_dom_face(uBody);
    float3 tu;
    float3 n;
    float3 tv;
    far_face_basis(face, tu, n, tv);
    float beta = dot(uBody, n);
    if (abs(beta) < 1e-6)
        return slopeRad;
    float3 duWorld = (ray - uWorld * slopeRad) / rad;
    float3 du = far_inv_rotate(rot, duWorld);
    float alpha = dot(uBody, tu);
    float gamma = dot(uBody, tv);
    float alphaT = dot(du, tu);
    float betaT = dot(du, n);
    float gammaT = dot(du, tv);
    float beta2 = beta * beta;
    float x = alpha / beta;
    float xT = (beta * alphaT - alpha * betaT) / beta2;
    float y = gamma / beta;
    float yT = (beta * gammaT - gamma * betaT) / beta2;
    float scale = 4.0 / 3.14159265358979323846;
    float dxiDt = scale * (xT / (1.0 + x * x));
    float detaDt = scale * (yT / (1.0 + y * y));
    float ddatumDt = drDxi * dxiDt + drDeta * detaDt;
    return slopeRad - ddatumDt / distance;
}

// (f, r, df/dt). Slope is filled only when |f| < 0.05: that is the Newton
// gate, and the chain rule is wasted on a sample that is still far from the
// surface. A degenerate chart reports the radial slope.
void far_march_sample(inout FarMarch m, float t, out float f, out float r, out float slope)
{
    m.evals += 1;
    float3 p = m.ray * t - m.center;
    float rad = length(p);
    f = -m.rho;
    r = 0.0;
    slope = 0.0;
    if (rad < 1e-8)
        return;
    float3 uWorld = p / rad;
    float3 body = far_inv_rotate(m.rot, uWorld);
    float value;
    float drDxi;
    float drDeta;
    bool ok = far_datum_d(m.map, body, value, drDxi, drDeta);
    r = m.rho + value / m.distance;
    f = rad - r;
    float slopeRad = dot(uWorld, m.ray);
    if (!ok || abs(f) >= 0.05)
    {
        slope = slopeRad;
        return;
    }
    slope = far_residual_slope(m.ray, m.center, t, m.distance, m.rot, uWorld, body, drDxi, drDeta);
}

bool far_positive_root(float facing, float radius, out float t)
{
    float tNear;
    float tFar;
    t = 0.0;
    if (!far_sphere_roots(facing, radius, tNear, tFar))
        return false;
    if (tNear > 0.0)
    {
        t = tNear;
        return true;
    }
    if (tFar > 0.0)
    {
        t = tFar;
        return true;
    }
    return false;
}

// Illinois regula falsi. The retained endpoint's weight is halved so a stuck
// side does not burn the remaining samples. A small residual takes a Newton
// step from the chart slope; with no sample left, sphere curvature corrects it.
float far_regula(inout FarMarch m, float a0, float b0, float fa0, float fb0)
{
    float a = a0;
    float b = b0;
    float fa = fa0;
    float fb = fb0;
    float wa = fa;
    float wb = fb;
    [loop]
    while (m.evals < FAR_MAP_CAP)
    {
        float den = wb - wa;
        if (abs(den) < 1e-20 || abs(b - a) < 1e-7)
            break;
        float tn = b - wb * (b - a) / den;
        if (!(tn > min(a, b) && tn < max(a, b)))
            break;
        float ft;
        float rt;
        float st;
        far_march_sample(m, tn, ft, rt, st);
        if (abs(ft) <= FAR_MAP_FTOL)
            return tn;
        if (abs(st) > 1e-4 && abs(ft) < 2e-3)
        {
            float t2 = tn - ft / st;
            float lo = min(a, b);
            float hi = max(a, b);
            if (t2 > lo && t2 < hi && abs(t2 - tn) < 5e-3)
            {
                if (!(m.evals < FAR_MAP_CAP))
                {
                    float3 p = m.ray * tn - m.center;
                    float rad = length(p);
                    float srad = dot(p / rad, m.ray);
                    float curve = (1.0 - srad * srad) / rad;
                    float hden = st - ft * curve / (2.0 * st);
                    float tH = (abs(hden) > 1e-4 && rad > 1e-8) ? (tn - ft / hden) : t2;
                    if (abs(tH - tn) < 5e-3)
                        return tH;
                    return t2;
                }
                float f2;
                float r2;
                float s2;
                far_march_sample(m, t2, f2, r2, s2);
                if (abs(s2) > 1e-4 && abs(f2) < 1e-3)
                {
                    float d2 = abs(t2 - tn) > 1e-6 ? (s2 - st) / (t2 - tn) : 0.0;
                    float hden = s2 - f2 * d2 / (2.0 * s2);
                    float t3 = abs(hden) > 1e-4 ? (t2 - f2 / hden) : (t2 - f2 / s2);
                    if (t3 > 0.0 && abs(t3 - t2) < 1e-3)
                        return t3;
                }
                if (abs(f2) <= abs(ft))
                    return t2;
            }
        }
        if (fa * ft <= 0.0)
        {
            b = tn;
            fb = ft;
            wb = ft;
            wa *= 0.5;
        }
        else
        {
            a = tn;
            fa = ft;
            wa = ft;
            wb *= 0.5;
        }
    }
    if (!(m.evals < FAR_MAP_CAP))
    {
        float den = fb - fa;
        if (abs(den) > 1e-20)
        {
            float tn = b - fb * (b - a) / den;
            if (tn > min(a, b) && tn < max(a, b))
                return tn;
        }
    }
    return abs(fa) <= abs(fb) ? a : b;
}

#if 0
// Finite-difference normal kept for comparison. Two one-sided steps of half
// a datum cell. A flat datum is exactly radial.
float3 far_mapped_normal_fd(float4 rot, float3 uWorld, float rho, float distance, uint map)
{
    float3 u = far_inv_rotate(rot, uWorld);
    uint g = max(farMaps[map * 8u], 2u);
    float eps = 3.14159265358979323846 / (4.0 * float(g - 1u));
    float3 axis = abs(u.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 t1 = normalize(cross(u, axis));
    float3 t2 = cross(u, t1);
    float r0 = rho + far_datum(map, u) / distance;
    float r1 = rho + far_datum(map, normalize(u + t1 * eps)) / distance;
    float r2 = rho + far_datum(map, normalize(u + t2 * eps)) / distance;
    float dr1 = (r1 - r0) / eps;
    float dr2 = (r2 - r0) / eps;
    return far_rotate(rot, normalize(u * r0 - t1 * dr1 - t2 * dr2));
}
#endif

// Outward normal of the bilinear patch at uWorld. On the +X face centre,
// d(uHat)/d(xi) × d(uHat)/d(eta) points inward, so the outward normal is
// dp/d(eta) × dp/d(xi). A flat patch is exactly radial.
float3 far_mapped_normal(float4 rot, float3 uWorld, float rho, float distance, uint map)
{
    float3 u = far_inv_rotate(rot, uWorld);
    float value;
    float drDxi;
    float drDeta;
    bool ok = far_datum_d(map, u, value, drDxi, drDeta);
    if (!ok || !(distance > 0.0))
        return uWorld;
    float r = rho + value / distance;
    float dr_dxi = drDxi / distance;
    float dr_deta = drDeta / distance;
    uint face = far_dom_face(u);
    float3 tu;
    float3 n;
    float3 tv;
    far_face_basis(face, tu, n, tv);
    float den = dot(u, n);
    if (abs(den) < 1e-8)
        return uWorld;
    float quarter = 0.7853981633974483;
    float xi = (4.0 / 3.14159265358979323846) * far_atan_approx(dot(u, tu) / den);
    float eta = (4.0 / 3.14159265358979323846) * far_atan_approx(dot(u, tv) / den);
    float tx = tan(xi * quarter);
    float ty = tan(eta * quarter);
    float dtx = quarter * (1.0 + tx * tx);
    float dty = quarter * (1.0 + ty * ty);
    float3 q = n + tu * tx + tv * ty;
    float s = length(q);
    if (!(s > 1e-8))
        return uWorld;
    float3 uHat = q / s;
    float3 dqDxi = tu * dtx;
    float3 dqDeta = tv * dty;
    float3 duDxi = dqDxi / s - uHat * (dot(uHat, dqDxi) / s);
    float3 duDeta = dqDeta / s - uHat * (dot(uHat, dqDeta) / s);
    float3 dpDxi = uHat * dr_dxi + duDxi * r;
    float3 dpDeta = uHat * dr_deta + duDeta * r;
    float3 nBody = cross(dpDeta, dpDxi);
    if (dot(nBody, nBody) < 1e-20)
        return uWorld;
    return far_rotate(rot, normalize(nBody));
}

bool far_mapped_finish(
    float3 ray, float3 center, float tHit, float rho, float distance, float4 rot, uint map,
    out float t, out float3 nrm, out uint face)
{
    t = 0.0;
    nrm = float3(0.0, 0.0, 0.0);
    face = 0u;
    if (!(tHit > 0.0))
        return false;
    float3 p = ray * tHit - center;
    float rad = length(p);
    if (rad < 1e-8)
        return false;
    float3 uWorld = p / rad;
    t = tHit;
    nrm = far_mapped_normal(rot, uWorld, rho, distance, map);
    face = far_dom_face(far_inv_rotate(rot, uWorld));
    return true;
}

// Datum surface. Fixed-point pulls from the lo-sphere entrance
// (t <- ray ∩ sphere(r(dir))), then Illinois regula on the bracket those
// pulls form. A ray that misses the lo sphere walks in from the hi entrance
// and finishes a shallow graze with one Newton step. At most FAR_MAP_CAP
// datum samples, plus the analytic normal at the hit. The caller has already
// applied the horizon gate.
bool far_ray_mapped(
    float3 ray, float3 center, float rho, float distance, float4 rot, uint map,
    out float t, out float3 nrm, out uint face)
{
    t = 0.0;
    nrm = float3(0.0, 0.0, 0.0);
    face = 0u;
    float minOff = asfloat(farMaps[map * 8u + 2u]);
    float maxOff = asfloat(farMaps[map * 8u + 3u]);
    float rhoLo = rho + minOff / distance;
    float rhoHi = rho + maxOff / distance;
    float facing = dot(ray, center);
    float tHiIn;
    float tHiOut;
    if (!far_sphere_roots(facing, rhoHi, tHiIn, tHiOut) || !(tHiOut > 0.0))
        return false;
    float tStart = max(tHiIn, 0.0);
    FarMarch m;
    m.ray = ray;
    m.center = center;
    m.rho = rho;
    m.distance = distance;
    m.rot = rot;
    m.map = map;
    m.evals = 0;
    float tLo;
    float tLoFar;
    bool loHit = far_sphere_roots(facing, rhoLo, tLo, tLoFar) && tLo > 0.0;
    if (loHit)
    {
        float bSpan = max(tLo, tStart);
        if (bSpan - tStart < 1e-5)
            return far_mapped_finish(ray, center, tLo > 0.0 ? tLo : tStart, rho, distance, rot, map, t, nrm, face);
        float f;
        float r;
        float slope;
        far_march_sample(m, tLo, f, r, slope);
        if (abs(f) <= FAR_MAP_FTOL)
            return far_mapped_finish(ray, center, tLo, rho, distance, rot, map, t, nrm, face);
        float curT = tLo;
        bool hasBracket = false;
        float ba = 0.0;
        float bb = 0.0;
        float bfa = 0.0;
        float bfb = 0.0;
        [loop]
        for (int step = 0; step < 2; step++)
        {
            if (!(m.evals < FAR_MAP_CAP))
                break;
            float tn;
            if (!far_positive_root(facing, r, tn))
                break;
            if (tn <= tStart || tn >= tHiOut || abs(tn - curT) <= 1e-8)
                break;
            if (hasBracket && (tn <= ba || tn >= bb))
                break;
            float ft;
            float rt;
            float st;
            far_march_sample(m, tn, ft, rt, st);
            if (f * ft <= 0.0)
            {
                hasBracket = true;
                if (curT < tn)
                {
                    ba = curT;
                    bb = tn;
                    bfa = f;
                    bfb = ft;
                }
                else
                {
                    ba = tn;
                    bb = curT;
                    bfa = ft;
                    bfb = f;
                }
            }
            if (abs(ft) <= FAR_MAP_FTOL)
                return far_mapped_finish(ray, center, tn, rho, distance, rot, map, t, nrm, face);
            curT = tn;
            f = ft;
            r = rt;
        }
        bool needEntrance = !hasBracket || tStart == 0.0;
        if (needEntrance)
        {
            if (!(m.evals < FAR_MAP_CAP))
                return far_mapped_finish(ray, center, curT, rho, distance, rot, map, t, nrm, face);
            float fa;
            float ra;
            float sa;
            far_march_sample(m, tStart, fa, ra, sa);
            if (abs(fa) <= 1e-5)
                return far_mapped_finish(ray, center, tStart, rho, distance, rot, map, t, nrm, face);
            if (fa < 0.0)
                return false;
            if (!hasBracket)
            {
                if (fa * f <= 0.0)
                {
                    hasBracket = true;
                    if (tStart < curT)
                    {
                        ba = tStart;
                        bb = curT;
                        bfa = fa;
                        bfb = f;
                    }
                    else
                    {
                        ba = curT;
                        bb = tStart;
                        bfa = f;
                        bfb = fa;
                    }
                }
                else
                {
                    return far_mapped_finish(ray, center, curT, rho, distance, rot, map, t, nrm, face);
                }
            }
        }
        if (!hasBracket)
            return far_mapped_finish(ray, center, curT, rho, distance, rot, map, t, nrm, face);
        float tHit = far_regula(m, ba, bb, bfa, bfb);
        return far_mapped_finish(ray, center, tHit, rho, distance, rot, map, t, nrm, face);
    }
    float f0;
    float r0;
    float slope0;
    far_march_sample(m, tStart, f0, r0, slope0);
    if (f0 < -1e-5)
        return false;
    if (abs(f0) <= 1e-5)
        return far_mapped_finish(ray, center, tStart, rho, distance, rot, map, t, nrm, face);
    float fEnter = f0;
    float curT = tStart;
    float f = f0;
    float r = r0;
    float slope = slope0;
    bool hasBracket = false;
    float ba = 0.0;
    float bb = 0.0;
    float bfa = 0.0;
    float bfb = 0.0;
    [loop]
    while (m.evals + 1 < FAR_MAP_CAP)
    {
        float tn;
        if (!far_positive_root(facing, r, tn) || !(tn > tStart && tn <= tHiOut))
            break;
        if (abs(tn - curT) <= 1e-8)
            break;
        float ft;
        float rt;
        float st;
        far_march_sample(m, tn, ft, rt, st);
        if (f * ft <= 0.0)
        {
            hasBracket = true;
            if (curT < tn)
            {
                ba = curT;
                bb = tn;
                bfa = f;
                bfb = ft;
            }
            else
            {
                ba = tn;
                bb = curT;
                bfa = ft;
                bfb = f;
            }
            break;
        }
        if (abs(ft) <= FAR_MAP_FTOL)
            return far_mapped_finish(ray, center, tn, rho, distance, rot, map, t, nrm, face);
        curT = tn;
        f = ft;
        r = rt;
        slope = st;
    }
    if (!hasBracket && abs(slope) > 1e-4 && abs(f) < 0.05)
    {
        float ts = curT - f / slope;
        if (ts > tStart && ts < tHiOut && abs(ts - curT) > 1e-8 && abs(ts - curT) < 0.05)
        {
            if (m.evals < FAR_MAP_CAP)
            {
                float fs;
                float rs;
                float ss;
                far_march_sample(m, ts, fs, rs, ss);
                if (abs(fs) <= 1e-5)
                {
                    if (abs(ss) > 1e-4)
                    {
                        float t2 = ts - fs / ss;
                        if (t2 > tStart && t2 < tHiOut && abs(t2 - ts) < 2e-3)
                            return far_mapped_finish(ray, center, t2, rho, distance, rot, map, t, nrm, face);
                    }
                    return far_mapped_finish(ray, center, ts, rho, distance, rot, map, t, nrm, face);
                }
                if (f * fs <= 0.0)
                {
                    hasBracket = true;
                    if (curT < ts)
                    {
                        ba = curT;
                        bb = ts;
                        bfa = f;
                        bfb = fs;
                    }
                    else
                    {
                        ba = ts;
                        bb = curT;
                        bfa = fs;
                        bfb = f;
                    }
                }
                else if (abs(fs) < 1e-3 && abs(ss) > 1e-4)
                {
                    float t2 = ts - fs / ss;
                    if (t2 > tStart && t2 < tHiOut && abs(t2 - ts) < 2e-3)
                        return far_mapped_finish(ray, center, t2, rho, distance, rot, map, t, nrm, face);
                }
            }
            else if (abs(f) < 1e-3)
            {
                return far_mapped_finish(ray, center, ts, rho, distance, rot, map, t, nrm, face);
            }
        }
    }
    if (hasBracket)
    {
        float tHit = far_regula(m, ba, bb, bfa, bfb);
        return far_mapped_finish(ray, center, tHit, rho, distance, rot, map, t, nrm, face);
    }
    float prevT = tStart;
    float prevF = fEnter;
    int left = FAR_MAP_CAP - m.evals;
    if (left < 0)
        left = 0;
    [loop]
    for (int i = 1; i <= left; i++)
    {
        if (!(m.evals < FAR_MAP_CAP))
            break;
        float tn = tStart + (tHiOut - tStart) * (float(i) / float(left));
        float ft;
        float rt;
        float st;
        far_march_sample(m, tn, ft, rt, st);
        if (abs(ft) <= FAR_MAP_FTOL)
            return far_mapped_finish(ray, center, tn, rho, distance, rot, map, t, nrm, face);
        if (prevF * ft <= 0.0)
        {
            float tHit = far_regula(m, prevT, tn, prevF, ft);
            return far_mapped_finish(ray, center, tHit, rho, distance, rot, map, t, nrm, face);
        }
        prevT = tn;
        prevF = ft;
    }
    return false;
}

#ifndef FAR_NO_MAPPED
// Air shell for a ray that missed the datum. The direction from the centre
// at the ray's closest approach is the limb's body-space sample: rSurf is
// the surface radius there, and the shell is
// [rSurf, rSurf + max(air/distance, px)]. A shell on the hi sphere floats
// off a lower silhouette. False when the ray misses that shell, including
// a degenerate approach straight through the centre.
bool far_mapped_limb(
    float3 ray, float3 center, float facing, float rho, float distance,
    float4 rot, uint map, float airThick, float px,
    out float rSurf, out float outer, out float s)
{
    rSurf = 0.0;
    outer = 0.0;
    s = length(cross(ray, center));
    // Closest approach is behind the camera when facing is not forward.
    if (!(distance > 0.0) || !(facing > 0.0))
        return false;
    float maxOff = asfloat(farMaps[map * 8u + 3u]);
    float rhoHi = rho + maxOff / distance;
    float shell = max(airThick / distance, px);
    // The local surface is at most the hi sphere, so a miss of that outer
    // bound misses every local shell and can skip the datum sample.
    if (!(rhoHi > 0.0 && s < rhoHi + shell))
        return false;
    float3 fromCenter = ray * facing - center;
    if (!(dot(fromCenter, fromCenter) > 1e-16))
        return false;
    float3 bodyDir = far_inv_rotate(rot, normalize(fromCenter));
    rSurf = rho + far_datum(map, bodyDir) / distance;
    outer = rSurf + shell;
    return rSurf > 0.0 && s < outer;
}
#endif

// Colour of the one additive body `i` emitted. The caller has already decided
// the term is in front of the final hit. Same formulas as the old array stash.
float3 far_add_rgb(uint i, float3 ray, float px, float flip)
{
    float4 dirRho = farTable[0].body[i].dir_rho;
    float4 atmo = farTable[0].body[i].atmosphere;
    float rho = dirRho.w;
    float shape = atmo.w;
    float3 center = dirRho.xyz;
    float3 air = atmo.rgb;
    float distance = farTable[0].body[i].albedo2.w;
    float facing = dot(ray, center);
#ifndef FAR_NO_ROUNDED
    bool rounded = shape > 2.5 && shape < 3.5;
#endif
#ifndef FAR_NO_MAPPED
    bool mapped = shape > 3.5 && shape < 4.5;
#else
    bool mapped = false;
#endif
    bool sphere = shape > 0.5 && shape < 1.5;
    // A sub-pixel body is a point blob, except a mapped body, whose datum is
    // the shape. This is the same gate pass 1 used to set the bit.
    float ang = (rho * 2.0) / px;
    if (!mapped && ang < 0.7 && facing > 0.0)
    {
        float s = length(cross(ray, center));
        float blob = exp(-4.0 * (s * s) / (px * px));
        float phase = saturate(0.5 - 0.5 * dot(center, frame.sun_dir_elev.xyz) * flip);
        float gain = max(phase, 0.15);
        uint seed = farTable[0].body[i].seed.x;
        float tw = 0.75 + 0.25 * (float(seed & 255u) * (1.0 / 255.0));
        return farTable[0].body[i].albedo0.rgb * (frame.light.rgb * gain + frame.ambient_glow.rgb) * (blob * tw);
    }
    if (sphere)
    {
        float s = length(cross(ray, center));
        float width = max(rho * 0.05, px * 1.25);
        float u = saturate((s - rho) / width);
        float rim = 1.0 - u;
        return air * (rim * rim);
    }
#ifndef FAR_NO_ROUNDED
    if (rounded)
    {
        float pExp = asfloat(farTable[0].body[i].seed.y);
        float pRim = max(pExp, 2.0);
        float rhoB = rho * pow(3.0, 0.5 - 1.0 / pRim);
        float s = length(cross(ray, center));
        float width = max(rhoB * 0.05, px * 1.25);
        float u = saturate((s - rhoB) / width);
        float rim = 1.0 - u;
        return air * (rim * rim);
    }
#endif
#ifndef FAR_NO_MAPPED
    if (mapped)
    {
        uint mapPlus = farTable[0].body[i].seed.z;
        float airThick = farTable[0].body[i].albedo1.w;
        if (mapPlus == 0u || mapPlus > 8u || !(distance > 0.0))
            return float3(0.0, 0.0, 0.0);
        uint map = mapPlus - 1u;
        float4 rot = farTable[0].body[i].rot;
        float rSurf;
        float outer;
        float s;
        if (!far_mapped_limb(
                ray, center, facing, rho, distance, rot, map, airThick, px, rSurf, outer, s))
            return float3(0.0, 0.0, 0.0);
        // Shell on the local surface, at least one pixel thick. Chord is in
        // world units. The denominator stays at least a pixel so the
        // widened shell does not go opaque. Daylight follows far_lit.
        float outerChord = sqrt(max(outer * outer - s * s, 0.0));
        float chordN = s >= rSurf
            ? 2.0 * outerChord
            : 2.0 * (outerChord - sqrt(max(rSurf * rSurf - s * s, 0.0)));
        float chord = chordN * distance;
        float denom = max(airThick, px * distance);
        float trans = exp(-FAR_AIR_K * chord / denom);
        float3 limbN = normalize(ray * facing - center);
        float daylight = saturate(dot(limbN, frame.sun_dir_elev.xyz) * flip);
        return air * ((1.0 - trans) * frame.light.rgb * daylight);
    }
#endif
#ifndef FAR_NO_CUBE
    float rhoRim = rho * 1.035 + px * 0.5;
    float tRim;
    float3 nRim;
    uint faceRim;
    float4 rot = farTable[0].body[i].rot;
    if (rhoRim < 1.0 && far_ray_cube(ray, center, rhoRim, rot, tRim, nRim, faceRim))
    {
        float fres = 1.0 - saturate(dot(nRim, -ray));
        fres = fres * fres * fres;
        return air * fres;
    }
#endif
    return float3(0.0, 0.0, 0.0);
}

// `px` is the pixel angle, from fwidth in the entry point (uniform flow).
// `pix` is SV_Position.xy (framebuffer pixels, origin top-left).
// `inscatter` is the viewer's sky radiance along `ray` (no sun disc, moon or stars).
// Pass 1 only searches. Albedo and lighting run once, for the nearest hit, so a
// pixel a single body covers matches the old per-hit shade. Hits are ordered by
// world depth `t * distance` and rims and point blobs by `facing * distance`;
// the normalised `t` is kept for shading. The inner sphere's far wall is the
// background; anything with a smaller world depth composites over it. A mapped
// body is kind 4: one shade without the fresnel air rim, then the terrain fog
// in place of far_through_air, then the air shell along the view chord above
// the hit. Its miss-ray limb is an additive term on the local surface, not
// the hi sphere.
//
// Pass 2 accumulates those additive rims and blobs. This matches the old
// dynamically indexed `float4 addTerm[32]` (that array spilled to local
// memory). Pass 1 sets `addMask` when the array would have pushed, gated on
// the running depth and the geometric test: a point blob gates on
// `tPoint * distance`, a cube rim on `tRim * distance`, and a sphere, rounded
// or mapped rim on `facing * distance`. Every term's stored depth is
// `facing * distance`. Pass 2 adds the term only when its bit is set and that
// stored depth is still in front of the final hit. The running depth only
// shrinks and each body emits at most one term, so the scalar mask plus the
// final compare keeps and drops the same terms the array did. A final-depth
// test alone would not: a blob or cube rim can pass its gate and then sit
// behind the final hit on `tPoint` / `tRim` while its stored depth is still
// in front, or fail the gate against the running hit and still have a stored
// depth in front of the final hit.
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

    // Pass 2 walks the same bits. Pass 1 consumes `mask`.
    uint bits = mask;
    float flip = frame.light.w * 2.0 - 1.0;
    // World units. `bestT` is the winning hit's normalised parameter.
    float bestDepth = 1e30;
    float bestT = 0.0;
    uint bestI = 0u;
    uint bestFace = 0u;
    // 0 none, 1 sphere (coverage lerp), 2 inner wall, 3 opaque replace, 4 mapped.
    // Sphere and inner normals are rebuilt after the loop. A cube, rounded or
    // mapped hit still carries its normal: repeating that march would inline
    // it twice. A sphere-only compile has no such store, so the float3 is not
    // live across the search.
    uint bestKind = 0u;
#if !defined(FAR_NO_ROUNDED) || !defined(FAR_NO_MAPPED) || !defined(FAR_NO_CUBE)
#define FAR_KEEP_NORMAL
    float3 bestN = float3(0.0, 0.0, 0.0);
#endif
    // Bit i: body i would have pushed one additive (see far_bodies).
    uint addMask = 0u;

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
#ifndef FAR_NO_ROUNDED
        bool rounded = shape > 2.5 && shape < 3.5;
#else
        bool rounded = false;
#endif
#ifndef FAR_NO_MAPPED
        bool mapped = shape > 3.5 && shape < 4.5;
#else
        bool mapped = false;
#endif
        // Divergent shapes stay out of a flattened branch. A warp of spheres
        // does not issue the cube, rounded or mapped march.
        [branch]
        if (inner)
        {
            if (!(rho > 1.0))
                continue;
        }
        else if (mapped)
        {
            // Reference rho stays below 1, but the hi sphere can contain the
            // camera, so the outside-sphere gate does not apply.
            if (!(rho > 0.0))
                continue;
        }
        else if (!(rho > 0.0 && rho < 1.0))
            continue;
        float3 center = dirRho.xyz;
        float facing = dot(ray, center);
        // World distance. Normalised `t` and `facing` live in this body's unit space.
        float distance = farTable[0].body[i].albedo2.w;
        float facingDepth = facing * distance;
        // Inside the sphere every direction hits the far wall, including away from the centre.
        // A rounded body can contain the camera in its enclosing sphere while the camera is
        // still outside the solid, so a ray away from the centre may meet the bulge.
        // A mapped body can do the same once the datum's hi sphere contains the camera.
        if (!inner && !rounded && !mapped && facing <= 0.0)
            continue;

        float3 air = atmo.rgb;
        bool hasAir = dot(air, air) > 0.0;
        if (inner)
        {
            float t = 0.0;
            float3 nrm = float3(0.0, 0.0, 0.0);
            if (far_ray_inner(ray, center, rho, facing, t, nrm) && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestT = t;
                bestI = i;
                bestFace = 0u;
                bestKind = 2u;
            }
            continue;
        }
        bool sphere = shape > 0.5 && shape < 1.5;
        // Angular diameter in pixels. Under ~0.7 px the disc is a soft point.
        // A mapped body keeps its ray test: the datum is the shape, not a blob.
        float ang = (rho * 2.0) / px;
        [branch]
        if (!mapped && ang < 0.7 && facing > 0.0)
        {
            float tPoint = 1.0 - rho;
            // Gate on the point depth. The stored depth is still `facing * distance`.
            if (tPoint * distance < bestDepth)
                addMask |= 1u << i;
            continue;
        }

        float t = 0.0;
        float3 nrm = float3(0.0, 0.0, 0.0);
        uint face = 0u;
        bool hit = false;
        [branch]
        if (sphere)
        {
            float s = length(cross(ray, center));
            if (s <= rho)
            {
                hit = far_ray_sphere(ray, center, rho, facing, t, nrm);
                if (hit && t * distance < bestDepth)
                {
                    bestDepth = t * distance;
                    bestT = t;
                    bestI = i;
                    bestFace = 0u;
                    bestKind = 1u;
                }
            }
            else if (hasAir)
            {
                float width = max(rho * 0.05, px * 1.25);
                if (s < rho + width && facingDepth < bestDepth)
                    addMask |= 1u << i;
            }
        }
#ifndef FAR_NO_ROUNDED
        else if (rounded)
        {
            float pExp = asfloat(farTable[0].body[i].seed.y);
            float4 rot = farTable[0].body[i].rot;
            hit = far_ray_rounded(ray, center, rho, rot, pExp, facing, t, nrm, face);
            if (hit && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestT = t;
                bestI = i;
                bestFace = face;
                bestN = nrm;
                bestKind = 3u;
            }
            else if (!hit && hasAir)
            {
                float pRim = max(pExp, 2.0);
                float rhoB = rho * pow(3.0, 0.5 - 1.0 / pRim);
                float s = length(cross(ray, center));
                float width = max(rhoB * 0.05, px * 1.25);
                if (s > rhoB && s < rhoB + width && facingDepth < bestDepth)
                    addMask |= 1u << i;
            }
        }
#endif
#ifndef FAR_NO_MAPPED
        else if (mapped)
        {
            uint mapPlus = farTable[0].body[i].seed.z;
            float horizon = farTable[0].body[i].albedo0.w;
            float airThick = farTable[0].body[i].albedo1.w;
            // Above the horizon plane before any intersection, including the limb.
            if (mapPlus == 0u || mapPlus > 8u || !(distance > 0.0) || dot(ray, -center) > horizon)
                continue;
            uint map = mapPlus - 1u;
            float4 rot = farTable[0].body[i].rot;
            hit = far_ray_mapped(ray, center, rho, distance, rot, map, t, nrm, face);
            if (hit && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestT = t;
                bestI = i;
                bestFace = face;
                bestN = nrm;
                bestKind = 4u;
            }
            else if (!hit && airThick > 0.0 && facing > 0.0 && facingDepth < bestDepth)
            {
                // The colour is rebuilt in pass 2. The bit records that the
                // local shell test passed against the running depth.
                float rSurf;
                float outer;
                float s;
                if (far_mapped_limb(
                        ray, center, facing, rho, distance, rot, map, airThick, px, rSurf, outer, s))
                    addMask |= 1u << i;
            }
        }
#endif
#ifndef FAR_NO_CUBE
        else
        {
            float4 rot = farTable[0].body[i].rot;
            hit = far_ray_cube(ray, center, rho, rot, t, nrm, face);
            if (hit && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestT = t;
                bestI = i;
                bestFace = face;
                bestN = nrm;
                bestKind = 3u;
            }
            else if (!hit && hasAir)
            {
                float rhoRim = rho * 1.035 + px * 0.5;
                float tRim;
                float3 nRim;
                uint faceRim;
                // Gate on the rim-cube depth. The stored depth is `facing * distance`.
                if (rhoRim < 1.0 && far_ray_cube(ray, center, rhoRim, rot, tRim, nRim, faceRim) && tRim * distance < bestDepth)
                    addMask |= 1u << i;
            }
        }
#endif
    }

    float3 add = float3(0.0, 0.0, 0.0);
    // Pass 1 set addMask for every term the old array would have pushed.
    // Walking only those bits is the same test, without a miss per other body.
    bits &= addMask;
    [loop]
    while (bits != 0u)
    {
        uint i = firstbitlow(bits);
        bits &= bits - 1u;
        float distance = farTable[0].body[i].albedo2.w;
        float facingDepth = dot(ray, farTable[0].body[i].dir_rho.xyz) * distance;
        // Stored depth of every additive. Terms the running gate rejected
        // never have their bit set, so this is the array's post-loop test.
        if (facingDepth < bestDepth)
            add += far_add_rgb(i, ray, px, flip);
    }

    // Sphere and inner normals match far_ray_sphere / far_ray_inner exactly,
    // including the un-normalised scale. Coverage uses the same cross length
    // pass 1 tested. Heavy shapes already stored bestN.
#ifndef FAR_KEEP_NORMAL
    float3 bestN = float3(0.0, 0.0, 0.0);
#endif
    float bestCov = 1.0;
    if (bestKind == 1u || bestKind == 2u)
    {
        float4 wonRho = farTable[0].body[bestI].dir_rho;
        float wonR = wonRho.w;
        float3 wonC = wonRho.xyz;
        if (bestKind == 1u)
        {
            bestN = (ray * bestT - wonC) / wonR;
            float s = length(cross(ray, wonC));
            bestCov = saturate(0.5 + (wonR - s) / px);
        }
        else
            bestN = (wonC - ray * bestT) / wonR;
    }

    if (bestKind != 0u)
    {
        float4 dirRho = farTable[0].body[bestI].dir_rho;
        float rho = dirRho.w;
        float3 center = dirRho.xyz;
        float4 rot = farTable[0].body[bestI].rot;
        float3 air = farTable[0].body[bestI].atmosphere.rgb;
        float3 bodyP = far_inv_rotate(rot, ray * bestT - center);
        if (bestKind == 4u)
        {
#ifndef FAR_NO_MAPPED
            // Terrain fog, the same factor mesh3d applies, replaces far_through_air.
            uint mapPlus = farTable[0].body[bestI].seed.z;
            float distance = farTable[0].body[bestI].albedo2.w;
            float3 alb = far_face(bestI, bestFace);
            if (mapPlus >= 1u && mapPlus <= 8u)
            {
                uint map = mapPlus - 1u;
                uint landed = farMaps[map * 8u + 4u];
                uint albedoSize = farMaps[map * 8u + 5u];
                if (albedoSize != 0u && (landed & (1u << bestFace)) != 0u)
                {
                    float rad = length(bodyP);
                    float3 bodyDir = rad > 1e-8 ? bodyP / rad : float3(0.0, 1.0, 0.0);
                    float rWorld = rad * distance;
                    float footprint = px * bestT * distance;
                    float texel = (2.0 * rWorld) / float(albedoSize);
                    float lod = 0.0;
                    if (texel > 1e-20)
                        lod = log2(max(footprint / texel, 1e-20));
                    alb = far_cube(map, bodyDir, clamp(lod, 0.0, 16.0));
                }
            }
            float3 lit = far_lit(alb, bestN, ray, air, rho, false, false);
            float3 shaded = lit;
            if (frame.horizon.w > 0.0)
            {
                float fog = 1.0 - exp(-bestT * distance * frame.horizon.w);
                shaded = lerp(lit, inscatter, fog);
            }
            // Air shell on the sphere of radius r_s(hit) + air. The chord is
            // the view segment inside that shell, in world units (0 when the
            // ray never enters). Daylight matches the mapped limb.
            float airThick = farTable[0].body[bestI].albedo1.w;
            float trans = 1.0;
            float facing = dot(ray, center);
            if (airThick > 0.0 && distance > 0.0)
            {
                float rSurf = length(ray * bestT - center);
                float outer = rSurf + airThick / distance;
                float tEnter;
                float tExit;
                if (far_sphere_roots(facing, outer, tEnter, tExit))
                {
                    float chord = (bestT - max(tEnter, 0.0)) * distance;
                    if (chord > 0.0)
                        trans = exp(-FAR_AIR_K * chord / airThick);
                }
            }
            col = shaded;
            if (trans < 1.0)
            {
                float3 approach = ray * facing - center;
                float3 radial = ray * bestT - center;
                float3 limbN = dot(approach, approach) > 1e-16
                    ? normalize(approach)
                    : (dot(radial, radial) > 1e-16 ? normalize(radial) : -ray);
                float daylight = saturate(dot(limbN, frame.sun_dir_elev.xyz) * flip);
                col = shaded * trans + air * (1.0 - trans) * daylight;
            }
#endif
        }
        else
        {
            bool sphereAlb = bestKind == 1u || bestKind == 2u;
            float3 alb = far_albedo(bestI, bodyP, bestFace, sphereAlb, rho);
            bool core = bestKind == 2u;
            float3 lit = far_lit(alb, bestN, ray, air, rho, core, true);
            float3 shaded = core ? lit : far_through_air(lit, inscatter, rho);
            if (bestKind == 1u)
                col = lerp(col, shaded, bestCov);
            else
                col = shaded;
        }
    }
    col += add;
    return col;
}

#endif
