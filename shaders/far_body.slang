// Far-body impostors for the sky pass. Ray tests match the CPU mirror in
// src/far_body/mirror.rs.
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

// Set 0, binding 3. MAX_FAR_MAPS headers of FAR_MAP_HEADER_STRIDE uints, then
// the datum floats. Header of map m, at element m * FAR_MAP_HEADER_STRIDE: g,
// float-offset into the datum array, min offset bits, max offset bits, 6-bit
// landed mask, albedo edge, 0, 0. Datum float i of map m is element
// FAR_MAP_DATUM_BASE + offset + i. offset = m * 6 * 65 * 65.
[[vk::binding(3, 0)]]
StructuredBuffer<uint> farMaps;
static const uint FAR_MAP_DATUM_BASE = MAX_FAR_MAPS * FAR_MAP_HEADER_STRIDE;

// Set 0, binding 4: one cube per map. Empty slots are a 1x1 dummy.
// Sampled through far_cube so the index is a constant (no dynamic indexing).
[[vk::binding(4, 0)]]
SamplerCube farAlbedo[MAX_FAR_MAPS];

// Mapped-only from here to far_datum_d. FAR_NO_MAPPED compiles it out.
#ifndef FAR_NO_MAPPED
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
    uint g = farMaps[map * FAR_MAP_HEADER_STRIDE];
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
    uint off = farMaps[map * FAR_MAP_HEADER_STRIDE + 1u];
    uint base = FAR_MAP_DATUM_BASE + off + face * g * g;
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
#endif

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
    // A switch, not FACE_NORMAL[face]. A dynamic index of that array is a
    // function-scope copy, and that copy is the sky fragment's local memory.
    nrm = far_rotate(rot, far_face_normal(face));
    return true;
}

// Cube-face unit normal. Same order as FACE_NORMAL. Inlined as a branch so
// the fragment does not copy a 6-vector table into local memory.
float3 far_face_normal(uint face)
{
    if (face == 0u)
        return float3(1.0, 0.0, 0.0);
    if (face == 1u)
        return float3(-1.0, 0.0, 0.0);
    if (face == 2u)
        return float3(0.0, 1.0, 0.0);
    if (face == 3u)
        return float3(0.0, -1.0, 0.0);
    if (face == 4u)
        return float3(0.0, 0.0, 1.0);
    return float3(0.0, 0.0, -1.0);
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

// Sample at tHit when it is outside the solid. The face is the gradient's
// dominant axis. Shade rebuilds the world normal with one far_lp at the hit,
// so this march does not carry a float3 out.
bool far_round_at(float3 o, float3 d, float tHit, float p, float rho, out float t, out uint face)
{
    t = 0.0;
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
    face = far_dom_face(g);
    return true;
}

// Superellipsoid of face radius rho. Bound is rho * 3^(1/2 - 1/p), padded a hair
// so the entry stays outside the solid. A graze of that sphere, where it meets
// the body, falls back to the sphere point rather than opening a hole.
// No world normal: a carried float3 across this loop spilled. Shade rebuilds
// the gradient, or the enclosing-sphere radial when face bit 3 is set.
bool far_ray_rounded(float3 ray, float3 center, float rho, float4 rot, float exponent, float facing, out float t, out uint face)
{
    t = 0.0;
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
    float s;
    float3 g;
    far_lp(o, p, s, g);
    if (t0 == 0.0 && s <= rho)
        return false;

    float tC = clamp(facing, t0, tFar);
    far_lp(o + d * tC, p, s, g);
    if (s > rho)
    {
        if (s <= rho * 1.001 && tC > 0.0 && far_round_at(o, d, tC, p, rho, t, face))
            return true;
        if (tNear > 0.0 && sd <= rhoB * 0.02)
        {
            far_lp(o + d * tNear, p, s, g);
            if (s >= rho && s <= rho * 1.01)
            {
                t = tNear;
                // far_dom_face(inv_rotate(rot, (ray*t-center)/rhoB)). A positive
                // scale does not change the dominant axis.
                face = far_dom_face(o + d * tNear) | 8u;
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
        far_lp(o + d * lo, p, sL, g);
        if (sL >= rho && sL <= rho * (1.0 + 2.0e-5))
            break;
        if (hi - lo < 1e-6)
            break;
        float slope = dot(g, d);
        bool moved = false;
        if (slope < -1e-8)
        {
            float tN = lo - (sL - rho) / slope;
            if (tN > lo && tN < hi)
            {
                float sN;
                far_lp(o + d * tN, p, sN, g);
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
            far_lp(o + d * hi, p, sH, g);
            float tF = 0.5 * (lo + hi);
            float span = sL - sH;
            if (abs(span) > 1e-12)
            {
                float guess = lo + (hi - lo) * (sL - rho) / span;
                if (guess > lo && guess < hi)
                    tF = guess;
            }
            float sF;
            far_lp(o + d * tF, p, sF, g);
            if (sF >= rho)
                lo = tF;
            else
                hi = tF;
        }
    }
    if (far_round_at(o, d, lo, p, rho, t, face))
        return true;
    if (tNear > 0.0)
    {
        far_lp(o + d * tNear, p, s, g);
        if (s >= rho)
        {
            t = tNear;
            face = far_dom_face(o + d * tNear) | 8u;
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
// exp(-0.2) of the surface (~82% transmittance).
static const float FAR_AIR_K = 0.2;

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

// Mapped-only from here to far_mapped_air_chord.
#ifndef FAR_NO_MAPPED
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

// Five-sample fixed-point march. The only mapped intersection a GPU module
// compiles. The bracket march stays a CPU test reference (`ray_mapped`);
// the host mirror of this function is `ray_mapped_fast`, and the two stay
// sample-for-sample identical. No fallback.
// Datum samples the march may spend. The analytic normal is one more fetch
// after the march, outside this cap. Matches MAPPED_EVAL_CAP and
// MAPPED_F_TOL in src/far_body/mirror.rs.
static const int FAR_MAP_CAP = 5;
static const float FAR_MAP_FTOL = 1e-8;

// Value only. Derivatives stay in far_datum_d for the analytic normal.
// Components, not the three basis vectors: those nine floats are live
// across the bilinear loads, which is the regula sample's hottest point.
float far_datum_value(uint map, float3 d)
{
    uint g = farMaps[map * FAR_MAP_HEADER_STRIDE];
    if (g < 2u)
        return 0.0;
    uint face = far_dom_face(d);
    // Same dots as far_face_basis. d is dead once the three components exist.
    float den;
    float du;
    float dv;
    if (face == 0u)
    {
        den = d.x;
        du = -d.y;
        dv = d.z;
    }
    else if (face == 1u)
    {
        den = -d.x;
        du = d.y;
        dv = d.z;
    }
    else if (face == 2u)
    {
        den = d.y;
        du = d.x;
        dv = d.z;
    }
    else if (face == 3u)
    {
        den = -d.y;
        du = d.x;
        dv = -d.z;
    }
    else if (face == 4u)
    {
        den = d.z;
        du = d.x;
        dv = -d.y;
    }
    else
    {
        den = -d.z;
        du = d.x;
        dv = d.y;
    }
    if (abs(den) < 1e-8)
        return 0.0;
    float xi = (4.0 / 3.14159265358979323846) * far_atan_approx(du / den);
    float eta = (4.0 / 3.14159265358979323846) * far_atan_approx(dv / den);
    float scale = 0.5 * float(g - 1u);
    float u = clamp((xi + 1.0) * scale, 0.0, float(g - 1u));
    float v = clamp((eta + 1.0) * scale, 0.0, float(g - 1u));
    uint i0 = min(uint(floor(u)), g - 1u);
    uint j0 = min(uint(floor(v)), g - 1u);
    uint i1 = min(i0 + 1u, g - 1u);
    uint j1 = min(j0 + 1u, g - 1u);
    float fu = u - float(i0);
    float fv = v - float(j0);
    uint off = farMaps[map * FAR_MAP_HEADER_STRIDE + 1u];
    uint base = FAR_MAP_DATUM_BASE + off + face * g * g;
    // Two loads, then the lerp, so the four samples are not live together.
    float a = asfloat(farMaps[base + j0 * g + i0]);
    a = a + (asfloat(farMaps[base + j0 * g + i1]) - a) * fu;
    float b = asfloat(farMaps[base + j1 * g + i0]);
    b = b + (asfloat(farMaps[base + j1 * g + i1]) - b) * fu;
    return a + (b - a) * fv;
}

// Reloaded at each sample so a centre or rotation from an earlier sample is
// not kept live across the datum chart.
float4 far_body_dir(uint bodyIndex)
{
    return farTable[0].body[bodyIndex].dir_rho;
}

float4 far_body_rot(uint bodyIndex)
{
    return farTable[0].body[bodyIndex].rot;
}

// Radial slope u·ray at parameter t. Same f32 ops as far_march_sample.
float far_radial_slope(float3 ray, uint bodyIndex, float t)
{
    float3 center = far_body_dir(bodyIndex).xyz;
    float3 p = ray * t - center;
    float rad = length(p);
    float3 uWorld = p / rad;
    return dot(uWorld, ray);
}

// (f, r, df/dt). Slope is u·ray. Rays inside the lo-sphere disc match the
// robust march to 1e-6 relative hit distance with that term alone.
// Centre and rotation are reloaded here and dropped before the chart.
void far_march_sample(
    float3 ray, uint bodyIndex, float rho, float distance, uint map,
    inout int evals, float t, out float f, out float r, out float slope)
{
    evals += 1;
    f = -rho;
    r = 0.0;
    slope = 0.0;
    float facing;
    float centerSq;
    float raySq;
    float rad;
    float3 uBody;
    {
        float3 center = far_body_dir(bodyIndex).xyz;
        float4 rot = far_body_rot(bodyIndex);
        float3 p = ray * t - center;
        rad = length(p);
        if (!(rad < 1e-8))
        {
            float3 uWorld = p / rad;
            // These do not read the datum. Finishing them first drops p,
            // uWorld, centre and rotation before the chart.
            slope = dot(uWorld, ray);
            facing = dot(ray, center);
            centerSq = dot(center, center);
            raySq = dot(ray, ray);
            uBody = far_inv_rotate(rot, uWorld);
        }
    }
    if (rad < 1e-8)
        return;
    float value = far_datum_value(map, uBody);
    r = rho + value / distance;
    // |p| - r cancels in f32 when both are near 1. Same |p|^2 - r^2 form
    // as the robust march, so a hit a few metres out still agrees.
    float oneMinusR = (1.0 - rho) - value / distance;
    float diffSq = t * t * raySq - 2.0 * t * facing
        + (centerSq - 1.0) + oneMinusR * (1.0 + r);
    f = diffSq / (rad + r);
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

// Another Newton step would move t by less than 1e-8 relative, inside the
// 1e-6 agreement band of the robust march. A shallow
// slope falls back to FAR_MAP_FTOL: a graze stays small across a wide interval.
bool far_fast_settled(float t, float f, float slope)
{
    if (abs(slope) > 1e-4)
        return abs(f / slope) <= 1e-8 * max(abs(t), 1e-12);
    return abs(f) <= FAR_MAP_FTOL;
}

// Illinois regula falsi. The retained endpoint's weight is halved so a stuck
// side does not burn the remaining samples. A small residual takes a Newton
// step from the radial slope; with no sample left, sphere curvature corrects it.
float far_regula(
    float3 ray, uint bodyIndex, float rho, float distance, uint map,
    inout int evals, float a0, float b0, float fa0, float fb0)
{
    float a = a0;
    float b = b0;
    float fa = fa0;
    float fb = fb0;
    float wa = fa;
    float wb = fb;
    [loop]
    while (evals < FAR_MAP_CAP)
    {
        float den = wb - wa;
        // Relative, not absolute: a 1e-7 bracket is a large fraction of a
        // hit that sits a few metres in front of a planet-sized body.
        float span = max(max(abs(a), abs(b)), 1e-6);
        if (abs(den) < 1e-20 || abs(b - a) <= 1e-8 * span)
            break;
        float tn = b - wb * (b - a) / den;
        if (!(tn > min(a, b) && tn < max(a, b)))
            break;
        float ft;
        float rt;
        float st;
        far_march_sample(ray, bodyIndex, rho, distance, map, evals, tn, ft, rt, st);
        if (far_fast_settled(tn, ft, st))
            return tn;
        if (abs(st) > 1e-4 && abs(ft) < 2e-3)
        {
            float t2 = tn - ft / st;
            // lo/hi are min/max of the bracket. Recomputing them after the
            // sample keeps those two floats out of the chart.
            if (t2 > min(a, b) && t2 < max(a, b) && abs(t2 - tn) < 5e-3)
            {
                if (!(evals < FAR_MAP_CAP))
                {
                    float3 center = far_body_dir(bodyIndex).xyz;
                    float3 p = ray * tn - center;
                    float rad = length(p);
                    float srad = dot(p / rad, ray);
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
                far_march_sample(ray, bodyIndex, rho, distance, map, evals, t2, f2, r2, s2);
                if (far_fast_settled(t2, f2, s2))
                    return t2;
                if (abs(s2) > 1e-4 && abs(f2) < 1e-3)
                {
                    // Halley. Accept only inside the settle band; a wider
                    // step goes back into the bracket.
                    // The first sample's slope, recomputed so it is not live
                    // across this second datum fetch.
                    float stR = far_radial_slope(ray, bodyIndex, tn);
                    float d2 = abs(t2 - tn) > 1e-6 ? (s2 - stR) / (t2 - tn) : 0.0;
                    float hden = s2 - f2 * d2 / (2.0 * s2);
                    float t3 = abs(hden) > 1e-4 ? (t2 - f2 / hden) : (t2 - f2 / s2);
                    float lo = min(a, b);
                    float hi = max(a, b);
                    if (t3 > lo && t3 < hi && abs(f2 / s2) <= 1e-8 * max(abs(t2), 1e-12))
                        return t3;
                }
                if (ft * f2 <= 0.0)
                {
                    if (tn < t2)
                    {
                        a = tn;
                        b = t2;
                        fa = ft;
                        fb = f2;
                    }
                    else
                    {
                        a = t2;
                        b = tn;
                        fa = f2;
                        fb = ft;
                    }
                    wa = fa;
                    wb = fb;
                    continue;
                }
                if (abs(f2) < abs(ft))
                {
                    if (fa * f2 <= 0.0)
                    {
                        b = t2;
                        fb = f2;
                        wb = f2;
                    }
                    else
                    {
                        a = t2;
                        fa = f2;
                        wa = f2;
                    }
                    continue;
                }
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
    if (!(evals < FAR_MAP_CAP))
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

// Outward normal of the bilinear patch at uWorld. On the +X face centre,
// d(uHat)/d(xi) × d(uHat)/d(eta) points inward, so the outward normal is
// dp/d(eta) × dp/d(xi). A flat patch is exactly radial.
// Chart point q = n + tu*tx + tv*ty. Axis bases, so this is a permutation
// of (1, tx, ty), not three live basis vectors.
float3 far_chart_q(uint face, float tx, float ty)
{
    if (face == 0u)
        return float3(1.0, -tx, ty);
    if (face == 1u)
        return float3(-1.0, tx, ty);
    if (face == 2u)
        return float3(tx, 1.0, ty);
    if (face == 3u)
        return float3(tx, -1.0, -ty);
    if (face == 4u)
        return float3(tx, -ty, 1.0);
    return float3(tx, ty, -1.0);
}

// dq/d(xi) = tu * dtx, dq/d(eta) = tv * dty. One vector at a time.
float3 far_chart_dq(uint face, float scale, bool eta)
{
    float s = eta ? 0.0 : scale;
    float e = eta ? scale : 0.0;
    if (face == 0u)
        return float3(0.0, -s, e);
    if (face == 1u)
        return float3(0.0, s, e);
    if (face == 2u)
        return float3(s, 0.0, e);
    if (face == 3u)
        return float3(s, 0.0, -e);
    if (face == 4u)
        return float3(s, -e, 0.0);
    return float3(s, e, 0.0);
}

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
    float den;
    float du;
    float dv;
    if (face == 0u)
    {
        den = u.x;
        du = -u.y;
        dv = u.z;
    }
    else if (face == 1u)
    {
        den = -u.x;
        du = u.y;
        dv = u.z;
    }
    else if (face == 2u)
    {
        den = u.y;
        du = u.x;
        dv = u.z;
    }
    else if (face == 3u)
    {
        den = -u.y;
        du = u.x;
        dv = -u.z;
    }
    else if (face == 4u)
    {
        den = u.z;
        du = u.x;
        dv = -u.y;
    }
    else
    {
        den = -u.z;
        du = u.x;
        dv = u.y;
    }
    if (abs(den) < 1e-8)
        return uWorld;
    float quarter = 0.7853981633974483;
    float xi = (4.0 / 3.14159265358979323846) * far_atan_approx(du / den);
    float eta = (4.0 / 3.14159265358979323846) * far_atan_approx(dv / den);
    float tx = tan(xi * quarter);
    float ty = tan(eta * quarter);
    float dtx = quarter * (1.0 + tx * tx);
    float dty = quarter * (1.0 + ty * ty);
    float3 q = far_chart_q(face, tx, ty);
    float s = length(q);
    if (!(s > 1e-8))
        return uWorld;
    float3 uHat = q / s;
    float3 dq = far_chart_dq(face, dtx, false);
    float3 duD = dq / s - uHat * (dot(uHat, dq) / s);
    float3 dpDxi = uHat * dr_dxi + duD * r;
    dq = far_chart_dq(face, dty, true);
    duD = dq / s - uHat * (dot(uHat, dq) / s);
    float3 dpDeta = uHat * dr_deta + duD * r;
    float3 nBody = cross(dpDeta, dpDxi);
    if (dot(nBody, nBody) < 1e-20)
        return uWorld;
    return far_rotate(rot, normalize(nBody));
}

bool far_mapped_finish(
    float3 ray, uint bodyIndex, float tHit, float rho, float distance, uint map,
    out float t, out uint face)
{
    t = 0.0;
    face = 0u;
    if (!(tHit > 0.0))
        return false;
    float3 center = far_body_dir(bodyIndex).xyz;
    float4 rot = far_body_rot(bodyIndex);
    float3 p = ray * tHit - center;
    float rad = length(p);
    if (rad < 1e-8)
        return false;
    t = tHit;
    // The analytic normal is one shade-time fetch. Returning it here kept a
    // float3 live on every exit of the march.
    face = far_dom_face(far_inv_rotate(rot, p / rad));
    return true;
}

// Datum surface. Fixed-point pulls from the lo-sphere entrance
// (t <- ray ∩ sphere(r(dir))), then Illinois regula on the bracket those
// pulls form. A ray that misses the lo sphere walks in from the hi entrance
// and finishes a shallow graze with one Newton step. At most FAR_MAP_CAP
// datum samples. The analytic normal is one fetch at shade time: inlining
// the patch derivative into every return blew the register budget. The
// caller has already applied the horizon gate.
bool far_ray_mapped(
    float3 ray, uint bodyIndex, float rho, float distance, uint map,
    out float t, out uint face)
{
    t = 0.0;
    face = 0u;
    float minOff = asfloat(farMaps[map * FAR_MAP_HEADER_STRIDE + 2u]);
    float maxOff = asfloat(farMaps[map * FAR_MAP_HEADER_STRIDE + 3u]);
    float rhoLo = rho + minOff / distance;
    float rhoHi = rho + maxOff / distance;
    float facing;
    {
        // Dies before the first datum sample. Later uses reload.
        float3 center = far_body_dir(bodyIndex).xyz;
        facing = dot(ray, center);
    }
    float tHiIn;
    float tHiOut;
    if (!far_sphere_roots(facing, rhoHi, tHiIn, tHiOut) || !(tHiOut > 0.0))
        return false;
    float tStart = max(tHiIn, 0.0);
    int evals = 0;
    float tLo;
    float tLoFar;
    bool loHit = far_sphere_roots(facing, rhoLo, tLo, tLoFar) && tLo > 0.0;
    if (loHit)
    {
        float bSpan = max(tLo, tStart);
        if (bSpan - tStart < 1e-5)
            return far_mapped_finish(ray, bodyIndex, tLo > 0.0 ? tLo : tStart, rho, distance, map, t, face);
        float f;
        float r;
        float slope;
        far_march_sample(ray, bodyIndex, rho, distance, map, evals, tLo, f, r, slope);
        if (far_fast_settled(tLo, f, slope))
            return far_mapped_finish(ray, bodyIndex, tLo, rho, distance, map, t, face);
        float curT = tLo;
        bool hasBracket = false;
        float ba = 0.0;
        float bb = 0.0;
        float bfa = 0.0;
        float bfb = 0.0;
        [loop]
        for (int step = 0; step < 2; step++)
        {
            if (!(evals < FAR_MAP_CAP))
                break;
            // The far root of a sphere that contains the camera is the back
            // of the planet. Step with the radial slope instead; that stays
            // on the near surface. A 1e-8 absolute stop froze hits near the
            // eye a step short of the root.
            float tNear;
            float tFar;
            float tn;
            bool haveNear = far_sphere_roots(facing, r, tNear, tFar) && tNear > tStart && tNear < tHiOut;
            if (haveNear)
                tn = tNear;
            else if (abs(slope) > 1e-4 && abs(f) < 0.05)
            {
                tn = curT - f / slope;
                if (!(tn > tStart && tn < tHiOut))
                    break;
            }
            else
                break;
            if (abs(tn - curT) <= 1e-12)
                break;
            if (hasBracket && (tn <= ba || tn >= bb))
                break;
            float ft;
            float rt;
            float st;
            far_march_sample(ray, bodyIndex, rho, distance, map, evals, tn, ft, rt, st);
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
            if (far_fast_settled(tn, ft, st))
                return far_mapped_finish(ray, bodyIndex, tn, rho, distance, map, t, face);
            curT = tn;
            f = ft;
            r = rt;
            slope = st;
        }
        // The camera sample catches an eye inside the body. A bracket whose
        // near end is already outside does not need it.
        float nearF = ba <= bb ? bfa : bfb;
        bool needEntrance = !hasBracket || (tStart == 0.0 && nearF < 0.0);
        if (needEntrance)
        {
            if (!(evals < FAR_MAP_CAP))
                return far_mapped_finish(ray, bodyIndex, curT, rho, distance, map, t, face);
            float fa;
            float ra;
            float sa;
            far_march_sample(ray, bodyIndex, rho, distance, map, evals, tStart, fa, ra, sa);
            // Same camera clamp as the robust march: |f| <= 1e-5 at t == 0
            // is clearance, not a root, and finishing there drops the ray.
            if (tStart == 0.0 && abs(fa) <= 1e-5)
            {
                if (fa <= 0.0)
                    return false;
            }
            else if (abs(fa) <= 1e-5)
                return far_mapped_finish(ray, bodyIndex, tStart, rho, distance, map, t, face);
            else if (fa < 0.0)
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
                    return far_mapped_finish(ray, bodyIndex, curT, rho, distance, map, t, face);
                }
            }
        }
        if (!hasBracket)
            return far_mapped_finish(ray, bodyIndex, curT, rho, distance, map, t, face);
        float tHit = far_regula(ray, bodyIndex, rho, distance, map, evals, ba, bb, bfa, bfb);
        return far_mapped_finish(ray, bodyIndex, tHit, rho, distance, map, t, face);
    }
    float f0;
    float r0;
    float slope0;
    far_march_sample(ray, bodyIndex, rho, distance, map, evals, tStart, f0, r0, slope0);
    if (f0 < -1e-5)
        return false;
    if (abs(f0) <= 1e-5 && tStart > 0.0)
        return far_mapped_finish(ray, bodyIndex, tStart, rho, distance, map, t, face);
    if (!(tStart > 0.0) && f0 <= 0.0)
        return false;
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
    while (evals + 1 < FAR_MAP_CAP)
    {
        float tn;
        if (!far_positive_root(facing, r, tn) || !(tn > tStart && tn <= tHiOut))
            break;
        if (abs(tn - curT) <= 1e-8)
            break;
        float ft;
        float rt;
        float st;
        far_march_sample(ray, bodyIndex, rho, distance, map, evals, tn, ft, rt, st);
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
        if (far_fast_settled(tn, ft, st))
            return far_mapped_finish(ray, bodyIndex, tn, rho, distance, map, t, face);
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
            if (evals < FAR_MAP_CAP)
            {
                float fs;
                float rs;
                float ss;
                far_march_sample(ray, bodyIndex, rho, distance, map, evals, ts, fs, rs, ss);
                if (abs(fs) <= 1e-5)
                {
                    if (abs(ss) > 1e-4)
                    {
                        float t2 = ts - fs / ss;
                        if (t2 > tStart && t2 < tHiOut && abs(t2 - ts) < 2e-3)
                            return far_mapped_finish(ray, bodyIndex, t2, rho, distance, map, t, face);
                    }
                    return far_mapped_finish(ray, bodyIndex, ts, rho, distance, map, t, face);
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
                        return far_mapped_finish(ray, bodyIndex, t2, rho, distance, map, t, face);
                }
            }
            else if (abs(f) < 1e-3)
            {
                return far_mapped_finish(ray, bodyIndex, ts, rho, distance, map, t, face);
            }
        }
    }
    if (hasBracket)
    {
        float tHit = far_regula(ray, bodyIndex, rho, distance, map, evals, ba, bb, bfa, bfb);
        return far_mapped_finish(ray, bodyIndex, tHit, rho, distance, map, t, face);
    }
    float prevT = tStart;
    float prevF = fEnter;
    int left = FAR_MAP_CAP - evals;
    if (left < 0)
        left = 0;
    [loop]
    for (int i = 1; i <= left; i++)
    {
        if (!(evals < FAR_MAP_CAP))
            break;
        float tn = tStart + (tHiOut - tStart) * (float(i) / float(left));
        float ft;
        float rt;
        float st;
        far_march_sample(ray, bodyIndex, rho, distance, map, evals, tn, ft, rt, st);
        if (far_fast_settled(tn, ft, st))
            return far_mapped_finish(ray, bodyIndex, tn, rho, distance, map, t, face);
        if (prevF * ft <= 0.0)
        {
            float tHit = far_regula(ray, bodyIndex, rho, distance, map, evals, prevT, tn, prevF, ft);
            return far_mapped_finish(ray, bodyIndex, tHit, rho, distance, map, t, face);
        }
        prevT = tn;
        prevF = ft;
    }
    return false;
}

// True when `ray` is strictly above body `bodyIndex`'s horizon table.
// No table (or an unused slot) is never above. The frame matches the host:
// up = normalise(centre → eye), the same east/north/atan2 as far_table.slang.
bool far_above_horizon(uint bodyIndex, float3 ray, float3 center)
{
    uint slot = 2u;
    if (farTable[0].horizon_id.x == bodyIndex)
        slot = 0u;
    else if (farTable[0].horizon_id.y == bodyIndex)
        slot = 1u;
    else
        return false;
    float3 up = -center;
    float upLen = length(up);
    if (!(upLen > 0.0))
        return false;
    up /= upLen;
    float mu = dot(ray, up);
    float3 refv = abs(up.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 east = normalize(cross(refv, up));
    float3 north = cross(up, east);
    float3 horiz = ray - up * mu;
    float az = atan2(dot(horiz, east), dot(horiz, north));
    if (az < 0.0)
        az += 6.283185307179586;
    float binsPerRad = float(FAR_HORIZON_BINS) / 6.283185307179586;
    uint bin = min(uint(floor(az * binsPerRad)), FAR_HORIZON_BINS - 1u);
    return mu > farTable[0].horizon_sin[slot * FAR_HORIZON_BINS + bin];
}

// A mapped body whose map id or distance cannot be marched or lit. The
// horizon half of the mapped gate stays inline at each site: as a second
// helper it regrouped the || chain and moved the SPIR-V.
bool far_map_bad(uint mapPlus, float distance)
{
    return mapPlus == 0u || mapPlus > MAX_FAR_MAPS || !(distance > 0.0);
}

// World metres from the closest approach. The ends are the cap chord.
// Zero is the approach itself. Scalar steps only: a function-scope table
// is the sky fragment's local memory. Lockstep with `limb_delta_m`.
float far_limb_delta(int i)
{
    if (i <= 0)
        return -1.0e30;
    if (i == 1)
        return -1200000.0;
    if (i == 2)
        return -400000.0;
    if (i == 3)
        return 0.0;
    if (i == 4)
        return 400000.0;
    if (i == 5)
        return 1200000.0;
    if (i == 6)
        return 2400000.0;
    return 1.0e30;
}

// Fraction of u in [0, 1] where qa u² + qb u + qc <= 0. Same stable
// quadratic as `limb_inside_fraction`. `sign` is not used: Slang's sign(0)
// is 0 and the host's signum(0) is 1.
float far_limb_inside_fraction(float qa, float qb, float qc)
{
    if (abs(qa) <= 1e-20)
    {
        if (abs(qb) <= 1e-20)
            return qc <= 0.0 ? 1.0 : 0.0;
        float u = -qc / qb;
        return qb > 0.0 ? clamp(u, 0.0, 1.0) : clamp(1.0 - u, 0.0, 1.0);
    }
    float disc = qb * qb - 4.0 * qa * qc;
    if (!(disc >= 0.0))
        return qa > 0.0 ? 0.0 : 1.0;
    float sd = sqrt(disc);
    float q = qb >= 0.0 ? -0.5 * (qb + sd) : -0.5 * (qb - sd);
    float u0;
    float u1;
    if (abs(q) <= 1e-20)
    {
        float inv = 0.5 / qa;
        u0 = (-qb - sd) * inv;
        u1 = (-qb + sd) * inv;
    }
    else
    {
        u0 = q / qa;
        u1 = qc / q;
    }
    float lo = min(u0, u1);
    float hi = max(u0, u1);
    if (qa > 0.0)
    {
        float a = clamp(lo, 0.0, 1.0);
        float b = clamp(hi, 0.0, 1.0);
        return max(b - a, 0.0);
    }
    return clamp(lo, 0.0, 1.0) + clamp(1.0 - hi, 0.0, 1.0);
}

// Normalised length of [t0, t1] on which |p| is at or below the air radius
// linearly interpolated from r0 to r1. The quadratic matches the measured
// |p| at both ends and the analytic curvature between them, so a segment
// that ends inside the air is not dropped when the analytic radius misses
// that end. Below the datum counts.
float far_limb_segment(float t0, float t1, float rad0, float rad1, float r0, float r1)
{
    float dt = t1 - t0;
    if (!(dt > 1e-20))
        return 0.0;
    float a0 = max(r0, 0.0);
    float a1 = max(r1, 0.0);
    float dr = a1 - a0;
    float qa = dt * dt - dr * dr;
    float qc = (rad0 - a0) * (rad0 + a0);
    float g1 = (rad1 - a1) * (rad1 + a1);
    float qb = g1 - qa - qc;
    float frac = far_limb_inside_fraction(qa, qb, qc);
    return frac == frac ? frac * dt : 0.0;
}

// World length of the air `ray` crosses. 0 when the eye's altitude above
// the bilinear datum in the eye's direction is below `airThick`, or the
// cap-sphere chord never enters R + datum + air. The horizon table still
// widens its shell by a pixel; this chord does not. A miss whose sample
// dips below the datum counts, so a graze the march missed stays lit.
float far_mapped_air_chord(
    float3 ray, float3 center, float rho, float distance, float4 rot, uint map, float airThick)
{
    if (!(distance > 0.0) || !(airThick > 0.0))
        return 0.0;
    float facing = dot(ray, center);
    if (!(facing > 0.0))
        return 0.0;
    float3 radial = -center;
    float eyeRad = length(radial);
    if (!(eyeRad > 1e-8))
        return 0.0;
    float hEye = far_datum_value(map, far_inv_rotate(rot, radial / eyeRad));
    if (distance - rho * distance - hEye < airThick)
        return 0.0;
    float maxOff = asfloat(farMaps[map * FAR_MAP_HEADER_STRIDE + 3u]);
    float rhoCap = rho + (maxOff + airThick) / distance;
    float disc = facing * facing - (1.0 - rhoCap * rhoCap);
    if (!(rhoCap > 0.0) || !(disc >= 0.0))
        return 0.0;
    float sd = sqrt(disc);
    float tFar = facing + sd;
    if (!(tFar > 0.0))
        return 0.0;
    float t0 = max(facing - sd, 0.0);
    if (!(tFar > t0))
        return 0.0;
    float ca = clamp(facing, t0, tFar);
    float sumT = 0.0;
    float prevT = 0.0;
    float prevRad = 0.0;
    float prevR = 0.0;
    // Eight samples. Lockstep with `LIMB_SAMPLES` in src/far_body/mirror.rs.
    [loop]
    for (int i = 0; i < 8; i++)
    {
        float t = clamp(ca + far_limb_delta(i) / distance, t0, tFar);
        float3 p = ray * t - center;
        float rad = length(p);
        float rAir;
        if (!(rad > 1e-8))
        {
            rad = 0.0;
            rAir = 1.0;
        }
        else
        {
            float h = far_datum_value(map, far_inv_rotate(rot, p / rad));
            rAir = rho + (h + airThick) / distance;
            if (rAir < 0.0)
                rAir = 0.0;
        }
        if (i > 0)
            sumT += far_limb_segment(prevT, t, prevRad, rad, prevR, rAir);
        prevT = t;
        prevRad = rad;
        prevR = rAir;
    }
    if (!(sumT > 0.0) || !(sumT < 1e20))
        return 0.0;
    return sumT * distance;
}

// Shade of mapped body `i`'s winning hit at normalised `t` on cube face
// `face`: the loop's kind-4 arm and the single-mapped variant. One shade
// without the fresnel air rim, then the terrain fog in place of
// far_through_air, then the air shell along the view chord above the hit.
// Runs after the march, so the chart derivative is not live inside a datum
// fetch, and the normal runs before air, inscatter and the body point exist.
float3 far_shade_mapped(float3 ray, float px, uint i, float t, uint face)
{
    float rho = farTable[0].body[i].dir_rho.w;
    uint mapPlus = farTable[0].body[i].seed.z;
    float distance = farTable[0].body[i].albedo2.w;
    float4 rot = farTable[0].body[i].rot;
    float3 uN;
    {
        float3 center = farTable[0].body[i].dir_rho.xyz;
        float3 bodyP = far_inv_rotate(rot, ray * t - center);
        float radN = length(bodyP);
        uN = radN > 1e-8 ? bodyP / radN : float3(0.0, 1.0, 0.0);
    }
    uint mapN = (mapPlus >= 1u && mapPlus <= MAX_FAR_MAPS) ? mapPlus - 1u : 0u;
    float3 bestN = far_mapped_normal(rot, uN, rho, distance, mapN);
    float3 center = farTable[0].body[i].dir_rho.xyz;
    float3 bodyP = far_inv_rotate(rot, ray * t - center);
    float3 air = farTable[0].body[i].atmosphere.rgb;
    float3 inscatter = sky_radiance(ray, frame);
    float3 alb = far_face(i, face);
    if (mapPlus >= 1u && mapPlus <= MAX_FAR_MAPS)
    {
        uint map = mapPlus - 1u;
        uint landed = farMaps[map * FAR_MAP_HEADER_STRIDE + 4u];
        uint albedoSize = farMaps[map * FAR_MAP_HEADER_STRIDE + 5u];
        if (albedoSize != 0u && (landed & (1u << face)) != 0u)
        {
            float rad = length(bodyP);
            float3 bodyDir = rad > 1e-8 ? bodyP / rad : float3(0.0, 1.0, 0.0);
            // Footprint of this pixel on the surface. A graze stretches
            // it by 1/|ray·n|; the floor keeps a silhouette finite.
            // Texel is the face-centre size, (π/2 R) / albedo edge.
            float ndot = abs(dot(ray, bestN));
            if (ndot < 0.05)
                ndot = 0.05;
            float footprint = px * t * distance / ndot;
            float texel = (1.5707963267948966 * rho * distance) / float(albedoSize);
            float lod = 0.0;
            if (texel > 1e-20)
                lod = log2(max(footprint / texel, 1e-20));
            float mipMax = log2(float(albedoSize));
            alb = far_cube(map, bodyDir, clamp(lod, 0.0, mipMax));
        }
    }
    float3 lit = far_lit(alb, bestN, ray, air, rho, false, false);
    float3 shaded = lit;
    // Terrain fog, the same factor mesh3d applies.
    if (frame.horizon.w > 0.0)
    {
        float fog = 1.0 - exp(-t * distance * frame.horizon.w);
        shaded = lerp(lit, inscatter, fog);
    }
    // Air shell on the sphere of radius r_s(hit) + air. The chord is the
    // view segment inside that shell, in world units (0 when the ray never
    // enters). Daylight matches far_limb_mapped.
    float airThick = farTable[0].body[i].albedo1.w;
    float trans = 1.0;
    float facing = dot(ray, center);
    if (airThick > 0.0 && distance > 0.0)
    {
        float rSurf = length(ray * t - center);
        float outer = rSurf + airThick / distance;
        float tEnter;
        float tExit;
        if (far_sphere_roots(facing, outer, tEnter, tExit))
        {
            float chord = (t - max(tEnter, 0.0)) * distance;
            if (chord > 0.0)
                trans = exp(-FAR_AIR_K * chord / airThick);
        }
    }
    float3 col = shaded;
    if (trans < 1.0)
    {
        float flip = frame.light.w * 2.0 - 1.0;
        float3 approach = ray * facing - center;
        float3 radial = ray * t - center;
        float3 limbN = dot(approach, approach) > 1e-16
            ? normalize(approach)
            : (dot(radial, radial) > 1e-16 ? normalize(radial) : -ray);
        float daylight = saturate(dot(limbN, frame.sun_dir_elev.xyz) * flip);
        col = shaded * trans + air * (1.0 - trans) * daylight;
    }
    return col;
}

// Miss-ray limb of mapped body `i`, for far_add_rgb and the single-mapped
// variant: the air `ray` crosses, in world units. Not a pixel-widened
// sphere: the denominator is the shell thickness. Daylight follows far_lit.
// Zero for a bad map or distance, or when the ray crosses no air.
float3 far_limb_mapped(float3 ray, uint i)
{
    float4 dirRho = farTable[0].body[i].dir_rho;
    float rho = dirRho.w;
    float3 center = dirRho.xyz;
    float3 air = farTable[0].body[i].atmosphere.rgb;
    float distance = farTable[0].body[i].albedo2.w;
    float facing = dot(ray, center);
    uint mapPlus = farTable[0].body[i].seed.z;
    float airThick = farTable[0].body[i].albedo1.w;
    if (far_map_bad(mapPlus, distance))
        return float3(0.0, 0.0, 0.0);
    uint map = mapPlus - 1u;
    float4 rot = farTable[0].body[i].rot;
    float chord = far_mapped_air_chord(ray, center, rho, distance, rot, map, airThick);
    if (!(chord > 0.0))
        return float3(0.0, 0.0, 0.0);
    float trans = exp(-FAR_AIR_K * chord / airThick);
    float3 approach = ray * facing - center;
    float3 limbN = dot(approach, approach) > 1e-16 ? normalize(approach) : -ray;
    float flip = frame.light.w * 2.0 - 1.0;
    float daylight = saturate(dot(limbN, frame.sun_dir_elev.xyz) * flip);
    return air * ((1.0 - trans) * frame.light.rgb * daylight);
}
#endif

// Cone reject for kept body `i`, before its 160-byte record is read: runs
// `reject` (a continue or a return) when the ray is outside the cone. The
// bound is the body's sine plus 3 px, which covers the 1.25 px rim, the
// 0.5 px AA edge and the blob tail. `f <= 0` also drops the rounded air rim's
// ghost ring at the antipode (that rim has no facing test). The cone dies in
// the block: a float4 carried into the march is four registers at the regula
// sample. A macro: a bool helper, even [ForceInline], added a phi and a
// second branch at every site.
#define FAR_CONE_REJECT(i, ray, px, reject) \
    { \
        float4 c = farTable[0].cone[i]; \
        if (c.w >= 0.0) \
        { \
            float f = dot(ray, c.xyz); \
            float3 x = cross(ray, c.xyz); \
            float lim = c.w + 3.0 * px; \
            if (f <= 0.0 || dot(x, x) > lim * lim) \
                reject; \
        } \
    }

#ifdef FAR_MAPPED_SOLO
// One mapped body, the lowest set bit of the tile mask. The CPU draws this
// variant only when that bit is the mask. No body loop and no shape dispatch,
// so the march does not carry a composite across a datum fetch.

// The loop variants' far_tile_mask with the count check folded in. It keeps
// its own returns: built on far_tile_mask, this variant's SPIR-V changed.
uint far_solo_mask(float2 pix)
{
    uint n = min(farTable[0].header.x, MAX_FAR_BODIES);
    if (n == 0u)
        return 0u;
    uint live = n == 32u ? 0xffffffffu : ((1u << n) - 1u);
    uint tilePx = farTable[0].header.y;
    uint tilesX = farTable[0].header.z;
    uint tilesY = farTable[0].header.w;
    if (tilePx == 0u || tilesX == 0u || tilesY == 0u)
        return live;
    float2 p = max(pix, float2(0.0, 0.0));
    uint tx = min(uint(p.x) / tilePx, tilesX - 1u);
    uint ty = min(uint(p.y) / tilePx, tilesY - 1u);
    return farTable[0].tile_mask[ty * tilesX + tx] & live;
}

// xyz is the shaded surface, or the limb (zero when the ray misses the air).
// w is 1 when the surface replaces the background and 0 when the caller adds
// xyz onto the background it builds after this returns. The background is not
// an argument: a float3 carried into the march stays live on every fetch.
float4 far_mapped_solo(float3 ray, float px, float2 pix)
{
    uint mask = far_solo_mask(pix);
    if (mask == 0u)
        return float4(0.0, 0.0, 0.0, 0.0);
    uint i = firstbitlow(mask);
    FAR_CONE_REJECT(i, ray, px, return float4(0.0, 0.0, 0.0, 0.0));

    float rho;
    float distance;
    uint map;
    {
        rho = farTable[0].body[i].dir_rho.w;
        if (!(rho > 0.0))
            return float4(0.0, 0.0, 0.0, 0.0);
        distance = farTable[0].body[i].albedo2.w;
        uint mapPlus = farTable[0].body[i].seed.z;
        float horizon = farTable[0].body[i].albedo0.w;
        float3 center = farTable[0].body[i].dir_rho.xyz;
        if (far_map_bad(mapPlus, distance) || dot(ray, -center) > horizon
            || far_above_horizon(i, ray, center))
            return float4(0.0, 0.0, 0.0, 0.0);
        map = mapPlus - 1u;
    }

    float t = 0.0;
    uint face = 0u;
    bool hit = far_ray_mapped(ray, i, rho, distance, map, t, face);
    // bestDepth starts at 1e30. One body, so the hit replaces the sky when
    // its world depth wins that sentinel.
    if (hit && t * distance < 1e30)
        return float4(far_shade_mapped(ray, px, i, t, face), 1.0);
    if (!hit)
    {
        // Recomputed after the march. A copy from the horizon test would sit
        // live across every datum sample.
        float facing = dot(ray, far_body_dir(i).xyz);
        if (facing > 0.0 && facing * distance < 1e30)
        {
            float airThick = farTable[0].body[i].albedo1.w;
            float3 limbC = far_body_dir(i).xyz;
            float4 rot = far_body_rot(i);
            if (airThick > 0.0
                && far_mapped_air_chord(ray, limbC, rho, distance, rot, map, airThick) > 0.0)
                return float4(far_limb_mapped(ray, i), 0.0);
        }
    }
    return float4(0.0, 0.0, 0.0, 0.0);
}

// Source key for one mapped body. Same channels as far_sky_debug_rgb.
float3 far_sky_debug_rgb(float3 ray, float px, float2 pix)
{
    uint mask = far_solo_mask(pix);
    if (mask == 0u)
        return float3(0.25, 0.25, 0.25);
    uint i = firstbitlow(mask);
    FAR_CONE_REJECT(i, ray, px, return float3(0.25, 0.25, 0.25));
    float rho = farTable[0].body[i].dir_rho.w;
    if (!(rho > 0.0))
        return float3(0.25, 0.25, 0.25);
    float distance = farTable[0].body[i].albedo2.w;
    uint mapPlus = farTable[0].body[i].seed.z;
    float horizon = farTable[0].body[i].albedo0.w;
    float3 center = farTable[0].body[i].dir_rho.xyz;
    if (far_map_bad(mapPlus, distance))
        return float3(0.25, 0.25, 0.25);
    if (dot(ray, -center) > horizon || far_above_horizon(i, ray, center))
        return float3(0.25, 0.25, 1.0);
    uint map = mapPlus - 1u;
    float t = 0.0;
    uint face = 0u;
    bool hit = far_ray_mapped(ray, i, rho, distance, map, t, face);
    // The key is built after the march, so it is not live on the fetches.
    float3 dbg = float3(0.25, 0.25, 0.25);
    if (hit && t * distance < 1e30)
    {
        dbg.r = 1.0;
        return dbg;
    }
    if (!hit)
    {
        float facing = dot(ray, far_body_dir(i).xyz);
        if (facing > 0.0 && facing * distance < 1e30)
        {
            float airThick = farTable[0].body[i].albedo1.w;
            float3 limbC = far_body_dir(i).xyz;
            float4 rot = far_body_rot(i);
            if (airThick > 0.0
                && far_mapped_air_chord(ray, limbC, rho, distance, rot, map, airThick) > 0.0)
            {
                float3 limb = far_limb_mapped(ray, i);
                if (max(limb.x, max(limb.y, limb.z)) > 1e-6)
                    dbg.g = 1.0;
            }
        }
    }
    return dbg;
}
#else
// Kept bodies that may cover the tile under `pix`, for n = the kept count
// (1..MAX_FAR_BODIES). Bit k is kept body k, so walking the lowest set bit
// keeps the composite order. Every body is live when the host has no view
// (tile size 0). ForceInline: left to the SPIR-V optimizer, far_bodies and
// far_sky_debug_rgb shared one float2 zero constant and sky.frag changed.
[ForceInline]
uint far_tile_mask(uint n, float2 pix)
{
    // 32 is the word width: 1u << 32 is undefined.
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
    return mask;
}

// Colour of the one additive term body `i` emits: a point blob, or a sphere,
// rounded, mapped or cube rim. The caller has already decided the term is in
// front of the final hit.
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
        return far_limb_mapped(ray, i);
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
// The viewer's sky radiance is recomputed at shade time. Carrying that
// float3 through the march was live across every regula sample.
// Pass 1 only searches. Albedo and lighting run once, for the nearest hit, so a
// pixel a single body covers matches the old per-hit shade. Hits are ordered by
// world depth `t * distance` and rims and point blobs by `facing * distance`;
// the normalised `t` is kept for shading. The inner sphere's far wall is the
// background; anything with a smaller world depth composites over it. A mapped
// body is kind 4, shaded by far_shade_mapped. Its miss-ray limb is
// far_limb_mapped: the air that ray crosses, and nothing when the eye is
// already inside that air. Not the hi sphere.
//
// Pass 2 accumulates those additive rims and blobs. Each body emits at most
// one term. Pass 1 sets the body's `addMask` bit when the term passes its
// geometric test and the running depth: a point blob gates on
// `tPoint * distance`, a cube rim on `tRim * distance`, and a sphere, rounded
// or mapped rim on `facing * distance`. Every term's stored depth is
// `facing * distance`. Pass 2 adds the term only when its bit is set and that
// stored depth is still in front of the final hit. Both tests count; a
// final-depth test alone keeps other terms: a blob or cube rim can pass its
// gate and then sit behind the final hit on `tPoint` / `tRim` while its
// stored depth is still in front, or fail the gate against the running hit
// and still have a stored depth in front of the final hit.
float3 far_bodies(float3 ray, float3 col, float px, float2 pix)
{
    uint n = min(farTable[0].header.x, MAX_FAR_BODIES);
    if (n == 0u)
        return col;

    // One mask for the whole tile.
    uint mask = far_tile_mask(n, pix);
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
    // 0 none, 1 sphere (coverage lerp), 2 inner wall, 3 cube, 4 mapped,
    // 5 rounded gradient, 6 rounded enclosing-sphere fallback.
    // Every normal is rebuilt once after the loop, before the additive
    // pass. Carrying a float3 across the march or the limb spilled.
    uint bestKind = 0u;
    // Bit i: body i emits one additive term (pass 2, above).
    uint addMask = 0u;

    [loop]
    while (mask != 0u)
    {
        uint i = firstbitlow(mask);
        mask &= mask - 1u;
        // Reject before touching the 160-byte record.
        FAR_CONE_REJECT(i, ray, px, continue);

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
        // Each rim forms facing * distance at its gate; the product is not
        // stored. `facing` itself stays live across the mapped march, since
        // the limb gate after it reads it.
        float distance = farTable[0].body[i].albedo2.w;
        // Inside the sphere every direction hits the far wall, including away from the centre.
        // A rounded body can contain the camera in its enclosing sphere while the camera is
        // still outside the solid, so a ray away from the centre may meet the bulge.
        // A mapped body can do the same once the datum's hi sphere contains the camera.
        if (!inner && !rounded && !mapped && facing <= 0.0)
            continue;

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
        uint face = 0u;
        bool hit = false;
        [branch]
        if (sphere)
        {
            float s = length(cross(ray, center));
            if (s <= rho)
            {
                float3 nrm = float3(0.0, 0.0, 0.0);
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
            else if (dot(atmo.rgb, atmo.rgb) > 0.0)
            {
                float width = max(rho * 0.05, px * 1.25);
                if (s < rho + width && facing * distance < bestDepth)
                    addMask |= 1u << i;
            }
        }
#ifndef FAR_NO_ROUNDED
        else if (rounded)
        {
            float pExp = asfloat(farTable[0].body[i].seed.y);
            float4 rot = farTable[0].body[i].rot;
            hit = far_ray_rounded(ray, center, rho, rot, pExp, facing, t, face);
            if (hit && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestT = t;
                bestI = i;
                bestFace = face & 7u;
                // Bit 3 is the enclosing-sphere fallback. The gradient case
                // is far_lp at bestT, same as far_round_at.
                bestKind = (face & 8u) != 0u ? 6u : 5u;
            }
            else if (!hit && dot(atmo.rgb, atmo.rgb) > 0.0)
            {
                float pRim = max(pExp, 2.0);
                float rhoB = rho * pow(3.0, 0.5 - 1.0 / pRim);
                float s = length(cross(ray, center));
                float width = max(rhoB * 0.05, px * 1.25);
                if (s > rhoB && s < rhoB + width && facing * distance < bestDepth)
                    addMask |= 1u << i;
            }
        }
#endif
#ifndef FAR_NO_MAPPED
        else if (mapped)
        {
            uint mapPlus = farTable[0].body[i].seed.z;
            float horizon = farTable[0].body[i].albedo0.w;
            // Above the scalar horizon plane, or above the azimuth table,
            // before any intersection, including the limb. Air thickness is
            // loaded after the march so it is not live across the chart.
            if (far_map_bad(mapPlus, distance) || dot(ray, -center) > horizon
                || far_above_horizon(i, ray, center))
                continue;
            uint map = mapPlus - 1u;
            // Centre and rotation are reloaded inside the march. Keeping the
            // vectors here would hold them across every datum sample.
            hit = far_ray_mapped(ray, i, rho, distance, map, t, face);
            if (hit && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestT = t;
                bestI = i;
                bestFace = face;
                // Radial placeholder only. The analytic normal is fetched
                // once at shade time, so it is not live across this loop.
                bestKind = 4u;
            }
            else if (!hit && facing > 0.0 && facing * distance < bestDepth)
            {
                float airThick = farTable[0].body[i].albedo1.w;
                // The colour is rebuilt in pass 2. The bit records that the
                // ray crosses air and that crossing is in front of the hit.
                float3 limbC = far_body_dir(i).xyz;
                float4 rot = far_body_rot(i);
                if (airThick > 0.0
                    && far_mapped_air_chord(ray, limbC, rho, distance, rot, map, airThick) > 0.0)
                    addMask |= 1u << i;
            }
        }
#endif
#ifndef FAR_NO_CUBE
        else
        {
            float4 rot = farTable[0].body[i].rot;
            float3 nrm = float3(0.0, 0.0, 0.0);
            hit = far_ray_cube(ray, center, rho, rot, t, nrm, face);
            if (hit && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestT = t;
                bestI = i;
                bestFace = face;
                bestKind = 3u;
            }
            else if (!hit && dot(atmo.rgb, atmo.rgb) > 0.0)
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

    // Shade before the additive pass so the rebuilt normal is dead before
    // the limb. Sphere and inner normals match far_ray_sphere / far_ray_inner
    // exactly, including the un-normalised scale. Coverage uses the same
    // cross length pass 1 tested.
    float3 bestN = float3(0.0, 0.0, 0.0);
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
        float rho = farTable[0].body[bestI].dir_rho.w;
        if (bestKind == 4u)
        {
#ifndef FAR_NO_MAPPED
            col = far_shade_mapped(ray, px, bestI, bestT, bestFace);
#endif
        }
        else
        {
            // Reloaded here so the mapped shade above does not carry them.
            float3 center = farTable[0].body[bestI].dir_rho.xyz;
            float4 rot = farTable[0].body[bestI].rot;
            float3 air = farTable[0].body[bestI].atmosphere.rgb;
            float3 bodyP = far_inv_rotate(rot, ray * bestT - center);
            float3 inscatter = sky_radiance(ray, frame);
#ifndef FAR_NO_CUBE
            if (bestKind == 3u)
                bestN = far_rotate(rot, far_face_normal(bestFace));
#endif
#ifndef FAR_NO_ROUNDED
            if (bestKind == 5u || bestKind == 6u)
            {
                float p = clamp(asfloat(farTable[0].body[bestI].seed.y), 2.0, 32.0);
                if (bestKind == 6u)
                {
                    float rhoB = rho * pow(3.0, 0.5 - 1.0 / p) * (1.0 + 2.0e-4);
                    bestN = (ray * bestT - center) / rhoB;
                }
                else
                {
                    float3 o = far_inv_rotate(rot, -center);
                    float3 d = far_inv_rotate(rot, ray);
                    float sLp;
                    float3 gLp;
                    far_lp(o + d * bestT, p, sLp, gLp);
                    float glen = length(gLp);
                    if (glen > 1e-8)
                        bestN = far_rotate(rot, gLp / glen);
                }
            }
#endif
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

    float3 add = float3(0.0, 0.0, 0.0);
    // Pass 1 set addMask for every term that passed its gate. Walking only
    // those bits skips a miss per other body.
    // The normal is already consumed, so this limb does not keep a float3.
    bits &= addMask;
    [loop]
    while (bits != 0u)
    {
        uint i = firstbitlow(bits);
        bits &= bits - 1u;
        float distance = farTable[0].body[i].albedo2.w;
        float facingDepth = dot(ray, farTable[0].body[i].dir_rho.xyz) * distance;
        // Stored depth of every additive. Terms the running gate rejected
        // never have their bit set.
        if (facingDepth < bestDepth)
            add += far_add_rgb(i, ray, px, flip);
    }
    col += add;
    return col;
}

// Source key for VOXEL_SKY_DEBUG. The fragment calls this only from the
// uniform branch on sky_tangent.w, so the colour path does not run it. It is
// still inlined into the shipping module (each sky .spv is one function).
// Channels are 1 or 0.25 before the steep remap add in sky.frag:
//   R — a mapped surface hit won and would be shaded (bestKind == 4)
//   G — mapped miss-ray limbs that passed the depth test summed to a channel
//       above 1e-6 (the air the ray crosses; stars stay under it)
//   B — a mapped ray was skipped by far_above_horizon or by the per-body
//       gate dot(ray, -center) > albedo0.w, after the same cone and tile mask
//       as far_bodies
// The rejects match far_bodies, including a closer non-mapped hit clearing R.
#ifndef FAR_NO_MAPPED
float3 far_sky_debug_rgb(float3 ray, float px, float2 pix)
{
    float3 dbg = float3(0.25, 0.25, 0.25);
    uint n = min(farTable[0].header.x, MAX_FAR_BODIES);
    if (n == 0u)
        return dbg;
    uint mask = far_tile_mask(n, pix);
    if (mask == 0u)
        return dbg;

    uint bits = mask;
    float flip = frame.light.w * 2.0 - 1.0;
    float bestDepth = 1e30;
    uint bestKind = 0u;
    uint addMask = 0u;
    bool horizonSkip = false;

    [loop]
    while (mask != 0u)
    {
        uint i = firstbitlow(mask);
        mask &= mask - 1u;
        FAR_CONE_REJECT(i, ray, px, continue);

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
        bool mapped = shape > 3.5 && shape < 4.5;
        [branch]
        if (inner)
        {
            if (!(rho > 1.0))
                continue;
        }
        else if (mapped)
        {
            if (!(rho > 0.0))
                continue;
        }
        else if (!(rho > 0.0 && rho < 1.0))
            continue;
        float3 center = dirRho.xyz;
        float facing = dot(ray, center);
        float distance = farTable[0].body[i].albedo2.w;
        if (!inner && !rounded && !mapped && facing <= 0.0)
            continue;

        if (inner)
        {
            float t = 0.0;
            float3 nrm = float3(0.0, 0.0, 0.0);
            if (far_ray_inner(ray, center, rho, facing, t, nrm) && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestKind = 2u;
            }
            continue;
        }
        bool sphere = shape > 0.5 && shape < 1.5;
        float ang = (rho * 2.0) / px;
        [branch]
        if (!mapped && ang < 0.7 && facing > 0.0)
        {
            float tPoint = 1.0 - rho;
            if (tPoint * distance < bestDepth)
                addMask |= 1u << i;
            continue;
        }

        float t = 0.0;
        uint face = 0u;
        bool hit = false;
        [branch]
        if (sphere)
        {
            float s = length(cross(ray, center));
            if (s <= rho)
            {
                float3 nrm = float3(0.0, 0.0, 0.0);
                hit = far_ray_sphere(ray, center, rho, facing, t, nrm);
                if (hit && t * distance < bestDepth)
                {
                    bestDepth = t * distance;
                    bestKind = 1u;
                }
            }
            else if (dot(atmo.rgb, atmo.rgb) > 0.0)
            {
                float width = max(rho * 0.05, px * 1.25);
                if (s < rho + width && facing * distance < bestDepth)
                    addMask |= 1u << i;
            }
        }
#ifndef FAR_NO_ROUNDED
        else if (rounded)
        {
            float pExp = asfloat(farTable[0].body[i].seed.y);
            float4 rot = farTable[0].body[i].rot;
            hit = far_ray_rounded(ray, center, rho, rot, pExp, facing, t, face);
            if (hit && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestKind = (face & 8u) != 0u ? 6u : 5u;
            }
            else if (!hit && dot(atmo.rgb, atmo.rgb) > 0.0)
            {
                float pRim = max(pExp, 2.0);
                float rhoB = rho * pow(3.0, 0.5 - 1.0 / pRim);
                float s = length(cross(ray, center));
                float width = max(rhoB * 0.05, px * 1.25);
                if (s > rhoB && s < rhoB + width && facing * distance < bestDepth)
                    addMask |= 1u << i;
            }
        }
#endif
        else if (mapped)
        {
            uint mapPlus = farTable[0].body[i].seed.z;
            float horizon = farTable[0].body[i].albedo0.w;
            // Same short-circuit as far_bodies: a bad map or distance is not
            // a horizon skip. The gate and the azimuth table are.
            if (far_map_bad(mapPlus, distance))
                continue;
            if (dot(ray, -center) > horizon || far_above_horizon(i, ray, center))
            {
                horizonSkip = true;
                continue;
            }
            uint map = mapPlus - 1u;
            hit = far_ray_mapped(ray, i, rho, distance, map, t, face);
            if (hit && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestKind = 4u;
            }
            else if (!hit && facing > 0.0 && facing * distance < bestDepth)
            {
                float airThick = farTable[0].body[i].albedo1.w;
                float3 limbC = far_body_dir(i).xyz;
                float4 rot = far_body_rot(i);
                if (airThick > 0.0
                    && far_mapped_air_chord(ray, limbC, rho, distance, rot, map, airThick) > 0.0)
                    addMask |= 1u << i;
            }
        }
#ifndef FAR_NO_CUBE
        else
        {
            float4 rot = farTable[0].body[i].rot;
            float3 nrm = float3(0.0, 0.0, 0.0);
            hit = far_ray_cube(ray, center, rho, rot, t, nrm, face);
            if (hit && t * distance < bestDepth)
            {
                bestDepth = t * distance;
                bestKind = 3u;
            }
            else if (!hit && dot(atmo.rgb, atmo.rgb) > 0.0)
            {
                float rhoRim = rho * 1.035 + px * 0.5;
                float tRim;
                float3 nRim;
                uint faceRim;
                if (rhoRim < 1.0 && far_ray_cube(ray, center, rhoRim, rot, tRim, nRim, faceRim) && tRim * distance < bestDepth)
                    addMask |= 1u << i;
            }
        }
#endif
    }

    if (bestKind == 4u)
        dbg.r = 1.0;
    if (horizonSkip)
        dbg.b = 1.0;

    float3 limb = float3(0.0, 0.0, 0.0);
    bits &= addMask;
    [loop]
    while (bits != 0u)
    {
        uint i = firstbitlow(bits);
        bits &= bits - 1u;
        float distance = farTable[0].body[i].albedo2.w;
        float facingDepth = dot(ray, farTable[0].body[i].dir_rho.xyz) * distance;
        if (facingDepth < bestDepth)
        {
            float shape = farTable[0].body[i].atmosphere.w;
            if (shape > 3.5 && shape < 4.5)
                limb += far_add_rgb(i, ray, px, flip);
        }
    }
    if (max(limb.x, max(limb.y, limb.z)) > 1e-6)
        dbg.g = 1.0;
    return dbg;
}
#endif

#endif // FAR_MAPPED_SOLO
#endif
