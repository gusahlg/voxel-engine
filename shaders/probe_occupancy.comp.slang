// Occupancy skeleton: validate mega-kernel register pressure with representative live state.
// Not the real renderer—shape only. Output reduced to prevent dead-code elimination.

[[vk::binding(0, 0)]] RWStructuredBuffer<float> sink;

struct Params
{
    uint2 dim;
    uint steps;    // march loop length
    float t0;      // constant-fold prevention seed
};
[[vk::push_constant]] Params pc;

// 8-corner trilinear shape (representative live state).
float fake_trilinear(float3 p)
{
    float c[8];
    [unroll] for (uint i = 0; i < 8; ++i)
        c[i] = frac(sin(p.x * (i + 1) + p.y * 1.7 + p.z * 2.3) * 43758.5453);
    float3 f = frac(p);
    float x0 = lerp(c[0], c[1], f.x), x1 = lerp(c[2], c[3], f.x);
    float x2 = lerp(c[4], c[5], f.x), x3 = lerp(c[6], c[7], f.x);
    return lerp(lerp(x0, x1, f.y), lerp(x2, x3, f.y), f.z);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dim.x || gid.y >= pc.dim.y)
        return;

    // Primary-ray accumulators.
    float3 pos = float3(gid.xy, pc.t0);
    float3 dir = normalize(float3(0.3, 0.5, 1.0));
    float C = 0.0, T = 1.0, medium = 0.0;

    float light = fake_trilinear(pos * 0.01);
    // Shadow march loop with independent DDA state.
    float3 sp = pos; float3 sd = normalize(float3(1.0, 2.0, 0.5));
    float vis = 1.0;
    [loop] for (uint s = 0; s < pc.steps; ++s)
    {
        sp += sd;
        float occ = fake_trilinear(sp * 0.02);
        vis *= (occ > 0.5) ? 0.98 : 1.0;
        if (medium > 0.5)
            T *= exp(-0.35 * length(sd));
    }

    // 4-corner AO shape.
    float ao = 0.0;
    [unroll] for (uint k = 0; k < 4; ++k)
        ao += fake_trilinear(pos * 0.05 + float3(k, k * 2, k * 3));
    ao *= 0.25;

    C = light * vis * ao * T;
    sink[gid.y * pc.dim.x + gid.x] = C;
}
