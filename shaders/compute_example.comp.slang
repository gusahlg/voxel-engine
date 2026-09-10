// Example game compute job: 32-bit integer hash, wrapping arithmetic only.
// Each thread writes hash(input[i] ^ seed). No shared memory; index-guarded.
//
// Bindings (engine contract): 0 = read-only input storage, 1 = output storage.
// Push: seed, count. Workgroup: 64x1x1.

struct Push
{
    uint seed;
    uint count;
};
[[vk::push_constant]] Push pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;

// lowbias32 (Wellons): wrapping 32-bit integer hash. CPU parity uses the
// same ops (`wrapping_mul` / `wrapping_shr` / xor).
uint hash32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pc.count)
        return;
    dst[i] = hash32(src[i] ^ pc.seed);
}
