// BDA three-site round-trip test: deref device address and write pattern.
// Requires SPIR-V physical-storage-buffer (spirv_1_5).

struct Params
{
    uint64_t addr; // target buffer device address
    uint n;        // element count
};
[[vk::push_constant]] Params pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.n)
        return;
    // Cast device address to uint pointer and write test pattern.
    uint *target = (uint *)pc.addr;
    target[i] = 0xC0DE0000u | i;
}
