// Subgroup QUAD ops test: verify quad shuffles work (zero-reference primitive).
// Host validates quad lane layout is consecutive groups of 4.

struct Out
{
    uint lane;     // subgroup lane index
    uint bcast0;   // quad broadcast from member 0
    uint swap_x;   // horizontal quad shuffle
    uint swap_y;   // vertical quad shuffle
};
[[vk::binding(0, 0)]] RWStructuredBuffer<Out> results;

struct Params { uint2 dim; };
[[vk::push_constant]] Params pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 gid: SV_DispatchThreadID)
{
    if (gid.x >= pc.dim.x || gid.y >= pc.dim.y)
        return;
    uint idx = gid.y * pc.dim.x + gid.x;
    uint lane = WaveGetLaneIndex();
    Out o;
    o.lane = lane;
    o.bcast0 = QuadReadLaneAt(lane, 0);
    o.swap_x = QuadReadAcrossX(lane);
    o.swap_y = QuadReadAcrossY(lane);
    results[idx] = o;
}
