// mesh3d.vert.slang — camera-relative vertices: a per-draw float4 offset is
// added to the input position before the view_proj multiply, so meshes can
// carry small local-space coordinates while the world spans huge distances.
// The offset lives in a per-draw storage buffer indexed by the raw Vulkan
// InstanceIndex builtin (= firstInstance + instance; every draw uses
// instance_count 1 with first_instance = its slot in the buffer). Slot 0 is
// reserved as a zero offset for immediate geometry (cubes, lines).
//
// SV_VulkanInstanceID, NOT SV_InstanceID: Slang lowers SV_InstanceID with
// D3D semantics (InstanceIndex - BaseInstance — verified via spirv-dis:
// OpISub of the two builtins, plus an unwanted DrawParameters capability),
// which would always read slot 0 under the first_instance-as-index scheme.
// SV_VulkanInstanceID lowers to the plain InstanceIndex builtin.
//
// Passes uv + color through; color.a carries the texture-array layer index.

[[vk::binding(0, 0)]]
StructuredBuffer<float4> draw_offsets;

struct VertexInput
{
    [[vk::location(0)]]
    float3 position;

    [[vk::location(1)]]
    float2 uv;

    [[vk::location(2)]]
    float4 color; // R8G8B8A8_UNORM, arrives as 0..1; a = layer / 255

    uint instance : SV_VulkanInstanceID; // raw SPIR-V InstanceIndex
};

struct VertexOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 uv;

    [[vk::location(1)]]
    float4 color;
};

struct PushConstants
{
    float4x4 view_proj;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    float3 offset = draw_offsets[input.instance].xyz;
    output.position = mul(pc.view_proj, float4(input.position + offset, 1.0));
    output.uv = input.uv;
    output.color = input.color;
    return output;
}
