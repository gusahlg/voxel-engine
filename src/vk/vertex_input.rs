/// Vertex layout generation. A vertex struct declared via [`vertex_struct!`]
/// automatically derives its GPU binding and attribute descriptions from its
/// fields: location = declaration order, offset from `offset_of!`, format
/// from field type (via [`VertexFormat`]). A field type without a
/// `VertexFormat` impl fails to compile.
///
/// Shader layouts (`layout(location = N) in <glsl_type>`) must still match
/// the format chosen here. This is validated at pipeline creation by
/// validation layers, not at compile time.
use ash::vk;

/// Maps a vertex field's Rust type to the GPU vertex-fetch format.
pub trait VertexFormat {
    const FORMAT: vk::Format;
}

impl VertexFormat for [f32; 2] {
    const FORMAT: vk::Format = vk::Format::R32G32_SFLOAT;
}
impl VertexFormat for [f32; 3] {
    const FORMAT: vk::Format = vk::Format::R32G32B32_SFLOAT;
}
impl VertexFormat for [f32; 4] {
    const FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
}
impl VertexFormat for [u8; 4] {
    const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
}
impl VertexFormat for [u32; 2] {
    const FORMAT: vk::Format = vk::Format::R32G32_UINT;
}

/// GPU input layout for a vertex struct. Implemented by [`vertex_struct!`].
pub trait VertexInput {
    const STRIDE: u32;
    const ATTRIBUTES: &'static [vk::VertexInputAttributeDescription];

    fn binding() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: Self::STRIDE,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }
}

/// Defines a `#[repr(C)]`, `Pod`/`Zeroable` vertex struct and its `VertexInput` impl.
/// Locations, offsets, and formats are derived from field types and declaration order.
macro_rules! vertex_struct {
    (
        $(#[$m:meta])*
        $v:vis struct $name:ident {
            $($(#[$fm:meta])* $fv:vis $field:ident : $ty:ty),+ $(,)?
        }
    ) => {
        $(#[$m])*
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, ::bytemuck::Pod, ::bytemuck::Zeroable)]
        $v struct $name {
            $($(#[$fm])* $fv $field: $ty),+
        }

        impl $crate::vk::vertex_input::VertexInput for $name {
            const STRIDE: u32 = ::std::mem::size_of::<Self>() as u32;
            const ATTRIBUTES: &'static [::ash::vk::VertexInputAttributeDescription] = &{
                // Placeholder locations; actual locations assigned by the loop.
                let mut attrs = [$(
                    ::ash::vk::VertexInputAttributeDescription {
                        binding: 0,
                        location: 0,
                        format: <$ty as $crate::vk::vertex_input::VertexFormat>::FORMAT,
                        offset: ::std::mem::offset_of!($name, $field) as u32,
                    }
                ),+];
                let mut i = 0;
                while i < attrs.len() {
                    attrs[i].location = i as u32;
                    i += 1;
                }
                attrs
            };
        }
    };
}
pub(crate) use vertex_struct;
