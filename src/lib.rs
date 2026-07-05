//! voxel_engine — a small, fast Vulkan 1.3 renderer for voxel games.
//!
//! Design goals: raylib-shaped polling API on top of winit + ash, unlit
//! textured rendering (24-byte vertices: pos + uv + rgb shade, with the
//! vertex alpha channel selecting a layer of a runtime-swappable block
//! texture array — see [`Engine::set_block_textures`]), device-local
//! suballocated mesh memory with same-frame uploads and deferred frees,
//! reversed-Z depth, automatic frustum culling, an embedded 8x8 font for the
//! 2D overlay, and runtime graphics settings (fullscreen, vsync, MSAA).
//!
//! Entry point: [`run`] with a per-frame callback over [`Engine`].

mod camera;
mod color;
mod engine;
mod font;
mod frame;
mod input;
mod mesh;
mod vk;

pub use camera::{Camera3D, Frustum, Z_NEAR, world_to_screen};
pub use color::Color;
pub use engine::{Config, Engine, run};
pub use frame::{Frame, Frame3D};
pub use glam::{DVec2, DVec3, Mat4, Vec2, Vec3};
pub use input::{Key, MouseButton};
pub use mesh::{MeshData, MeshHandle, Vertex};

/// Text metrics for the embedded font, usable without an [`Engine`].
pub use font::measure_text;
