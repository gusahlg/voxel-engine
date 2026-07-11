//! voxel_engine — a Vulkan 1.3 renderer for voxel games.
//!
//! Features: raylib-style API with winit + ash, unlit textured rendering
//! (8-byte packed mesh vertices, shader-derived UV + face shade),
//! runtime-swappable block texture arrays, device-local
//! mesh memory with same-frame uploads and deferred frees, reversed-Z depth,
//! frustum culling, an embedded 8x8 font for 2D overlay, and runtime graphics
//! settings (fullscreen, vsync, MSAA).
//!
//! Entry point: [`run`] with a per-frame callback over [`Engine`].

mod camera;
mod color;
mod engine;
mod font;
mod frame;
mod input;
mod mesh;
pub mod profile;
mod screenshot;
mod surface;
mod vk;

pub use camera::{
    Aspect, Camera3D, Frustum, Lens, WarpMap, WarpPush, WarpStrength, Z_NEAR, world_to_screen,
};
pub use color::Color;
pub use engine::{Config, Engine, run};
pub use frame::{Frame, Frame3D, SkyDesc};
pub use glam::{DVec2, DVec3, IVec2, Mat3, Mat4, Vec2, Vec3};
pub use input::{Key, MouseButton};
pub use mesh::{Ao, DebugVertex, Light, MeshData, MeshHandle, MeshVertex, Normal, Pass};
pub use surface::{SurfaceData, SurfaceHandle, SurfaceVertex};
pub use vk::RENDER_SCALE_RANGE;

/// Text metrics for the embedded font, usable without an [`Engine`].
pub use font::measure_text;
