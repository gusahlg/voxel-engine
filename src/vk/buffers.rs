//! GPU mesh registry and per-frame immediate-geometry buffers.
//!
//! Meshes live in device-local memory suballocated from `GpuAllocator`
//! blocks: one allocation per mesh holding vertices. Indices are the shared
//! per-quad pattern in [`QuadIbo`]. On unified-memory devices uploads are
//! direct memcpys; otherwise they go through a staging allocation and a
//! `cmd_copy_buffer` recorded at the next frame's start (so a mesh uploaded
//! mid-update is drawable the same frame). Frees are deferred until the GPU
//! provably finished the last frame that could have referenced the mesh.

pub use super::handles::DrawDyn;
pub use super::host_buffer::HostBuffer;
pub use super::mesh3d_desc::{create_mesh3d_set_layout, push_mesh3d_descriptors, push_prev_depth};
pub use super::records::{DrawIndexedIndirect, MeshRecord};
pub use super::retire::RetireQueue;
pub use crate::rev::{FRAMES_IN_FLIGHT, SUBMIT_BATCH_MAX};

pub(crate) use super::handles::{MeshHandles, MeshMeta, PlacementState};
pub(crate) use super::host_buffer::{HOST_BAR_BYTES, HOST_COHERENT};
pub(crate) use super::mesh_residency::{MESH_CONSUMER_STAGES, MeshResidency};
pub(crate) use super::mesh_resident::{
    GpuResident, build_mesh_resident, build_mesh_resident_staged,
};
pub(crate) use super::quad_ibo::QuadIbo;
pub(crate) use super::records::{MESH_FLAG_FACE_RUNS, RecordBuffers, RecordTable};
