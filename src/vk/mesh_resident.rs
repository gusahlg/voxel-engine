use ash::vk;
use glam::Vec3;

use super::alloc::{Allocation, GpuAllocator};
use super::handles::{DrawDyn, MeshMeta, PlacementState};
use super::mesh_staging::{MeshStaging, MeshStagingPool, StagingLease, Stamp};
use super::timeline::TimelineValue;
use crate::mesh::{FACE_UPLOAD_ORDER, MeshData, Pass};

/// GPU storage-buffer offset alignment; the 256 half of `MESH_ALIGN`.
const GPU_OFFSET_ALIGN: u64 = 256;

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Suballocation alignment: must divide both GPU offset alignment (256) and vertex stride.
const MESH_ALIGN: u64 = {
    let stride = std::mem::size_of::<crate::mesh::MeshVertex>() as u64;
    stride / gcd(stride, GPU_OFFSET_ALIGN) * GPU_OFFSET_ALIGN
};
const _: () = {
    assert!(MESH_ALIGN % std::mem::size_of::<crate::mesh::MeshVertex>() as u64 == 0);
    assert!(MESH_ALIGN % GPU_OFFSET_ALIGN == 0);
};

/// Vertex stride shared by the mesh pipelines (must divide [`MESH_ALIGN`]).
const VERTEX_STRIDE: u64 = std::mem::size_of::<crate::mesh::MeshVertex>() as u64;

pub(crate) enum CopySource {
    Alloc(Allocation),
    Pool(StagingLease),
}

pub(crate) struct PendingCopy {
    pub(crate) src_buffer: vk::Buffer,
    pub(crate) src_offset: u64,
    pub(crate) dst_buffer: vk::Buffer,
    pub(crate) dst_offset: u64,
    pub(crate) size: u64,
    pub(crate) source: CopySource,
}

/// Render-owned GPU residency for one mesh: the device buffer plus its
/// deferred staging copy. `Send` because [`Allocation`] is now `Send`.
pub(crate) struct GpuResident {
    pub(crate) buffer: vk::Buffer,
    /// Device-arena suballocation to return to [`GpuAllocator`].
    pub(crate) arena: Allocation,
    pub(crate) copy: Option<PendingCopy>,
    /// Timeline value ordering copy before reads; `None` while budget-deferred.
    pub(crate) arrived_at: Option<TimelineValue>,
}

impl GpuResident {
    /// Get the device buffer.
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }
}

/// Copies each direction's vertices into `dst` in [`FACE_UPLOAD_ORDER`], one
/// `copy_nonoverlapping` per direction. Returns bytes written.
///
/// # Safety
/// `dst` must be valid for `data.vertex_bytes()` writes.
pub(crate) unsafe fn write_vertices_upload_order(data: &MeshData, dst: *mut u8) -> usize {
    let mut cursor = 0usize;
    for &dir in &FACE_UPLOAD_ORDER {
        let bytes: &[u8] = bytemuck::cast_slice(&data.vertices[dir]);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.add(cursor), bytes.len());
        }
        cursor += bytes.len();
    }
    debug_assert_eq!(
        cursor,
        data.vertex_bytes(),
        "upload must cover every vertex"
    );
    cursor
}

/// Local index boundaries into the shared quad IBO from per-[`crate::mesh::Normal`]
/// quad counts: `bounds[k]..bounds[k+1]` is upload-order face `k` (`6*quads`).
pub(crate) fn index_bounds_from_quad_counts(counts: [u32; 6]) -> [u32; 7] {
    let mut bounds = [0u32; 7];
    for (k, &dir) in FACE_UPLOAD_ORDER.iter().enumerate() {
        bounds[k + 1] = bounds[k] + counts[dir] * 6;
    }
    bounds
}

/// Shared meta + residency tail after a device allocation and optional copy.
/// Vertex bytes and AABB are the caller's: this only records them.
fn finish_resident(
    alloc: Allocation,
    bounds: [u32; 7],
    aabb: (Vec3, Vec3),
    pass: Pass,
    copy: Option<PendingCopy>,
) -> (MeshMeta, GpuResident) {
    debug_assert_eq!(alloc.offset % VERTEX_STRIDE, 0);
    let vertex_offset = (alloc.offset / VERTEX_STRIDE) as i32;
    let (aabb_min, aabb_max) = aabb;
    let meta = MeshMeta {
        aabb_min,
        aabb_max,
        bounds,
        vertex_offset,
        pass,
        placement: PlacementState::Tracked(None),
        dyn_lane: DrawDyn::resting(),
    };
    // No staged copy (unified memory: already written) is immediately
    // drawable; a staged copy gates drawability until `flush_copies` submits
    // it (see [`GpuResident::arrived_at`]).
    let arrived_at = copy.is_none().then_some(TimelineValue::START);
    (
        meta,
        GpuResident {
            buffer: alloc.buffer,
            arena: alloc,
            copy,
            arrived_at,
        },
    )
}

/// Allocates a device buffer for `data`, writes/stages its bytes, and returns
/// the main-owned [`MeshMeta`] plus render-owned [`GpuResident`]. Main-thread
/// only: touches the allocator + persistent mapping, never the timeline.
/// `None` on empty data or OOM (the partial device alloc is freed on failure).
/// Stores vertices only; indices are the shared per-quad pattern in [`QuadIbo`].
pub(crate) unsafe fn build_mesh_resident(
    device: &ash::Device,
    allocator: &mut GpuAllocator,
    data: &MeshData,
) -> Option<(MeshMeta, GpuResident)> {
    if data.is_empty() {
        return None;
    }

    let vertex_bytes_len = data.vertex_bytes();
    let total = vertex_bytes_len as u64;
    let bounds = index_bounds_from_quad_counts(data.quad_counts());
    debug_assert_eq!(
        bounds[6] as usize / 6 * 4,
        vertex_bytes_len / VERTEX_STRIDE as usize,
        "vertex count stays 4 * quads"
    );

    let alloc = unsafe { allocator.alloc_device(device, total, MESH_ALIGN) }
        .map_err(|err| log::error!("mesh allocation failed: {err:?}"))
        .ok()?;

    let copy = if let Some(mapped) = alloc.mapped {
        // Unified memory: write straight into the device-local block.
        let written = unsafe { write_vertices_upload_order(data, mapped.as_ptr()) };
        debug_assert_eq!(written, vertex_bytes_len);
        None
    } else {
        let staging = match unsafe { allocator.alloc_staging(device, total, 4) } {
            Ok(staging) => staging,
            Err(err) => {
                log::error!("staging allocation failed: {err:?}");
                unsafe { allocator.free(alloc) };
                return None;
            }
        };
        let mapped = staging
            .mapped
            .expect("staging memory is always host-visible");
        let written = unsafe { write_vertices_upload_order(data, mapped.as_ptr()) };
        debug_assert_eq!(written, vertex_bytes_len);
        Some(PendingCopy {
            src_buffer: staging.buffer,
            src_offset: staging.offset,
            dst_buffer: alloc.buffer,
            dst_offset: alloc.offset,
            size: total,
            source: CopySource::Alloc(staging),
        })
    };

    let (aabb_min, aabb_max) = data.aabb();
    let aabb_min = Vec3::from_array(aabb_min);
    let aabb_max = Vec3::from_array(aabb_max);
    debug_assert!({
        let mut scan_min = Vec3::splat(f32::INFINITY);
        let mut scan_max = Vec3::splat(f32::NEG_INFINITY);
        for bucket in &data.vertices {
            for v in bucket {
                let p = Vec3::from_array(v.local_pos());
                scan_min = scan_min.min(p);
                scan_max = scan_max.max(p);
            }
        }
        scan_min == aabb_min && scan_max == aabb_max
    });

    Some(finish_resident(
        alloc,
        bounds,
        (aabb_min, aabb_max),
        data.pass,
        copy,
    ))
}

/// Installs a worker-written staging region as a mesh: always one device-arena
/// allocation. When the arena is host-mapped (unified / ReBAR) the vertex bytes
/// are copied with one `copy_nonoverlapping` and the region is stamped
/// `Stamp::Render(0)` so the next reclaim frees it. Otherwise one pending
/// `vkCmdCopyBuffer` is batched with the rest of the frame by
/// [`MeshResidency::flush_copies`] and the region is stamped with the transfer
/// timeline (`Stamp::Transfer` on a separate queue, `Stamp::Render` on the
/// fallback). The staging ring is transient in both modes.
///
/// The AABB comes from the region (tracked as vertices were written). This
/// function does not scan staging memory except as a documented fallback
/// when no AABB was recorded.
pub(crate) unsafe fn build_mesh_resident_staged(
    device: &ash::Device,
    allocator: &mut GpuAllocator,
    pool: &MeshStagingPool,
    staging: MeshStaging,
    quads: [u32; 6],
    pass: Pass,
) -> Option<(MeshMeta, GpuResident)> {
    let vertex_count: usize = quads.iter().map(|&q| q as usize * 4).sum();
    if vertex_count == 0 {
        return None;
    }
    let vertex_bytes_len = vertex_count * VERTEX_STRIDE as usize;
    if staging.requested_bytes() < vertex_bytes_len {
        log::error!(
            "mesh staging region ({} bytes) smaller than vertex payload ({vertex_bytes_len})",
            staging.requested_bytes()
        );
        return None;
    }
    let (min, max) = staging.aabb_for_upload(vertex_bytes_len);
    let (aabb_min, aabb_max) = (Vec3::from_array(min), Vec3::from_array(max));
    let bounds = index_bounds_from_quad_counts(quads);
    debug_assert_eq!(
        bounds[6] as usize / 6 * 4,
        vertex_count,
        "vertex count stays 4 * quads"
    );

    let total = vertex_bytes_len as u64;
    let alloc = match unsafe { allocator.alloc_device(device, total, MESH_ALIGN) } {
        Ok(alloc) => alloc,
        Err(err) => {
            log::error!("mesh allocation failed: {err:?}");
            return None;
        }
    };

    let copy = if let Some(mapped) = alloc.mapped {
        // Unified / ReBAR: both the ring region and the arena block are
        // host-mapped (~5 KB per chunk). One memcpy, then free the region
        // at the next reclaim — the FIFO must not pin live meshes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                staging.as_bytes().as_ptr(),
                mapped.as_ptr(),
                vertex_bytes_len,
            );
        }
        staging.into_lease().stamp(Stamp::Render(0));
        None
    } else {
        let lease = staging.into_lease();
        Some(PendingCopy {
            src_buffer: pool.buffer(),
            src_offset: lease.offset(),
            dst_buffer: alloc.buffer,
            dst_offset: alloc.offset,
            size: total,
            source: CopySource::Pool(lease),
        })
    };

    Some(finish_resident(
        alloc,
        bounds,
        (aabb_min, aabb_max),
        pass,
        copy,
    ))
}
