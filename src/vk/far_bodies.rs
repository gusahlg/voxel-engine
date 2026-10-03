//! Host ring of far-body records for the sky pass (set 0, binding 2).
//! One coherent buffer per frame-in-flight, written before the sky draw.
//! The record matches `FarGpu` in `shaders/far_body.slang` (160 bytes, std430).

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::far_body::{FarBody, FarShape, MAX_FAR_BODIES};
use crate::rev::{FrameSlot, PerSlot};
use crate::vk::buffers::HostBuffer;

/// One body on the GPU. Ten 16-byte lanes, 160 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct FarBodyGpu {
    pub dir_rho: [f32; 4],
    pub rot: [f32; 4],
    pub albedo0: [f32; 4],
    pub albedo1: [f32; 4],
    pub albedo2: [f32; 4],
    pub albedo3: [f32; 4],
    pub albedo4: [f32; 4],
    pub albedo5: [f32; 4],
    /// rgb rim tint, a = shape (0 cube, 1 sphere, 2 inner sphere).
    pub atmosphere: [f32; 4],
    pub seed: [u32; 4],
}

/// Header plus the fixed table. `header[0]` is the live count.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct FarTableGpu {
    pub header: [u32; 4],
    pub body: [FarBodyGpu; MAX_FAR_BODIES],
}

const _: () = assert!(std::mem::size_of::<FarBodyGpu>() == 160);
const _: () = assert!(std::mem::size_of::<FarTableGpu>() == 16 + 160 * MAX_FAR_BODIES);
const _: () = assert!(std::mem::offset_of!(FarTableGpu, body) == 16);

fn rgb4(c: crate::color::LinearRgb) -> [f32; 4] {
    [c.0[0], c.0[1], c.0[2], 0.0]
}

fn pack_one(body: &FarBody) -> FarBodyGpu {
    let dir = body.dir;
    let q = body.rotation;
    let shape = match body.shape {
        FarShape::Cube => 0.0,
        FarShape::Sphere => 1.0,
        FarShape::InnerSphere => 2.0,
    };
    FarBodyGpu {
        dir_rho: [dir.x, dir.y, dir.z, body.radius / body.distance],
        rot: [q.x, q.y, q.z, q.w],
        albedo0: rgb4(body.albedo[0]),
        albedo1: rgb4(body.albedo[1]),
        albedo2: rgb4(body.albedo[2]),
        albedo3: rgb4(body.albedo[3]),
        albedo4: rgb4(body.albedo[4]),
        albedo5: rgb4(body.albedo[5]),
        atmosphere: [
            body.atmosphere.0[0],
            body.atmosphere.0[1],
            body.atmosphere.0[2],
            shape,
        ],
        seed: [body.seed, 0, 0, 0],
    }
}

pub(crate) fn pack_table(bodies: &[FarBody]) -> FarTableGpu {
    let mut table = FarTableGpu::zeroed();
    let n = bodies.len().min(MAX_FAR_BODIES);
    table.header[0] = n as u32;
    for (i, body) in bodies.iter().take(n).enumerate() {
        table.body[i] = pack_one(body);
    }
    table
}

/// Per-slot far-body SSBO. Identical bytes skip the map write.
pub(crate) struct FarBodyRing {
    bufs: PerSlot<HostBuffer>,
    last: PerSlot<Option<FarTableGpu>>,
}

impl FarBodyRing {
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Self {
        let size = std::mem::size_of::<FarTableGpu>() as u64;
        let make = || {
            let mut b = HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER);
            unsafe { b.maintain(instance, device, physical, size) };
            b
        };
        Self {
            bufs: PerSlot::new(std::array::from_fn(|_| make())),
            last: PerSlot::new(std::array::from_fn(|_| None)),
        }
    }

    pub(crate) fn write(&mut self, slot: FrameSlot, bodies: &[FarBody]) {
        let table = pack_table(bodies);
        if self.last[slot].as_ref() == Some(&table) {
            return;
        }
        unsafe { self.bufs[slot].write(0, bytemuck::bytes_of(&table)) };
        self.last[slot] = Some(table);
    }

    pub(crate) fn buffer(&self, slot: FrameSlot) -> vk::Buffer {
        self.bufs[slot]
            .bound()
            .expect("the far-body table is written before the sky pass binds it")
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for buf in self.bufs.iter_mut() {
            unsafe { buf.destroy(device) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::LinearRgb;
    use crate::far_body::{store, FarBody, FarShape};
    use glam::{Quat, Vec3};

    #[test]
    fn packs_a_billion_block_sphere_exactly() {
        let body = FarBody {
            dir: Vec3::new(0.0, 0.0, 4.0),
            distance: 1.0e9,
            radius: 2.5e8,
            shape: FarShape::Sphere,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.3, 0.4]); 6],
            atmosphere: LinearRgb([0.0, 0.1, 0.0]),
            seed: 0xA11CE,
        };
        let mut slot = [FarBody::default(); MAX_FAR_BODIES];
        let n = store(std::slice::from_ref(&body), &mut slot);
        assert_eq!(n, 1);
        let table = pack_table(&slot[..n as usize]);
        assert_eq!(table.header[0], 1);
        assert_eq!(table.header[1], 0);
        let gpu = &table.body[0];
        assert_eq!(gpu.dir_rho[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(gpu.dir_rho[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(gpu.dir_rho[2].to_bits(), 1.0f32.to_bits());
        assert_eq!(gpu.dir_rho[3].to_bits(), 0.25f32.to_bits());
        assert_eq!(gpu.rot, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(gpu.seed[0], 0xA11CE);
        assert_eq!(gpu.atmosphere[3].to_bits(), 1.0f32.to_bits());
        assert_eq!(gpu.albedo0[0].to_bits(), 0.2f32.to_bits());

        let mut cube = body;
        cube.shape = FarShape::Cube;
        cube.distance = 4.0;
        cube.radius = 1.0;
        let packed = pack_one(&cube);
        assert_eq!(packed.atmosphere[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(packed.dir_rho[3].to_bits(), 0.25f32.to_bits());

        let mut wall = body;
        wall.shape = FarShape::InnerSphere;
        wall.dir = Vec3::Z;
        wall.distance = 2.0;
        wall.radius = 5.0;
        let packed = pack_one(&wall);
        assert_eq!(packed.atmosphere[3].to_bits(), 2.0f32.to_bits());
        assert_eq!(packed.dir_rho[3].to_bits(), 2.5f32.to_bits());
        assert_eq!(packed.dir_rho[2].to_bits(), 1.0f32.to_bits());
    }
}
