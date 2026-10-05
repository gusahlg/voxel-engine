//! Host ring of far-body records for the sky pass (set 0, binding 2).
//! One coherent buffer per frame-in-flight, written before the sky draw.
//! The record matches `FarGpu` in `shaders/far_body.slang` (160 bytes, std430).
//! A 16-byte cone sits in front of each record so a pixel can reject the body
//! before loading it.

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::far_body::{FarBody, FarShape, MAX_FAR_BODIES};
use crate::rev::{FrameSlot, PerSlot};
use crate::vk::buffers::HostBuffer;

/// `VOXEL_FAR_CULL=0` disables far-body culling. Any other value, including
/// unset, leaves it on. Read once.
fn far_cull_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| !std::env::var("VOXEL_FAR_CULL").is_ok_and(|v| v == "0"))
}

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
    /// rgb rim tint, a = shape (0 cube, 1 sphere, 2 inner sphere, 3 rounded).
    pub atmosphere: [f32; 4],
    /// x = noise seed. y = rounded exponent as f32 bits (0 for other shapes).
    pub seed: [u32; 4],
}

/// Header, then one cone per slot, then the body records.
/// `header[0]` is the live count. `cone[i] = (dir.xyz, sine bound)` and a
/// bound of `-1` means the pixel test must not reject that body.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct FarTableGpu {
    pub header: [u32; 4],
    pub cone: [[f32; 4]; MAX_FAR_BODIES],
    pub body: [FarBodyGpu; MAX_FAR_BODIES],
}

const _: () = assert!(std::mem::size_of::<FarBodyGpu>() == 160);
const _: () =
    assert!(std::mem::size_of::<FarTableGpu>() == 16 + 16 * MAX_FAR_BODIES + 160 * MAX_FAR_BODIES);
const _: () = assert!(std::mem::offset_of!(FarTableGpu, cone) == 16);
const _: () = assert!(std::mem::offset_of!(FarTableGpu, body) == 16 + 16 * MAX_FAR_BODIES);

/// Sine of the angular radius the shader can draw (`s = ‖ray × center‖`), or
/// `-1` when a ray facing away can still hit. Matches `far_bodies()`: the
/// sphere and rounded rims reach `1.05` times the body, the cube's corners
/// sit at `√3` times the enlarged half-extent, and an inner sphere is the sky.
fn cone_bound(body: &FarBody) -> f32 {
    if !far_cull_enabled() {
        return -1.0;
    }
    let rho = body.radius / body.distance;
    let bound = match body.shape {
        FarShape::InnerSphere => return -1.0,
        FarShape::Sphere => 1.05 * rho,
        FarShape::Cube => 3.0f32.sqrt() * 1.035 * rho,
        FarShape::Rounded { exponent } => {
            let p = exponent.clamp(2.0, 32.0);
            let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
            1.05 * rho_b
        }
    };
    // The camera is inside, or nearly inside, the bounding sphere: a ray
    // facing away can still meet the body, so the cone must not reject it.
    if !bound.is_finite() || bound >= 0.99 {
        -1.0
    } else {
        bound
    }
}

fn rgb4(c: crate::color::LinearRgb) -> [f32; 4] {
    [c.0[0], c.0[1], c.0[2], 0.0]
}

fn pack_one(body: &FarBody) -> FarBodyGpu {
    let dir = body.dir;
    let q = body.rotation;
    let (shape, exponent) = match body.shape {
        FarShape::Cube => (0.0, 0.0),
        FarShape::Sphere => (1.0, 0.0),
        FarShape::InnerSphere => (2.0, 0.0),
        FarShape::Rounded { exponent } => (3.0, exponent),
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
        seed: [body.seed, exponent.to_bits(), 0, 0],
    }
}

pub(crate) fn pack_table(bodies: &[FarBody]) -> FarTableGpu {
    let mut table = FarTableGpu::zeroed();
    let n = bodies.len().min(MAX_FAR_BODIES);
    table.header[0] = n as u32;
    for (i, body) in bodies.iter().take(n).enumerate() {
        let d = body.dir;
        table.cone[i] = [d.x, d.y, d.z, cone_bound(body)];
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
    use crate::far_body::{FarBody, FarShape, store};
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
        assert_eq!(packed.seed[1], 0);
    }

    #[test]
    fn packs_rounded_exponent_round_trip() {
        let body = FarBody {
            dir: Vec3::new(0.0, 0.0, 8.0),
            distance: 8.0,
            radius: 2.0,
            shape: FarShape::Rounded { exponent: 2.17 },
            rotation: Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
            albedo: [LinearRgb([0.4, 0.5, 0.6]); 6],
            atmosphere: LinearRgb([0.1, 0.2, 0.3]),
            seed: 0xB0D1,
        };
        let mut slot = [FarBody::default(); MAX_FAR_BODIES];
        let n = store(std::slice::from_ref(&body), &mut slot);
        assert_eq!(n, 1);
        assert_eq!(slot[0].shape, FarShape::Rounded { exponent: 2.17 });
        let table = pack_table(&slot[..n as usize]);
        let gpu = &table.body[0];
        assert_eq!(gpu.atmosphere[3].to_bits(), 3.0f32.to_bits());
        assert_eq!(f32::from_bits(gpu.seed[1]).to_bits(), 2.17f32.to_bits());
        assert_eq!(gpu.seed[0], 0xB0D1);
        assert_eq!(gpu.seed[2], 0);
        assert_eq!(gpu.dir_rho[3].to_bits(), 0.25f32.to_bits());
        assert_eq!(gpu.rot, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(std::mem::size_of::<FarBodyGpu>(), 160);

        let again = f32::from_bits(pack_one(&slot[0]).seed[1]);
        assert_eq!(again.to_bits(), 2.17f32.to_bits());
    }

    fn sample(shape: FarShape, distance: f32, radius: f32) -> FarBody {
        FarBody {
            dir: Vec3::Z,
            distance,
            radius,
            shape,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.3, 0.4]); 6],
            atmosphere: LinearRgb([0.0, 0.0, 0.0]),
            seed: 1,
        }
    }

    #[test]
    fn cone_bound_matches_what_the_shader_can_draw() {
        let sphere = sample(FarShape::Sphere, 1.0, 0.25);
        let table = pack_table(std::slice::from_ref(&sphere));
        assert_eq!(table.header[0], 1);
        assert_eq!(table.cone[0][0].to_bits(), 0.0f32.to_bits());
        assert_eq!(table.cone[0][1].to_bits(), 0.0f32.to_bits());
        assert_eq!(table.cone[0][2].to_bits(), 1.0f32.to_bits());
        assert_eq!(table.cone[0][3].to_bits(), (1.05f32 * 0.25).to_bits());
        assert_eq!(table.cone[0][0..3], table.body[0].dir_rho[0..3]);

        // 1.05 * rho >= 0.99: the camera is nearly inside the bounding sphere.
        let inside = sample(FarShape::Sphere, 1.0, 0.95);
        assert!(1.05 * 0.95 >= 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&inside)).cone[0][3].to_bits(),
            (-1.0f32).to_bits()
        );

        let cube = sample(FarShape::Cube, 1.0, 0.2);
        let cube_bound = 3.0f32.sqrt() * 1.035 * 0.2;
        assert!(cube_bound < 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&cube)).cone[0][3].to_bits(),
            cube_bound.to_bits()
        );

        let rounded = sample(FarShape::Rounded { exponent: 4.0 }, 1.0, 0.2);
        let p = 4.0f32;
        let rho_b = 0.2 * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
        assert_eq!(
            pack_table(std::slice::from_ref(&rounded)).cone[0][3].to_bits(),
            (1.05 * rho_b).to_bits()
        );

        // The shader clamps the exponent to 32; the cone uses that same p.
        let steep = sample(FarShape::Rounded { exponent: 80.0 }, 1.0, 0.1);
        let p32 = 32.0f32;
        let rho_b32 = 0.1 * 3.0f32.powf(0.5 - 1.0 / p32) * (1.0 + 2.0e-4);
        assert!(1.05 * rho_b32 < 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&steep)).cone[0][3].to_bits(),
            (1.05 * rho_b32).to_bits()
        );

        let wall = sample(FarShape::InnerSphere, 2.0, 5.0);
        assert_eq!(
            pack_table(std::slice::from_ref(&wall)).cone[0][3].to_bits(),
            (-1.0f32).to_bits()
        );

        assert_eq!(std::mem::offset_of!(FarTableGpu, cone), 16);
        assert_eq!(
            std::mem::offset_of!(FarTableGpu, body),
            16 + 16 * MAX_FAR_BODIES
        );
    }
}
