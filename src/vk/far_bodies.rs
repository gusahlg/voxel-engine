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

/// `VOXEL_FAR_CULL=0` disables the cone reject and the CPU frustum compaction.
/// Any other value, including unset, leaves culling on. Read once.
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

/// What the sky pass needs to drop bodies that cannot meet this frame's view.
/// `view_proj` is the clean (unjittered) camera matrix. `px_max` is an upper
/// bound on the pixel angle, with a factor of two over the projection.
#[derive(Clone, Copy)]
pub(crate) struct FarView {
    pub view_proj: glam::Mat4,
    pub px_max: f32,
}

/// Vertical pixel angle `2 * tan(fovy/2) / height`, the horizontal equivalent
/// from the projection, the max of those, times two.
pub(crate) fn px_max(fovy_tan_half: f32, view_proj: glam::Mat4, width: u32, height: u32) -> f32 {
    let h = height.max(1) as f32;
    let w = width.max(1) as f32;
    let vert = 2.0 * fovy_tan_half / h;
    // Row 0 of `proj * view` has length `1 / tan(fovx/2)` (the view is rigid).
    let row0 = view_proj.transpose().x_axis.truncate().length().max(1e-20);
    let horiz = 2.0 / (row0 * w);
    vert.max(horiz) * 2.0
}

pub(crate) fn far_view(
    fovy_tan_half: f32,
    view_proj: glam::Mat4,
    width: u32,
    height: u32,
) -> FarView {
    FarView {
        view_proj,
        px_max: px_max(fovy_tan_half, view_proj, width, height),
    }
}

/// Inward normals of the four side planes, for directions through the eye.
/// Gribb-Hartmann on the rows of `view_proj` (row3 ± row0, row3 ± row1); the
/// translation drops out because each side plane contains the camera.
fn side_normals(view_proj: glam::Mat4) -> [glam::Vec3; 4] {
    let rows = view_proj.transpose();
    let (r0, r1, r3) = (rows.x_axis, rows.y_axis, rows.w_axis);
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1].map(|plane| {
        let n = plane.truncate();
        let len = n.length();
        if len > 1e-20 {
            n / len
        } else {
            glam::Vec3::ZERO
        }
    })
}

/// `bound` is the cone's sine, or `-1` to always keep the body. A cone of sine
/// `a = min(bound + 3 px, 1)` lies outside an inward plane `n` when
/// `dot(n, dir) < -a`. `a >= 1` covers the view, so the body stays.
fn cone_in_view(dir: glam::Vec3, bound: f32, view: &FarView) -> bool {
    if bound < 0.0 {
        return true;
    }
    let a_sin = (bound + 3.0 * view.px_max).min(1.0);
    if a_sin >= 1.0 {
        return true;
    }
    let len2 = dir.length_squared();
    if !(len2 > 0.0) {
        return true;
    }
    let dir = dir / len2.sqrt();
    for n in side_normals(view.view_proj) {
        if n.length_squared() == 0.0 {
            continue;
        }
        if n.dot(dir) < -a_sin {
            return false;
        }
    }
    true
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

/// Pack `bodies` in order. With culling on and a view, bodies whose cone cannot
/// meet the frustum are dropped; the rest stay in their original relative order
/// (the sky composite is order-dependent). `None` keeps every body.
pub(crate) fn pack_table(bodies: &[FarBody], view: Option<FarView>) -> FarTableGpu {
    let mut table = FarTableGpu::zeroed();
    let n = bodies.len().min(MAX_FAR_BODIES);
    let cull = far_cull_enabled();
    let mut kept = 0usize;
    for body in bodies.iter().take(n) {
        let bound = cone_bound(body);
        if cull {
            if let Some(view) = view.as_ref() {
                if !cone_in_view(body.dir, bound, view) {
                    continue;
                }
            }
        }
        let d = body.dir;
        table.cone[kept] = [d.x, d.y, d.z, bound];
        table.body[kept] = pack_one(body);
        kept += 1;
    }
    table.header[0] = kept as u32;
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

    pub(crate) fn write(&mut self, slot: FrameSlot, bodies: &[FarBody], view: Option<FarView>) {
        let table = pack_table(bodies, view);
        let offered = bodies.len().min(MAX_FAR_BODIES) as u64;
        crate::profile::gauge(crate::profile::Gauge::FarBodies, offered);
        crate::profile::gauge(crate::profile::Gauge::FarDrawn, u64::from(table.header[0]));
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
    use crate::camera::{Camera3D, Lens, WarpMap, WarpStrength};
    use crate::color::LinearRgb;
    use crate::far_body::{
        FarBody, FarShape, ray_cube, ray_inner_sphere, ray_rounded, ray_sphere, store,
    };
    use glam::{Quat, Vec3, Vec4};

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
        let table = pack_table(&slot[..n as usize], None);
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
        let table = pack_table(&slot[..n as usize], None);
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
        let table = pack_table(std::slice::from_ref(&sphere), None);
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
            pack_table(std::slice::from_ref(&inside), None).cone[0][3].to_bits(),
            (-1.0f32).to_bits()
        );

        let cube = sample(FarShape::Cube, 1.0, 0.2);
        let cube_bound = 3.0f32.sqrt() * 1.035 * 0.2;
        assert!(cube_bound < 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&cube), None).cone[0][3].to_bits(),
            cube_bound.to_bits()
        );

        let rounded = sample(FarShape::Rounded { exponent: 4.0 }, 1.0, 0.2);
        let p = 4.0f32;
        let rho_b = 0.2 * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
        assert_eq!(
            pack_table(std::slice::from_ref(&rounded), None).cone[0][3].to_bits(),
            (1.05 * rho_b).to_bits()
        );

        // The shader clamps the exponent to 32; the cone uses that same p.
        let steep = sample(FarShape::Rounded { exponent: 80.0 }, 1.0, 0.1);
        let p32 = 32.0f32;
        let rho_b32 = 0.1 * 3.0f32.powf(0.5 - 1.0 / p32) * (1.0 + 2.0e-4);
        assert!(1.05 * rho_b32 < 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&steep), None).cone[0][3].to_bits(),
            (1.05 * rho_b32).to_bits()
        );

        let wall = sample(FarShape::InnerSphere, 2.0, 5.0);
        assert_eq!(
            pack_table(std::slice::from_ref(&wall), None).cone[0][3].to_bits(),
            (-1.0f32).to_bits()
        );

        assert_eq!(std::mem::offset_of!(FarTableGpu, cone), 16);
        assert_eq!(
            std::mem::offset_of!(FarTableGpu, body),
            16 + 16 * MAX_FAR_BODIES
        );
    }

    fn view_along_neg_z(fovy: f32, width: u32, height: u32) -> FarView {
        let cam = Camera3D {
            position: Vec3::new(12.0, 3.0, -4.0),
            target: Vec3::new(12.0, 3.0, -5.0),
            up: Vec3::Y,
            fovy,
            lens: Lens::Rectilinear,
        };
        let aspect = width as f32 / height as f32;
        let tan_half = (fovy.to_radians() * 0.5).tan();
        far_view(tan_half, cam.view_proj(aspect), width, height)
    }

    fn placed(dir: Vec3, rho: f32, shape: FarShape, seed: u32) -> FarBody {
        FarBody {
            dir,
            distance: 1.0,
            radius: rho,
            shape,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
            atmosphere: LinearRgb([0.1, 0.0, 0.0]),
            seed,
        }
    }

    fn kept_seeds(table: &FarTableGpu) -> Vec<u32> {
        (0..table.header[0] as usize)
            .map(|i| table.body[i].seed[0])
            .collect()
    }

    #[test]
    fn px_max_is_twice_the_larger_pixel_angle() {
        let fovy = 60.0f32;
        let w = 3440u32;
        let h = 1440u32;
        let aspect = w as f32 / h as f32;
        let tan_half = (fovy.to_radians() * 0.5).tan();
        let cam = Camera3D {
            position: Vec3::ZERO,
            target: -Vec3::Z,
            up: Vec3::Y,
            fovy,
            lens: Lens::Rectilinear,
        };
        let got = px_max(tan_half, cam.view_proj(aspect), w, h);
        let vert = 2.0 * tan_half / h as f32;
        let horiz = 2.0 * tan_half * aspect / w as f32;
        assert!((vert - horiz).abs() < 1e-5, "{vert} vs {horiz}");
        assert!((got - vert * 2.0).abs() < 1e-5 * got, "{got}");

        let moved = Camera3D {
            position: Vec3::new(80.0, -3.0, 12.0),
            target: Vec3::new(80.0, -3.0, 11.0),
            ..cam
        };
        let got_moved = px_max(tan_half, moved.view_proj(aspect), w, h);
        assert!((got - got_moved).abs() < 1e-4 * got);

        let lens = Lens::WideFov {
            strength: WarpStrength::new(1.5).unwrap(),
        };
        let wide_aspect = aspect * WarpMap::from_lens(lens).fov_scale();
        let wide_cam = Camera3D { lens, ..cam };
        let wide = px_max(tan_half, wide_cam.view_proj(wide_aspect), w, h);
        let wide_horiz = 2.0 * tan_half * wide_aspect / w as f32;
        assert!(wide_horiz > vert * 1.2);
        assert!(
            (wide - wide_horiz * 2.0).abs() < 1e-4 * wide,
            "{wide} vs {wide_horiz}"
        );
    }

    #[test]
    fn frustum_keeps_ahead_and_a_crossing_rim_and_drops_the_rest() {
        let view = view_along_neg_z(60.0, 1920, 1080);
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let back = -forward;
        let ahead = placed(forward, 0.05, FarShape::Sphere, 1);
        let behind = placed(back, 0.05, FarShape::Sphere, 2);

        let right = side_normals(view.view_proj)[1];
        assert!(
            right.dot(forward) > 0.0,
            "forward must sit inside the right plane"
        );
        let on_plane = (forward - right * forward.dot(right)).normalize();
        assert!(right.dot(on_plane).abs() < 1e-4);
        let bound = 1.05 * 0.02;
        let a_sin = (bound + 3.0 * view.px_max).min(1.0);
        let outward = -right;
        let beta_in = (a_sin * 0.5).asin();
        let crossing_dir = (on_plane * beta_in.cos() + outward * beta_in.sin()).normalize();
        assert!(right.dot(crossing_dir) < 0.0);
        assert!(right.dot(crossing_dir) >= -a_sin);
        let crossing = placed(crossing_dir, 0.02, FarShape::Sphere, 3);

        let beta_out = (a_sin + 0.08).min(0.95).asin();
        let outside_dir = (on_plane * beta_out.cos() + outward * beta_out.sin()).normalize();
        assert!(right.dot(outside_dir) < -a_sin);
        let outside = placed(outside_dir, 0.02, FarShape::Sphere, 4);

        let table = pack_table(&[behind, ahead, outside, crossing], Some(view));
        assert_eq!(kept_seeds(&table), vec![1, 3]);
        assert_eq!(table.header[0], 2);
    }

    #[test]
    fn inner_sphere_and_near_camera_rounded_are_always_kept() {
        let view = view_along_neg_z(50.0, 1280, 720);
        let back = Vec3::new(0.0, 0.0, 1.0);
        let wall = FarBody {
            dir: back,
            distance: 2.0,
            radius: 6.0,
            shape: FarShape::InnerSphere,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
            atmosphere: LinearRgb([0.0, 0.0, 0.0]),
            seed: 7,
        };
        let rho = 0.96f32;
        let p = 2.0f32;
        let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
        assert!(1.05 * rho_b >= 0.99);
        let round = placed(back, rho, FarShape::Rounded { exponent: p }, 8);
        let table = pack_table(&[wall, round], Some(view));
        assert_eq!(table.header[0], 2);
        assert_eq!(kept_seeds(&table), vec![7, 8]);
        assert_eq!(table.cone[0][3].to_bits(), (-1.0f32).to_bits());
        assert_eq!(table.cone[1][3].to_bits(), (-1.0f32).to_bits());
    }

    #[test]
    fn compaction_preserves_relative_order_and_the_header_count() {
        let view = view_along_neg_z(60.0, 800, 600);
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let back = -forward;
        let wall = FarBody {
            dir: back,
            distance: 1.5,
            radius: 4.0,
            shape: FarShape::InnerSphere,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
            atmosphere: LinearRgb([0.0, 0.0, 0.0]),
            seed: 13,
        };
        let bodies = [
            placed(back, 0.04, FarShape::Sphere, 10),
            placed(forward, 0.04, FarShape::Cube, 11),
            placed(back, 0.05, FarShape::Sphere, 12),
            wall,
            placed(forward, 0.03, FarShape::Rounded { exponent: 6.0 }, 14),
        ];
        let table = pack_table(&bodies, Some(view));
        assert_eq!(table.header[0], 3);
        assert_eq!(kept_seeds(&table), vec![11, 13, 14]);
        assert_eq!(table.cone[0][2].to_bits(), forward.z.to_bits());
        assert_eq!(table.body[0].dir_rho[2].to_bits(), forward.z.to_bits());
        assert_eq!(table.cone[2][2].to_bits(), forward.z.to_bits());
    }

    struct Rng(u32);

    impl Rng {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = if x == 0 { 0xA5A5_A5A5 } else { x };
            self.0
        }

        fn unit_f32(&mut self) -> f32 {
            (self.next() >> 8) as f32 * (1.0 / 16_777_216.0)
        }

        fn range(&mut self, a: f32, b: f32) -> f32 {
            a + (b - a) * self.unit_f32()
        }

        fn unit_vec(&mut self) -> Vec3 {
            let z = self.range(-1.0, 1.0);
            let t = self.range(0.0, std::f32::consts::TAU);
            let r = (1.0 - z * z).max(0.0).sqrt();
            Vec3::new(r * t.cos(), r * t.sin(), z)
        }

        fn quat(&mut self) -> Quat {
            Quat::from_axis_angle(self.unit_vec(), self.range(0.0, std::f32::consts::TAU))
        }
    }

    fn random_body(rng: &mut Rng, seed: u32) -> FarBody {
        let dir = rng.unit_vec();
        let (shape, distance, radius) = match rng.next() % 4 {
            0 => (FarShape::Sphere, 1.0, rng.range(0.002, 0.9)),
            1 => (FarShape::Cube, 1.0, rng.range(0.002, 0.9)),
            2 => (
                FarShape::Rounded {
                    exponent: rng.range(2.0, 32.0),
                },
                1.0,
                rng.range(0.002, 0.85),
            ),
            _ => (FarShape::InnerSphere, 2.0, 5.0),
        };
        FarBody {
            dir,
            distance,
            radius,
            shape,
            rotation: rng.quat(),
            albedo: [LinearRgb([0.4, 0.3, 0.2]); 6],
            atmosphere: LinearRgb([0.05, 0.08, 0.1]),
            seed,
        }
    }

    fn pixel_rejects(ray: Vec3, dir: Vec3, bound: f32, px: f32) -> bool {
        if bound < 0.0 {
            return false;
        }
        let dir = dir.normalize();
        let facing = ray.dot(dir);
        let s2 = ray.cross(dir).length_squared();
        let lim = bound + 3.0 * px;
        facing <= 0.0 || s2 > lim * lim
    }

    fn solid_hit(ray: Vec3, body: &FarBody) -> bool {
        let dir = body.dir.normalize();
        let rho = body.radius / body.distance;
        match body.shape {
            FarShape::Sphere => ray_sphere(ray, dir, rho).is_some(),
            FarShape::Cube => ray_cube(ray, dir, rho, body.rotation).is_some(),
            FarShape::Rounded { exponent } => {
                ray_rounded(ray, dir, rho, body.rotation, exponent).is_some()
            }
            FarShape::InnerSphere => {
                ray_inner_sphere(ray, dir, body.distance, body.radius).is_some()
            }
        }
    }

    /// Front contribution only. `facing <= 0` is past any rim the shader keeps
    /// after the cone reject (the rounded antipodal ghost is not a real rim).
    fn beyond_rim(ray: Vec3, body: &FarBody, px: f32) -> bool {
        let dir = body.dir.normalize();
        let rho = body.radius / body.distance;
        let facing = ray.dot(dir);
        let s = ray.cross(dir).length();
        match body.shape {
            FarShape::Sphere => {
                let rim = rho + (0.05 * rho).max(1.25 * px);
                facing <= 0.0 || s >= rim
            }
            FarShape::Rounded { exponent } => {
                let p = exponent.clamp(2.0, 32.0);
                let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p);
                let rim = rho_b + (0.05 * rho_b).max(1.25 * px);
                facing <= 0.0 || s >= rim
            }
            FarShape::Cube => {
                if facing <= 0.0 {
                    return true;
                }
                let rho_rim = rho * 1.035 + 0.5 * px;
                if !(rho_rim > 0.0 && rho_rim < 1.0) {
                    return true;
                }
                ray_cube(ray, dir, rho_rim, body.rotation).is_none()
            }
            FarShape::InnerSphere => false,
        }
    }

    #[test]
    fn cull_and_cone_reject_only_drop_rays_with_no_contribution() {
        let mut rng = Rng(0xF4A5_C011);
        let width = 3440u32;
        let height = 1440u32;
        let fovy = 70.0f32;
        let cam = Camera3D {
            position: Vec3::new(4.0, -2.0, 9.0),
            target: Vec3::new(4.0, -2.0, 8.0),
            up: Vec3::Y,
            fovy,
            lens: Lens::Rectilinear,
        };
        let aspect = width as f32 / height as f32;
        let tan_half = (fovy.to_radians() * 0.5).tan();
        let view_proj = cam.view_proj(aspect);
        let view = far_view(tan_half, view_proj, width, height);
        let vert = 2.0 * tan_half / height as f32;
        let row0 = view_proj.transpose().x_axis.truncate().length();
        let horiz = 2.0 / (row0 * width as f32);
        let px_values = [vert, horiz, vert.hypot(horiz)];

        let inv = view_proj.inverse();
        let mut rays = Vec::new();
        let push_ndc = |rays: &mut Vec<Vec3>, x: f32, y: f32| {
            let ray = (inv * Vec4::new(x, y, 0.0, 1.0)).truncate().normalize();
            let inside = side_normals(view_proj)
                .iter()
                .all(|n| n.length_squared() == 0.0 || n.dot(ray) >= -1e-4);
            if inside {
                rays.push(ray);
            }
        };
        push_ndc(&mut rays, 0.0, 0.0);
        for _ in 0..48 {
            push_ndc(&mut rays, rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
        }
        assert!(rays.len() >= 30, "rays inside the frustum: {}", rays.len());

        // The table keeps at most MAX_FAR_BODIES. Anything past that is omitted
        // by capacity, not by the frustum test, so each pack stays within the cap.
        for _batch in 0..3 {
            let bodies: Vec<FarBody> = (0..MAX_FAR_BODIES)
                .map(|i| random_body(&mut rng, i as u32))
                .collect();
            let table = pack_table(&bodies, Some(view));
            let kept = kept_seeds(&table);

            for body in &bodies {
                let culled = !kept.contains(&body.seed);
                let bound = cone_bound(body);
                for ray in &rays {
                    for px in px_values {
                        if !(culled || pixel_rejects(*ray, body.dir, bound, px)) {
                            continue;
                        }
                        let hit = solid_hit(*ray, body);
                        let past = matches!(body.shape, FarShape::InnerSphere)
                            || beyond_rim(*ray, body, px);
                        let dir = body.dir.normalize();
                        assert!(
                            !hit && past,
                            "seed {} {:?} culled {culled} bound {bound} px {px} hit {hit} past {past} facing {} s {}",
                            body.seed,
                            body.shape,
                            ray.dot(dir),
                            ray.cross(dir).length(),
                        );
                    }
                }
            }
        }
    }
}
