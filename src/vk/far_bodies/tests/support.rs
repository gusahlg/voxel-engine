//! Bodies, views and ray helpers shared by the far-table tests.

use glam::{Quat, Vec3, Vec4};

use crate::camera::{Camera3D, Lens};
use crate::color::LinearRgb;
use crate::far_body::mirror::chart_sample_dir;
use crate::far_body::{FarBody, FarMapId, FarShape, MAX_FAR_MAPS};
use crate::vk::far_bodies::table::FarTableGpu;
use crate::vk::far_bodies::view::{FarView, far_view};

pub(super) fn pack_table(bodies: &[FarBody], view: Option<FarView>) -> FarTableGpu {
    super::pack_table(bodies, view, &[0.0; MAX_FAR_MAPS])
}

pub(super) fn sample(shape: FarShape, distance: f32, radius: f32) -> FarBody {
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

pub(super) fn view_along_neg_z(fovy: f32, width: u32, height: u32) -> FarView {
    view_pitched(0.0, fovy, width, height)
}

/// `pitch_deg` is elevation from the horizon: 0 looks along −Z, positive looks up.
/// The eye is not at the origin, so a direction test that forgets the view
/// rotation fails.
pub(super) fn view_pitched(pitch_deg: f32, fovy: f32, width: u32, height: u32) -> FarView {
    let pitch = pitch_deg.to_radians();
    let forward = Vec3::new(0.0, pitch.sin(), -pitch.cos());
    let up = if forward.cross(Vec3::Y).length_squared() < 1e-6 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let position = Vec3::new(12.0, 3.0, -4.0);
    let cam = Camera3D {
        position,
        target: position + forward,
        up,
        fovy,
        lens: Lens::Rectilinear,
    };
    let aspect = width as f32 / height as f32;
    let tan_half = (fovy.to_radians() * 0.5).tan();
    far_view(tan_half, cam.view_proj(aspect), width, height)
}

pub(super) fn placed(dir: Vec3, rho: f32, shape: FarShape, seed: u32) -> FarBody {
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

pub(super) fn kept_seeds(table: &FarTableGpu) -> Vec<u32> {
    (0..table.header[0] as usize)
        .map(|i| table.body[i].seed[0])
        .collect()
}

pub(super) struct Rng(pub(super) u32);

impl Rng {
    pub(super) fn next(&mut self) -> u32 {
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

    pub(super) fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit_f32()
    }

    pub(super) fn unit_vec(&mut self) -> Vec3 {
        let z = self.range(-1.0, 1.0);
        let t = self.range(0.0, std::f32::consts::TAU);
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::new(r * t.cos(), r * t.sin(), z)
    }

    fn quat(&mut self) -> Quat {
        Quat::from_axis_angle(self.unit_vec(), self.range(0.0, std::f32::consts::TAU))
    }
}

pub(super) fn random_body(rng: &mut Rng, seed: u32) -> FarBody {
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

pub(super) fn ray_at(inv: &glam::Mat4, w: u32, h: u32, x: f32, y: f32) -> Vec3 {
    let ndc_x = (x / w as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - (y / h as f32) * 2.0;
    (*inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0))
        .truncate()
        .normalize()
}

pub(super) fn tile_at(x: f32, y: f32, tile: u32, tiles_x: u32) -> usize {
    let tx = (x.max(0.0) as u32) / tile;
    let ty = (y.max(0.0) as u32) / tile;
    (ty * tiles_x + tx) as usize
}

pub(super) fn mapped_down(distance: f32, radius: f32, horizon: f32, air: f32) -> FarBody {
    FarBody {
        dir: -Vec3::Y,
        distance,
        radius,
        shape: FarShape::Mapped {
            map: FarMapId(0),
            horizon,
            air,
        },
        rotation: Quat::IDENTITY,
        albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
        atmosphere: LinearRgb([0.1, 0.2, 0.3]),
        seed: 7,
    }
}

/// A `6·g²` datum, `height` of each chart-sample direction, in the order the
/// far map stores it (face, then row, then column).
pub(super) fn datum_from(g: u32, mut height: impl FnMut(Vec3) -> f32) -> Vec<f32> {
    let gg = g as usize;
    let mut datum = vec![0.0f32; 6 * gg * gg];
    for face in 0..6usize {
        for j in 0..g {
            for i in 0..g {
                datum[face * gg * gg + j as usize * gg + i as usize] =
                    height(chart_sample_dir(g, face, i, j));
            }
        }
    }
    datum
}
