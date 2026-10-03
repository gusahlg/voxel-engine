//! Bent-mesh cages. A cage is the eight physical corners of one chunk; the
//! vertex shader interpolates through them. Slot 0 of the GPU table is "no
//! cage", so a handle's table index is `slot + 1`.

use std::num::NonZeroU32;

use glam::{IVec3, Vec3};

use crate::vk::handles::GpuHandle;

/// Generational handle to one cage. `Option<CageHandle>` stays 8 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CageHandle {
    pub(crate) slot: u32,
    pub(crate) generation: NonZeroU32,
}

impl CageHandle {
    /// Index in the cage SSBO. Zero is reserved for "no cage".
    pub(crate) fn gpu_index(self) -> u32 {
        self.slot.wrapping_add(1)
    }
}

impl GpuHandle for CageHandle {
    fn from_parts(slot: u32, generation: NonZeroU32) -> Self {
        Self { slot, generation }
    }
    fn slot(self) -> u32 {
        self.slot
    }
    fn generation(self) -> NonZeroU32 {
        self.generation
    }
}

/// One cage in the SSBO. `corners` is a std430 `float3[8]` (16-byte stride);
/// `.w` is padding. Corner `i`: bit 0 = +x, bit 1 = +y, bit 2 = +z.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CageGpu {
    pub anchor: [i32; 3],
    pub _pad: u32,
    pub corners: [[f32; 4]; 8],
}

const _: () = assert!(std::mem::size_of::<CageGpu>() == 144);
const _: () = assert!(std::mem::offset_of!(CageGpu, corners) == 16);

impl CageGpu {
    pub(crate) const ZERO: Self = Self {
        anchor: [0; 3],
        _pad: 0,
        corners: [[0.0; 4]; 8],
    };

    pub(crate) fn from_corners(anchor: IVec3, corners: [Vec3; 8]) -> Self {
        let mut packed = [[0.0; 4]; 8];
        for (i, c) in corners.into_iter().enumerate() {
            packed[i] = [c.x, c.y, c.z, 0.0];
        }
        Self {
            anchor: anchor.to_array(),
            _pad: 0,
            corners: packed,
        }
    }

    pub(crate) fn corners3(self) -> [[f32; 3]; 8] {
        std::array::from_fn(|i| [self.corners[i][0], self.corners[i][1], self.corners[i][2]])
    }

    /// True when the corner AABB, placed at `anchor`, meets an eye-centred sphere.
    pub(crate) fn hits_sphere(self, eye_block: [i32; 3], eye_frac: [f32; 3], radius: f32) -> bool {
        let (mn, mx) = corner_aabb(self.corners3());
        let mut d2 = 0.0f32;
        for i in 0..3 {
            let o = self.anchor[i].wrapping_sub(eye_block[i]) as f32 - eye_frac[i];
            let p = 0.0f32.clamp(mn[i] + o, mx[i] + o);
            d2 += p * p;
        }
        d2 <= radius * radius
    }
}

/// Identity cage: the mesh's own unit box, in blocks. Exact for scale 1.
#[cfg(test)]
pub(crate) fn identity_corners() -> [Vec3; 8] {
    std::array::from_fn(|i| {
        Vec3::new(
            if i & 1 != 0 { 16.0 } else { 0.0 },
            if i & 2 != 0 { 16.0 } else { 0.0 },
            if i & 4 != 0 { 16.0 } else { 0.0 },
        )
    })
}

/// `(x, y, z) → (z, y, −x)`. A 90° turn with determinant +1; no trig.
#[cfg(test)]
pub(crate) fn rotate_z_y_neg_x(p: Vec3) -> Vec3 {
    Vec3::new(p.z, p.y, -p.x)
}

#[cfg(test)]
fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Trilinear map. `t` is not clamped: a micro-offset may sit just outside [0, 1].
#[cfg(test)]
pub(crate) fn trilinear(corners: [[f32; 3]; 8], t: [f32; 3]) -> [f32; 3] {
    let c0 = lerp(corners[0], corners[1], t[0]);
    let c1 = lerp(corners[2], corners[3], t[0]);
    let c2 = lerp(corners[4], corners[5], t[0]);
    let c3 = lerp(corners[6], corners[7], t[0]);
    let d0 = lerp(c0, c1, t[1]);
    let d1 = lerp(c2, c3, t[1]);
    lerp(d0, d1, t[2])
}

#[cfg(test)]
fn dtx(corners: [[f32; 3]; 8], t: [f32; 3]) -> [f32; 3] {
    let e0 = sub(corners[1], corners[0]);
    let e1 = sub(corners[3], corners[2]);
    let e2 = sub(corners[5], corners[4]);
    let e3 = sub(corners[7], corners[6]);
    lerp(lerp(e0, e1, t[1]), lerp(e2, e3, t[1]), t[2])
}

#[cfg(test)]
fn dty(corners: [[f32; 3]; 8], t: [f32; 3]) -> [f32; 3] {
    let c0 = lerp(corners[0], corners[1], t[0]);
    let c1 = lerp(corners[2], corners[3], t[0]);
    let c2 = lerp(corners[4], corners[5], t[0]);
    let c3 = lerp(corners[6], corners[7], t[0]);
    lerp(sub(c1, c0), sub(c3, c2), t[2])
}

#[cfg(test)]
fn dtz(corners: [[f32; 3]; 8], t: [f32; 3]) -> [f32; 3] {
    let c0 = lerp(corners[0], corners[1], t[0]);
    let c1 = lerp(corners[2], corners[3], t[0]);
    let c2 = lerp(corners[4], corners[5], t[0]);
    let c3 = lerp(corners[6], corners[7], t[0]);
    sub(lerp(c2, c3, t[1]), lerp(c0, c1, t[1]))
}

#[cfg(test)]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(test)]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
const FACE_NORMAL: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
];

/// Unit face normal from the two tangents of that face. A collapsed tangent
/// falls back to the flat face normal.
#[cfg(test)]
pub(crate) fn face_normal(corners: [[f32; 3]; 8], t: [f32; 3], face: u32) -> [f32; 3] {
    let tx = dtx(corners, t);
    let ty = dty(corners, t);
    let tz = dtz(corners, t);
    let mut n = match face {
        0 | 1 => cross(ty, tz),
        2 | 3 => cross(tz, tx),
        _ => cross(tx, ty),
    };
    if face & 1 != 0 {
        n = [-n[0], -n[1], -n[2]];
    }
    let len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if len2 == 0.0 {
        return FACE_NORMAL[(face as usize) % 6];
    }
    let inv = 1.0 / len2.sqrt();
    [n[0] * inv, n[1] * inv, n[2] * inv]
}

pub(crate) fn corner_aabb(corners: [[f32; 3]; 8]) -> ([f32; 3], [f32; 3]) {
    let mut mn = corners[0];
    let mut mx = corners[0];
    for c in &corners[1..] {
        for k in 0..3 {
            mn[k] = mn[k].min(c[k]);
            mx[k] = mx[k].max(c[k]);
        }
    }
    (mn, mx)
}

/// Camera-relative position of local `q` through the cage. `q` is the cell
/// position times the detail scale, in `[0, 16·scale]³` before micro-offsets.
#[cfg(test)]
pub(crate) fn placed(
    anchor: [i32; 3],
    corners: [[f32; 3]; 8],
    q: [f32; 3],
    scale: f32,
    cam_block: [i32; 3],
    cam_frac: [f32; 3],
) -> [f32; 3] {
    let edge = 16.0 * scale;
    let t = [q[0] / edge, q[1] / edge, q[2] / edge];
    let p = trilinear(corners, t);
    let d = [
        anchor[0].wrapping_sub(cam_block[0]) as f32 - cam_frac[0],
        anchor[1].wrapping_sub(cam_block[1]) as f32 - cam_frac[1],
        anchor[2].wrapping_sub(cam_block[2]) as f32 - cam_frac[2],
    ];
    [p[0] + d[0], p[1] + d[1], p[2] + d[2]]
}

/// Camera-relative position of an uncaged mesh: `q + local_off + (block − cam)`.
#[cfg(test)]
pub(crate) fn uncaged(
    q: [f32; 3],
    block: [i32; 3],
    local_off: [f32; 3],
    cam_block: [i32; 3],
    cam_frac: [f32; 3],
) -> [f32; 3] {
    let d = [
        block[0].wrapping_sub(cam_block[0]) as f32 - cam_frac[0] + local_off[0],
        block[1].wrapping_sub(cam_block[1]) as f32 - cam_frac[1] + local_off[1],
        block[2].wrapping_sub(cam_block[2]) as f32 - cam_frac[2] + local_off[2],
    ];
    [q[0] + d[0], q[1] + d[1], q[2] + d[2]]
}

/// Whether a corner change must rebuild cascade shadows. No stored eye yet:
/// bump. Otherwise bump when the old or the new box meets a cascade sphere.
pub(crate) fn cage_change_bumps(
    old: Option<&CageGpu>,
    new: Option<&CageGpu>,
    eye: Option<([i32; 3], [f32; 3], [f32; 2])>,
) -> bool {
    let Some((block, frac, radii)) = eye else {
        return true;
    };
    let hit = |c: &CageGpu| radii.iter().any(|&r| c.hits_sphere(block, frac, r));
    old.is_some_and(|c| hit(c)) || new.is_some_and(|c| hit(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corners_of(vs: [Vec3; 8]) -> [[f32; 3]; 8] {
        vs.map(|v| v.to_array())
    }

    #[test]
    fn cage_gpu_is_std430_144_with_corners_at_16() {
        assert_eq!(std::mem::size_of::<CageGpu>(), 144);
        assert_eq!(std::mem::offset_of!(CageGpu, corners), 16);
        assert_eq!(std::mem::align_of::<CageGpu>(), 4);
    }

    #[test]
    fn identity_cage_matches_uncaged_positions_exactly() {
        let corners = corners_of(identity_corners());
        let (mn, mx) = corner_aabb(corners);
        assert_eq!(mn, [0.0; 3]);
        assert_eq!(mx, [16.0; 3]);
        let cam_block = [i32::MAX, -3, 40];
        let cam_frac = [0.25, 0.5, 0.0];
        let anchor = [i32::MIN, 100, -7];
        for x in 0..=16 {
            for y in [0, 3, 5, 9, 16] {
                for z in [0, 1, 8, 16] {
                    let q = [x as f32, y as f32, z as f32];
                    let bent = placed(anchor, corners, q, 1.0, cam_block, cam_frac);
                    let flat = uncaged(q, anchor, [0.0; 3], cam_block, cam_frac);
                    assert_eq!(bent, flat, "q={q:?}");
                }
            }
        }
    }

    #[test]
    fn rotated_cage_matches_the_rotated_mesh_and_its_face_normal() {
        let id = identity_corners();
        let rotated = corners_of(id.map(rotate_z_y_neg_x));
        let anchor = [4, -2, 9];
        let cam_block = [1, 2, 3];
        let cam_frac = [0.0; 3];
        for q in [[0.0, 0.0, 0.0], [16.0, 0.0, 0.0], [0.0, 16.0, 8.0], [4.0, 8.0, 12.0]] {
            let bent = placed(anchor, rotated, q, 1.0, cam_block, cam_frac);
            let spun = rotate_z_y_neg_x(Vec3::from(q)).to_array();
            let expect = uncaged(spun, anchor, [0.0; 3], cam_block, cam_frac);
            assert_eq!(bent, expect, "q={q:?}");
        }
        // +X (1,0,0) rotates to −Z. The face is the YZ tangents.
        let n = face_normal(rotated, [1.0, 0.5, 0.25], 0);
        assert_eq!(n, [0.0, 0.0, -1.0]);
        let id_n = face_normal(corners_of(id), [0.0, 0.0, 0.0], 1);
        assert_eq!(id_n, [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn collapsed_cage_falls_back_to_the_flat_face_normal() {
        let corners = [[0.0; 3]; 8];
        assert_eq!(face_normal(corners, [0.5; 3], 2), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn sphere_test_uses_the_corner_box_not_the_local_cell_box() {
        let eye = [0; 3];
        let frac = [0.0; 3];
        let near = CageGpu::from_corners(IVec3::ZERO, identity_corners());
        assert!(near.hits_sphere(eye, frac, 4.0));
        let far = CageGpu::from_corners(IVec3::new(10_000, 0, 0), identity_corners());
        assert!(!far.hits_sphere(eye, frac, 68.0));
        assert!(far.hits_sphere(eye, frac, 20_000.0));
        // Old inside, new outside still counts as a hit for the caller.
        assert!(cage_change_bumps(
            Some(&near),
            Some(&far),
            Some((eye, frac, [68.0, 260.0]))
        ));
        assert!(!cage_change_bumps(
            Some(&far),
            Some(&far),
            Some((eye, frac, [68.0, 260.0]))
        ));
        assert!(cage_change_bumps(Some(&far), Some(&near), None));
    }
}
