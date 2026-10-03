//! Render-side cage table. Uploads ride the ordered command stream, one
//! host-visible copy per frame in flight, same discipline as the record table.
//! Entry 0 stays zero ("no cage") so the descriptor is valid before any cage
//! is created.

use std::num::NonZeroU32;

use ash::vk;

use super::host_buffer::HostBuffer;
use super::pipeline::EyeSplit;
use crate::cage::{CageGpu, cage_change_bumps};
use crate::rev::FRAMES_IN_FLIGHT;

struct CageCopy {
    buf: HostBuffer,
    dirty: Vec<u32>,
}

/// Result of a generation-checked set or free.
pub(crate) enum CageEdit {
    Ignored,
    Unchanged,
    Changed {
        old: Option<CageGpu>,
        new: Option<CageGpu>,
    },
}

pub(crate) struct CageTable {
    /// Parallel to `entries`. `None` is an empty slot (entry bytes are zero).
    generations: Vec<Option<NonZeroU32>>,
    /// Contiguous so a flush casts this slice and does not allocate.
    entries: Vec<CageGpu>,
    gpu: [CageCopy; FRAMES_IN_FLIGHT as usize],
    eye: Option<(EyeSplit, [f32; 2])>,
}

impl CageTable {
    pub fn new() -> Self {
        Self {
            generations: vec![None],
            entries: vec![CageGpu::ZERO],
            gpu: std::array::from_fn(|_| CageCopy {
                buf: HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER),
                dirty: Vec::new(),
            }),
            eye: None,
        }
    }

    /// Eye and cascade-sphere radii from the frame that just prepared draws.
    /// The next `set`/`free` (drained before the following prepare) tests these.
    pub fn note_eye(&mut self, eye: EyeSplit, radii: [f32; 2]) {
        self.eye = Some((eye, radii));
    }

    pub fn entry(&self, index: u32) -> Option<CageGpu> {
        self.generations.get(index as usize)?.as_ref()?;
        Some(self.entries[index as usize])
    }

    pub fn bumps(&self, edit: &CageEdit) -> bool {
        let CageEdit::Changed { old, new } = edit else {
            return false;
        };
        let eye = self
            .eye
            .map(|(e, radii)| (e.block, e.frac, radii));
        cage_change_bumps(old.as_ref(), new.as_ref(), eye)
    }

    pub fn set(&mut self, index: u32, generation: NonZeroU32, entry: CageGpu) -> CageEdit {
        if index == 0 {
            return CageEdit::Ignored;
        }
        self.grow(index);
        let i = index as usize;
        if let Some(stored) = self.generations[i]
            && stored != generation
        {
            return CageEdit::Ignored;
        }
        if self.generations[i] == Some(generation) && self.entries[i] == entry {
            return CageEdit::Unchanged;
        }
        let old = self.generations[i].map(|_| self.entries[i]);
        self.generations[i] = Some(generation);
        self.entries[i] = entry;
        self.mark(index);
        CageEdit::Changed {
            old,
            new: Some(entry),
        }
    }

    pub fn free(&mut self, index: u32, generation: NonZeroU32) -> CageEdit {
        let Some(stored) = self.generations.get(index as usize).copied() else {
            return CageEdit::Ignored;
        };
        if stored != Some(generation) {
            return CageEdit::Ignored;
        }
        let i = index as usize;
        let old = self.entries[i];
        self.generations[i] = None;
        self.entries[i] = CageGpu::ZERO;
        self.mark(index);
        CageEdit::Changed {
            old: Some(old),
            new: None,
        }
    }

    fn grow(&mut self, index: u32) {
        let n = index as usize + 1;
        if self.entries.len() < n {
            self.entries.resize(n, CageGpu::ZERO);
            self.generations.resize(n, None);
        }
    }

    fn mark(&mut self, index: u32) {
        for copy in &mut self.gpu {
            copy.dirty.push(index);
        }
    }

    /// Upload dirty slots into this frame's copy. `None` if the device buffer
    /// could not be allocated. Entry 0 is always the zero cage.
    pub unsafe fn flush(
        &mut self,
        slot: usize,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Option<vk::Buffer> {
        let copy = &mut self.gpu[slot];
        let bytes: &[u8] = bytemuck::cast_slice(&self.entries);
        let bound = unsafe {
            let grew = copy
                .buf
                .maintain(instance, device, physical, bytes.len() as u64);
            if grew {
                copy.buf.write(0, bytes);
            } else {
                const STRIDE: usize = std::mem::size_of::<CageGpu>();
                for &s in &copy.dirty {
                    let s = s as usize;
                    copy.buf
                        .write((s * STRIDE) as u64, &bytes[s * STRIDE..(s + 1) * STRIDE]);
                }
            }
            copy.buf.bound()
        };
        copy.dirty.clear();
        bound
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for copy in &mut self.gpu {
            unsafe { copy.buf.destroy(device) };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use glam::IVec3;

    use super::*;
    use crate::cage::identity_corners;

    fn nz(v: u32) -> NonZeroU32 {
        NonZeroU32::new(v).unwrap()
    }

    fn eye() -> EyeSplit {
        EyeSplit {
            block: [0; 3],
            _pad0: 0,
            frac: [0.0; 3],
            _pad1: 0.0,
        }
    }

    #[test]
    fn set_bumps_only_when_a_corner_box_meets_a_cascade_sphere() {
        let mut t = CageTable::new();
        let inside = CageGpu::from_corners(IVec3::ZERO, identity_corners());
        let outside = CageGpu::from_corners(IVec3::new(10_000, 0, 0), identity_corners());
        let further = CageGpu::from_corners(IVec3::new(20_000, 0, 0), identity_corners());
        let g = nz(1);

        // No eye stored yet: any real change bumps.
        let edit = t.set(1, g, inside);
        assert!(t.bumps(&edit));
        t.note_eye(eye(), [68.0, 260.0]);

        assert!(matches!(t.set(1, g, inside), CageEdit::Unchanged));

        // Leaving the spheres still bumps, because the old box was inside.
        let edit = t.set(1, g, outside);
        assert!(t.bumps(&edit));
        // Both boxes outside: the cascades do not care.
        let edit = t.set(1, g, further);
        assert!(!t.bumps(&edit));
        // Entering again bumps.
        let edit = t.set(1, g, inside);
        assert!(t.bumps(&edit));

        assert!(matches!(t.set(1, nz(2), outside), CageEdit::Ignored));
        assert!(matches!(t.free(1, nz(2)), CageEdit::Ignored));
        let edit = t.free(1, g);
        assert!(t.bumps(&edit));
        assert!(t.entry(1).is_none());
    }
}
