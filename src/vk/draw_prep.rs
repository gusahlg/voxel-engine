//! CPU Blend draw prep and packed immediate vertices. Split out of `frame_loop`
//! (move only).

use ash::vk;

use crate::frame::DrawLists;
use crate::mesh::Pass;
use crate::skeleton::FrameSlot;

use super::Renderer;
use super::buffers::DrawIndexedIndirect;
use super::pipeline;
use super::shadow;

/// Byte offsets within a frame's packed immediate buffer.
#[derive(Clone, Copy)]
pub(crate) struct ImmOffsets {
    pub(super) line: u64,
    pub(super) shadow: u64,
    pub(super) d2: u64,
    pub(super) d2_tex: u64,
}

/// Resolved mesh draw (one direction-run or whole mesh), pre-sort scratch.
/// Placement/style live in the persistent record SSBOs, reached through
/// `slot`; this carries only what the CPU sort/batch needs.
#[derive(Clone, Copy)]
pub(crate) struct DrawEntry {
    buffer: vk::Buffer,
    pass: Pass,
    first: u32,
    count: u32,
    vertex_offset: i32,
    /// Mesh slot: the emitted command's `first_instance`, indexing the
    /// record/dyn SSBOs in the vertex shader.
    slot: u32,
    /// Squared distance to AABB center (monotonic; back-to-front sort key).
    dist2: f32,
}
/// Contiguous indirect commands sharing one buffer and pass.
#[derive(Clone, Copy)]
pub(crate) struct DrawRun {
    pub(super) buffer: vk::Buffer,
    pub(super) pass: Pass,
    pub(super) first: u32,
    pub(super) count: u32,
}
/// Shadow-map content key: sun/eye-snap/occluders plus hashed avatar casters.
fn shadow_key(
    eye: glam::DVec3,
    sun: glam::DVec3,
    occluders: u64,
    lists: &DrawLists,
    cfg: &crate::skeleton::ShadowCfg,
) -> shadow::ShadowKey {
    shadow::ShadowKey::of(
        eye,
        sun,
        occluders,
        shadow::hash_casters(bytemuck::cast_slice(&lists.cube_verts)),
        cfg,
    )
}

impl Renderer {
    /// Packs frame immediates (cubes, lines, 2D) into host buffer and returns offsets.
    ///
    /// 2D verts (`d2` / `d2_tex`) are read solely by the present overlay
    /// (`present.rs::record_overlay_present`). Skip those two writes when this
    /// frame will not present; a forced capture always presents. Offset math and
    /// `imm.maintain(total)` stay identical so the buffer layout is stable.
    pub(super) fn write_immediates(
        &mut self,
        slot: usize,
        lists: &DrawLists,
        will_present: bool,
    ) -> ImmOffsets {
        let cube_bytes: &[u8] = bytemuck::cast_slice(&lists.cube_verts);
        let line_bytes: &[u8] = bytemuck::cast_slice(&lists.line_verts);
        let shadow_bytes: &[u8] = bytemuck::cast_slice(&lists.shadow_verts);
        let d2_bytes: &[u8] = bytemuck::cast_slice(&lists.verts_2d);
        let d2_tex_bytes: &[u8] = bytemuck::cast_slice(&lists.tex_verts_2d);
        let line = (cube_bytes.len() as u64).next_multiple_of(16);
        let shadow = (line + line_bytes.len() as u64).next_multiple_of(16);
        let d2 = (shadow + shadow_bytes.len() as u64).next_multiple_of(16);
        let d2_tex = (d2 + d2_bytes.len() as u64).next_multiple_of(16);
        let total = d2_tex + d2_tex_bytes.len() as u64;
        let imm = &mut self.slots[FrameSlot::new(slot)].imm;
        unsafe {
            imm.maintain(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                total,
            );
            if total > 0 {
                imm.write(0, cube_bytes);
                imm.write(line, line_bytes);
                imm.write(shadow, shadow_bytes);
                if will_present {
                    imm.write(d2, d2_bytes);
                    imm.write(d2_tex, d2_tex_bytes);
                }
            }
        }
        ImmOffsets {
            line,
            shadow,
            d2,
            d2_tex,
        }
    }

    /// CPU Blend re-source: transparency needs exact far→near ordering the GPU
    /// cull does not provide, so Blend is the ONE pass still resolved CPU-side.
    /// Sourced from the same persistent records, arena directory, and
    /// `visible_mask` — not a per-frame draw list — by iterating resident,
    /// visible Blend-pass slots, frustum-culling, sorting by distance, and
    /// emitting whole-mesh indirect commands (`first_instance = slot` so
    /// placement/style come from the record/dyn SSBOs).
    ///
    /// Safe to run before the slot fence: everything it reads is render-thread
    /// state that cannot change until the next command drain (reclaim frees
    /// only already-retired allocations). The indirect buffer write stays in
    /// [`Self::prepare_mesh_draws`] after the wait.
    pub(super) fn prepare_blend_draws(
        &mut self,
        lists: &DrawLists,
        camera: &crate::camera::Frustum,
        eye: pipeline::EyeSplit,
    ) {
        use ash::vk::Handle;

        self.draw_scratch.clear();
        self.draw_commands.clear();
        self.draw_runs.clear();

        // Walk the resident, visible, Blend-pass records. The directory's Blend
        // set is exactly that candidate list, so this is O(transparent meshes),
        // never a sweep of the whole slot table.
        let Some(scene) = &lists.scene else {
            return;
        };
        for &s in self.arena_dir.blend_slots() {
            // Arena word (0 = not resident) is the arena index + 1, giving
            // the vertex buffer without a residency-handle lookup. Gated
            // on `is_arrived` too: a budget-deferred copy is registered in
            // `arena_dir` (capacity/ref-count bookkeeping happens at
            // upload) before its bytes actually land — reading it here
            // early would source the CPU Blend draw from uninitialized
            // arena memory, same hazard `RecordTable::flush` guards for
            // the GPU cull path.
            let arena = self.arena_dir.arena_word(s as usize);
            if arena == 0 || !self.mesh_res.is_arrived(s) {
                continue;
            }
            // The same persistent mask the GPU cull reads.
            let visible = self
                .visible_mask
                .get((s >> 5) as usize)
                .is_some_and(|w| w & (1 << (s & 31)) != 0);
            if !visible {
                continue;
            }
            let Some(rec) = self.records.record(s) else {
                continue;
            };
            debug_assert_eq!(
                rec.pass(),
                Pass::Blend,
                "Blend set holds a non-Blend record"
            );
            // Camera-relative placement reconstructed exactly as the vertex
            // shader does (integer block minus camera block, then the
            // fractional remainder), so the CPU sort/cull agrees with the GPU
            // draw. `detail_scale` decodes the BIASED detail field — never
            // decode `detail_pass` here by hand (it carries a to_gpu_bits offset).
            let scale = rec.detail_scale();
            let offset = glam::Vec3::new(
                (rec.block[0] - eye.block[0]) as f32 - eye.frac[0] + rec.local_off[0],
                (rec.block[1] - eye.block[1]) as f32 - eye.frac[1] + rec.local_off[1],
                (rec.block[2] - eye.block[2]) as f32 - eye.frac[2] + rec.local_off[2],
            );
            let amin = glam::Vec3::from(rec.aabb_min);
            let amax = glam::Vec3::from(rec.aabb_max);
            if !camera.intersects_aabb(amin * scale + offset, amax * scale + offset) {
                continue;
            }
            let center = offset + (amin + amax) * 0.5 * scale;
            let dist2 = (center - scene.cam_pos).length_squared();
            self.draw_scratch.push(DrawEntry {
                buffer: self.arena_dir.arena_buffer((arena - 1) as usize),
                pass: Pass::Blend,
                first: 0,
                count: rec.index_count,
                vertex_offset: rec.vertex_offset,
                slot: s,
                dist2,
            });
        }
        // Blend far→near for correct back-to-front alpha compositing.
        self.draw_scratch.sort_unstable_by(|a, b| {
            b.dist2
                .total_cmp(&a.dist2)
                // Deterministic tiebreak; keeps equidistant same-arena draws batched.
                .then_with(|| a.buffer.as_raw().cmp(&b.buffer.as_raw()))
        });

        for entry in &self.draw_scratch {
            let command_index = self.draw_commands.len() as u32;
            self.draw_commands.push(DrawIndexedIndirect {
                index_count: entry.count,
                instance_count: 1,
                first_index: entry.first,
                vertex_offset: entry.vertex_offset,
                first_instance: entry.slot,
            });
            match self.draw_runs.last_mut() {
                Some(run) if run.buffer == entry.buffer && run.pass == entry.pass => run.count += 1,
                _ => self.draw_runs.push(DrawRun {
                    buffer: entry.buffer,
                    pass: entry.pass,
                    first: command_index,
                    count: 1,
                }),
            }
        }
        crate::profile::gauge(
            crate::profile::Gauge::DrawsBlend,
            self.draw_commands.len() as u64,
        );
    }

    /// GPU-cull prep, shadow fits, and the Blend indirect write. Runs after the
    /// slot fence: flush and HostBuffer maintains must not race the previous
    /// use of this slot. `camera`/`eye` were computed before the wait.
    pub(super) fn prepare_mesh_draws(
        &mut self,
        slot: usize,
        lists: &DrawLists,
        sun: glam::DVec3,
        camera_eye: Option<&(crate::camera::Frustum, pipeline::EyeSplit)>,
    ) {
        // Flush record/dyn patches into this slot's copies (post-fence, same
        // discipline as the HostBuffer maintains below).
        self.record_buffers = unsafe {
            self.records.flush(
                slot,
                &self.arena_dir,
                &self.mesh_res,
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
            )
        };

        // Cull emits shadow casters only when the shared map will actually be
        // rewritten this frame. A cache hit skips cascade `fit()` and sets
        // `shadow_enabled = 0` so invisible slots bail before the AABB load.
        // Far cascade radius follows full-res coverage (`lod_clip`); a render-
        // distance change snaps `ShadowKey` from `cfg.splits` and rebuilds once.
        let cfg = crate::skeleton::ShadowCfg::for_coverage(lists.lod_clip);
        let shadow_frusta = lists.scene.as_ref().and_then(|scene| {
            let rebuild = if self.flags.shadows {
                let key = shadow_key(scene.eye, sun, self.records.occluder_rev(), lists, &cfg);
                self.shadow_cache.prepare(Some((key, &cfg)))
            } else {
                self.shadow_cache.prepare(None)
            };
            if !rebuild {
                return None;
            }
            let fits = shadow::PerCascade::new(
                shadow::CASCADES.map(|c| shadow::fit(scene.eye, sun, c, &cfg)),
            );
            self.shadow_cache.store_fits(fits);
            self.flags.shadows.then(|| {
                shadow::CASCADES
                    .map(|c| crate::camera::Frustum::from_view_proj(&fits[c].view_proj.0))
            })
        });

        // GPU cull prep: the persistent visibility mask IS the dispatch's
        // visibility input, grown to cover every live slot (zero = hidden).
        // The dispatch (and the mask it reads) stops at the directory's live
        // end, not the table length: a freed tail is all dead words.
        // Last frame's partition table is handed back so its allocation is
        // reused rather than rebuilt from scratch every frame.
        let recycled = self
            .cull_frame
            .take()
            .map_or_else(Vec::new, |f| f.partitions);
        self.cull_frame = if let Some((camera, eye)) = camera_eye {
            if let Some(records) = self.record_buffers {
                let slot_count = records.slots.min(self.arena_dir.live_end());
                let need = slot_count.div_ceil(32) as usize;
                if self.visible_mask.len() < need {
                    self.visible_mask.resize(need, 0);
                }
                unsafe {
                    self.cull.prepare(
                        slot,
                        &self.instance.instance,
                        &self.device.device,
                        self.device.physical,
                        &mut self.arena_dir,
                        records,
                        self.records.records(),
                        |s| self.mesh_res.is_arrived(s),
                        slot_count,
                        camera,
                        shadow_frusta.as_ref(),
                        *eye,
                        lists.lod_clip,
                        lists.lod_clip_v,
                        &self.visible_mask[..need],
                        recycled,
                    )
                }
            } else {
                None
            }
        } else {
            None
        };

        let indirect_bytes: &[u8] = bytemuck::cast_slice(&self.draw_commands);
        unsafe {
            let indirect = &mut self.slots[FrameSlot::new(slot)].indirect;
            indirect.maintain(
                &self.instance.instance,
                &self.device.device,
                self.device.physical,
                indirect_bytes.len() as u64,
            );
            if !indirect_bytes.is_empty() {
                indirect.write(0, indirect_bytes);
            }
        }
        crate::profile::gauge(
            crate::profile::Gauge::DrawsPacked,
            self.draw_commands.len() as u64,
        );
    }
}
