//! Compatibility shim: re-exports from owning modules under skeleton:: path for backward compatibility.
pub use crate::rev::{FrameSlot, PerSlot};
pub use crate::vk::exposure::{Exposure, ExposureRead, ExposureWrite};
pub use crate::vk::shadow::{
    CASCADE_UNIFORMS_BINDING, Cascade, CascadeFit, CascadeUniformsGpu, PerCascade, ShadowCfg,
};
pub use crate::vk::taa::{
    CleanViewProj, JitterOffset, Reprojection, TAA_HISTORY_FORMAT, TEMPORAL_SEQ_LEN, jitter_at,
};
#[allow(deprecated)]
pub use crate::vk::taa::{TAA_RESOLVE_CURRENT_BINDING, TAA_RESOLVE_HISTORY_BINDING};
pub use crate::vk::uniforms::{
    FRAME_UNIFORMS_BINDING, FRAME_UNIFORMS_SET, FRAME_UNIFORMS_VERSION, FrameUniformsGpu,
};
// Backward compatibility re-exports.
pub use crate::{Screenshot, load_png, screenshot_to};
