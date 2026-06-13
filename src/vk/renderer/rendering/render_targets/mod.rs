mod depth_image;
pub use depth_image::*;

pub struct RenderTargets {
    depth_image: DepthImage,
}
