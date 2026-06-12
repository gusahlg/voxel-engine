/// This file has the defintion for the depth image that is used during rendering
use ash::vk;
pub struct DepthImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub fotmat: vk::Format,
}
