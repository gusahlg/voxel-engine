/// The file that exposes all of the device functionality, is supposed to act as a thin and
/// practical api that keeps complicated details hidden.

struct Device {
    physical_device: ash::vk::PhysicalDevice,
    logical_device: ash::vk::Device,

    graphics_queue: ash::vk::Queue,
    present_queue: ash::vk::Queue,

    command_pool: ash::vk::CommandPool,
}
impl Device {
    pub fn get_graphics_queue() ->
}

