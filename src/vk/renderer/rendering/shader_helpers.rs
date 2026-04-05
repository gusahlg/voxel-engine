use ash::Device;
use ash::{vk, util};

use std::{fs::File, io::BufReader, path::Path};

pub fn read_spv(path: impl AsRef<Path>) -> Vec<u32> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    util::read_spv(&mut reader).unwrap()
}

pub unsafe fn create_shader_module(device: &Device, code: &[u32]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}
