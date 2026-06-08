/// This is the main file for all the resources related to how new frame data is submitted.
use ash::vk;

// A buffer contains GPU accessable data, this buffer type will be used to easily pass vertex data
// to the shaders after command buffers have been recorded. A new buffer will have to created every
// frame since the contents of it may change from frame to frame in a practical application.
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub element_count: u32,
}
impl Buffer {
    pub fn new<T>(
        ctx: &BufferContext,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self, vk::Result> {
        let size = std::mem::size_of_val(data) as vk::DeviceSize;

        let element_count = data.len() as u32;

        // Creating the buffer (handle)
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            ctx.device.create_buffer(&buffer_info, None)?
        };

        // Determining compatibility and type   
        let requirements = unsafe {
            ctx.device.get_buffer_memory_requirements(buffer)
        };
        let memory_type_index = find_memory_type(
            ctx.instance,
            ctx.physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        // Create the buffer memory (the actual data)
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe {
            ctx.device.allocate_memory(&alloc_info, None)?
        };

        unsafe {
            ctx.device.bind_buffer_memory(buffer, memory, 0)?;
        }

        unsafe {
            let ptr = ctx.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;

            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr as *mut u8,
                size as usize,
            );

            ctx.device.unmap_memory(memory);
        }

        Ok(Self {
            buffer,
            memory,
            size,
            element_count,
        })
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

pub struct BufferContext<'a> {
    pub instance: &'a ash::Instance,
    pub device: &'a ash::Device,
    pub physical_device: vk::PhysicalDevice,
}
impl<'a> BufferContext<'a> {
    pub fn new(instance: &'a ash::Instance,
               device: &'a ash::Device,
               physical_device: vk::PhysicalDevice,
              ) -> Self {
        Self { instance, device, physical_device }
    }
}

// Macro for creating a vertex buffer more easily, JUS SOME SYNTAX SUGAAAARRR BABAY!!!
#[macro_export]
macro_rules! vertex_buffer {
    ($ctx:expr, $($vertex:expr),* $(,)?) => {{
        let data = vec![$($vertex),*];

        Buffer::new(
            $ctx,
            &data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )
    }};
}

#[macro_export]
macro_rules! index_buffer {
    ($ctx:expr, $($index:expr),* $(,)?) => {{
        let data = vec![$($index as u32),*];

        Buffer::new(
            $ctx,
            &data,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )
    }};
}
fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let memory_properties = unsafe {
        instance.get_physical_device_memory_properties(physical_device)
    };

    for i in 0..memory_properties.memory_type_count {
        let suitable_type = (type_filter & (1 << i)) != 0;

        let has_properties = memory_properties.memory_types[i as usize]
            .property_flags
            .contains(properties);

        if suitable_type && has_properties {
            return i;
        }
    }

    panic!("failed to find suitable memory type");
}
