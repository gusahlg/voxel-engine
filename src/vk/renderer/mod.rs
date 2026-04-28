use winit::event_loop::ActiveEventLoop;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use ash::vk;
use std::mem::ManuallyDrop;

mod device;
use device::Device;

mod swapchain;
use swapchain::SwapchainInfo;

mod rendering;
use rendering::*;

mod constants;
use constants as VkConsts;

mod frame;
use frame::*;

// Holds all core Vulkan state and the window. Created in App::resumed() once
// the event loop is active and we can obtain platform display/window handles.
pub struct Renderer {
    // Abstraction that stores:
    //      * Graphics and present queue
    //      * Physical device
    //      * Logical device
    //      * Command pool
    device: ManuallyDrop<Device>,

    _vk_entry: ash::Entry,
    vk_instance: ash::Instance,

    swapchain_info: SwapchainInfo,

    // For rendering
    pipeline_bundle: RenderingBundle,

    // Synchronization
    frames: Vec<FrameSlot>,
    current_frame: usize,

    // For interacting with the screen
    surface_loader: ash::khr::surface::Instance,
    surface: ash::vk::SurfaceKHR,
    pub window: winit::window::Window,
}

impl Renderer {
    pub fn new(event_loop: &ActiveEventLoop, title: &str) -> Self {
        // Load libvulkan.so and get global function pointers
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan") };

        // Ask the platform which Vulkan extensions are needed for surface creation
        let display_handle = event_loop.display_handle().unwrap().as_raw();
        let extensions = ash_window::enumerate_required_extensions(display_handle)
            .expect("Failed to enumerate required extensions");

        // Create the Vulkan instance with the required surface extensions enabled
        let app_info = ash::vk::ApplicationInfo::default()
            .application_name(c"voxel_engine")
            .application_version(ash::vk::make_api_version(0, 0, 1, 0))
            .engine_version(ash::vk::make_api_version(0, 0, 1, 0))
            .api_version(ash::vk::API_VERSION_1_3);

        let create_info = ash::vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(extensions);

        let instance = unsafe {
            entry.create_instance(&create_info, None)
                .expect("Failed to create Vulkan instance")
        };

        // Load the VK_KHR_surface function pointers (destroy_surface, query capabilities, etc.)
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // Create the window
        let attrs = winit::window::WindowAttributes::default().with_title(title);
        let window = event_loop.create_window(attrs).expect("Failed to create window");

        // Create the Vulkan surface from the window's platform handles
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .expect("Failed to create Vulkan surface")
        };

        // Below is for finding the physical devices and then also adding on the abstraction of the
        // logical device to make the renderer ready to actually interact with the gpu.
        let device = ManuallyDrop::new(Device::new(&instance, &surface_loader, surface));

        let size = window.inner_size();
        let window_extent = ash::vk::Extent2D {
            width: size.width,
            height: size.height,
        };

        // Create swapchain
        let swapchain_info = SwapchainInfo::new(
            &instance,
            device.physical_device,
            &device.logical_device,
            &surface_loader,
            surface,
            window_extent,
            device.graphics_queue_family,
            device.present_queue_family,
        );

        let pipeline_bundle = RenderingBundle::new(
            &device.logical_device,
            swapchain_info.format,
            swapchain_info.extent,
        );

        // COMMAND BUFFERS AND SYNCHRONIZATION
        let mut frames: Vec<FrameSlot> = Vec::with_capacity(VkConsts::MAX_FRAMES_IN_FLIGHT as usize);

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(VkConsts::MAX_FRAMES_IN_FLIGHT.into());

        let command_buffers = unsafe {
            device.logical_device.allocate_command_buffers(&allocate_info).expect("Failed to allocate command buffers")
        };
        
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let semaphore_info = vk::SemaphoreCreateInfo::default();

        for i in 0..command_buffers.len() {
            let fence = unsafe {
                device.logical_device.create_fence(&fence_info, None).expect("Failed to create fence")
            };

            let image_available_semaphore = unsafe {
                device.logical_device.create_semaphore(&semaphore_info, None).expect("Failed to create semaphore")
            };
            let render_finished_semaphore = unsafe {
                device.logical_device.create_semaphore(&semaphore_info, None).expect("Failed to create semaphore")
            };
            frames.push(FrameSlot {
                command_buffer: command_buffers[i],
                in_flight_fence: fence,
                image_available_semaphore,
                render_finished_semaphore,
            });
        }

        Self {
            _vk_entry: entry,
            vk_instance: instance,
            swapchain_info,
            pipeline_bundle,
            frames, 
            current_frame: 0,
            surface_loader,
            surface,
            window,
            device,
        }
    }

    // Function that synchronizes rendering and drawing as a whole
    pub fn draw_frame(&mut self) -> Result<(), vk::Result> {
        if self.swapchain_info.dirty {
            self.swapchain_info.recreate(
                &mut self.pipeline_bundle,
                &mut self.device,
                &self.vk_instance,
                self.surface,
                &self.surface_loader,
                &self.window,
            )?;
        }

        let frame = &self.frames[self.current_frame];

        // Refactor
        unsafe {
            self.device.logical_device.wait_for_fences(&[frame.in_flight_fence], true, u64::MAX,)?;

            let (image_index, suboptimal) = match self.swapchain_info.swapchain_loader.acquire_next_image(
                self.swapchain_info.swapchain,
                u64::MAX,
                frame.image_available_semaphore,
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swapchain_info.recreate(
                        &mut self.pipeline_bundle,
                        &mut self.device,
                        &self.vk_instance,
                        self.surface,
                        &self.surface_loader,
                        &self.window,
                    )?;
                    return Ok(());
                }
                Err(err) => return Err(err),
            };

            // Maybe make resets a function
            self.device.logical_device.reset_fences(&[frame.in_flight_fence])?;

            self.device.logical_device.reset_command_buffer(
                frame.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )?;

            record_command_buffer(&self.device, &self.swapchain_info, self.pipeline_bundle.graphics_pipeline, frame.command_buffer, image_index as usize)?;

            let frame_submit_info = create_submit_info(frame);
            let submit_infos = frame_submit_info.submit_infos();

            self.device.logical_device.queue_submit2(
                self.device.graphics_queue,
                &submit_infos,
                frame.in_flight_fence,
            )?;

            let present_wait_semaphores = [frame.render_finished_semaphore];
            let swapchains = [self.swapchain_info.swapchain];
            let image_indices = [image_index];

            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&present_wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            match self.swapchain_info.swapchain_loader.queue_present(self.device.present_queue, &present_info) {
                Ok(present_suboptimal) => {
                    if suboptimal || present_suboptimal {
                        self.swapchain_info.recreate(
                            &mut self.pipeline_bundle,
                            &mut self.device,
                            &self.vk_instance,
                            self.surface,
                            &self.surface_loader,
                            &self.window,
                        )?;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swapchain_info.recreate(
                        &mut self.pipeline_bundle,
                        &mut self.device,
                        &self.vk_instance,
                        self.surface,
                        &self.surface_loader,
                        &self.window,
                    )?;
                }
                Err(err) => return Err(err),
            }
        }

        self.current_frame = (self.current_frame + 1) % self.frames.len();
        Ok(())
    }

    pub fn request_swapchain_recreation(&mut self) {
        self.swapchain_info.dirty = true;
    }
}

// Destroy in reverse creation order.
// Entry doesn't need explicit cleanup (it's just loaded function pointers).
impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.logical_device.device_wait_idle();

            self.pipeline_bundle.destroy(&self.device.logical_device);

            for frame in &self.frames {
                self.device.logical_device.destroy_semaphore(frame.render_finished_semaphore, None);
                self.device.logical_device.destroy_semaphore(frame.image_available_semaphore, None);
                self.device.logical_device.destroy_fence(frame.in_flight_fence, None);
            }

            for &view in &self.swapchain_info.image_views {
                self.device.logical_device.destroy_image_view(view, None);
            }
            self.swapchain_info.swapchain_loader
                .destroy_swapchain(self.swapchain_info.swapchain, None);
            ManuallyDrop::drop(&mut self.device);
            self.surface_loader.destroy_surface(self.surface, None);
            self.vk_instance.destroy_instance(None);
        }
    }
}
