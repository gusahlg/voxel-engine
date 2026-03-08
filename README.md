SUPER COOL VULKAN VOXEL ENGINE

Step 1 — Find Queue Family Indices
- Call instance.get_physical_device_queue_family_properties(device) to list queue families.
- Find indices for: a graphics family (supports QUEUE_GRAPHICS_BIT) and a present family (supports presenting to your surface, checked via surface_loader.get_physical_device_surface_support(device, index, surface)).
- Store both indices — they might be the same family or different.

Step 2 — Pick a Physical Device
- Call instance.enumerate_physical_devices() to get all GPUs.
- For each, check that it has valid graphics + present queue families (from step 1).
- Optionally prefer DISCRETE_GPU over INTEGRATED_GPU.
- Store the chosen VkPhysicalDevice.

Step 3 — Create the Logical Device
- Build a list of DeviceQueueCreateInfo for each unique queue family index you need (graphics, present). Set priority to 1.0.
- Specify PhysicalDeviceFeatures (can be empty/default for now).
- Enable the VK_KHR_swapchain extension (ash::khr::swapchain::NAME), since you'll need it right after this step.
- Call instance.create_device(physical_device, &create_info, None) to get a VkDevice.
- Retrieve queue handles with device.get_device_queue(family_index, 0).

Step 4 — Wire it into Renderer
- Add the Device (or its individual fields) to the Renderer struct.
- Call your setup code in Renderer::new() after surface creation.
- Add device.destroy_device(None) to Renderer::drop() before the surface/instance destroy calls.

Key things to keep in mind

- All ash device-creation calls are unsafe.
- If graphics and present families share the same index, only create one DeviceQueueCreateInfo (no duplicates).
- Destroy order in Drop matters: logical device first, then surface, then instance.
- You don't need a command pool yet — that comes later with the render loop.
