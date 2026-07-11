/// Vulkan instance creation: entry loading, optional validation layer with a
/// debug-utils messenger, and MoltenVK portability enumeration on macOS.
use ash::{ext, vk};
#[cfg(target_os = "macos")]
use ash::khr;
use raw_window_handle::RawDisplayHandle;
use std::ffi::{CStr, c_char, c_void};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

pub struct InstanceBundle {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    debug: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

/// Validation defaults to on for debug builds; VOXEL_ENGINE_VALIDATION=0/1
/// overrides in either direction.
fn validation_requested() -> bool {
    match std::env::var("VOXEL_ENGINE_VALIDATION") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false") && !v.eq_ignore_ascii_case("off"),
        Err(_) => cfg!(debug_assertions),
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe {
        if data.is_null() || (*data).p_message.is_null() {
            return vk::FALSE;
        }
        CStr::from_ptr((*data).p_message).to_string_lossy()
    };
    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("[vulkan] {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("[vulkan] {message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::info!("[vulkan] {message}"),
        _ => log::debug!("[vulkan] {message}"),
    }
    vk::FALSE
}

impl InstanceBundle {
    pub fn new(display_handle: RawDisplayHandle) -> Self {
        let entry = load_entry();

        #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
        let mut extensions = ash_window::enumerate_required_extensions(display_handle)
            .expect("Failed to enumerate required surface extensions")
            .to_vec();

        #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
        let mut create_flags = vk::InstanceCreateFlags::empty();
        // MoltenVK is a non-conformant "portability" driver; the loader hides
        // it unless the instance opts in.
        #[cfg(target_os = "macos")]
        {
            extensions.push(khr::portability_enumeration::NAME.as_ptr());
            extensions.push(khr::get_physical_device_properties2::NAME.as_ptr());
            create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let mut validation = validation_requested() && has_validation_layer(&entry);

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"voxel_engine")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"voxel_engine")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let mut instance = None;
        if validation {
            let mut ext_with_debug = extensions.clone();
            ext_with_debug.push(ext::debug_utils::NAME.as_ptr());
            let layers: Vec<*const c_char> = vec![VALIDATION_LAYER.as_ptr()];
            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&ext_with_debug)
                .enabled_layer_names(&layers)
                .flags(create_flags);
            match unsafe { entry.create_instance(&create_info, None) } {
                Ok(inst) => {
                    log::info!("Vulkan validation layer enabled");
                    instance = Some(inst);
                }
                Err(err) => {
                    // Enumerable but unloadable layers happen (e.g. Homebrew
                    // manifests without a loadable dylib) — fall back cleanly.
                    log::warn!("validation layer unavailable ({err:?}); continuing without it");
                    validation = false;
                }
            }
        }
        let instance = instance.unwrap_or_else(|| {
            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extensions)
                .flags(create_flags);
            unsafe {
                entry
                    .create_instance(&create_info, None)
                    .expect("Failed to create Vulkan instance")
            }
        });

        let debug = if validation {
            let loader = ext::debug_utils::Instance::new(&entry, &instance);
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe {
                loader
                    .create_debug_utils_messenger(&messenger_info, None)
                    .expect("Failed to create debug messenger")
            };
            Some((loader, messenger))
        } else {
            None
        };

        Self {
            entry,
            instance,
            debug,
        }
    }

    pub unsafe fn destroy(&mut self) {
        unsafe {
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Loads libvulkan, falling back to well-known install locations that are
/// not on the default linker path (Homebrew on macOS, VULKAN_SDK).
fn load_entry() -> ash::Entry {
    if let Ok(entry) = unsafe { ash::Entry::load() } {
        return entry;
    }
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        candidates.push(std::path::Path::new(&sdk).join("lib/libvulkan.dylib"));
        candidates.push(std::path::Path::new(&sdk).join("lib/libvulkan.so.1"));
    }
    #[cfg(target_os = "macos")]
    {
        candidates.push("/opt/homebrew/lib/libvulkan.dylib".into());
        candidates.push("/usr/local/lib/libvulkan.dylib".into());
    }
    for path in candidates {
        if path.exists()
            && let Ok(entry) = unsafe { ash::Entry::load_from(&path) }
        {
            log::info!("Loaded Vulkan from {}", path.display());
            return entry;
        }
    }
    panic!("Failed to load the Vulkan library (is a Vulkan driver or MoltenVK installed?)");
}

fn has_validation_layer(entry: &ash::Entry) -> bool {
    let Ok(layers) = (unsafe { entry.enumerate_instance_layer_properties() }) else {
        return false;
    };
    layers.iter().any(|layer| {
        layer
            .layer_name_as_c_str()
            .is_ok_and(|name| name == VALIDATION_LAYER)
    })
}
