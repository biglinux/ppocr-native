//! The part of the Vulkan 1.2 C interface the engine calls, laid out as in
//! `vulkan_core.h` for 64-bit Linux, and its functions looked up at run time
//! from `libvulkan.so.1`.

use std::ffi::{c_char, c_int, c_void, CStr};

pub type Handle = *mut c_void;
/// A non-dispatchable handle: buffers, memory, pipelines and the rest.
pub type Object = u64;
pub type Pfn = Option<unsafe extern "system" fn()>;

pub const SUCCESS: i32 = 0;
pub const API_1_2: u32 = (1 << 22) | (2 << 12);
pub const WHOLE_SIZE: u64 = !0;

pub const TYPE_APPLICATION_INFO: u32 = 0;
pub const TYPE_INSTANCE_CREATE_INFO: u32 = 1;
pub const TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub const TYPE_DEVICE_CREATE_INFO: u32 = 3;
pub const TYPE_SUBMIT_INFO: u32 = 4;
pub const TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
pub const TYPE_FENCE_CREATE_INFO: u32 = 8;
pub const TYPE_BUFFER_CREATE_INFO: u32 = 12;
pub const TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
pub const TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
pub const TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
pub const TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
pub const TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
pub const TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
pub const TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
pub const TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
pub const TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub const TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub const TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub const TYPE_MEMORY_BARRIER: u32 = 46;

pub const DEVICE_TYPE_INTEGRATED: u32 = 1;
pub const DEVICE_TYPE_DISCRETE: u32 = 2;
pub const DEVICE_TYPE_VIRTUAL: u32 = 3;
pub const QUEUE_GRAPHICS: u32 = 1;
pub const QUEUE_COMPUTE: u32 = 2;
pub const MEMORY_DEVICE_LOCAL: u32 = 1;
pub const MEMORY_HOST_VISIBLE: u32 = 2;
pub const MEMORY_HOST_COHERENT: u32 = 4;
pub const MEMORY_HOST_CACHED: u32 = 8;
pub const BUFFER_TRANSFER_SRC: u32 = 1;
pub const BUFFER_TRANSFER_DST: u32 = 2;
pub const BUFFER_STORAGE: u32 = 0x20;
pub const DESCRIPTOR_STORAGE_BUFFER: u32 = 7;
pub const STAGE_COMPUTE: u32 = 0x20;
pub const BIND_POINT_COMPUTE: u32 = 1;
pub const PIPELINE_STAGE_COMPUTE: u32 = 0x800;
pub const PIPELINE_STAGE_TRANSFER: u32 = 0x1000;
pub const PIPELINE_STAGE_HOST: u32 = 0x4000;
pub const ACCESS_SHADER_READ: u32 = 0x20;
pub const ACCESS_SHADER_WRITE: u32 = 0x40;
pub const ACCESS_TRANSFER_READ: u32 = 0x800;
pub const ACCESS_TRANSFER_WRITE: u32 = 0x1000;
pub const ACCESS_HOST_READ: u32 = 0x2000;
pub const ACCESS_HOST_WRITE: u32 = 0x4000;
pub const POOL_RESET_COMMAND_BUFFER: u32 = 2;
pub const ONE_TIME_SUBMIT: u32 = 1;

#[repr(C)]
pub struct ApplicationInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub application_name: *const c_char,
    pub application_version: u32,
    pub engine_name: *const c_char,
    pub engine_version: u32,
    pub api_version: u32,
}

#[repr(C)]
pub struct InstanceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub application_info: *const ApplicationInfo,
    pub layer_count: u32,
    pub layers: *const *const c_char,
    pub extension_count: u32,
    pub extensions: *const *const c_char,
}

/// `VkPhysicalDeviceLimits`, naming only the fields read here.
#[repr(C)]
pub struct Limits {
    _images_to_uniform_range: [u32; 7],
    pub max_storage_buffer_range: u32,
    _push_constants_to_samplers: [u32; 3],
    _granularity_and_sparse: [u64; 2],
    _descriptors_to_fragment: [u32; 38],
    pub max_compute_shared_memory_size: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size: [u32; 3],
    _rest: [u64; 32],
}

#[repr(C)]
pub struct PhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub device_name: [c_char; 256],
    pub pipeline_cache_uuid: [u8; 16],
    pub limits: Limits,
    pub sparse_properties: [u32; 5],
}

const _: () = assert!(size_of::<Limits>() == 504);
const _: () = assert!(std::mem::offset_of!(Limits, max_storage_buffer_range) == 28);
const _: () = assert!(std::mem::offset_of!(Limits, max_compute_shared_memory_size) == 216);
const _: () = assert!(size_of::<PhysicalDeviceProperties>() == 824);

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QueueFamilyProperties {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MemoryType {
    pub property_flags: u32,
    pub heap_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MemoryHeap {
    pub size: u64,
    pub flags: u32,
}

#[repr(C)]
pub struct MemoryProperties {
    pub type_count: u32,
    pub types: [MemoryType; 32],
    pub heap_count: u32,
    pub heaps: [MemoryHeap; 16],
}

#[repr(C)]
pub struct DeviceQueueCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
    pub queue_count: u32,
    pub priorities: *const f32,
}

#[repr(C)]
pub struct DeviceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_info_count: u32,
    pub queue_infos: *const DeviceQueueCreateInfo,
    pub layer_count: u32,
    pub layers: *const *const c_char,
    pub extension_count: u32,
    pub extensions: *const *const c_char,
    pub features: *const c_void,
}

#[repr(C)]
pub struct BufferCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub size: u64,
    pub usage: u32,
    pub sharing_mode: u32,
    pub queue_family_count: u32,
    pub queue_families: *const u32,
}

#[repr(C)]
#[derive(Default)]
pub struct MemoryRequirements {
    pub size: u64,
    pub alignment: u64,
    pub memory_type_bits: u32,
}

#[repr(C)]
pub struct MemoryAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub size: u64,
    pub memory_type_index: u32,
}

#[repr(C)]
pub struct ShaderModuleCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub code_size: usize,
    pub code: *const u32,
}

#[repr(C)]
pub struct DescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: u32,
    pub descriptor_count: u32,
    pub stage_flags: u32,
    pub immutable_samplers: *const Object,
}

#[repr(C)]
pub struct DescriptorSetLayoutCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub binding_count: u32,
    pub bindings: *const DescriptorSetLayoutBinding,
}

#[repr(C)]
pub struct PushConstantRange {
    pub stage_flags: u32,
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
pub struct PipelineLayoutCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub set_layout_count: u32,
    pub set_layouts: *const Object,
    pub push_range_count: u32,
    pub push_ranges: *const PushConstantRange,
}

#[repr(C)]
pub struct PipelineShaderStageCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: u32,
    pub module: Object,
    pub name: *const c_char,
    pub specialization: *const c_void,
}

#[repr(C)]
pub struct ComputePipelineCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: PipelineShaderStageCreateInfo,
    pub layout: Object,
    pub base_pipeline: Object,
    pub base_pipeline_index: i32,
}

#[repr(C)]
pub struct DescriptorPoolSize {
    pub descriptor_type: u32,
    pub count: u32,
}

#[repr(C)]
pub struct DescriptorPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub max_sets: u32,
    pub size_count: u32,
    pub sizes: *const DescriptorPoolSize,
}

#[repr(C)]
pub struct DescriptorSetAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub pool: Object,
    pub set_count: u32,
    pub set_layouts: *const Object,
}

#[repr(C)]
pub struct DescriptorBufferInfo {
    pub buffer: Object,
    pub offset: u64,
    pub range: u64,
}

#[repr(C)]
pub struct WriteDescriptorSet {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub set: Object,
    pub binding: u32,
    pub array_element: u32,
    pub count: u32,
    pub descriptor_type: u32,
    pub image_info: *const c_void,
    pub buffer_info: *const DescriptorBufferInfo,
    pub texel_buffer_view: *const Object,
}

#[repr(C)]
pub struct CommandPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
}

#[repr(C)]
pub struct CommandBufferAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub pool: Object,
    pub level: u32,
    pub count: u32,
}

#[repr(C)]
pub struct CommandBufferBeginInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub inheritance: *const c_void,
}

#[repr(C)]
pub struct MemoryBarrier {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub src_access: u32,
    pub dst_access: u32,
}

#[repr(C)]
pub struct BufferCopy {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

#[repr(C)]
pub struct FenceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct SubmitInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub wait_count: u32,
    pub wait_semaphores: *const Object,
    pub wait_stages: *const u32,
    pub command_buffer_count: u32,
    pub command_buffers: *const Handle,
    pub signal_count: u32,
    pub signal_semaphores: *const Object,
}

type Alloc = *const c_void;

/// Declares `Functions` with one pointer per Vulkan command, and its loader.
macro_rules! functions {
    ($($field:ident = $name:literal: fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        pub struct Functions {
            $(pub $field: unsafe extern "system" fn($($arg),*) $(-> $ret)?,)*
        }

        impl Functions {
            /// Look every command up through `lookup`, which returns null for
            /// one the driver lacks.
            ///
            /// # Safety
            ///
            /// `lookup` must return either null or the address of the named
            /// Vulkan command.
            pub unsafe fn load(lookup: impl Fn(&CStr) -> Pfn) -> Result<Self, String> {
                Ok(Self {
                    $($field: {
                        let f = lookup($name).ok_or(concat!("no ", stringify!($field)))?;
                        // SAFETY: the caller guarantees `f` is this command,
                        // whose C signature the field's type spells out.
                        unsafe {
                            std::mem::transmute::<
                                unsafe extern "system" fn(),
                                unsafe extern "system" fn($($arg),*) $(-> $ret)?,
                            >(f)
                        }
                    },)*
                })
            }
        }
    };
}

/// The commands of an instance.
pub mod instance {
    use super::*;
    functions! {
        destroy_instance = c"vkDestroyInstance": fn(Handle, Alloc);
        enumerate_physical_devices = c"vkEnumeratePhysicalDevices": fn(Handle, *mut u32, *mut Handle) -> i32;
        get_properties = c"vkGetPhysicalDeviceProperties": fn(Handle, *mut PhysicalDeviceProperties);
        get_queue_families = c"vkGetPhysicalDeviceQueueFamilyProperties": fn(Handle, *mut u32, *mut QueueFamilyProperties);
        get_memory_properties = c"vkGetPhysicalDeviceMemoryProperties": fn(Handle, *mut MemoryProperties);
        create_device = c"vkCreateDevice": fn(Handle, *const DeviceCreateInfo, Alloc, *mut Handle) -> i32;
        get_device_proc_addr = c"vkGetDeviceProcAddr": fn(Handle, *const c_char) -> Pfn;
    }
}

/// The commands of a device.
pub mod device {
    use super::*;
    functions! {
        destroy_device = c"vkDestroyDevice": fn(Handle, Alloc);
        device_wait_idle = c"vkDeviceWaitIdle": fn(Handle) -> i32;
        get_queue = c"vkGetDeviceQueue": fn(Handle, u32, u32, *mut Handle);
        create_buffer = c"vkCreateBuffer": fn(Handle, *const BufferCreateInfo, Alloc, *mut Object) -> i32;
        destroy_buffer = c"vkDestroyBuffer": fn(Handle, Object, Alloc);
        get_buffer_requirements = c"vkGetBufferMemoryRequirements": fn(Handle, Object, *mut MemoryRequirements);
        allocate_memory = c"vkAllocateMemory": fn(Handle, *const MemoryAllocateInfo, Alloc, *mut Object) -> i32;
        free_memory = c"vkFreeMemory": fn(Handle, Object, Alloc);
        bind_buffer_memory = c"vkBindBufferMemory": fn(Handle, Object, Object, u64) -> i32;
        map_memory = c"vkMapMemory": fn(Handle, Object, u64, u64, u32, *mut *mut c_void) -> i32;
        create_shader_module = c"vkCreateShaderModule": fn(Handle, *const ShaderModuleCreateInfo, Alloc, *mut Object) -> i32;
        destroy_shader_module = c"vkDestroyShaderModule": fn(Handle, Object, Alloc);
        create_set_layout = c"vkCreateDescriptorSetLayout": fn(Handle, *const DescriptorSetLayoutCreateInfo, Alloc, *mut Object) -> i32;
        destroy_set_layout = c"vkDestroyDescriptorSetLayout": fn(Handle, Object, Alloc);
        create_pipeline_layout = c"vkCreatePipelineLayout": fn(Handle, *const PipelineLayoutCreateInfo, Alloc, *mut Object) -> i32;
        destroy_pipeline_layout = c"vkDestroyPipelineLayout": fn(Handle, Object, Alloc);
        create_compute_pipelines = c"vkCreateComputePipelines": fn(Handle, Object, u32, *const ComputePipelineCreateInfo, Alloc, *mut Object) -> i32;
        destroy_pipeline = c"vkDestroyPipeline": fn(Handle, Object, Alloc);
        create_descriptor_pool = c"vkCreateDescriptorPool": fn(Handle, *const DescriptorPoolCreateInfo, Alloc, *mut Object) -> i32;
        destroy_descriptor_pool = c"vkDestroyDescriptorPool": fn(Handle, Object, Alloc);
        reset_descriptor_pool = c"vkResetDescriptorPool": fn(Handle, Object, u32) -> i32;
        allocate_descriptor_sets = c"vkAllocateDescriptorSets": fn(Handle, *const DescriptorSetAllocateInfo, *mut Object) -> i32;
        update_descriptor_sets = c"vkUpdateDescriptorSets": fn(Handle, u32, *const WriteDescriptorSet, u32, *const c_void);
        create_command_pool = c"vkCreateCommandPool": fn(Handle, *const CommandPoolCreateInfo, Alloc, *mut Object) -> i32;
        destroy_command_pool = c"vkDestroyCommandPool": fn(Handle, Object, Alloc);
        allocate_command_buffers = c"vkAllocateCommandBuffers": fn(Handle, *const CommandBufferAllocateInfo, *mut Handle) -> i32;
        begin_command_buffer = c"vkBeginCommandBuffer": fn(Handle, *const CommandBufferBeginInfo) -> i32;
        end_command_buffer = c"vkEndCommandBuffer": fn(Handle) -> i32;
        cmd_bind_pipeline = c"vkCmdBindPipeline": fn(Handle, u32, Object);
        cmd_bind_descriptor_sets = c"vkCmdBindDescriptorSets": fn(Handle, u32, Object, u32, u32, *const Object, u32, *const u32);
        cmd_push_constants = c"vkCmdPushConstants": fn(Handle, Object, u32, u32, u32, *const c_void);
        cmd_copy_buffer = c"vkCmdCopyBuffer": fn(Handle, Object, Object, u32, *const BufferCopy);
        cmd_dispatch = c"vkCmdDispatch": fn(Handle, u32, u32, u32);
        cmd_pipeline_barrier = c"vkCmdPipelineBarrier": fn(Handle, u32, u32, u32, u32, *const MemoryBarrier, u32, *const c_void, u32, *const c_void);
        create_fence = c"vkCreateFence": fn(Handle, *const FenceCreateInfo, Alloc, *mut Object) -> i32;
        destroy_fence = c"vkDestroyFence": fn(Handle, Object, Alloc);
        reset_fences = c"vkResetFences": fn(Handle, u32, *const Object) -> i32;
        wait_for_fences = c"vkWaitForFences": fn(Handle, u32, *const Object, u32, u64) -> i32;
        queue_submit = c"vkQueueSubmit": fn(Handle, u32, *const SubmitInfo, Object) -> i32;
    }
}

const RTLD_NOW: c_int = 2;

unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

/// `vkGetInstanceProcAddr` of the system's Vulkan loader, which stays loaded
/// for the life of the process.
pub fn loader() -> Result<unsafe extern "system" fn(Handle, *const c_char) -> Pfn, String> {
    // SAFETY: both names are NUL-terminated; dlopen and dlsym only read them.
    let symbol = unsafe {
        let lib = dlopen(c"libvulkan.so.1".as_ptr(), RTLD_NOW);
        if lib.is_null() {
            return Err("no Vulkan loader (libvulkan.so.1)".into());
        }
        dlsym(lib, c"vkGetInstanceProcAddr".as_ptr())
    };
    if symbol.is_null() {
        return Err("libvulkan.so.1 has no vkGetInstanceProcAddr".into());
    }
    // SAFETY: the loader exports vkGetInstanceProcAddr with this signature.
    Ok(unsafe {
        std::mem::transmute::<*mut c_void, unsafe extern "system" fn(Handle, *const c_char) -> Pfn>(
            symbol,
        )
    })
}
