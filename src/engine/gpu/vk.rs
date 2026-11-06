//! Compute shaders on a Vulkan GPU, with nothing but `std`: the system's
//! `libvulkan.so.1` is opened at run time, so a machine without it still runs
//! the CPU engine. Adapted from image-engine-native's `native-vulkan`.
//!
//! [`Gpu::open`] picks a Vulkan 1.2 GPU, and [`Gpu::run`] records dispatches
//! of [`Kernel`]s over [`Buffer`]s into one submission and waits for it.
//! Dropping a buffer or a kernel waits until the GPU is idle, so none is freed
//! while a submission uses it.

use std::ffi::{c_void, CStr};
use std::ptr::{null, null_mut};
use std::sync::Arc;

use super::ffi::{self, Handle, Object};

/// What a submission's dispatches and copies do, for its barriers.
const GPU_STAGES: u32 = ffi::PIPELINE_STAGE_COMPUTE | ffi::PIPELINE_STAGE_TRANSFER;
const GPU_READS: u32 = ffi::ACCESS_SHADER_READ | ffi::ACCESS_TRANSFER_READ;
const GPU_WRITES: u32 = ffi::ACCESS_SHADER_WRITE | ffi::ACCESS_TRANSFER_WRITE;

/// Descriptor sets one submission may use, one per dispatch, and storage
/// buffers per set.
pub(super) const SETS: u32 = 4096;
const BINDINGS: u32 = 8;

fn check(what: &str, result: i32) -> Result<(), String> {
    let name = match result {
        ffi::SUCCESS => return Ok(()),
        -1 => "out of host memory",
        -2 => "out of device memory",
        -3 => "initialization failed",
        -4 => "device lost",
        -9 => "incompatible driver",
        _ => return Err(format!("{what}: Vulkan error {result}")),
    };
    Err(format!("{what}: {name}"))
}

/// An instance, destroyed when dropped.
struct Instance {
    handle: Handle,
    i: ffi::instance::Functions,
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: the device made from it, if any, is already destroyed: it
        // is owned by the `Device` holding this instance, dropped first.
        unsafe { (self.i.destroy_instance)(self.handle, null()) };
    }
}

/// The logical device and its instance, destroyed once the last object made
/// from them is.
struct Device {
    handle: Handle,
    d: ffi::device::Functions,
    memory: ffi::MemoryProperties,
    // Dropped after the device, by field order.
    _instance: Instance,
}

// SAFETY: Vulkan handles may be used from any thread.
unsafe impl Send for Device {}
// SAFETY: Vulkan synchronizes object creation and destruction itself;
// vkDeviceWaitIdle runs only from the owning `Gpu`, which is not `Sync`.
unsafe impl Sync for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: every object made from the device is gone, since each holds
        // an Arc of it, and the device is idle once vkDeviceWaitIdle returns.
        unsafe {
            (self.d.device_wait_idle)(self.handle);
            (self.d.destroy_device)(self.handle, null());
        }
    }
}

impl Device {
    /// The first memory type in `bits` with all of `flags`, trying each set
    /// of flags in turn.
    fn memory_type(&self, bits: u32, flags: &[u32]) -> Option<u32> {
        let types = &self.memory.types[..self.memory.type_count as usize];
        flags
            .iter()
            .find_map(|&want| {
                (0..types.len())
                    .find(|&i| bits & (1 << i) != 0 && types[i].property_flags & want == want)
            })
            .map(|i| i as u32)
    }
}

/// The chosen physical device.
struct Choice {
    handle: Handle,
    family: u32,
    props: Box<ffi::PhysicalDeviceProperties>,
}

/// A GPU, its queue and the objects of a submission.
pub struct Gpu {
    dev: Arc<Device>,
    queue: Handle,
    commands: Object,
    cb: Handle,
    fence: Object,
    descriptors: Object,
    name: String,
    vendor: u32,
    integrated: bool,
    groups: [u32; 3],
    /// Bytes a storage buffer may span in a shader.
    storage_range: u32,
}

// SAFETY: the queue, command buffer, fence and descriptor pool are only used
// through `&mut self`, which keeps their use on one thread at a time.
unsafe impl Send for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: `run` waits for each submission, so nothing is pending;
        // destroying a null handle does nothing.
        unsafe {
            let d = &self.dev.d;
            (d.destroy_fence)(self.dev.handle, self.fence, null());
            (d.destroy_descriptor_pool)(self.dev.handle, self.descriptors, null());
            (d.destroy_command_pool)(self.dev.handle, self.commands, null());
        }
    }
}

impl Gpu {
    /// Open the best GPU with Vulkan 1.2 and workgroups of 256 invocations:
    /// a discrete one first, then an integrated one. CPU implementations
    /// such as llvmpipe are left out.
    pub fn open() -> Result<Self, String> {
        let instance = create_instance()?;
        let choice = choose(&instance)?;
        let (handle, d) = create_device(&instance, &choice)?;
        let i = &instance.i;
        // SAFETY: the physical device belongs to the instance; the call
        // writes the whole struct.
        let memory = unsafe {
            let mut memory = std::mem::MaybeUninit::<ffi::MemoryProperties>::zeroed();
            (i.get_memory_properties)(choice.handle, memory.as_mut_ptr());
            memory.assume_init()
        };
        let dev = Arc::new(Device {
            handle,
            d,
            memory,
            _instance: instance,
        });
        let limits = &choice.props.limits;
        let mut gpu = Self {
            dev,
            queue: null_mut(),
            commands: 0,
            cb: null_mut(),
            fence: 0,
            descriptors: 0,
            name: device_name(&choice.props),
            vendor: choice.props.vendor_id,
            integrated: choice.props.device_type == ffi::DEVICE_TYPE_INTEGRATED,
            groups: limits.max_compute_work_group_count,
            storage_range: limits.max_storage_buffer_range,
        };
        let (d, h) = (&gpu.dev.d, gpu.dev.handle);
        // SAFETY: the device is valid and each create-info lives across its
        // call; on failure `gpu`'s Drop frees what was made.
        unsafe {
            (d.get_queue)(h, choice.family, 0, &mut gpu.queue);
            let command_pool_info = ffi::CommandPoolCreateInfo {
                s_type: ffi::TYPE_COMMAND_POOL_CREATE_INFO,
                p_next: null(),
                flags: ffi::POOL_RESET_COMMAND_BUFFER,
                queue_family_index: choice.family,
            };
            check(
                "creating a command pool",
                (d.create_command_pool)(h, &command_pool_info, null(), &mut gpu.commands),
            )?;
            let command_buffer_info = ffi::CommandBufferAllocateInfo {
                s_type: ffi::TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: null(),
                pool: gpu.commands,
                level: 0,
                count: 1,
            };
            check(
                "allocating command buffers",
                (d.allocate_command_buffers)(h, &command_buffer_info, &mut gpu.cb),
            )?;
            let fence_info = ffi::FenceCreateInfo {
                s_type: ffi::TYPE_FENCE_CREATE_INFO,
                p_next: null(),
                flags: 0,
            };
            check(
                "creating a fence",
                (d.create_fence)(h, &fence_info, null(), &mut gpu.fence),
            )?;
            let size = ffi::DescriptorPoolSize {
                descriptor_type: ffi::DESCRIPTOR_STORAGE_BUFFER,
                count: SETS * BINDINGS,
            };
            let descriptor_pool_info = ffi::DescriptorPoolCreateInfo {
                s_type: ffi::TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                p_next: null(),
                flags: 0,
                max_sets: SETS,
                size_count: 1,
                sizes: &size,
            };
            check(
                "creating a descriptor pool",
                (d.create_descriptor_pool)(h, &descriptor_pool_info, null(), &mut gpu.descriptors),
            )?;
        }
        Ok(gpu)
    }

    /// The GPU's name as its driver gives it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The PCI vendor ID: 0x1002 AMD, 0x10de NVIDIA, 0x8086 Intel.
    pub fn vendor(&self) -> u32 {
        self.vendor
    }

    /// Whether the GPU shares the CPU's package and memory.
    pub fn integrated(&self) -> bool {
        self.integrated
    }

    /// The largest storage buffer, in bytes.
    pub fn max_buffer(&self) -> usize {
        self.storage_range as usize
    }

    /// A storage buffer of `size` bytes in the GPU's own memory.
    pub fn buffer(&self, size: usize) -> Result<Buffer, String> {
        self.check_range(size)?;
        Buffer::new(&self.dev, size, &[ffi::MEMORY_DEVICE_LOCAL, 0], false)
    }

    /// Shaders may not reach past a storage buffer's range: refuse larger
    /// buffers rather than let a dispatch run out of bounds.
    fn check_range(&self, size: usize) -> Result<(), String> {
        if size > self.storage_range as usize {
            return Err(format!(
                "a buffer of {size} bytes, more than the GPU's {}",
                self.storage_range
            ));
        }
        Ok(())
    }

    /// A storage buffer of `size` bytes the CPU can also read and write,
    /// cached on the CPU side when `read` is set: the GPU's output is read
    /// back from one.
    pub fn host_buffer(&self, size: usize, read: bool) -> Result<Buffer, String> {
        self.check_range(size)?;
        let coherent = ffi::MEMORY_HOST_VISIBLE | ffi::MEMORY_HOST_COHERENT;
        let flags: &[u32] = if read {
            &[coherent | ffi::MEMORY_HOST_CACHED, coherent]
        } else {
            &[coherent]
        };
        Buffer::new(&self.dev, size, flags, true)
    }

    /// A compute kernel from SPIR-V, with `bindings` storage buffers in set
    /// 0 and `push` bytes of push constants.
    pub fn kernel(&self, spirv: &[u8], bindings: u32, push: u32) -> Result<Kernel, String> {
        Kernel::new(&self.dev, spirv, bindings, push)
    }

    /// Record a submission with `record`, run it and wait until it is done.
    /// Writes by the CPU before `run` are visible to it, and its writes to
    /// host buffers are visible to the CPU after.
    pub fn run(
        &mut self,
        record: impl FnOnce(&mut Commands<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        match self.record(record).and_then(|()| self.submit()) {
            Ok(()) => self.wait(),
            Err(e) => Err(match self.reset() {
                Ok(()) => e,
                Err(reset) => format!("{e}; {reset}"),
            }),
        }
    }

    /// Record the command buffer with `record`, after a barrier
    /// that orders it after the CPU's writes and every earlier submission,
    /// and before one that makes its writes visible to the CPU.
    fn record(
        &mut self,
        record: impl FnOnce(&mut Commands<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        let d = &self.dev.d;
        let cb = self.cb;
        let begin = ffi::CommandBufferBeginInfo {
            s_type: ffi::TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: null(),
            flags: ffi::ONE_TIME_SUBMIT,
            inheritance: null(),
        };
        // SAFETY: the command buffer is not pending: each submission is
        // waited for before it is recorded again.
        check("recording", unsafe { (d.begin_command_buffer)(cb, &begin) })?;
        let mut commands = Commands { gpu: self, sets: 0 };
        commands.barrier_from(
            ffi::PIPELINE_STAGE_HOST | GPU_STAGES,
            ffi::ACCESS_HOST_WRITE | GPU_WRITES,
        );
        let recorded = record(&mut commands);
        // Make the last writes visible to the CPU.
        let barrier = ffi::MemoryBarrier {
            s_type: ffi::TYPE_MEMORY_BARRIER,
            p_next: null(),
            src_access: GPU_WRITES,
            dst_access: ffi::ACCESS_HOST_READ,
        };
        // SAFETY: the command buffer is recording; `barrier` lives across the
        // call.
        let ended = unsafe {
            (d.cmd_pipeline_barrier)(
                cb,
                GPU_STAGES,
                ffi::PIPELINE_STAGE_HOST,
                0,
                1,
                &barrier,
                0,
                null(),
                0,
                null(),
            );
            check("recording", (d.end_command_buffer)(cb))
        };
        recorded.and(ended)
    }

    /// Queue the recorded command buffer.
    fn submit(&self) -> Result<(), String> {
        let submit = ffi::SubmitInfo {
            s_type: ffi::TYPE_SUBMIT_INFO,
            p_next: null(),
            wait_count: 0,
            wait_semaphores: null(),
            wait_stages: null(),
            command_buffer_count: 1,
            command_buffers: &self.cb,
            signal_count: 0,
            signal_semaphores: null(),
        };
        // SAFETY: the command buffer is recorded and its fence unsignalled;
        // `submit` lives across the call.
        check("submitting", unsafe {
            (self.dev.d.queue_submit)(self.queue, 1, &submit, self.fence)
        })
    }

    /// Wait for the submission, then make its fence and descriptor sets
    /// reusable.
    fn wait(&self) -> Result<(), String> {
        let (d, h) = (&self.dev.d, self.dev.handle);
        // SAFETY: the fence belongs to the device and went with the
        // submission.
        let waited = check("waiting for the GPU", unsafe {
            (d.wait_for_fences)(h, 1, &self.fence, 1, u64::MAX)
        });
        // SAFETY: the submission has finished, or the device is lost.
        let reset = check("resetting a fence", unsafe {
            (d.reset_fences)(h, 1, &self.fence)
        });
        waited.and(reset).and(self.reset())
    }

    /// Free the descriptor sets, whose submission has finished or never ran.
    fn reset(&self) -> Result<(), String> {
        // SAFETY: no pending submission uses the descriptor sets.
        check("resetting descriptors", unsafe {
            (self.dev.d.reset_descriptor_pool)(self.dev.handle, self.descriptors, 0)
        })
    }
}

/// A submission being recorded.
pub struct Commands<'a> {
    gpu: &'a Gpu,
    sets: u32,
}

impl Commands<'_> {
    fn barrier_from(&mut self, stages: u32, access: u32) {
        let barrier = ffi::MemoryBarrier {
            s_type: ffi::TYPE_MEMORY_BARRIER,
            p_next: null(),
            src_access: access,
            dst_access: GPU_READS | GPU_WRITES,
        };
        // SAFETY: the command buffer is recording; `barrier` lives across the call.
        unsafe {
            (self.gpu.dev.d.cmd_pipeline_barrier)(
                self.gpu.cb,
                stages,
                GPU_STAGES,
                0,
                1,
                &barrier,
                0,
                null(),
                0,
                null(),
            );
        }
    }

    /// Make the dispatches and copies recorded so far finish, and their
    /// writes visible, before the next ones start.
    pub fn barrier(&mut self) {
        self.barrier_from(GPU_STAGES, GPU_WRITES);
    }

    /// Copy the first `bytes` of `src` to the start of `dst`.
    pub fn copy(&mut self, src: &Buffer, dst: &Buffer, bytes: usize) -> Result<(), String> {
        if bytes > src.size || bytes > dst.size {
            return Err(format!(
                "copying {bytes} bytes from {} into {}",
                src.size, dst.size
            ));
        }
        let region = ffi::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: bytes as u64,
        };
        // SAFETY: the command buffer is recording; both buffers are borrowed
        // until `Gpu::run` has waited for the submission.
        unsafe {
            (self.gpu.dev.d.cmd_copy_buffer)(self.gpu.cb, src.buffer, dst.buffer, 1, &region)
        };
        Ok(())
    }

    /// Run `kernel` on `groups` workgroups, its binding `i` being
    /// `buffers[i]` and its push constants `push`.
    pub fn dispatch(
        &mut self,
        kernel: &Kernel,
        buffers: &[&Buffer],
        push: &[u8],
        groups: [u32; 3],
    ) -> Result<(), String> {
        if buffers.len() != kernel.bindings as usize || push.len() != kernel.push as usize {
            return Err(format!(
                "a kernel of {} buffers and {} bytes of constants given {} and {}",
                kernel.bindings,
                kernel.push,
                buffers.len(),
                push.len()
            ));
        }
        if groups.iter().zip(self.gpu.groups).any(|(&g, max)| g > max) || groups.contains(&0) {
            return Err(format!(
                "{groups:?} workgroups, at most {:?}",
                self.gpu.groups
            ));
        }
        if self.sets == SETS {
            return Err(format!("more than {SETS} dispatches in one submission"));
        }
        self.sets += 1;
        let (d, h, cb) = (&self.gpu.dev.d, self.gpu.dev.handle, self.gpu.cb);
        let descriptor_set_info = ffi::DescriptorSetAllocateInfo {
            s_type: ffi::TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: null(),
            pool: self.gpu.descriptors,
            set_count: 1,
            set_layouts: &kernel.set_layout,
        };
        let mut set = 0;
        // SAFETY: the pool has room for SETS sets of BINDINGS buffers, and
        // `bindings` is at most BINDINGS (checked by `Kernel::new`).
        check("allocating descriptors", unsafe {
            (d.allocate_descriptor_sets)(h, &descriptor_set_info, &mut set)
        })?;
        let buffer_infos: Vec<_> = buffers
            .iter()
            .map(|b| ffi::DescriptorBufferInfo {
                buffer: b.buffer,
                offset: 0,
                range: ffi::WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<_> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| ffi::WriteDescriptorSet {
                s_type: ffi::TYPE_WRITE_DESCRIPTOR_SET,
                p_next: null(),
                set,
                binding: i as u32,
                array_element: 0,
                count: 1,
                descriptor_type: ffi::DESCRIPTOR_STORAGE_BUFFER,
                image_info: null(),
                buffer_info: info,
                texel_buffer_view: null(),
            })
            .collect();
        // SAFETY: the set is new and unused; the buffers are borrowed until
        // `Gpu::run` has waited for the submission that reads them.
        unsafe {
            (d.update_descriptor_sets)(h, writes.len() as u32, writes.as_ptr(), 0, null());
            (d.cmd_bind_pipeline)(cb, ffi::BIND_POINT_COMPUTE, kernel.pipeline);
            (d.cmd_bind_descriptor_sets)(
                cb,
                ffi::BIND_POINT_COMPUTE,
                kernel.layout,
                0,
                1,
                &set,
                0,
                null(),
            );
            if !push.is_empty() {
                (d.cmd_push_constants)(
                    cb,
                    kernel.layout,
                    ffi::STAGE_COMPUTE,
                    0,
                    push.len() as u32,
                    push.as_ptr().cast(),
                );
            }
            (d.cmd_dispatch)(cb, groups[0], groups[1], groups[2]);
        }
        Ok(())
    }
}

/// A storage buffer, mapped for the CPU when made by [`Gpu::host_buffer`].
pub struct Buffer {
    dev: Arc<Device>,
    buffer: Object,
    memory: Object,
    map: *mut u8,
    size: usize,
}

// SAFETY: the mapping is plain memory owned by the buffer, reached only
// through `&self` or `&mut self`.
unsafe impl Send for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: once the device is idle no submission uses the buffer, and
        // freeing the memory unmaps it.
        unsafe {
            (self.dev.d.device_wait_idle)(self.dev.handle);
            (self.dev.d.destroy_buffer)(self.dev.handle, self.buffer, null());
            (self.dev.d.free_memory)(self.dev.handle, self.memory, null());
        }
    }
}

impl Buffer {
    fn new(dev: &Arc<Device>, size: usize, flags: &[u32], map: bool) -> Result<Self, String> {
        let (d, h) = (&dev.d, dev.handle);
        let buffer_info = ffi::BufferCreateInfo {
            s_type: ffi::TYPE_BUFFER_CREATE_INFO,
            p_next: null(),
            flags: 0,
            size: size.max(4) as u64,
            usage: ffi::BUFFER_STORAGE | ffi::BUFFER_TRANSFER_SRC | ffi::BUFFER_TRANSFER_DST,
            sharing_mode: 0,
            queue_family_count: 0,
            queue_families: null(),
        };
        let mut buffer = Self {
            dev: dev.clone(),
            buffer: 0,
            memory: 0,
            map: null_mut(),
            size,
        };
        // SAFETY: each create-info lives across its call; on failure Drop
        // frees what was made, null handles being no-ops.
        unsafe {
            check(
                "creating a buffer",
                (d.create_buffer)(h, &buffer_info, null(), &mut buffer.buffer),
            )?;
            let mut needs = ffi::MemoryRequirements::default();
            (d.get_buffer_requirements)(h, buffer.buffer, &mut needs);
            let index = dev
                .memory_type(needs.memory_type_bits, flags)
                .ok_or("no memory type for a buffer")?;
            let alloc = ffi::MemoryAllocateInfo {
                s_type: ffi::TYPE_MEMORY_ALLOCATE_INFO,
                p_next: null(),
                size: needs.size,
                memory_type_index: index,
            };
            check(
                &format!("allocating {size} bytes"),
                (d.allocate_memory)(h, &alloc, null(), &mut buffer.memory),
            )?;
            check(
                "binding memory",
                (d.bind_buffer_memory)(h, buffer.buffer, buffer.memory, 0),
            )?;
            if map {
                let mut ptr = null_mut();
                check(
                    "mapping memory",
                    (d.map_memory)(h, buffer.memory, 0, ffi::WHOLE_SIZE, 0, &mut ptr),
                )?;
                buffer.map = ptr.cast();
            }
        }
        Ok(buffer)
    }

    /// The buffer's bytes, which only a host buffer has.
    pub fn bytes(&self) -> &[u8] {
        assert!(!self.map.is_null(), "a device buffer has no CPU mapping");
        // SAFETY: the mapping covers `size` bytes, lives as long as the
        // buffer, and the GPU is not running (`Gpu::run` waits).
        unsafe { std::slice::from_raw_parts(self.map, self.size) }
    }

    /// The buffer's bytes, to fill before a run; only a host buffer has them.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        assert!(!self.map.is_null(), "a device buffer has no CPU mapping");
        // SAFETY: as for `bytes`, and `&mut self` makes the slice unique.
        unsafe { std::slice::from_raw_parts_mut(self.map, self.size) }
    }

    pub fn len(&self) -> usize {
        self.size
    }
}

/// A compute pipeline and its layout.
pub struct Kernel {
    dev: Arc<Device>,
    set_layout: Object,
    layout: Object,
    pipeline: Object,
    bindings: u32,
    push: u32,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: once the device is idle no submission uses the kernel, as
        // for a buffer; null handles are no-ops.
        unsafe {
            let (d, h) = (&self.dev.d, self.dev.handle);
            (d.device_wait_idle)(h);
            (d.destroy_pipeline)(h, self.pipeline, null());
            (d.destroy_pipeline_layout)(h, self.layout, null());
            (d.destroy_set_layout)(h, self.set_layout, null());
        }
    }
}

impl Kernel {
    fn new(dev: &Arc<Device>, spirv: &[u8], bindings: u32, push: u32) -> Result<Self, String> {
        if !spirv.len().is_multiple_of(4) || spirv.get(..4) != Some(&0x0723_0203u32.to_le_bytes()) {
            return Err("not SPIR-V".into());
        }
        if bindings > BINDINGS {
            return Err(format!("{bindings} buffers, at most {BINDINGS}"));
        }
        let words: Vec<u32> = spirv
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| u32::from_le_bytes(w))
            .collect();
        let (d, h) = (&dev.d, dev.handle);
        let mut kernel = Self {
            dev: dev.clone(),
            set_layout: 0,
            layout: 0,
            pipeline: 0,
            bindings,
            push,
        };
        let slots: Vec<_> = (0..bindings)
            .map(|i| ffi::DescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: ffi::DESCRIPTOR_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ffi::STAGE_COMPUTE,
                immutable_samplers: null(),
            })
            .collect();
        let range = ffi::PushConstantRange {
            stage_flags: ffi::STAGE_COMPUTE,
            offset: 0,
            size: push,
        };
        // SAFETY: each create-info and what it points to lives across its
        // call; on failure Drop frees what was made.
        unsafe {
            let descriptor_set_layout_info = ffi::DescriptorSetLayoutCreateInfo {
                s_type: ffi::TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                p_next: null(),
                flags: 0,
                binding_count: bindings,
                bindings: slots.as_ptr(),
            };
            check(
                "creating a set layout",
                (d.create_set_layout)(
                    h,
                    &descriptor_set_layout_info,
                    null(),
                    &mut kernel.set_layout,
                ),
            )?;
            let pipeline_layout_info = ffi::PipelineLayoutCreateInfo {
                s_type: ffi::TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                p_next: null(),
                flags: 0,
                set_layout_count: 1,
                set_layouts: &kernel.set_layout,
                push_range_count: u32::from(push > 0),
                push_ranges: &range,
            };
            check(
                "creating a pipeline layout",
                (d.create_pipeline_layout)(h, &pipeline_layout_info, null(), &mut kernel.layout),
            )?;
            let shader_module_info = ffi::ShaderModuleCreateInfo {
                s_type: ffi::TYPE_SHADER_MODULE_CREATE_INFO,
                p_next: null(),
                flags: 0,
                code_size: spirv.len(),
                code: words.as_ptr(),
            };
            let mut module = 0;
            check(
                "loading a shader",
                (d.create_shader_module)(h, &shader_module_info, null(), &mut module),
            )?;
            let compute_pipeline_info = ffi::ComputePipelineCreateInfo {
                s_type: ffi::TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                p_next: null(),
                flags: 0,
                stage: ffi::PipelineShaderStageCreateInfo {
                    s_type: ffi::TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: null(),
                    flags: 0,
                    stage: ffi::STAGE_COMPUTE,
                    module,
                    name: c"main".as_ptr(),
                    specialization: null(),
                },
                layout: kernel.layout,
                base_pipeline: 0,
                base_pipeline_index: -1,
            };
            let made = (d.create_compute_pipelines)(
                h,
                0,
                1,
                &compute_pipeline_info,
                null(),
                &mut kernel.pipeline,
            );
            (d.destroy_shader_module)(h, module, null());
            check("compiling a shader", made)?;
        }
        Ok(kernel)
    }
}

/// A device's name as its driver gives it.
fn device_name(props: &ffi::PhysicalDeviceProperties) -> String {
    let bytes: Vec<u8> = props.device_name.iter().map(|&c| c as u8).collect();
    CStr::from_bytes_until_nul(&bytes).map_or_else(
        |_| String::from_utf8_lossy(&bytes).into_owned(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// The instance, made through the system's Vulkan loader.
fn create_instance() -> Result<Instance, String> {
    let get = ffi::loader()?;
    // SAFETY: vkGetInstanceProcAddr with a null instance returns global
    // commands, vkCreateInstance among them, or null.
    let create =
        unsafe { get(null_mut(), c"vkCreateInstance".as_ptr()) }.ok_or("no vkCreateInstance")?;
    // SAFETY: that is vkCreateInstance's signature.
    let create: unsafe extern "system" fn(
        *const ffi::InstanceCreateInfo,
        *const c_void,
        *mut Handle,
    ) -> i32 = unsafe { std::mem::transmute(create) };
    let app = ffi::ApplicationInfo {
        s_type: ffi::TYPE_APPLICATION_INFO,
        p_next: null(),
        application_name: c"ppocr-native".as_ptr(),
        application_version: 0,
        engine_name: c"ppocr-native".as_ptr(),
        engine_version: 0,
        api_version: ffi::API_1_2,
    };
    let instance_info = ffi::InstanceCreateInfo {
        s_type: ffi::TYPE_INSTANCE_CREATE_INFO,
        p_next: null(),
        flags: 0,
        application_info: &app,
        layer_count: 0,
        layers: null(),
        extension_count: 0,
        extensions: null(),
    };
    let mut handle = null_mut();
    // SAFETY: `instance_info` and what it points to outlive the call.
    check("creating an instance", unsafe {
        create(&instance_info, null(), &mut handle)
    })?;
    // SAFETY: the instance is valid, and vkGetInstanceProcAddr returns its
    // commands or null.
    let i = unsafe { ffi::instance::Functions::load(|name| get(handle, name.as_ptr())) };
    match i {
        Ok(i) => Ok(Instance { handle, i }),
        Err(e) => {
            // SAFETY: as above; vkDestroyInstance has this signature.
            unsafe {
                if let Some(f) = get(handle, c"vkDestroyInstance".as_ptr()) {
                    let destroy: unsafe extern "system" fn(Handle, *const c_void) =
                        std::mem::transmute(f);
                    destroy(handle, null());
                }
            }
            Err(e)
        }
    }
}

/// The best GPU of `instance` for the engine, and a queue family on it with
/// compute, or why there is none.
fn choose(instance: &Instance) -> Result<Choice, String> {
    let (i, h) = (&instance.i, instance.handle);
    let mut count = 0;
    // SAFETY: the two-call enumeration, the second with room for the count
    // the first gave.
    let devices = unsafe {
        check(
            "listing GPUs",
            (i.enumerate_physical_devices)(h, &mut count, null_mut()),
        )?;
        let mut devices = vec![null_mut(); count as usize];
        check(
            "listing GPUs",
            (i.enumerate_physical_devices)(h, &mut count, devices.as_mut_ptr()),
        )?;
        devices.truncate(count as usize);
        devices
    };
    let mut best: Option<(u32, Choice)> = None;
    let mut rejected = Vec::new();
    for pd in devices {
        // SAFETY: `pd` came from this instance, and the output is a zeroed
        // struct of the type the command writes.
        let props = unsafe {
            let mut props = Box::<ffi::PhysicalDeviceProperties>::new_zeroed().assume_init();
            (i.get_properties)(pd, &mut *props);
            props
        };
        let name = device_name(&props);
        let rank = match props.device_type {
            ffi::DEVICE_TYPE_DISCRETE => 3,
            ffi::DEVICE_TYPE_INTEGRATED => 2,
            ffi::DEVICE_TYPE_VIRTUAL => 1,
            _ => {
                rejected.push(format!("{name}: not a GPU"));
                continue;
            }
        };
        if props.api_version < ffi::API_1_2 {
            rejected.push(format!("{name}: Vulkan older than 1.2"));
            continue;
        }
        // The matrix and CTC shaders run 256 invocations a workgroup, all
        // along x; Vulkan only promises 128.
        let limits = &props.limits;
        if limits.max_compute_work_group_invocations < 256
            || limits.max_compute_work_group_size[0] < 256
        {
            rejected.push(format!("{name}: workgroups narrower than 256 invocations"));
            continue;
        }
        // SAFETY: the two-call enumeration again.
        let families = unsafe {
            let mut n = 0;
            (i.get_queue_families)(pd, &mut n, null_mut());
            let mut families = vec![ffi::QueueFamilyProperties::default(); n as usize];
            (i.get_queue_families)(pd, &mut n, families.as_mut_ptr());
            families.truncate(n as usize);
            families
        };
        // A compute-only family runs on the asynchronous compute rings, apart
        // from the desktop's graphics work.
        let compute = |q: &ffi::QueueFamilyProperties| q.queue_flags & ffi::QUEUE_COMPUTE != 0;
        let Some(family) = families
            .iter()
            .position(|q| compute(q) && q.queue_flags & ffi::QUEUE_GRAPHICS == 0)
            .or_else(|| families.iter().position(compute))
        else {
            rejected.push(format!("{name}: no compute queue"));
            continue;
        };
        if best.as_ref().is_none_or(|b| rank > b.0) {
            best = Some((
                rank,
                Choice {
                    handle: pd,
                    family: family as u32,
                    props,
                },
            ));
        }
    }
    match best {
        Some((_, choice)) => Ok(choice),
        None if rejected.is_empty() => Err("no Vulkan device".into()),
        None => Err(rejected.join("; ")),
    }
}

/// The logical device on `choice` and its commands.
fn create_device(
    instance: &Instance,
    choice: &Choice,
) -> Result<(Handle, ffi::device::Functions), String> {
    let i = &instance.i;
    let priority = 1.0f32;
    let queue_info = ffi::DeviceQueueCreateInfo {
        s_type: ffi::TYPE_DEVICE_QUEUE_CREATE_INFO,
        p_next: null(),
        flags: 0,
        queue_family_index: choice.family,
        queue_count: 1,
        priorities: &priority,
    };
    let device_info = ffi::DeviceCreateInfo {
        s_type: ffi::TYPE_DEVICE_CREATE_INFO,
        p_next: null(),
        flags: 0,
        queue_info_count: 1,
        queue_infos: &queue_info,
        layer_count: 0,
        layers: null(),
        extension_count: 0,
        extensions: null(),
        features: null(),
    };
    let mut handle = null_mut();
    // SAFETY: `device_info` and the queue info it points to outlive the call.
    check("creating the device", unsafe {
        (i.create_device)(choice.handle, &device_info, null(), &mut handle)
    })?;
    // SAFETY: the device is valid, and vkGetDeviceProcAddr returns its
    // commands or null.
    let d = unsafe {
        ffi::device::Functions::load(|name| (i.get_device_proc_addr)(handle, name.as_ptr()))
    };
    let d = match d {
        Ok(d) => d,
        Err(e) => {
            // SAFETY: as above; vkDestroyDevice has this signature.
            unsafe {
                if let Some(f) = (i.get_device_proc_addr)(handle, c"vkDestroyDevice".as_ptr()) {
                    let destroy: unsafe extern "system" fn(Handle, *const c_void) =
                        std::mem::transmute(f);
                    destroy(handle, null());
                }
            }
            return Err(e);
        }
    };
    Ok((handle, d))
}
