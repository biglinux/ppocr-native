//! The networks on a Vulkan GPU: each operation of a plan is one compute
//! dispatch, all of a run in one submission, in 32-bit floats. A convolution
//! takes in the squeeze-and-excite product it reads, the addition that reads
//! it, and the nearest upsamplings and channel concatenations that feed it,
//! when nothing else reads what they pass between them; an upsampled sum
//! takes in a squeeze-and-excite residual the same way.
//!
//! The results are close to the CPU engine's, not equal: sums run in another
//! order, and the channel mean in `f32`. Tensors stay in `f32`, since half
//! floats move the detector's boxes. The weights are the plan's own, as the
//! file stores them. docs/development.md lists the kernel layouts measured
//! slower.

mod ffi;
mod vk;

use std::path::Path;

use super::plan::{Fused, Kind, Op, Plan};
use vk::{Buffer, Commands, Gpu, Kernel};

const GEMM: [[&[u8]; 3]; 2] = [
    [
        include_bytes!(concat!(env!("OUT_DIR"), "/gemm64.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/gemm32.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/gemm16.spv")),
    ],
    [
        include_bytes!(concat!(env!("OUT_DIR"), "/gemm64_vec4.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/gemm32_vec4.spv")),
        include_bytes!(concat!(env!("OUT_DIR"), "/gemm16_vec4.spv")),
    ],
];
/// The PCI vendor ID of AMD, whose GPUs take the scalar shared-memory reads of
/// `gemm.comp`.
const AMD: u32 = 0x1002;
const DWCONV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dwconv.spv"));
const DW7: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dw7.spv"));

const ELTWISE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/eltwise.spv"));
const POOL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pool.spv"));
const MEAN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mean.spv"));
const CONVT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/convt.spv"));
const LAYERNORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/layernorm.spv"));
const ATTENTION: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv"));
const CTC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ctc.spv"));

/// Invocations per workgroup of the one-value-per-invocation kernels.
const GROUP: usize = 64;
/// Workgroups along x of a dispatch, the most every Vulkan device allows.
const MAX_GROUPS: usize = 65_535;
/// Outputs of a depthwise row an invocation of `dwconv.comp` computes, and
/// the widest kernel and stride its row segment holds.
const DW_RUN: usize = 4;
const DW_MAX_KW: usize = 7;
const DW_MAX_STRIDE: usize = 2;
/// Side of the square of outputs an invocation of `dw7.comp` computes.
const DW7_BLOCK: usize = 4;
/// Head size `attention.comp` holds in registers.
const MAX_HEAD: usize = 64;

struct Kernels {
    /// `gemm.comp` with tiles of 64 x 64, 128 x 32 and 256 x 16, its shared
    /// memory read as the GPU prefers.
    gemm: [Kernel; 3],
    dwconv: Kernel,
    dw7: Kernel,
    eltwise: Kernel,
    pool: Kernel,
    mean: Kernel,
    convt: Kernel,
    layernorm: Kernel,
    attention: Kernel,
    ctc: Kernel,
}

/// A plan loaded onto the GPU, run over one or more images at a time
/// ([`GpuModel::set_sizes`]): each image is a lane with device memory of its
/// own for every tensor, and the lanes go through each step together, so
/// that small images fill the GPU between two barriers.
pub struct GpuModel {
    gpu: Gpu,
    /// The plan at the size it was loaded at, which each lane's starts from.
    plan: Plan,
    weight_floats: usize,
    weights: Buffer,
    k: Kernels,
    steps: Vec<Step>,
    /// Every lane a run has had, so that a smaller run keeps the memory of
    /// the lanes it leaves idle; the first `active` run.
    lanes: Vec<Lane>,
    active: usize,
}

/// One image of a run: its plan at its size and the memory of its tensors.
struct Lane {
    plan: Plan,
    /// The device buffer, an index into `buffers`, each plan buffer lives in.
    home: Vec<usize>,
    buffers: Vec<Buffer>,
    /// The channel means' partial sums.
    partial: Buffer,
    /// The CTC head's logits.
    logits: Buffer,
    /// The image, written by the CPU, and the results, read by it.
    input: Buffer,
    outputs: Vec<Buffer>,
}

impl GpuModel {
    /// Load a plan onto the best GPU, with one lane at the input size it was
    /// exported at.
    pub fn load(path: &Path) -> Result<Self, String> {
        let (plan, values) = Plan::load(path).map_err(|e| e.to_string())?;
        for (i, op) in plan.ops.iter().enumerate() {
            supported(op).map_err(|e| format!("operation {i}: {e}"))?;
        }
        one_writer(&plan)?;
        let mut gpu = Gpu::open()?;
        let weights = gpu.buffer(values.len() * 4)?;
        let mut staging = gpu.host_buffer(values.len() * 4, false)?;
        for (out, v) in staging
            .bytes_mut()
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&values)
        {
            *out = v.to_le_bytes();
        }
        gpu.run(|c| c.copy(&staging, &weights, values.len() * 4))?;
        let kernel = |spirv, bindings, ints: u32| gpu.kernel(spirv, bindings, 4 * ints);
        let gemm = &GEMM[usize::from(gpu.vendor() != AMD)];
        let k = Kernels {
            gemm: [
                kernel(gemm[0], 8, 28)?,
                kernel(gemm[1], 8, 28)?,
                kernel(gemm[2], 8, 28)?,
            ],
            dwconv: kernel(DWCONV, 3, 14)?,
            dw7: kernel(DW7, 3, 14)?,
            eltwise: kernel(ELTWISE, 4, 10)?,
            pool: kernel(POOL, 2, 10)?,
            mean: kernel(MEAN, 3, 4)?,
            convt: kernel(CONVT, 3, 10)?,
            layernorm: kernel(LAYERNORM, 3, 3)?,
            attention: kernel(ATTENTION, 3, 4)?,
            ctc: kernel(CTC, 2, 2)?,
        };
        let steps = fuse(&plan);
        let mut model = Self {
            steps,
            weight_floats: values.len(),
            lanes: Vec::new(),
            active: 0,
            weights,
            k,
            plan,
            gpu,
        };
        model.set_sizes(&[(model.plan.input.h, model.plan.input.w)])?;
        Ok(model)
    }

    /// The GPU's name as its driver gives it.
    pub fn device(&self) -> &str {
        self.gpu.name()
    }

    /// Whether the GPU is integrated with the CPU, sharing its memory.
    pub fn integrated(&self) -> bool {
        self.gpu.integrated()
    }

    /// Make the following runs one image `height` by `width` pixels; as
    /// [`super::Model::set_input_size`], but device memory is reserved
    /// afresh when a buffer has to grow.
    pub fn set_input_size(&mut self, height: usize, width: usize) -> Result<(), String> {
        self.set_sizes(&[(height, width)])
    }

    /// The most lanes one run takes: every step of every lane is a dispatch
    /// or two, and a submission holds [`vk::SETS`] of them.
    pub fn max_lanes(&self) -> usize {
        (vk::SETS as usize / (2 * self.steps.len())).max(1)
    }

    /// Make the following runs one lane per image size in `sizes`, height
    /// and width; a lane keeps the memory it had. On error the lanes are
    /// those that could be set, the first ones.
    pub fn set_sizes(&mut self, sizes: &[(usize, usize)]) -> Result<(), String> {
        self.active = 0;
        if sizes.len() > self.max_lanes() {
            return Err(format!(
                "{} lanes, more than the {} a run takes",
                sizes.len(),
                self.max_lanes()
            ));
        }
        for (i, &(h, w)) in sizes.iter().enumerate() {
            if i == self.lanes.len() {
                self.lanes.push(Lane {
                    plan: self.plan.clone(),
                    home: Vec::new(),
                    buffers: Vec::new(),
                    partial: self.gpu.buffer(4)?,
                    logits: self.gpu.buffer(4)?,
                    input: self.gpu.host_buffer(4, false)?,
                    outputs: Vec::new(),
                });
            } else if (h, w) == (self.lanes[i].plan.input.h, self.lanes[i].plan.input.w)
                && !self.lanes[i].home.is_empty()
            {
                self.active = i + 1;
                continue;
            }
            let lane = &mut self.lanes[i];
            let sized = lane
                .plan
                .set_size(h, w, self.weight_floats)
                .map_err(|e| e.to_string());
            if let Err(e) = sized.and_then(|()| lane.arrange(&self.gpu, &self.steps)) {
                // Arranged again before its next run.
                lane.home.clear();
                return Err(e);
            }
            self.active = i + 1;
        }
        Ok(())
    }

    /// The image of the first lane, HWC, to be filled before each
    /// [`GpuModel::run`].
    pub fn input_mut(&mut self) -> &mut [f32] {
        self.lane_input_mut(0)
    }

    /// The image of lane `lane`, as [`GpuModel::input_mut`].
    pub fn lane_input_mut(&mut self, lane: usize) -> &mut [f32] {
        let lane = &mut self.lanes[lane];
        let input = &lane.plan.input;
        let n = input.h * input.w * input.c;
        &mut floats_mut(lane.input.bytes_mut())[..n]
    }

    /// One result of the first lane's last run, as [`super::Model::output`]
    /// gives it.
    pub fn output(&self, index: usize) -> &[f32] {
        self.lane_output(0, index)
    }

    /// One result of lane `lane`'s last run.
    pub fn lane_output(&self, lane: usize, index: usize) -> &[f32] {
        let lane = &self.lanes[lane];
        &floats(lane.outputs[index].bytes())[..lane.plan.outputs[index].n]
    }

    /// Run the network once over each lane's image.
    pub fn run(&mut self) -> Result<(), String> {
        let Self {
            gpu,
            weights,
            k,
            steps,
            lanes,
            active,
            ..
        } = self;
        let lanes = &lanes[..*active];
        gpu.run(|c| {
            for lane in lanes.iter() {
                let input = &lane.plan.input;
                let image = 4 * input.h * input.w * input.c;
                c.copy(&lane.input, &lane.buffers[lane.home[input.slot]], image)?;
            }
            for step in steps.iter() {
                c.barrier();
                for lane in lanes.iter() {
                    dispatch(c, k, step, weights, lane)?;
                }
            }
            c.barrier();
            for lane in lanes.iter() {
                for (out, host) in lane.plan.outputs.iter().zip(&lane.outputs) {
                    c.copy(&lane.buffers[lane.home[out.slot]], host, 4 * out.n)?;
                }
            }
            Ok(())
        })
    }
}

impl Lane {
    /// Place every plan buffer in a device buffer, reusing one whose
    /// tensors are no longer read, and size the device buffers and the host
    /// ones for the lane's input size.
    fn arrange(&mut self, gpu: &Gpu, steps: &[Step]) -> Result<(), String> {
        let plan = &self.plan;
        let mut last = vec![0; plan.pool.len()];
        for (i, step) in steps.iter().enumerate() {
            for s in step.reads(&plan.ops) {
                last[s] = i;
            }
        }
        for out in &plan.outputs {
            last[out.slot] = usize::MAX;
        }
        let mut home = vec![usize::MAX; plan.pool.len()];
        let mut sizes: Vec<usize> = Vec::new();
        let mut free: Vec<usize> = Vec::new();
        // The smallest free buffer that holds `n` floats, else the largest
        // free one grown, else a new one.
        let place = |n: usize, free: &mut Vec<usize>, sizes: &mut Vec<usize>| {
            let fits = free
                .iter()
                .enumerate()
                .filter(|&(_, &b)| sizes[b] >= n)
                .min_by_key(|&(_, &b)| sizes[b]);
            let chosen = fits.or_else(|| free.iter().enumerate().max_by_key(|&(_, &b)| sizes[b]));
            match chosen.map(|(i, _)| i) {
                Some(i) => {
                    let b = free.swap_remove(i);
                    sizes[b] = sizes[b].max(n);
                    b
                }
                None => {
                    sizes.push(n);
                    sizes.len() - 1
                }
            }
        };
        home[plan.input.slot] = place(plan.pool[plan.input.slot], &mut free, &mut sizes);
        for (i, step) in steps.iter().enumerate() {
            let y = step.writes(&plan.ops);
            home[y] = place(plan.pool[y], &mut free, &mut sizes);
            for s in step.reads(&plan.ops) {
                if last[s] == i && !free.contains(&home[s]) {
                    free.push(home[s]);
                }
            }
        }
        // A buffer made or grown takes a quarter more than it needs, so the
        // next lines, a little wider or narrower, fit without a new one.
        let room = |bytes: usize| (bytes + bytes / 4).min(gpu.max_buffer()).max(bytes);
        for (b, &n) in sizes.iter().enumerate() {
            let bytes = 4 * n.max(1);
            match self.buffers.get(b) {
                Some(buffer) if buffer.len() >= bytes => {}
                Some(_) => self.buffers[b] = gpu.buffer(room(bytes))?,
                None => self.buffers.push(gpu.buffer(room(bytes))?),
            }
        }
        let partial = plan
            .ops
            .iter()
            .filter(|op| op.kind == Kind::Mean)
            .map(|op| 4 * mean_chunks(op).0 * op.ic)
            .max()
            .unwrap_or(0);
        if self.partial.len() < partial {
            self.partial = gpu.buffer(room(partial))?;
        }
        let logits = plan
            .ops
            .iter()
            .filter(|op| op.kind == Kind::Ctcargmax)
            .map(|op| 4 * op.iw * ctc_columns(op))
            .max()
            .unwrap_or(0);
        if self.logits.len() < logits {
            self.logits = gpu.buffer(room(logits))?;
        }
        let input = 4 * plan.input.h * plan.input.w * plan.input.c;
        if self.input.len() < input {
            self.input = gpu.host_buffer(room(input), false)?;
        }
        self.outputs.truncate(plan.outputs.len());
        for (i, out) in plan.outputs.iter().enumerate() {
            let bytes = 4 * out.n.max(1);
            match self.outputs.get(i) {
                Some(buffer) if buffer.len() >= bytes => {}
                Some(_) => self.outputs[i] = gpu.host_buffer(room(bytes), true)?,
                None => self.outputs.push(gpu.host_buffer(room(bytes), true)?),
            }
        }
        self.home = home;
        Ok(())
    }
}

/// One dispatch: operation `op`, for a convolution with the gate of the
/// squeeze-and-excite product `gate` applied to its input and the addition
/// `add` applied to its output, and reading `sources` when it has them: the
/// inputs of the nearest upsamplings and channel concatenations folded in,
/// each a buffer, its channels and the power of two it is upsampled by. An
/// upsampled sum's addend may be the channel multiply `residual`'s
/// `x + x * gate`, folded in.
struct Step {
    op: usize,
    gate: Option<usize>,
    add: Option<usize>,
    sources: Vec<(usize, usize, u32)>,
    residual: Option<usize>,
}

impl Step {
    /// The input of a gated convolution: the product's input and its gate.
    fn input(&self, ops: &[Op]) -> usize {
        match self.sources.first() {
            Some(&(slot, _, _)) => slot,
            None => self.gate.map_or(ops[self.op].x, |g| ops[g].x),
        }
    }

    /// The addend of an addition folded in: its input that the convolution
    /// does not write.
    fn addend(&self, ops: &[Op]) -> Option<usize> {
        self.add.map(|a| {
            let (add, y) = (&ops[a], ops[self.op].y);
            if add.x == y {
                add.x2
            } else {
                add.x
            }
        })
    }

    fn reads(&self, ops: &[Op]) -> Vec<usize> {
        let op = &ops[self.op];
        if let Some(m) = self.residual {
            return vec![op.x, ops[m].x, ops[m].x2];
        }
        let second = if self.sources.is_empty() {
            op.second()
        } else {
            None
        };
        [
            Some(self.input(ops)),
            second,
            self.gate.map(|g| ops[g].x2),
            self.addend(ops),
        ]
        .into_iter()
        .flatten()
        .chain(self.sources.iter().skip(1).map(|s| s.0))
        .collect()
    }

    fn writes(&self, ops: &[Op]) -> usize {
        self.add.map_or(ops[self.op].y, |a| ops[a].y)
    }
}

/// The plan's operations as dispatches, folding a squeeze-and-excite
/// product into the convolution that alone reads it, and an addition into
/// the convolution whose output only it reads.
fn fuse(plan: &Plan) -> Vec<Step> {
    let ops = &plan.ops;
    let mut reads = vec![0usize; plan.pool.len()];
    let mut writer = vec![usize::MAX; plan.pool.len()];
    for (i, op) in ops.iter().enumerate() {
        reads[op.x] += 1;
        if let Some(s) = op.second() {
            reads[s] += 1;
        }
        writer[op.y] = i;
    }
    for out in &plan.outputs {
        reads[out.slot] += 1;
    }
    let mut folded = vec![false; ops.len()];
    // The products, upsamplings and concatenations folded into a
    // convolution, which no longer run on their own.
    let mut inside = Vec::new();
    let mut steps = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        if folded[i] {
            continue;
        }
        let mut step = Step {
            op: i,
            gate: None,
            add: None,
            sources: Vec::new(),
            residual: None,
        };
        let next = ops.get(i + 1);
        if residual_gate(ops, &reads, i) {
            // x + x * gate, the neck's residual squeeze-and-excite.
            step.add = Some(i + 1);
            folded[i + 1] = true;
        }
        if op.kind == Kind::Upsampleadd && reads[op.x2] == 1 {
            // The addend, when it is such a residual that only this sum reads.
            let a = writer[op.x2];
            if a < i && a > 0 && residual_gate(ops, &reads, a - 1) {
                step.residual = Some(a - 1);
            }
        }
        if op.kind == Kind::Conv {
            let p = writer[op.x];
            if p < i && ops[p].kind == Kind::Channelmul && reads[op.x] == 1 && op.x != op.x2 {
                step.gate = Some(p);
            }
            if step.gate.is_none() {
                let (mut sources, mut producers) = (Vec::new(), Vec::new());
                let mut read = |slot, channels| {
                    let found = (&mut sources, &mut producers);
                    expand(ops, &reads, &writer, i, slot, channels, 0, found);
                };
                read(op.x, op.ic);
                if let Some(x2) = op.second() {
                    read(x2, op.n2);
                }
                let plain = sources.len() <= 2 && sources.iter().all(|s| s.2 == 0);
                if sources.len() <= 4 && !plain {
                    step.sources = sources;
                    inside.extend(producers);
                }
            }
            let adds = |a: &Op| a.kind == Kind::Add && (a.x == op.y) != (a.x2 == op.y);
            if reads[op.y] == 1 && next.is_some_and(adds) {
                step.add = Some(i + 1);
                folded[i + 1] = true;
            }
        }
        steps.push(step);
    }
    inside.extend(steps.iter().filter_map(|s| s.gate.or(s.residual)));
    steps.retain(|s| !inside.contains(&s.op));
    steps
}

/// The concatenation or nearest upsampling by a power of two that alone
/// writes `slot` for operation `i` to read, if one does.
fn foldable(ops: &[Op], reads: &[usize], writer: &[usize], i: usize, slot: usize) -> Option<usize> {
    let p = writer[slot];
    if p >= i || reads[slot] != 1 {
        return None;
    }
    let op = &ops[p];
    let upsample = op.kind == Kind::Upsample
        && op.sh == op.sw
        && op.sh.is_power_of_two()
        && (op.ih * op.sh, op.iw * op.sw) == (op.oh, op.ow);
    (op.kind == Kind::Concatc || upsample).then_some(p)
}

/// The inputs `slot`, of `channels` channels upsampled by `2^shift`, reads
/// through the concatenations and upsamplings it folds in, in channel order,
/// into `found.0`, and those operations into `found.1`.
#[allow(clippy::too_many_arguments)]
fn expand(
    ops: &[Op],
    reads: &[usize],
    writer: &[usize],
    i: usize,
    slot: usize,
    channels: usize,
    shift: u32,
    found: (&mut Vec<(usize, usize, u32)>, &mut Vec<usize>),
) {
    let (sources, producers) = found;
    let Some(p) = foldable(ops, reads, writer, i, slot) else {
        sources.push((slot, channels, shift));
        return;
    };
    producers.push(p);
    let op = &ops[p];
    if op.kind == Kind::Concatc {
        expand(
            ops,
            reads,
            writer,
            p,
            op.x,
            op.ic,
            shift,
            (sources, producers),
        );
        expand(
            ops,
            reads,
            writer,
            p,
            op.x2,
            op.n2,
            shift,
            (sources, producers),
        );
    } else {
        let shift = shift + op.sh.trailing_zeros();
        expand(
            ops,
            reads,
            writer,
            p,
            op.x,
            channels,
            shift,
            (sources, producers),
        );
    }
}

/// Whether operation `m` is a channel multiply only the addition after it
/// reads, which adds the product to the multiply's own input: `x + x * gate`.
fn residual_gate(ops: &[Op], reads: &[usize], m: usize) -> bool {
    let gate = &ops[m];
    gate.kind == Kind::Channelmul
        && gate.fused == Fused::None
        && reads[gate.y] == 1
        && ops.get(m + 1).is_some_and(|a| {
            a.kind == Kind::Add && a.fused == Fused::None && a.x == gate.x && a.x2 == gate.y
        })
}

/// `GpuModel::arrange` gives each buffer of the plan one place on the GPU
/// for the whole run, so a buffer written twice would be read where its
/// last writer left it. The exporter writes each buffer once.
fn one_writer(plan: &Plan) -> Result<(), String> {
    let mut written = vec![false; plan.pool.len()];
    for (i, op) in plan.ops.iter().enumerate() {
        if std::mem::replace(&mut written[op.y], true) {
            return Err(format!("operation {i} writes buffer {} again", op.y));
        }
    }
    Ok(())
}

/// Why `op` cannot run on the GPU, if it cannot.
fn supported(op: &Op) -> Result<(), String> {
    let fused = matches!(
        op.kind,
        Kind::Conv | Kind::Dwconv | Kind::Convtranspose | Kind::Add
    );
    if op.fused != Fused::None && !fused {
        return Err(format!("{:?} with {:?} folded in", op.kind, op.fused));
    }
    match op.kind {
        Kind::Ctchead => Err("the CTC head with probabilities".into()),
        Kind::Attention if op.oc / op.n2 > MAX_HEAD => Err("attention heads too large".into()),
        Kind::Dwconv if op.kw > DW_MAX_KW || op.sw > DW_MAX_STRIDE => {
            Err("a depthwise window too wide".into())
        }
        _ => Ok(()),
    }
}

/// The CTC head's logits per pixel: its classes padded to whole panels.
fn ctc_columns(op: &Op) -> usize {
    op.n2.div_ceil(16) * 16
}

/// How many runs of pixels `mean.comp` sums a mean's input in, and the
/// pixels in each.
fn mean_chunks(op: &Op) -> (usize, usize) {
    let pixels = op.ih * op.iw;
    let chunk = pixels.div_ceil(MAX_GROUPS).max(256);
    (pixels.div_ceil(chunk).max(1), chunk)
}

/// The integers of a push-constant block, as `i32` bytes: a value wrapped from
/// a negative `isize` (a pad, `offset`'s -1) keeps its sign.
fn push(values: &[usize]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| {
            i32::try_from(v as isize)
                .expect("push constants fit i32")
                .to_le_bytes()
        })
        .collect()
}

/// A weight block's start, or the shaders' -1 for none.
fn offset(block: super::plan::Block) -> usize {
    if block.1 == 0 {
        // -1, through `push`.
        usize::MAX
    } else {
        block.0
    }
}

/// The `gemm.comp` variant, by index, whose tile width fits `columns` best,
/// and its tile's rows and columns.
fn tile(columns: usize) -> (usize, usize, usize) {
    match columns {
        33.. => (0, 64, 64),
        17..=32 => (1, 128, 32),
        _ => (2, 256, 16),
    }
}

/// `gemm.comp`'s push constants, named as in its `P`; `shifts` are s1 to s4.
#[derive(Default)]
struct Gemm {
    ih: usize,
    iw: usize,
    ic: usize,
    oh: usize,
    ow: usize,
    oc: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    pt: usize,
    pl: usize,
    n2: usize,
    w_off: usize,
    b_off: usize,
    act: usize,
    gated: usize,
    residual: usize,
    act2: usize,
    panels: usize,
    spread_kw: usize,
    spread_oc: usize,
    c3: usize,
    c4: usize,
    shifts: [usize; 4],
}

impl Gemm {
    fn push(&self) -> Vec<u8> {
        let [s1, s2, s3, s4] = self.shifts;
        push(&[
            self.ih,
            self.iw,
            self.ic,
            self.oh,
            self.ow,
            self.oc,
            self.kh,
            self.kw,
            self.sh,
            self.sw,
            self.pt,
            self.pl,
            self.n2,
            self.w_off,
            self.b_off,
            self.act,
            self.gated,
            self.residual,
            self.act2,
            self.panels,
            self.spread_kw,
            self.spread_oc,
            self.c3,
            self.c4,
            s1,
            s2,
            s3,
            s4,
        ])
    }
}

/// Workgroups for `n` invocations, `group` to a workgroup, as a grid the
/// shaders flatten again (see `invocation()` in `act.glsl`).
fn groups(n: usize, group: usize) -> [u32; 3] {
    let groups = n.div_ceil(group).max(1);
    let x = groups.min(MAX_GROUPS);
    [x as u32, groups.div_ceil(x) as u32, 1]
}

/// Record `step` of `lane`'s run.
fn dispatch(
    c: &mut Commands<'_>,
    k: &Kernels,
    step: &Step,
    weights: &Buffer,
    lane: &Lane,
) -> Result<(), String> {
    let at = |slot: usize| &lane.buffers[lane.home[slot]];
    let (ops, partial) = (&lane.plan.ops, &lane.partial);
    let op = &ops[step.op];
    let act = op.fused as usize;
    let (x, y) = (at(step.input(ops)), at(step.writes(ops)));
    // A second input folded away has no buffer of its own.
    let x2 = match op.second() {
        Some(slot) if step.residual.is_none() && step.sources.is_empty() => at(slot),
        _ => x,
    };
    let n = op.out_len();
    match op.kind {
        Kind::Conv => {
            let (tile, bm, bn) = tile(op.oc);
            let kernel = &k.gemm[tile];
            let gate = step.gate.map_or(x, |g| at(ops[g].x2));
            let addend = step.addend(ops).map_or(x, &at);
            let act2 = step.add.map_or(0, |a| ops[a].fused as usize);
            // Each input: its buffer, channels and upsampling shift.
            let mut inputs = [
                (x, op.ic, 0),
                (x2, op.n2 * usize::from(op.second().is_some()), 0),
                (x, 0, 0),
                (x, 0, 0),
            ];
            for (input, &(slot, channels, shift)) in inputs.iter_mut().zip(&step.sources) {
                *input = (at(slot), channels, shift as usize);
            }
            c.dispatch(
                kernel,
                &[
                    inputs[0].0,
                    inputs[1].0,
                    weights,
                    y,
                    gate,
                    addend,
                    inputs[2].0,
                    inputs[3].0,
                ],
                &Gemm {
                    ih: op.ih,
                    iw: op.iw,
                    ic: inputs[0].1,
                    oh: op.oh,
                    ow: op.ow,
                    oc: op.oc,
                    kh: op.kh,
                    kw: op.kw,
                    sh: op.sh,
                    sw: op.sw,
                    pt: op.pt as usize,
                    pl: op.pl as usize,
                    n2: inputs[1].1,
                    w_off: op.w.0,
                    b_off: offset(op.b),
                    act,
                    gated: usize::from(step.gate.is_some()),
                    residual: usize::from(step.add.is_some()),
                    act2,
                    c3: inputs[2].1,
                    c4: inputs[3].1,
                    shifts: inputs.map(|input| input.2),
                    ..Gemm::default()
                }
                .push(),
                groups((op.oh * op.ow).div_ceil(bm) * op.oc.div_ceil(bn), 1),
            )
        }
        Kind::Dwconv if (op.kh, op.kw, op.sh, op.sw) == (7, 7, 1, 1) => c.dispatch(
            &k.dw7,
            &[x, weights, y],
            &push(&[
                op.ih,
                op.iw,
                op.oc,
                op.oh,
                op.ow,
                op.kh,
                op.kw,
                op.sh,
                op.sw,
                op.pt as usize,
                op.pl as usize,
                op.w.0,
                offset(op.b),
                act,
            ]),
            groups(
                op.oh.div_ceil(DW7_BLOCK) * op.ow.div_ceil(DW7_BLOCK) * op.oc,
                GROUP,
            ),
        ),
        Kind::Dwconv => c.dispatch(
            &k.dwconv,
            &[x, weights, y],
            &push(&[
                op.ih,
                op.iw,
                op.oc,
                op.oh,
                op.ow,
                op.kh,
                op.kw,
                op.sh,
                op.sw,
                op.pt as usize,
                op.pl as usize,
                op.w.0,
                offset(op.b),
                act,
            ]),
            groups(op.oh * op.ow.div_ceil(DW_RUN) * op.oc, GROUP),
        ),
        Kind::Add | Kind::Channelmul | Kind::Upsample | Kind::Upsampleadd | Kind::Concatc => {
            let mode = match op.kind {
                Kind::Add => 0,
                Kind::Channelmul if step.add.is_some() => 5,
                Kind::Channelmul => 1,
                Kind::Upsample => 2,
                Kind::Upsampleadd if step.residual.is_some() => 6,
                Kind::Upsampleadd => 3,
                _ => 4,
            };
            let (x, x2, gate) = match step.residual {
                Some(m) => (at(op.x), at(ops[m].x), at(ops[m].x2)),
                None => (x, x2, x),
            };
            let (sh, sw) = if matches!(mode, 2 | 3 | 6) {
                (op.sh, op.sw)
            } else {
                (1, 1)
            };
            c.dispatch(
                &k.eltwise,
                &[x, x2, y, gate],
                &push(&[mode, n, op.iw, op.oc, op.ow, sh, sw, op.ic, op.n2, act]),
                groups(n, GROUP),
            )
        }
        Kind::Maxpool | Kind::Avgpool => c.dispatch(
            &k.pool,
            &[x, y],
            &push(&[
                usize::from(op.kind == Kind::Avgpool),
                op.ih,
                op.iw,
                op.oc,
                op.oh,
                op.ow,
                op.kh,
                op.kw,
                op.sh,
                op.sw,
            ]),
            groups(n, GROUP),
        ),
        Kind::Mean => {
            let (chunks, chunk) = mean_chunks(op);
            let pixels = op.ih * op.iw;
            let lanes = op.ic.div_ceil(GROUP) as u32;
            c.dispatch(
                &k.mean,
                &[x, partial, y],
                &push(&[0, pixels, op.ic, chunk]),
                [lanes, chunks as u32, 1],
            )?;
            c.barrier();
            c.dispatch(
                &k.mean,
                &[x, partial, y],
                &push(&[1, pixels, op.ic, chunk]),
                [lanes, 1, 1],
            )
        }
        // A product over the input pixels whose columns are the taps'
        // outputs, each written to its own output pixel; one too narrow for
        // a tile of 16 columns stays a dot product per output.
        Kind::Convtranspose if op.kh * op.kw * op.oc >= 16 => {
            let columns = op.kh * op.kw * op.oc;
            let (tile, bm, bn) = tile(columns);
            c.dispatch(
                &k.gemm[tile],
                &[x, x, weights, y, x, x, x, x],
                &Gemm {
                    ih: op.ih,
                    iw: op.iw,
                    ic: op.ic,
                    oh: op.ih,
                    ow: op.iw,
                    oc: columns,
                    kh: 1,
                    kw: 1,
                    sh: 1,
                    sw: 1,
                    w_off: op.w.0,
                    b_off: offset(op.b),
                    act,
                    spread_kw: op.kw,
                    spread_oc: op.oc,
                    ..Gemm::default()
                }
                .push(),
                groups((op.ih * op.iw).div_ceil(bm) * columns.div_ceil(bn), 1),
            )
        }
        Kind::Convtranspose => c.dispatch(
            &k.convt,
            &[x, weights, y],
            &push(&[
                op.iw,
                op.ic,
                op.oh,
                op.ow,
                op.oc,
                op.kh,
                op.kw,
                op.w.0,
                offset(op.b),
                act,
            ]),
            groups(n, GROUP),
        ),
        Kind::Layernorm => c.dispatch(
            &k.layernorm,
            &[x, weights, y],
            &push(&[op.oh * op.ow, op.oc, op.w.0]),
            groups(op.oh * op.ow, GROUP),
        ),
        Kind::Attention => c.dispatch(
            &k.attention,
            &[x, weights, y],
            &push(&[op.iw, op.oc, op.n2, op.w.0]),
            groups(op.iw * op.n2, GROUP),
        ),
        Kind::Ctcargmax => {
            // The logits as a product with the weights' panels, the bias
            // padded past the classes with -1e30, then each pixel's choice.
            let columns = ctc_columns(op);
            let logits = &lane.logits;
            let gemm = Gemm {
                ih: 1,
                iw: op.iw,
                ic: op.ic,
                oh: 1,
                ow: op.iw,
                oc: columns,
                kh: 1,
                kw: 1,
                sh: 1,
                sw: 1,
                w_off: op.w.0,
                b_off: op.b.0,
                panels: 1,
                ..Gemm::default()
            };
            c.dispatch(
                &k.gemm[0],
                &[x, x, weights, logits, x, x, x, x],
                &gemm.push(),
                groups(op.iw.div_ceil(64) * columns.div_ceil(64), 1),
            )?;
            c.barrier();
            c.dispatch(
                &k.ctc,
                &[logits, y],
                &push(&[op.n2, columns]),
                [op.iw as u32, 1, 1],
            )
        }
        Kind::Ctchead => Err("the CTC head with probabilities".into()),
    }
}

fn floats(bytes: &[u8]) -> &[f32] {
    assert!(bytes.as_ptr().cast::<f32>().is_aligned());
    // SAFETY: the mapping is aligned for f32 (checked) and every bit
    // pattern is a float; the length rounds down to whole floats.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / 4) }
}

fn floats_mut(bytes: &mut [u8]) -> &mut [f32] {
    assert!(bytes.as_ptr().cast::<f32>().is_aligned());
    // SAFETY: as for `floats`, and the borrow is unique.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len() / 4) }
}

#[cfg(test)]
mod tests {
    use super::super::plan::{encode, op, Kind, Plan};
    use super::one_writer;

    #[test]
    fn rewritten_buffer_is_refused() {
        let plan = |last_y| {
            let mut ops = [
                op(Kind::Add, 0, 1, 3, 3),
                op(Kind::Add, 1, 2, 3, 3),
                op(Kind::Add, 2, last_y, 3, 3),
            ];
            for op in &mut ops {
                op.x2 = op.x;
            }
            Plan::parse(&encode((0, 2, 2, 3), 4, &[last_y], &ops, &[]))
                .unwrap()
                .0
        };
        assert!(one_writer(&plan(3)).is_ok());
        assert!(one_writer(&plan(1)).is_err());
    }
}
