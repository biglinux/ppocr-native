//! PP-OCRv6's detector and recogniser, run on the CPU in plain Rust.
//!
//! Both networks are built from a few operations repeated a hundred times:
//! a 1x1 convolution, a depthwise convolution, an activation, an addition,
//! plus the recogniser's attention and CTC head and the detector's
//! upsampling. This engine specializes those operations, chooses a kernel
//! tier for the CPU once per process, and reserves its tensor storage ahead
//! of time; no inference library is linked.
//!
//! ```no_run
//! use std::path::Path;
//! use ppocr_native::engine::Model;
//!
//! let mut model = Model::load(Path::new("plans/PP-OCRv6_rec_small_argmax.plan"))?;
//! model.set_input_size(48, 640)?;
//! let (h, w, c) = model.input_shape();
//! model.input_mut().copy_from_slice(&vec![0.0; h * w * c]);
//! model.run();
//! let best_classes_and_probabilities = model.output(0);
//! # Ok::<(), ppocr_native::engine::Error>(())
//! ```
//!
//! Pixels are `f32`, row by row, and within a row pixel by pixel, with the
//! colour channels of one pixel next to each other (HWC): transpose
//! RapidOCR's CHW tensor first. Plans come from `tools/export_ocr.py`; one
//! plan serves every input size, see [`Model::set_input_size`].
//!
//! One thread, one image at a time, 32-bit floats throughout: no thread
//! pool, batching or quantisation.

#[macro_use]
mod kernels;
mod layers;
mod plan;

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::Range;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

pub use kernels::simd_tier;
pub use plan::{Block, Error, Fused, Kind, Op};
use plan::{Plan, MAX_SCRATCH_FLOATS};

/// Cache-line-aligned allocation base. Individual NHWC rows may be unaligned;
/// all generic SIMD paths therefore continue to use unaligned loads/stores.
struct Buffer {
    ptr: NonNull<f32>,
    len: usize,
}

impl Buffer {
    fn new(len: usize) -> Self {
        Self::try_new(len).unwrap_or_else(|| std::alloc::handle_alloc_error(Self::layout(len)))
    }

    fn try_new(len: usize) -> Option<Self> {
        // SAFETY: `layout` has a non-zero size and a valid power-of-two align.
        let raw = unsafe { alloc_zeroed(Self::layout(len)) }.cast::<f32>();
        NonNull::new(raw).map(|ptr| Self { ptr, len })
    }

    fn layout(len: usize) -> Layout {
        Layout::from_size_align(
            len.max(1).checked_mul(4).expect("buffer size overflows"),
            64,
        )
        .expect("buffer size overflows")
    }

    /// Replace this buffer with a zeroed one of `len` floats if it holds
    /// fewer. The new one is allocated first, so on failure this one stays.
    fn reserve(&mut self, len: usize) -> Result<(), Error> {
        if self.len < len {
            *self = Self::try_new(len).ok_or_else(|| {
                Error::Invalid(format!("cannot allocate {} bytes of scratch", len * 4))
            })?;
        }
        Ok(())
    }

    /// # Safety
    /// No mutable view of this buffer may exist for the returned lifetime.
    unsafe fn get(&self) -> &[f32] {
        // SAFETY: the allocation holds `len` initialised floats; the caller
        // promises no mutable view overlaps this one.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// # Safety
    /// No other view of this buffer may exist for the returned lifetime.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut [f32] {
        // SAFETY: the allocation holds `len` initialised floats; the caller
        // promises this is the only live view.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: allocated by `Buffer::try_new` with exactly this layout.
        unsafe { dealloc(self.ptr.as_ptr().cast(), Self::layout(self.len)) };
    }
}

/// A plan buffer's floats in the model's arena: `len` of them from `ptr`,
/// or none for a buffer no step writes.
#[derive(Clone, Copy)]
struct View {
    ptr: NonNull<f32>,
    len: usize,
}

impl View {
    const EMPTY: Self = Self {
        ptr: NonNull::dangling(),
        len: 0,
    };

    /// # Safety
    /// No mutable view of these floats may exist for the returned lifetime.
    unsafe fn get(&self) -> &[f32] {
        // SAFETY: `Model::arrange` points every view at `len` initialised
        // floats of the arena; the caller promises no mutable view overlaps.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// # Safety
    /// No other view of these floats may exist for the returned lifetime.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut [f32] {
        // SAFETY: as for `get`; the caller promises this is the only view.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

/// The weights: written once while loading, only read after, so models made
/// with [`Model::try_clone`] share one copy across threads.
struct Weights(Buffer);

// SAFETY: `Weights::new` fills the buffer before the value exists, and
// afterwards only `get` reads it, so no thread ever sees a write.
unsafe impl Send for Weights {}
// SAFETY: as for `Send`: shared references only ever read.
unsafe impl Sync for Weights {}

impl Weights {
    fn new(values: &[f32]) -> Self {
        let buffer = Buffer::new(values.len());
        // SAFETY: freshly allocated, no other view exists.
        unsafe { buffer.get_mut() }.copy_from_slice(values);
        Self(buffer)
    }

    fn get(&self) -> &[f32] {
        // SAFETY: never written after `new`, so no mutable view exists.
        unsafe { self.0.get() }
    }
}

/// A loaded model with its scratch memory reserved.
///
/// Running allocates nothing; changing the input size allocates only when a
/// buffer has to grow.
pub struct Model {
    plan: Plan,
    /// Held in cache-line-aligned storage, not a plain vector: the exporter
    /// starts every weight block on a 64-byte boundary, and that only buys
    /// aligned vector loads if the block the offsets are measured from is
    /// aligned too.
    weights: Arc<Weights>,
    /// The memory of every tensor a run writes, laid out by [`lay_out`].
    arena: Buffer,
    /// Each plan buffer's place in `arena`. While a step runs, the buffers
    /// it names have memory apart (see [`lay_out`]), but for an output
    /// written over an input, which [`Model::in_place`] runs on one view.
    pool: Vec<View>,
    /// [`lay_out`]'s lists, kept so that laying out allocates nothing.
    layout: Placement,
    /// Pointwise layers whose weights `pack_panels` reordered.
    panels: Vec<bool>,
    /// How a run goes through the operations, decided by `schedule`.
    steps: Vec<Step>,
    /// A group of rows of a private pair's convolution, see [`Pair`].
    rows: Buffer,
    /// A run of an expanded pair's widened pixels.
    mid: Buffer,
    /// A group of rows of a depthwise convolution feeding the layer after it.
    fed: Buffer,
    /// Scratch of the attention, CTC head, padded-window and transposed
    /// convolution layers, and the gate of a gated step.
    scratch: Buffer,
}

/// What a run does next: one operation, or consecutive operations that
/// run together a part of the image at a time, so that what passes between
/// them stays in cache and, when nothing else reads it, never reaches its
/// tensor.
#[derive(Clone, Copy)]
enum Step {
    /// Operation `i`; with `mean`, the channel mean after it too, summed as
    /// `i` writes (see [`can_sum_mean`]).
    Op {
        i: usize,
        mean: bool,
    },
    /// The windowed convolution `conv` reading `inputs` in place of what the
    /// chain of layers from `start` builds, which never runs (see
    /// [`chained_inputs`]); `mean` as for `Op`.
    Chained {
        start: usize,
        conv: usize,
        inputs: Inputs,
        mean: bool,
    },
    Pair(Pair),
    /// Depthwise convolution `dw` a group of rows at a time into `fed`, each
    /// group straight through the pointwise convolution after it (see
    /// [`can_feed`]); with `mean`, that convolution's channel mean summed
    /// as it goes.
    Fed {
        dw: usize,
        mean: bool,
    },
    /// Channel multiply `i` and the add of its product to its input, as one
    /// pass (see [`can_gate_add`]).
    GateAdd(usize),
    /// Transposed convolution `i` handing each output row to the one after
    /// it (see [`can_chain_transposed`]).
    Transposed(usize),
}

impl Step {
    /// The operations this step runs.
    fn ops(&self) -> Range<usize> {
        match *self {
            Step::Op { i, mean } => i..i + 1 + usize::from(mean),
            Step::Chained {
                start, conv, mean, ..
            } => start..conv + 1 + usize::from(mean),
            Step::Pair(pair) => {
                let first = pair.producer.or(pair.gate).or(pair.expand);
                first.unwrap_or(pair.conv)..pair.conv + 2
            }
            Step::Fed { dw, mean } => dw..dw + 2 + usize::from(mean),
            Step::GateAdd(i) | Step::Transposed(i) => i..i + 2,
        }
    }

    /// The buffers this step writes. The other outputs of its operations
    /// stay in `rows`, `mid`, `fed` or `scratch`, or are never made.
    fn writes(&self, ops: &[Op]) -> [Option<usize>; 2] {
        match *self {
            Step::Op { i, mean } | Step::Chained { conv: i, mean, .. } => {
                [Some(ops[i].y), mean.then(|| ops[i + 1].y)]
            }
            Step::Pair(pair) => [
                Some(ops[pair.conv + 1].y),
                (!pair.private).then(|| ops[pair.conv].y),
            ],
            Step::Fed { dw, mean } => [Some(ops[dw + 1].y), mean.then(|| ops[dw + 2].y)],
            Step::GateAdd(i) | Step::Transposed(i) => [Some(ops[i + 1].y), None],
        }
    }

    /// The output this step may write over one of its inputs, if the input
    /// is not read afterwards, and the inputs it may write over, in order
    /// of preference: a max pool over its input at stride 1 and its input's
    /// size, an add over either of two distinct inputs, an upsampled sum
    /// over its addend, and a gated sum over the gated input.
    fn over(&self, ops: &[Op]) -> Option<(usize, [Option<usize>; 2])> {
        match *self {
            Step::Op { i, mean: false } => {
                let op = &ops[i];
                match op.kind {
                    Kind::Maxpool if (op.sh, op.sw, op.oh, op.ow) == (1, 1, op.ih, op.iw) => {
                        Some((op.y, [Some(op.x), None]))
                    }
                    Kind::Add if op.x != op.x2 => Some((op.y, [Some(op.x), Some(op.x2)])),
                    Kind::Upsampleadd => Some((op.y, [Some(op.x2), None])),
                    _ => None,
                }
            }
            Step::GateAdd(i) => Some((ops[i + 1].y, [Some(ops[i].x), None])),
            _ => None,
        }
    }
}

/// The pointwise convolution `conv` and the residual add after it, a group
/// of rows at a time (see [`can_pair`]), with the layers before them that
/// run inside.
#[derive(Clone, Copy)]
struct Pair {
    /// A depthwise convolution computing each group's input rows into `fed`
    /// (see [`can_feed`]).
    producer: Option<usize>,
    /// A channel multiply gating the expansion's input, a run of pixels at a
    /// time into `scratch` (see [`can_gate_expand`]).
    gate: Option<usize>,
    /// A pointwise convolution widening the pair's input, a run of pixels at
    /// a time into `mid` (see [`can_expand`]).
    expand: Option<usize>,
    conv: usize,
    /// Only the add reads the convolution's output: it goes through `rows`
    /// and never reaches its tensor, which nothing would observe.
    private: bool,
}

// SAFETY: a `Model` owns its scratch buffers outright (nothing else holds a
// pointer into them, and every method that writes takes `&mut self`), and the
// weights it may share are never written (`Weights`). Moving one to another
// thread is therefore sound, which is what lets a caller build a model on one
// thread and run it on another.
unsafe impl Send for Model {}

impl Model {
    /// Load a plan written by `tools/export_ocr.py`, at the input size it
    /// was exported at.
    ///
    /// Validation completes before tensor buffers are allocated or kernels
    /// execute.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let (plan, values) = Plan::load(path)?;
        Self::new(plan, values)
    }

    /// Schedule and store a plan that `Plan::parse` has checked.
    fn new(plan: Plan, mut values: Vec<f32>) -> Result<Self, Error> {
        let panels = pack_panels(&plan, &mut values);
        let weights = Arc::new(Weights::new(&values));
        // No more steps than operations, so scheduling never allocates.
        let steps = Vec::with_capacity(plan.ops.len());
        let mut model = Self {
            pool: vec![View::EMPTY; plan.pool.len()],
            layout: Placement::new(plan.pool.len()),
            plan,
            weights,
            arena: Buffer::new(0),
            panels,
            steps,
            rows: Buffer::new(0),
            mid: Buffer::new(0),
            fed: Buffer::new(0),
            scratch: Buffer::new(0),
        };
        model.schedule()?;
        Ok(model)
    }

    /// Another model sharing this one's weights, with scratch memory of its
    /// own, at the same input size: a model per thread costs the weights once.
    pub fn try_clone(&self) -> Result<Self, Error> {
        let mut model = Self {
            plan: self.plan.clone(),
            weights: Arc::clone(&self.weights),
            arena: Buffer::new(0),
            pool: vec![View::EMPTY; self.plan.pool.len()],
            layout: Placement::new(self.plan.pool.len()),
            panels: self.panels.clone(),
            steps: Vec::with_capacity(self.plan.ops.len()),
            rows: Buffer::new(0),
            mid: Buffer::new(0),
            fed: Buffer::new(0),
            scratch: Buffer::new(0),
        };
        model.schedule()?;
        Ok(model)
    }

    /// Decide from the plan's current shapes the steps of a run, then
    /// [`Model::arrange`] the memory for them. Runs at load and on each
    /// change of input size.
    fn schedule(&mut self) -> Result<(), Error> {
        self.steps.clear();
        let mut i = 0;
        while i < self.plan.ops.len() {
            let step = next_step(&self.plan, i);
            i = step.ops().end;
            self.steps.push(step);
        }
        self.arrange()
    }

    /// Lay out the tensors the steps write, size the buffers they work in,
    /// and grow every allocation those outgrew; it allocates only to grow,
    /// and fails when the total would pass the scratch ceiling or an
    /// allocation fails.
    fn arrange(&mut self) -> Result<(), Error> {
        let plan = &self.plan;
        let (mut rows, mut mid, mut fed) = (0, 0, 0);
        let mut scratch = 0;
        for op in &plan.ops {
            scratch = scratch.max(match op.kind {
                Kind::Attention => layers::attention_scratch(op),
                Kind::Ctchead | Kind::Ctcargmax => layers::head_state(op.iw),
                // A pixel of zeros for the taps outside the image.
                Kind::Conv if reads_windows(op) => op.ic.max(op.n2),
                Kind::Convtranspose => layers::transpose_scratch(op),
                _ => 0,
            });
        }
        for step in &self.steps {
            match *step {
                Step::Pair(pair) => {
                    let conv = &plan.ops[pair.conv];
                    let group = rows_per_group(conv) * conv.ow;
                    if pair.private {
                        rows = rows.max(group * conv.oc);
                    }
                    if let Some(d) = pair.producer {
                        fed = fed.max(group * plan.ops[d].oc);
                    }
                    if let Some(e) = pair.expand {
                        let (e, gate) = (&plan.ops[e], pair.gate.map_or(0, |g| plan.ops[g].oc));
                        mid = mid.max(expand_run(e) * e.oc);
                        // A run of the gated product, then the gate.
                        scratch = scratch.max((expand_run(e) + 1) * gate);
                    }
                }
                Step::Fed { dw, .. } => {
                    let conv = &plan.ops[dw + 1];
                    fed = fed.max(rows_per_group(conv) * conv.ow * plan.ops[dw].oc);
                }
                Step::GateAdd(i) => scratch = scratch.max(plan.ops[i].oc),
                Step::Transposed(i) => {
                    let pair = layers::transpose_pair_scratch(&plan.ops[i], &plan.ops[i + 1]);
                    scratch = scratch.max(pair);
                }
                Step::Op { .. } | Step::Chained { .. } => {}
            }
        }
        let arena = lay_out(plan, &self.steps, &mut self.layout);
        let total = arena + rows + mid + fed + scratch;
        if total > MAX_SCRATCH_FLOATS {
            return Err(Error::Invalid(format!(
                "scratch wants {total} floats, ceiling is {MAX_SCRATCH_FLOATS}"
            )));
        }
        self.rows.reserve(rows)?;
        self.mid.reserve(mid)?;
        self.fed.reserve(fed)?;
        self.scratch.reserve(scratch)?;
        self.arena.reserve(arena)?;
        for (slot, view) in self.pool.iter_mut().enumerate() {
            *view = match self.layout.offset[slot] {
                Some(at) => View {
                    // SAFETY: `lay_out` keeps `at + len` within `arena` floats.
                    ptr: unsafe { self.arena.ptr.add(at) },
                    len: self.plan.pool[slot],
                },
                None => View::EMPTY,
            };
        }
        Ok(())
    }

    /// Make the following runs `height` by `width` pixels. Shapes, buffer
    /// sizes and the schedule are derived again; a buffer grows when a size
    /// needs more than any before, and nothing is allocated otherwise. On
    /// error the model keeps its size.
    pub fn set_input_size(&mut self, height: usize, width: usize) -> Result<(), Error> {
        if (height, width) == (self.plan.input.h, self.plan.input.w) {
            return Ok(());
        }
        let old = (self.plan.input.h, self.plan.input.w);
        self.plan.set_size(height, width, self.weights.0.len)?;
        if let Err(e) = self.schedule() {
            // Every buffer still holds at least the old size, so going back
            // allocates nothing and cannot fail.
            self.plan
                .set_size(old.0, old.1, self.weights.0.len)
                .expect("a size the model ran at derives again");
            self.schedule()
                .expect("a size the model ran at fits its buffers");
            return Err(e);
        }
        Ok(())
    }

    /// Height, width and channel count of the image this model expects.
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (self.plan.input.h, self.plan.input.w, self.plan.input.c)
    }

    /// The image buffer, to be filled before each call to [`Model::run`].
    ///
    /// Running the model overwrites scratch memory, and the image shares that
    /// memory once it has been consumed. Fill this again for every run; do
    /// not assume the last image survived.
    pub fn input_mut(&mut self) -> &mut [f32] {
        let n = self.plan.input.h * self.plan.input.w * self.plan.input.c;
        // SAFETY: `&mut self` rules out any other live view of the pool.
        let buffer = unsafe { self.pool[self.plan.input.slot].get_mut() };
        &mut buffer[..n]
    }

    /// Run the network once over the image in [`Model::input_mut`].
    pub fn run(&mut self) {
        for step in &self.steps {
            self.run_step(step);
        }
    }

    /// [`Model::run`], recording the time of each step in milliseconds of
    /// `clock`, with the operations it ran: one, or several that run
    /// together a part of the image at a time. A CPU-time clock keeps a
    /// loaded host's descheduling out.
    ///
    /// `times` is cleared and refilled in plan order; the ranges index
    /// [`Model::ops`]. Reading the clock that often adds a little to the
    /// total, so use this to compare steps against each other, not to quote
    /// a run time.
    pub fn run_timed(
        &mut self,
        times: &mut Vec<(Range<usize>, f64)>,
        mut clock: impl FnMut() -> f64,
    ) {
        times.clear();
        for step in &self.steps {
            let started = clock();
            self.run_step(step);
            times.push((step.ops(), clock() - started));
        }
    }

    fn run_step(&self, step: &Step) {
        match *step {
            Step::Op { i, mean } => self.alone(i, &Inputs::default(), mean),
            Step::Chained {
                conv,
                ref inputs,
                mean,
                ..
            } => self.alone(conv, inputs, mean),
            Step::Pair(ref pair) => self.row_pair(pair),
            Step::Fed { dw, mean } => self.fed_pointwise(dw, mean),
            Step::GateAdd(i) => self.gate_add(&self.plan.ops[i], &self.plan.ops[i + 1]),
            Step::Transposed(i) => self.transposed_pair(i),
        }
    }

    /// Operation `i`, a windowed convolution reading `inputs` if they name
    /// any; with `mean`, the channel mean after it summed as `i` writes.
    fn alone(&self, i: usize, inputs: &Inputs, mean: bool) {
        if !mean {
            self.step(i, inputs, &mut |_| {});
            return;
        }
        let (op, mean) = (&self.plan.ops[i], &self.plan.ops[i + 1]);
        let mut sums = layers::MeanSums::new(op.oc);
        self.step(i, inputs, &mut |rows| sums.add(rows));
        // SAFETY: the mean writes a buffer of its own; its input, the
        // layer's output, is no longer written.
        sums.finish(unsafe { self.pool[mean.y].get_mut() }, op.oh * op.ow);
    }

    /// The pointwise convolution and residual add of `pair`, a group of rows
    /// at a time. With an expansion the convolution before them runs too,
    /// its output only ever in `mid`, a run of pixels at a time; with a gate
    /// the channel multiply before that, its product only in `scratch`; with
    /// a producer the depthwise convolution before the first of them, a
    /// group of rows at a time into `fed`. Every other output is written to
    /// its tensor, except a private convolution's.
    fn row_pair(&self, pair: &Pair) {
        let Pair {
            producer,
            gate,
            expand,
            conv: at,
            private,
        } = *pair;
        let ops = &self.plan.ops;
        let (conv, add) = (&ops[at], &ops[at + 1]);
        let panels = self.panels[at];
        // The add is allowed to reuse the convolution's input allocation. When
        // channels expand, visit rows backwards, like memmove: writing an
        // expanded row must never destroy input belonging to a future row.
        // A pair reading what a layer before it left in a scratch buffer
        // goes forwards: `can_expand` and `can_feed` keep the pool safe.
        let backwards =
            producer.is_none() && expand.is_none() && add.y == conv.x && conv.oc > conv.ic;
        let weights = self.weights.get();
        let block = |b: Block| &weights[b.0..b.0 + b.1];
        let rows = rows_per_group(conv);
        // The gated run goes at the start of `scratch` and the gate after
        // it, copied: a layer of the pair may write over the gate, which the
        // multiply read whole before it.
        let gated = match (gate, expand) {
            (Some(g), Some(e)) => expand_run(&ops[e]) * ops[g].oc,
            _ => 0,
        };
        if let Some(g) = gate {
            let g = &ops[g];
            // SAFETY: `scratch` is an allocation of its own, the pool is only
            // read here.
            unsafe {
                self.scratch.get_mut()[gated..gated + g.oc]
                    .copy_from_slice(&self.pool[g.x2].get()[..g.oc]);
            }
        }
        let groups = conv.oh.div_ceil(rows);
        for group in 0..groups {
            let group = if backwards { groups - 1 - group } else { group };
            let first = group * rows;
            let pixels = rows.min(conv.oh - first) * conv.ow;
            if let Some(d) = producer {
                let dw = &ops[d];
                // SAFETY: `fed` is an allocation of its own, written only
                // here; the pool is only read.
                let (x, y) = unsafe {
                    (
                        self.pool[dw.x].get(),
                        &mut self.fed.get_mut()[..pixels * dw.oc],
                    )
                };
                let (w, b, tail) = (block(dw.w), opt(dw.b, weights), Tail::of(dw));
                depthwise(
                    dw,
                    x,
                    y,
                    w,
                    b,
                    tail,
                    &mut |_| {},
                    first..first + pixels / conv.ow,
                );
            }
            // An expanded group goes in runs of `expand_run` pixels: a
            // multiple of four, so every call but the last tiles all its
            // pixels and the last ends where the group's call would.
            let run = expand.map_or(pixels, |e| expand_run(&ops[e]));
            for start in (0..pixels).step_by(run) {
                let count = run.min(pixels - start);
                let pixel = first * conv.ow + start;
                // This run of `slot`, `c` channels a pixel: from the scratch
                // buffer a layer before left it in, else from the pool.
                // SAFETY: shared views only, of buffers no layer of the pair
                // writes while they are read (see each writer below).
                let read = |slot: usize, c: usize| unsafe {
                    if gate.is_some_and(|g| slot == ops[g].y) {
                        &self.scratch.get()[..count * c]
                    } else if producer.is_some_and(|d| slot == ops[d].y) {
                        &self.fed.get()[start * c..][..count * c]
                    } else {
                        &self.pool[slot].get()[pixel * c..][..count * c]
                    }
                };
                if let Some(g) = gate {
                    let g = &ops[g];
                    let x = read(g.x, g.oc);
                    // SAFETY: `scratch` is an allocation of its own; the gate
                    // reads the pool or `fed`.
                    let (y, gates) = unsafe { self.scratch.get_mut().split_at_mut(gated) };
                    kernels::channel_mul(y, x, &gates[..g.oc], count, g.oc);
                }
                if let Some(i) = expand {
                    let e = &ops[i];
                    let x = read(e.x, e.ic);
                    // SAFETY: `mid` is an allocation of its own, written only
                    // here.
                    let y = unsafe { &mut self.mid.get_mut()[..count * e.oc] };
                    let (clip, tail) = Tail::of(e).fold_clip();
                    let (w, b) = (block(e.w), opt(e.b, weights));
                    kernels::pointwise_with(y, w, x, b, count, e.ic, e.oc, self.panels[i], clip);
                    tail.apply(y);
                }
                let (dst, n) = (pixel * conv.oc, count * conv.oc);
                {
                    let x = if expand.is_some() {
                        // SAFETY: `mid` is only read from here on.
                        unsafe { &self.mid.get()[..count * conv.ic] }
                    } else {
                        read(conv.x, conv.ic)
                    };
                    // SAFETY: the validated convolution has distinct x/y
                    // buffers, and `rows` is an allocation of its own. Drop
                    // both views BEFORE the add can overwrite conv.x.
                    let y = unsafe {
                        if private {
                            &mut self.rows.get_mut()[..n]
                        } else {
                            &mut self.pool[conv.y].get_mut()[dst..dst + n]
                        }
                    };
                    let (clip, tail) = Tail::of(conv).fold_clip();
                    let (w, b) = (block(conv.w), opt(conv.b, weights));
                    kernels::pointwise_with(y, w, x, b, count, conv.ic, conv.oc, panels, clip);
                    tail.apply(y);
                }
                {
                    // SAFETY: add.y is distinct from both add inputs and from
                    // `rows`, `scratch` and `fed`. No view of the previous
                    // convolution scope survives.
                    let input = |slot: usize| {
                        if private && slot == conv.y {
                            unsafe { &self.rows.get()[..n] }
                        } else {
                            read(slot, conv.oc)
                        }
                    };
                    let (x, x2) = (input(add.x), input(add.x2));
                    // SAFETY: as above.
                    let y = unsafe { &mut self.pool[add.y].get_mut()[dst..dst + n] };
                    add_with_tail(y, x, x2, n, Tail::of(add));
                }
            }
        }
    }

    /// Depthwise convolution `i` a group of rows at a time into `fed`, each
    /// group straight through the pointwise convolution after it, and with
    /// `mean` that convolution's channel mean summed as it goes.
    fn fed_pointwise(&self, i: usize, mean: bool) {
        let ops = &self.plan.ops;
        let (dw, conv) = (&ops[i], &ops[i + 1]);
        let weights = self.weights.get();
        let block = |b: Block| &weights[b.0..b.0 + b.1];
        let mut sums = layers::MeanSums::new(if mean { conv.oc } else { 0 });
        let rows = rows_per_group(conv);
        let (clip, tail) = Tail::of(conv).fold_clip();
        for first in (0..conv.oh).step_by(rows) {
            let pixels = rows.min(conv.oh - first) * conv.ow;
            // SAFETY: `fed` is an allocation of its own; `can_feed` keeps the
            // convolution's output apart from the depthwise input.
            let (x, fed, y) = unsafe {
                (
                    self.pool[dw.x].get(),
                    &mut self.fed.get_mut()[..pixels * dw.oc],
                    &mut self.pool[conv.y].get_mut()[first * conv.ow * conv.oc..]
                        [..pixels * conv.oc],
                )
            };
            let (w, b, dw_tail) = (block(dw.w), opt(dw.b, weights), Tail::of(dw));
            depthwise(
                dw,
                x,
                fed,
                w,
                b,
                dw_tail,
                &mut |_| {},
                first..first + pixels / conv.ow,
            );
            let (w, b) = (block(conv.w), opt(conv.b, weights));
            kernels::pointwise_with(
                y,
                w,
                fed,
                b,
                pixels,
                conv.ic,
                conv.oc,
                self.panels[i + 1],
                clip,
            );
            tail.apply(y);
            if mean {
                sums.add(y);
            }
        }
        if mean {
            // SAFETY: the mean writes a buffer of its own.
            sums.finish(
                unsafe { self.pool[ops[i + 2].y].get_mut() },
                conv.oh * conv.ow,
            );
        }
    }

    /// Transposed convolutions `i` and `i + 1`, row by row.
    fn transposed_pair(&self, i: usize) {
        let (first, second) = (&self.plan.ops[i], &self.plan.ops[i + 1]);
        // SAFETY: the first layer's input, the second's output and the
        // scratch are three buffers apart (`can_chain_transposed`).
        let (x, y, scratch) = unsafe {
            (
                self.pool[first.x].get(),
                self.pool[second.y].get_mut(),
                self.scratch.get_mut(),
            )
        };
        let weights = self.weights.get();
        let layer = |i: usize| {
            let op = &self.plan.ops[i];
            let (clip, tail) = Tail::of(op).fold_clip();
            let w = &weights[op.w.0..op.w.0 + op.w.1];
            let bias = opt(op.b, weights);
            (
                layers::Transposed {
                    op,
                    w,
                    bias,
                    clip,
                    panels: self.panels[i],
                },
                tail,
            )
        };
        let ((a, tail_a), (b, tail_b)) = (layer(i), layer(i + 1));
        a.run_into(&b, x, y, scratch, &|out| tail_a.apply(out), &|out| {
            tail_b.apply(out)
        });
    }

    /// A channel multiply and the add of its product to its input.
    fn gate_add(&self, gate: &Op, add: &Op) {
        // The add may write over the gate, which the multiply read whole
        // before it: work from a copy.
        // SAFETY: `scratch` is an allocation of its own, the pool is only
        // read, and the views end before the add's output is taken.
        let g = unsafe {
            let g = &mut self.scratch.get_mut()[..gate.oc];
            g.copy_from_slice(&self.pool[gate.x2].get()[..gate.oc]);
            &self.scratch.get()[..gate.oc]
        };
        let (clip, tail) = Tail::of(add).fold_clip();
        let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
        let pixels = gate.oh * gate.ow;
        // SAFETY: the add's output has the multiply's input's memory, or
        // memory apart from both its inputs, live at the same step.
        let y = unsafe { self.pool[add.y].get_mut() };
        if self.pool[add.y].ptr == self.pool[gate.x].ptr {
            kernels::gate_add_in_place(y, g, pixels, gate.oc, bounds);
        } else {
            // SAFETY: as above.
            let x = unsafe { self.pool[gate.x].get() };
            kernels::gate_add(y, x, g, pixels, gate.oc, bounds);
        }
        tail.apply(&mut y[..pixels * gate.oc]);
    }

    /// The input whose memory `lay_out` gave `op`'s output, which `op`
    /// then writes over.
    fn written_over(&self, op: &Op) -> Option<usize> {
        let inputs = match op.kind {
            Kind::Maxpool => [Some(op.x), None],
            Kind::Add => [Some(op.x), op.second()],
            Kind::Upsampleadd => [op.second(), None],
            _ => return None,
        };
        let at = self.pool[op.y].ptr;
        inputs
            .into_iter()
            .flatten()
            .find(|&x| self.pool[x].ptr == at)
    }

    /// Operation `op` writing over its input `over`, on one view of it.
    fn in_place(&self, op: &Op, over: usize) {
        // SAFETY: the caller holds `&mut self`; the other input of an add or
        // upsampled sum is live at the same step, so its memory is apart.
        let y = unsafe { self.pool[op.y].get_mut() };
        let other = || unsafe { self.pool[if over == op.x { op.x2 } else { op.x }].get() };
        match op.kind {
            Kind::Maxpool => maxpool_in_place(op, y),
            Kind::Add => {
                let (clip, tail) = Tail::of(op).fold_clip();
                let n = op.out_len();
                kernels::add_in_place(y, other(), n, clip);
                tail.apply(&mut y[..n]);
            }
            Kind::Upsampleadd => layers::upsample_add_in_place(op, other(), y),
            _ => unreachable!("{:?} never writes over an input", op.kind),
        }
    }

    /// Run operation `i`, a windowed convolution reading `inputs` if they
    /// name any. A convolution hands `finished` each run of whole output
    /// pixels once it is final, in order.
    fn step(&self, i: usize, inputs: &Inputs, finished: &mut dyn FnMut(&[f32])) {
        let (op, panels) = (&self.plan.ops[i], self.panels[i]);
        if let Some(over) = self.written_over(op) {
            self.in_place(op, over);
            return;
        }
        // SAFETY: `Plan::check` proved that no operation writes a buffer it
        // also reads, and `lay_out` keeps the buffers apart in memory but
        // for the output written over an input, handled above; so the output
        // view never overlaps an input view, and the caller holds `&mut
        // self`, ruling out views from outside.
        let (x, y) = unsafe { (self.pool[op.x].get(), self.pool[op.y].get_mut()) };
        let weights = self.weights.get();
        // SAFETY: as for `x`; `Plan::check` proved `x2` distinct from `y`.
        let x2 = || unsafe { self.pool[op.x2].get() };
        // SAFETY: the scratch is its own allocation, used by no other view
        // while one operation runs.
        let scratch = || unsafe { self.scratch.get_mut() };
        let block = |b: Block| &weights[b.0..b.0 + b.1];
        let tail = Tail::of(op);
        match op.kind {
            Kind::Conv if reads_windows(op) => {
                let source = |data, c, shift| kernels::Source { data, c, shift };
                let mut sources = [source(x, op.ic, 0); kernels::SOURCES];
                let count = match *inputs {
                    Inputs { count: 0, .. } if op.n2 > 0 => {
                        sources[1] = source(x2(), op.n2, 0);
                        2
                    }
                    Inputs { count: 0, .. } => 1,
                    Inputs { ref slots, count } => {
                        for (s, &(slot, c, shift)) in sources.iter_mut().zip(&slots[..count]) {
                            // SAFETY: as for `x`: a chain's sources are buffers
                            // this convolution would have read through it, and
                            // it writes none of them.
                            *s = source(unsafe { self.pool[slot].get() }, c, shift);
                        }
                        count
                    }
                };
                let (w, b) = (block(op.w), opt(op.b, weights));
                let (clip, tail) = tail.fold_clip();
                let zeros = &mut scratch()[..op.ic.max(op.n2)];
                zeros.fill(0.0);
                let len = op.ow * op.oc;
                for (oy, row) in y[..op.oh * len].chunks_exact_mut(len).enumerate() {
                    kernels::conv_row(row, &sources[..count], zeros, w, b, op, oy, clip);
                    tail.apply(row);
                    finished(row);
                }
            }
            Kind::Conv => {
                let (w, b) = (block(op.w), opt(op.b, weights));
                convolve(op, x, y, w, b, tail, panels, finished);
            }
            Kind::Dwconv => {
                let (w, b) = (block(op.w), opt(op.b, weights));
                depthwise(op, x, y, w, b, tail, finished, 0..op.oh);
            }
            Kind::Add => add_with_tail(y, x, x2(), op.out_len(), tail),
            Kind::Channelmul => {
                kernels::channel_mul(y, x, &x2()[..op.oc], op.oh * op.ow, op.oc);
            }
            Kind::Maxpool => maxpool(op, x, y),
            Kind::Avgpool => layers::average_pool(op, x, y),
            Kind::Concatc => layers::concat_channels(op, x, x2(), y),
            Kind::Layernorm => layers::layer_norm(op, x, y, block(op.w)),
            Kind::Mean => layers::mean(op, x, y),
            Kind::Attention => layers::attention(op, x, y, block(op.w)[0], scratch()),
            Kind::Upsample => layers::upsample(op, x, y),
            Kind::Upsampleadd => layers::upsample_add(op, x, x2(), y),
            Kind::Convtranspose => {
                let (clip, tail) = tail.fold_clip();
                let (w, bias) = (block(op.w), opt(op.b, weights));
                let transposed = layers::Transposed {
                    op,
                    w,
                    bias,
                    clip,
                    panels,
                };
                transposed.run(x, y, scratch(), &|out| tail.apply(out));
            }
            Kind::Ctchead | Kind::Ctcargmax => {
                let probs = op.kind == Kind::Ctchead;
                let vocab = if probs { op.oc } else { op.n2 };
                let (w, b) = (block(op.w), block(op.b));
                layers::ctc_head(y, x, w, b, op.iw, op.ic, vocab, probs, scratch());
            }
        }
    }

    /// How many results the model produces.
    pub fn output_count(&self) -> usize {
        self.plan.outputs.len()
    }

    /// One result at the current input size: the detector's probability map,
    /// HxW; the recogniser's softmax over every class at each time step, or
    /// for an argmax plan each step's best class (as a float) and its
    /// probability.
    ///
    /// # Panics
    /// If `index` is past [`Model::output_count`].
    pub fn output(&self, index: usize) -> &[f32] {
        let out = &self.plan.outputs[index];
        // SAFETY: `&self` rules out any mutable view of the pool.
        let buffer = unsafe { self.pool[out.slot].get() };
        &buffer[..out.n]
    }

    /// Bytes of scratch memory reserved, not counting the weights.
    pub fn scratch_bytes(&self) -> usize {
        (self.arena.len + self.rows.len + self.mid.len + self.fed.len + self.scratch.len) * 4
    }

    /// The operations at the current input size, for callers that time or
    /// report them.
    pub fn ops(&self) -> &[Op] {
        &self.plan.ops
    }
}

/// Whether a convolution goes through `kernels::conv_row`: every one with a
/// window, except RGB stems, whose register tiles in [`convolve`] keep three
/// channels per tap.
fn reads_windows(op: &Op) -> bool {
    !op.is_pointwise() && (op.ic != 3 || op.n2 > 0)
}

fn opt(b: Block, weights: &[f32]) -> Option<&[f32]> {
    (b.1 > 0).then(|| &weights[b.0..b.0 + b.1])
}

/// Smallest pixel run a pointwise call is given.
const GROUP_PIXELS: usize = 48;

/// Output rows per pointwise call: enough for [`GROUP_PIXELS`]. Short rows
/// are too short for the 12-pixel tiles, while whole large images would fall
/// out of cache before the folded activation or the paired addition reads
/// them.
fn rows_per_group(op: &Op) -> usize {
    GROUP_PIXELS.div_ceil(op.ow).min(op.oh)
}

/// The activation folded into an operation's tail, applied to each finished
/// run of outputs while it is still in cache. A ReLU is the clamp to
/// `[0, inf]`, which the convolution kernels can apply as they store.
#[derive(Clone, Copy)]
struct Tail {
    clip: Option<(f32, f32)>,
    /// What runs after the clamp; never `Relu`.
    fused: Fused,
}

impl Tail {
    fn of(op: &Op) -> Self {
        match op.fused {
            Fused::Relu => Self {
                clip: Some((0.0, f32::INFINITY)),
                fused: Fused::None,
            },
            fused => Self { clip: None, fused },
        }
    }

    /// Hand the clamp to a kernel that applies it as it stores; what remains
    /// is still to be applied.
    fn fold_clip(self) -> (Option<(f32, f32)>, Self) {
        (self.clip, Self { clip: None, ..self })
    }

    fn apply(self, y: &mut [f32]) {
        if let Some((low, high)) = self.clip {
            kernels::clip_inplace(y, low, high);
        }
        match self.fused {
            Fused::Sigmoid => kernels::sigmoid_inplace(y),
            Fused::Gelu => layers::gelu_inplace(y),
            Fused::Swish => layers::swish_inplace(y),
            Fused::Hardsigmoid => layers::hardsigmoid_inplace(y),
            Fused::Hardsigmoidfifth => layers::hardsigmoid_fifth_inplace(y),
            Fused::None | Fused::Relu => {}
        }
    }
}

/// `y = x1 + x2` with the addition's folded activation.
fn add_with_tail(y: &mut [f32], x1: &[f32], x2: &[f32], n: usize, tail: Tail) {
    let (clip, rest) = tail.fold_clip();
    kernels::add_with(y, x1, x2, n, clip);
    rest.apply(&mut y[..n]);
}

/// Output columns for which this kernel column reads a real pixel rather than
/// padding. Hoisting the question out of the inner loop means padding costs a
/// shorter loop instead of a test per pixel.
fn live_columns(op: &Op, kx: usize) -> (usize, usize) {
    let stride = op.sw as isize;
    let shift = op.pl - kx as isize; // source column = out * stride - shift
    let first = if shift <= 0 {
        0
    } else {
        ((shift + stride - 1) / stride) as usize
    };
    let last_source = op.iw as isize - 1 + shift;
    if last_source < 0 {
        return (0, 0);
    }
    let end = ((last_source / stride) as usize + 1).min(op.ow);
    (first.min(end), end)
}

/// Dense convolution that does not gather windows: pointwise layers and RGB
/// stems.
///
/// A 1x1 kernel is the whole layer in one matrix multiply per group of rows,
/// so it takes the dedicated pointwise kernel. A stem keeps complete interior
/// sums in registers over small output tiles; borders keep the
/// tap-at-a-time fallback, seeded with bias. No path changes the per-output
/// reduction order.
///
/// Either way a run of output rows is finished before the next begins, so the
/// activation that follows reads it from the nearest cache.
#[allow(clippy::too_many_arguments)]
fn convolve(
    op: &Op,
    x: &[f32],
    y: &mut [f32],
    w: &[f32],
    bias: Option<&[f32]>,
    tail: Tail,
    panels: bool,
    finished: &mut dyn FnMut(&[f32]),
) {
    let (ci, co) = (op.ic, op.oc);

    if op.is_pointwise() {
        // Unit stride keeps every row contiguous, so rows can be grouped.
        let rows = rows_per_group(op);
        let (clip, tail) = tail.fold_clip();
        for first in (0..op.oh).step_by(rows) {
            let pixels = rows.min(op.oh - first) * op.ow;
            let (dst, src) = (first * op.ow * co, first * op.ow * ci);
            let run = &mut y[dst..dst + pixels * co];
            let x = &x[src..src + pixels * ci];
            kernels::pointwise_with(run, w, x, bias, pixels, ci, co, panels, clip);
            tail.apply(run);
            finished(run);
        }
        return;
    }

    let tap_len = ci * co;
    #[cfg(target_arch = "x86_64")]
    let tiled = kernels::simd_tier() >= 1;
    for oy in 0..op.oh {
        let out_row = oy * op.ow;
        let mut interior = (0, 0);
        let top = (oy * op.sh) as isize - op.pt;
        if top >= 0 && top + op.kh as isize <= op.ih as isize {
            let (a, b) = live_columns(op, 0);
            let (d, e) = live_columns(op, op.kw - 1);
            let (first, end) = (a.max(d), b.min(e));
            if first < end {
                let ix = ((first * op.sw) as isize - op.pl) as usize;
                let src = (top as usize * op.iw + ix) * ci;
                let done = kernels::rgb_interior(
                    &mut y[(out_row + first) * co..],
                    &x[src..],
                    w,
                    bias,
                    op,
                    end - first,
                );
                interior = (first, first + done);
            }
        }
        kernels::fill_bias(&mut y[out_row * co..], bias, interior.0, co);
        kernels::fill_bias(
            &mut y[(out_row + interior.1) * co..],
            bias,
            op.ow - interior.1,
            co,
        );
        for ky in 0..op.kh {
            let iy = (oy * op.sh) as isize + ky as isize - op.pt;
            if iy < 0 || iy as usize >= op.ih {
                continue;
            }
            let in_row = iy as usize * op.iw;
            for kx in 0..op.kw {
                let (first, end) = live_columns(op, kx);
                for (first, end) in [(first, end.min(interior.0)), (first.max(interior.1), end)] {
                    if first >= end {
                        continue;
                    }
                    let shift = op.pl - kx as isize;
                    let tap = &w[(ky * op.kw + kx) * tap_len..][..tap_len];
                    let mut ox = first;
                    #[cfg(target_arch = "x86_64")]
                    if tiled {
                        let (xb, yb) = (x.as_ptr(), y.as_mut_ptr());
                        while ox + 4 <= end {
                            let ix = ((ox * op.sw) as isize - shift) as usize;
                            // SAFETY: `live_columns` keeps every source column,
                            // including the three that follow, inside the row; the
                            // four destinations are distinct pixels of one row.
                            unsafe {
                                kernels::matvec_t_acc_x4(
                                    [
                                        yb.add((out_row + ox) * co),
                                        yb.add((out_row + ox + 1) * co),
                                        yb.add((out_row + ox + 2) * co),
                                        yb.add((out_row + ox + 3) * co),
                                    ],
                                    tap.as_ptr(),
                                    [
                                        xb.add((in_row + ix) * ci),
                                        xb.add((in_row + ix + op.sw) * ci),
                                        xb.add((in_row + ix + 2 * op.sw) * ci),
                                        xb.add((in_row + ix + 3 * op.sw) * ci),
                                    ],
                                    ci,
                                    co,
                                );
                            }
                            ox += 4;
                        }
                    }
                    while ox < end {
                        let ix = ((ox * op.sw) as isize - shift) as usize;
                        let dst = (out_row + ox) * co;
                        let pixel = &x[(in_row + ix) * ci..(in_row + ix) * ci + ci];
                        kernels::matvec_t_acc(&mut y[dst..dst + co], tap, pixel, ci, co);
                        ox += 1;
                    }
                }
            }
        }
        let dst = out_row * co;
        tail.apply(&mut y[dst..dst + op.ow * co]);
        finished(&y[dst..dst + op.ow * co]);
    }
}

/// Depthwise convolution: every channel is filtered on its own, so a kernel
/// tap is one multiply-add over a run of pixels with the channel weight held
/// in a register. Each output row is computed on its own: those in `rows`,
/// into `y`, which holds them alone.
#[allow(clippy::too_many_arguments)]
fn depthwise(
    op: &Op,
    x: &[f32],
    y: &mut [f32],
    w: &[f32],
    bias: Option<&[f32]>,
    tail: Tail,
    finished: &mut dyn FnMut(&[f32]),
    rows: std::ops::Range<usize>,
) {
    let c = op.oc;
    let windowed = c.is_multiple_of(8) && kernels::simd_tier() >= 2;
    let (clip, rest) = tail.fold_clip();
    for oy in rows.clone() {
        let row_start = (oy - rows.start) * op.ow * c;
        let row = row_start..row_start + op.ow * c;
        if windowed {
            depthwise_row(op, x, &mut y[row_start..], w, bias, oy, clip);
            rest.apply(&mut y[row.clone()]);
            finished(&y[row]);
            continue;
        }
        let mut interior = (0, 0);
        let top = (oy * op.sh) as isize - op.pt;
        if top >= 0 && top + op.kh as isize <= op.ih as isize && kernels::simd_tier() >= 1 {
            let (a, b) = live_columns(op, 0);
            let (d, e) = live_columns(op, op.kw - 1);
            let (first, end) = (a.max(d), b.min(e));
            if first < end {
                let ix = ((first * op.sw) as isize - op.pl) as usize;
                let src = (top as usize * op.iw + ix) * c;
                let (dst, src) = (&mut y[row_start + first * c..], &x[src..]);
                let (count, row, step) = (end - first, op.iw * c, op.sw * c);
                let done = if (op.kh, op.kw) == (3, 3) {
                    kernels::depthwise_3x3_interior(dst, src, w, bias, count, c, row, step, None)
                } else {
                    kernels::depthwise_window(
                        dst,
                        src,
                        w,
                        bias,
                        count,
                        c,
                        row,
                        step,
                        op.kw,
                        (0, op.kh),
                        (0, op.kw),
                        None,
                    )
                };
                if done {
                    interior = (first, end);
                }
            }
        }
        // The interior kernel initializes its own accumulators. Seed only
        // boundaries; when unsupported, (0,0) seeds the complete row.
        kernels::fill_bias(&mut y[row_start..], bias, interior.0, c);
        kernels::fill_bias(
            &mut y[row_start + interior.1 * c..],
            bias,
            op.ow - interior.1,
            c,
        );
        // Padding and unsupported shapes accumulate tap by tap, in kernel order.
        for ky in 0..op.kh {
            let iy = (oy * op.sh) as isize + ky as isize - op.pt;
            if iy < 0 || iy as usize >= op.ih {
                continue;
            }
            for kx in 0..op.kw {
                let (first, end) = live_columns(op, kx);
                for (first, end) in [(first, end.min(interior.0)), (first.max(interior.1), end)] {
                    if first >= end {
                        continue;
                    }
                    let ix = ((first * op.sw) as isize - (op.pl - kx as isize)) as usize;
                    let tap = &w[(ky * op.kw + kx) * c..][..c];
                    let src = (iy as usize * op.iw + ix) * c;
                    let dst = row_start + first * c;
                    kernels::depthwise_tap(
                        &mut y[dst..],
                        &x[src..],
                        tap,
                        end - first,
                        c,
                        op.sw * c,
                    );
                }
            }
        }
        tail.apply(&mut y[row.clone()]);
        finished(&y[row]);
    }
}

/// One depthwise output row in registers, borders included: the columns with
/// a whole kernel window go to one kernel call, each border column (a window
/// of its own) to another, and every call clamps to `clip` as it stores.
fn depthwise_row(
    op: &Op,
    x: &[f32],
    y: &mut [f32],
    w: &[f32],
    bias: Option<&[f32]>,
    oy: usize,
    clip: Option<(f32, f32)>,
) {
    let (c, top) = (op.oc, (oy * op.sh) as isize - op.pt);
    // Kernel taps [low, high) that land on real pixels of a `len`-long axis
    // when the window starts at `first`.
    let live = |first: isize, len: usize, k: usize| {
        let low = (-first).clamp(0, k as isize) as usize;
        (
            low,
            ((len as isize - first).clamp(0, k as isize) as usize).max(low),
        )
    };
    let rows = live(top, op.ih, op.kh);
    let (row, step) = (op.iw * c, op.sw * c);
    let run = |y: &mut [f32], ox: usize, count: usize| {
        let left = (ox * op.sw) as isize - op.pl;
        let cols = live(left, op.iw, op.kw);
        let dst = &mut y[ox * c..];
        let src = if rows.0 < rows.1 && cols.0 < cols.1 {
            let iy = (top + rows.0 as isize) as usize;
            &x[(iy * op.iw + (left + cols.0 as isize) as usize) * c..]
        } else {
            &x[..0]
        };
        let whole = rows == (0, op.kh) && cols == (0, op.kw);
        let done = (whole && (op.kh, op.kw, op.sw) == (3, 3, 1))
            && kernels::depthwise_3x3_interior(dst, src, w, bias, count, c, row, step, clip);
        if !done {
            let kw = op.kw;
            let done = kernels::depthwise_window(
                dst, src, w, bias, count, c, row, step, kw, rows, cols, clip,
            );
            assert!(done, "the windowed kernel covers c % 8 == 0 on AVX tiers");
        }
    };
    // Columns where the first and the last kernel column are both live.
    let (a, b) = live_columns(op, 0);
    let (d, e) = live_columns(op, op.kw - 1);
    let (first, end) = if a.max(d) < b.min(e) {
        (a.max(d), b.min(e))
    } else {
        (op.ow, op.ow)
    };
    for ox in (0..first).chain(end..op.ow) {
        run(y, ox, 1);
    }
    if first < end {
        run(y, first, end - first);
    }
}

/// Sliding-window maximum over each channel independently; windows are cut
/// short at the bottom and right edges.
fn maxpool(op: &Op, x: &[f32], y: &mut [f32]) {
    let c = op.oc;
    let (row, in_row) = (op.ow * c, op.iw * c);
    if kernels::streamed(&y[..op.oh * row], c) {
        for (oy, out) in y[..op.oh * row].chunks_exact_mut(row).enumerate() {
            let rows = oy * op.sh..(oy * op.sh + op.kh).min(op.ih);
            let x = &x[rows.start * in_row..rows.end * in_row];
            kernels::max_row_streamed(out, x, op.iw, op.kw, op.sw, c);
        }
        return;
    }
    for oy in 0..op.oh {
        for ox in 0..op.ow {
            let dst = (oy * op.ow + ox) * c;
            let mut seeded = false;
            for ky in 0..op.kh {
                let iy = oy * op.sh + ky;
                if iy >= op.ih {
                    continue;
                }
                for kx in 0..op.kw {
                    let ix = ox * op.sw + kx;
                    if ix >= op.iw {
                        continue;
                    }
                    let src = &x[(iy * op.iw + ix) * c..][..c];
                    if seeded {
                        kernels::max_into(&mut y[dst..dst + c], src, c);
                    } else {
                        y[dst..dst + c].copy_from_slice(src);
                        seeded = true;
                    }
                }
            }
        }
    }
}

/// Whether operation `i`'s weights share floats with any other block. Such a
/// region is never repacked, even if the exporter allowed overlapping blocks.
fn weights_shared(plan: &Plan, i: usize) -> bool {
    let (start, end) = (plan.ops[i].w.0, plan.ops[i].w.0 + plan.ops[i].w.1);
    let overlaps = |b: Block| b.1 > 0 && b.0 < end && start < b.0 + b.1;
    plan.ops
        .iter()
        .enumerate()
        .any(|(j, other)| (i != j && overlaps(other.w)) || overlaps(other.b))
}

/// Reorder the weights of eligible pointwise layers into 16-column panels
/// (see `kernels::pointwise_with`). A permutation only, on AVX tiers, where
/// every kernel that will read them understands panels.
fn pack_panels(plan: &Plan, values: &mut [f32]) -> Vec<bool> {
    let mut packed = vec![false; plan.ops.len()];
    if kernels::simd_tier() < 2 {
        return packed;
    }
    for (i, op) in plan.ops.iter().enumerate() {
        // A transposed convolution is one pointwise layer per input row.
        let co = match op.kind {
            Kind::Conv if op.is_pointwise() => op.oc,
            Kind::Convtranspose => op.kh * op.kw * op.oc,
            _ => continue,
        };
        if !co.is_multiple_of(16) || weights_shared(plan, i) {
            continue;
        }
        let (start, ci) = (op.w.0, op.ic);
        let mut panels = Vec::with_capacity(op.w.1);
        for j in (0..co).step_by(16) {
            for row in 0..ci {
                panels.extend_from_slice(&values[start + row * co + j..][..16]);
            }
        }
        values[start..start + op.w.1].copy_from_slice(&panels);
        packed[i] = true;
    }
    packed
}

/// [`maxpool`] writing over its input, at stride 1 with the input's size:
/// every window starts at its own output pixel and reaches only down and
/// right, so going forwards each pixel is written after the last window that
/// reads it. The taps are taken in the order [`maxpool`] takes them.
fn maxpool_in_place(op: &Op, y: &mut [f32]) {
    let (c, row) = (op.oc, op.iw * op.oc);
    for oy in 0..op.oh {
        let rows = op.kh.min(op.ih - oy);
        let window = &mut y[oy * row..(oy + rows) * row];
        if kernels::max_row_in_place(window, op.iw, op.kw, c) {
            continue;
        }
        for ox in 0..op.ow {
            let dst = ox * c;
            for ky in 0..rows {
                for kx in 0..op.kw.min(op.iw - ox) {
                    if (ky, kx) == (0, 0) {
                        continue;
                    }
                    let (out, rest) = window.split_at_mut(dst + c);
                    let src = (ky * op.iw + ox + kx) * c - dst - c;
                    kernels::max_into(&mut out[dst..], &rest[src..], c);
                }
            }
        }
    }
}

/// Whether the first `n` floats of `slot`, left unwritten before operation
/// `from`, can never be observed. Every write starts at element 0, so what
/// is still stale is `clean..n`; a read of `k` floats sees it when `k > clean`.
fn dead_after(plan: &Plan, slot: usize, from: usize, n: usize) -> bool {
    let mut clean = 0;
    for op in &plan.ops[from..] {
        if clean >= n {
            return true;
        }
        // Floats each operation reads from its first and second input.
        let (read, read2) = match op.kind {
            Kind::Conv => (op.in_len(), op.ih * op.iw * op.n2),
            Kind::Dwconv
            | Kind::Maxpool
            | Kind::Avgpool
            | Kind::Layernorm
            | Kind::Mean
            | Kind::Attention
            | Kind::Ctchead
            | Kind::Ctcargmax
            | Kind::Upsample
            | Kind::Convtranspose => (op.in_len(), 0),
            Kind::Upsampleadd => (op.in_len(), op.out_len()),
            Kind::Concatc => (op.in_len(), op.oh * op.ow * op.n2),
            Kind::Add => (op.in_len(), op.out_len()),
            Kind::Channelmul => (op.in_len(), op.oc),
        };
        if (op.x == slot && read > clean) || (op.x2 == slot && read2 > clean) {
            return false;
        }
        if op.y == slot {
            clean = clean.max(op.out_len());
        }
    }
    clean >= n
        || !plan
            .outputs
            .iter()
            .any(|out| out.slot == slot && out.n > clean)
}

/// Pixels per run of an expanded pair: the expansion's output for them stays
/// in the second-level cache until the next convolution reads it.
const EXPAND_RUN: usize = 96;

fn expand_run(op: &Op) -> usize {
    EXPAND_RUN.min(rows_per_group(op) * op.ow)
}

/// The step a run takes at operation `i`. Each kind of step needs its own
/// kinds of operation at `i` and after it, so at most one applies.
fn next_step(plan: &Plan, i: usize) -> Step {
    let ops = &plan.ops;
    let alone = Step::Op {
        i,
        mean: can_sum_mean(plan, i),
    };
    let chained = |k: &usize| matches!(ops[*k].kind, Kind::Concatc | Kind::Upsample);
    match ops[i].kind {
        Kind::Concatc | Kind::Upsample => {
            let Some(conv) = (i..ops.len()).find(|k| !chained(k)) else {
                return alone;
            };
            match chained_inputs(plan, conv) {
                Some((inputs, start)) if start == i => Step::Chained {
                    start,
                    conv,
                    inputs,
                    mean: can_sum_mean(plan, conv),
                },
                _ => alone,
            }
        }
        Kind::Dwconv => {
            let pair = row_pair_from(plan, i + 1);
            if !can_feed(plan, i, pair.as_ref()) {
                alone
            } else if let Some(pair) = pair {
                Step::Pair(Pair {
                    producer: Some(i),
                    ..pair
                })
            } else {
                Step::Fed {
                    dw: i,
                    mean: can_sum_mean(plan, i + 1),
                }
            }
        }
        Kind::Channelmul => match row_pair_from(plan, i + 1) {
            Some(pair) if pair.expand.is_some() && can_gate_expand(plan, i) => Step::Pair(Pair {
                gate: Some(i),
                ..pair
            }),
            _ if can_gate_add(plan, i) => Step::GateAdd(i),
            _ => alone,
        },
        Kind::Convtranspose if can_chain_transposed(plan, i) => Step::Transposed(i),
        _ => row_pair_from(plan, i).map_or(alone, Step::Pair),
    }
}

/// The row pair from operation `i`, if one starts there: an expansion and
/// the pair after it, or a pair.
fn row_pair_from(plan: &Plan, i: usize) -> Option<Pair> {
    let ops = &plan.ops;
    let expand = can_expand(plan, i);
    let conv = i + usize::from(expand);
    if conv + 1 >= ops.len() || !can_pair(&ops[conv], &ops[conv + 1]) {
        return None;
    }
    let out = &ops[conv];
    Some(Pair {
        producer: None,
        gate: None,
        expand: expand.then_some(i),
        conv,
        private: dead_after(plan, out.y, conv + 2, out.out_len()),
    })
}

/// Whether the pointwise convolution `i` can run inside the row-local pair
/// after it, a run of pixels at a time: the pair's convolution reads its
/// output, of the same size, and nothing after the pair reads it. Going
/// forwards, a group's add or convolution output must not overwrite input
/// of a later group, so neither may write more channels into this
/// convolution's input buffer than it reads from there.
fn can_expand(plan: &Plan, i: usize) -> bool {
    let ops = &plan.ops;
    if i + 2 >= ops.len() {
        return false;
    }
    let (e, conv, add) = (&ops[i], &ops[i + 1], &ops[i + 2]);
    let overwrites = |op: &Op| op.y == e.x && op.oc > e.ic;
    e.is_pointwise()
        && conv.x == e.y
        && (conv.ih, conv.iw) == (e.oh, e.ow)
        && can_pair(conv, add)
        && !overwrites(conv)
        && !overwrites(add)
        && dead_after(plan, e.y, i + 2, e.out_len())
}

/// Up to `kernels::SOURCES` buffers a windowed convolution reads side by
/// side, each `(slot, channels, shift)`; none when `count` is 0.
#[derive(Clone, Copy, Default)]
struct Inputs {
    slots: [(usize, usize, u32); kernels::SOURCES],
    count: usize,
}

/// The buffers the windowed convolution `i` can read in place of what the
/// unbroken run of upsamplings (by powers of two) and channel
/// concatenations just before it builds, and where that chain starts.
///
/// Every layer of the chain must feed the convolution, and its output must
/// go unread afterwards. The chain layers being the only ones between, a
/// buffer the chain reads still holds at the convolution what it held when
/// the chain would have read it, the chain's own writes being skipped. The
/// convolution must not write any buffer it now reads: `Plan::check` proved
/// that only of the buffers it read through the chain.
fn chained_inputs(plan: &Plan, i: usize) -> Option<(Inputs, usize)> {
    let ops = &plan.ops;
    let op = &ops[i];
    let chained = |k: usize| matches!(ops[k].kind, Kind::Concatc | Kind::Upsample);
    let start = (0..i).rev().take_while(|&k| chained(k)).last()?;
    if !(op.kind == Kind::Conv && reads_windows(op)) || i - start > 64 {
        return None;
    }
    let mut inputs = Inputs::default();
    // Bit `k - start` set once chain layer `k` is found feeding it.
    let mut used = 0u64;
    // `slot` as layer `before` reads it, `shift` the upsampling so far.
    fn resolve(
        plan: &Plan,
        slot: usize,
        before: usize,
        shift: u32,
        start: usize,
        inputs: &mut Inputs,
        used: &mut u64,
    ) -> Option<()> {
        let writer = (0..before).rev().find(|&k| plan.ops[k].y == slot);
        match writer {
            Some(k) if k >= start => {
                let src = &plan.ops[k];
                *used |= 1 << (k - start);
                if src.kind == Kind::Concatc {
                    resolve(plan, src.x, k, shift, start, inputs, used)?;
                    return resolve(plan, src.x2, k, shift, start, inputs, used);
                }
                let factor = src.sh;
                if src.sw != factor || !factor.is_power_of_two() {
                    return None;
                }
                let shift = shift + factor.trailing_zeros();
                (shift < 16).then_some(())?;
                resolve(plan, src.x, k, shift, start, inputs, used)
            }
            writer => {
                let c = writer.map_or(plan.input.c, |k| plan.ops[k].oc);
                *inputs.slots.get_mut(inputs.count)? = (slot, c, shift);
                inputs.count += 1;
                Some(())
            }
        }
    }
    resolve(plan, op.x, i, 0, start, &mut inputs, &mut used)?;
    if op.n2 > 0 {
        resolve(plan, op.x2, i, 0, start, &mut inputs, &mut used)?;
    }
    let unread = |k: usize| {
        let (slot, n) = (ops[k].y, ops[k].out_len());
        (op.y == slot && op.out_len() >= n) || dead_after(plan, slot, i + 1, n)
    };
    let all = u64::MAX >> (64 - (i - start));
    let apart = inputs.slots[..inputs.count]
        .iter()
        .all(|&(slot, ..)| slot != op.y);
    (used == all && apart && (start..i).all(unread)).then_some((inputs, start))
}

/// Whether channel multiply `i` can run inside the expanded pair after it, a
/// run of pixels at a time: the expansion reads its product, as the pair's
/// add may, and nothing after the pair does. As in [`can_expand`], no layer
/// of the pair may write more channels into the multiply's input buffer
/// than it reads.
fn can_gate_expand(plan: &Plan, i: usize) -> bool {
    let ops = &plan.ops;
    if i + 3 >= ops.len() {
        return false;
    }
    let (gate, e) = (&ops[i], &ops[i + 1]);
    // The expansion writes only `mid`. A layer of the pair writing the
    // product's buffer would leave there what the pair's add, reading the
    // product from `scratch`, should read.
    let overwrites = |op: &Op| op.y == gate.y || (op.y == gate.x && op.oc > gate.oc);
    gate.kind == Kind::Channelmul
        && e.x == gate.y
        && (e.ih, e.iw) == (gate.oh, gate.ow)
        && !ops[i + 2..i + 4].iter().any(overwrites)
        && dead_after(plan, gate.y, i + 4, gate.out_len())
}

/// Whether the channel mean after layer `i` can be summed as `i` writes:
/// the mean reads `i`'s whole output, a convolution's or a depthwise one's
/// that `step` runs alone, of at most `MEAN_CHANNELS` channels.
fn can_sum_mean(plan: &Plan, i: usize) -> bool {
    let ops = &plan.ops;
    if i + 1 >= ops.len() {
        return false;
    }
    let (op, mean) = (&ops[i], &ops[i + 1]);
    matches!(op.kind, Kind::Conv | Kind::Dwconv)
        && mean.kind == Kind::Mean
        && mean.x == op.y
        && op.oc <= layers::MEAN_CHANNELS
}

/// Whether depthwise convolution `i` can feed the layers after it a group
/// of rows at a time: the row pair `pair` that starts after it, else a
/// pointwise convolution alone, which alone read its output, of the same
/// size. Going forwards, no layer of the group may write into the pool the
/// depthwise input, which later groups read, or its output's buffer, read
/// from `fed` meanwhile; an expansion writes only `mid`, a private pair's
/// convolution only `rows`.
fn can_feed(plan: &Plan, i: usize, pair: Option<&Pair>) -> bool {
    let ops = &plan.ops;
    if i + 1 >= ops.len() || ops[i].kind != Kind::Dwconv {
        return false;
    }
    let (dw, next) = (&ops[i], &ops[i + 1]);
    // The first layer of the group that writes the pool, and the group's end.
    let (writers, end) = match pair {
        Some(pair) => (pair.conv + usize::from(pair.private), pair.conv + 2),
        None => (i + 1, i + 2),
    };
    let overwrites = |k: usize| ops[k].y == dw.x || ops[k].y == dw.y;
    next.is_pointwise()
        && next.x == dw.y
        && (next.ih, next.iw) == (dw.oh, dw.ow)
        && !(writers..end).any(overwrites)
        && dead_after(plan, dw.y, end, dw.out_len())
}

/// Whether transposed convolution `i` can hand its output rows straight to
/// the transposed convolution after it, which alone reads them and writes a
/// buffer the first does not read.
fn can_chain_transposed(plan: &Plan, i: usize) -> bool {
    let ops = &plan.ops;
    if i + 1 >= ops.len() {
        return false;
    }
    let (first, second) = (&ops[i], &ops[i + 1]);
    first.kind == Kind::Convtranspose
        && second.kind == Kind::Convtranspose
        && second.x == first.y
        && second.y != first.x
        && dead_after(plan, first.y, i + 2, first.out_len())
}

/// Whether channel multiply `i` and the add after it can run as one pass:
/// the add sums the multiply's input and product, and nothing after it
/// reads the product.
fn can_gate_add(plan: &Plan, i: usize) -> bool {
    let ops = &plan.ops;
    if i + 1 >= ops.len() {
        return false;
    }
    let (gate, add) = (&ops[i], &ops[i + 1]);
    gate.kind == Kind::Channelmul
        && add.kind == Kind::Add
        && (add.x == gate.x && add.x2 == gate.y || add.x == gate.y && add.x2 == gate.x)
        && (add.oh, add.ow, add.oc) == (gate.oh, gate.ow, gate.oc)
        && dead_after(plan, gate.y, i + 2, gate.out_len())
}

/// Whether `conv` and the `add` after it can run a group of rows at a time,
/// with every tensor ending as the two operations in sequence leave it. Shape
/// equality is essential to both row addressing and alias order.
fn can_pair(conv: &Op, add: &Op) -> bool {
    conv.is_pointwise()
        && conv.ih == conv.oh
        && conv.iw == conv.ow
        && add.kind == Kind::Add
        && (add.x == conv.y || add.x2 == conv.y)
        && (add.ih, add.iw, add.ic) == (conv.oh, conv.ow, conv.oc)
        && (add.oh, add.ow, add.oc) == (conv.oh, conv.ow, conv.oc)
}

/// [`lay_out`]'s working lists, one entry per plan buffer.
struct Placement {
    /// The first and last time a step names the buffer: 0 is before the
    /// run, step `k` is `k + 1`, and the outputs are read after the last.
    life: Vec<Option<(usize, usize)>>,
    /// Some step writes the buffer, so it needs memory.
    written: Vec<bool>,
    /// The buffer whose memory this one takes over, itself if none.
    with: Vec<usize>,
    /// Where the buffer's memory starts in the arena, in floats.
    offset: Vec<Option<usize>>,
    /// Floats from `offset` taken while the buffer is live, its group's.
    size: Vec<usize>,
    /// Buffers heading a group, to place largest first.
    order: Vec<usize>,
    /// Ranges of the arena taken by groups placed and live at the same
    /// time as the one being placed.
    taken: Vec<(usize, usize)>,
}

impl Placement {
    fn new(buffers: usize) -> Self {
        Self {
            life: vec![None; buffers],
            written: vec![false; buffers],
            with: (0..buffers).collect(),
            offset: vec![None; buffers],
            size: vec![0; buffers],
            order: Vec::with_capacity(buffers),
            taken: Vec::with_capacity(buffers),
        }
    }

    fn head(&self, mut slot: usize) -> usize {
        while self.with[slot] != slot {
            slot = self.with[slot];
        }
        slot
    }
}

/// Lay out the memory of a run of `steps`; returns the arena's length in
/// floats and leaves each written buffer's offset in `place.offset`.
///
/// A buffer needs memory only if some step writes it, from the first step
/// that names it to the last; a step is one unit of time, so every buffer a
/// step names has memory of its own while it runs, and a fused step may read
/// and write its buffers in any order. The exceptions are the outputs that
/// [`Step::over`] lets a step write over an input it names for the last
/// time; they share that input's memory. Groups are placed largest first, each
/// at the lowest offset clear of every group placed and live at the same
/// time, and every offset is a multiple of 16 floats, a cache line.
fn lay_out(plan: &Plan, steps: &[Step], place: &mut Placement) -> usize {
    let n = plan.pool.len();
    let end = steps.len() + 1;
    place.life.clear();
    place.life.resize(n, None);
    place.written.clear();
    place.written.resize(n, false);
    place.offset.clear();
    place.offset.resize(n, None);
    place.with.clear();
    place.with.extend(0..n);
    let life = &mut place.life;
    let mut name = |slot: usize, t: usize| {
        life[slot] = Some(life[slot].map_or((t, t), |(a, b)| (a.min(t), b.max(t))));
    };
    name(plan.input.slot, 0);
    for (k, step) in steps.iter().enumerate() {
        for op in &plan.ops[step.ops()] {
            name(op.x, k + 1);
            if let Some(x2) = op.second() {
                name(x2, k + 1);
            }
            name(op.y, k + 1);
        }
    }
    for out in &plan.outputs {
        name(out.slot, end);
    }
    place.written[plan.input.slot] = true;
    for step in steps {
        for slot in step.writes(&plan.ops).into_iter().flatten() {
            place.written[slot] = true;
        }
    }
    for (k, step) in steps.iter().enumerate() {
        let Some((y, inputs)) = step.over(&plan.ops) else {
            continue;
        };
        let t = Some(k + 1);
        let born = place.life[y].map(|(a, _)| a) == t;
        let dies = |x: usize| place.written[x] && place.life[x].map(|(_, b)| b) == t;
        let over = inputs.into_iter().flatten().find(|&x| born && dies(x));
        if let Some(x) = over {
            place.with[y] = place.head(x);
        }
    }
    // A group's life and size gather at its head.
    place.size.clear();
    place.size.resize(n, 0);
    for slot in (0..n).filter(|&s| place.written[s]) {
        let head = place.head(slot);
        let (a, b) = place.life[slot].expect("a written buffer is named");
        place.life[head] = place.life[head].map(|(c, d)| (a.min(c), b.max(d)));
        place.size[head] = place.size[head].max(plan.pool[slot].next_multiple_of(16));
    }
    place.order.clear();
    for slot in 0..n {
        if place.written[slot] && place.head(slot) == slot {
            place.order.push(slot);
        }
    }
    let (life, size) = (&place.life, &place.size);
    place
        .order
        .sort_unstable_by_key(|&s| (std::cmp::Reverse(size[s]), life[s], s));
    let mut arena = 0;
    for k in 0..place.order.len() {
        let slot = place.order[k];
        let (a, b) = place.life[slot].expect("a written buffer is named");
        place.taken.clear();
        for &other in &place.order[..k] {
            let (c, d) = place.life[other].expect("a written buffer is named");
            if c <= b && a <= d {
                let at = place.offset[other].expect("placed before");
                place.taken.push((at, at + place.size[other]));
            }
        }
        place.taken.sort_unstable();
        let mut at = 0;
        for &(start, stop) in &place.taken {
            if at + place.size[slot] <= start {
                break;
            }
            at = at.max(stop);
        }
        place.offset[slot] = Some(at);
        arena = arena.max(at + place.size[slot]);
    }
    for slot in (0..n).filter(|&s| place.written[s]) {
        place.offset[slot] = place.offset[place.head(slot)];
    }
    arena
}

#[cfg(test)]
mod tests {
    use super::plan::{encode, op};
    use super::*;

    /// A model of `ops` over an input `(slot, h, w, c)`, with `weights`
    /// floats of a smooth ramp.
    fn model(
        input: (usize, usize, usize, usize),
        buffers: usize,
        outputs: &[usize],
        ops: &[Op],
        weights: usize,
    ) -> Model {
        let values: Vec<f32> = (0..weights)
            .map(|i| (i as f32 * 0.37).sin() * 0.6)
            .collect();
        let bytes = encode(input, buffers, outputs, ops, &values);
        let (plan, values) = Plan::parse(&bytes).expect("the test plan is valid");
        Model::new(plan, values).expect("the test plan fits")
    }

    fn window(op: &mut Op, k: (usize, usize), s: usize, before: isize, after: isize) {
        (op.kh, op.kw, op.sh, op.sw) = (k.0, k.1, s, s);
        (op.pt, op.pl, op.pb, op.pr) = (before, before, after, after);
    }

    /// A convolution or transposed convolution with weights and bias at
    /// `at` in the weights; returns the floats it takes.
    fn weigh(op: &mut Op, at: usize) -> usize {
        let taps = op.kh * op.kw;
        let (w, b) = match op.kind {
            Kind::Dwconv => (taps * op.oc, op.oc),
            Kind::Convtranspose => (taps * op.ic * op.oc, taps * op.oc),
            _ => (taps * (op.ic + op.n2) * op.oc, op.oc),
        };
        (op.w, op.b) = (Block(at, w), Block(at + w, b));
        w + b
    }

    fn fill(model: &mut Model) {
        for (i, v) in model.input_mut().iter_mut().enumerate() {
            *v = (i as f32 * 0.29).cos() * 2.0;
        }
    }

    fn results(model: &Model) -> Vec<Vec<u32>> {
        (0..model.output_count())
            .map(|o| model.output(o).iter().map(|v| v.to_bits()).collect())
            .collect()
    }

    /// The step that starts at operation `i`.
    fn step_at(model: &Model, i: usize) -> Step {
        *model
            .steps
            .iter()
            .find(|step| step.ops().start == i)
            .expect("a step starts there")
    }

    /// Run `model` an operation at a time, its memory laid out for that,
    /// then as scheduled, from the same input; asserts both leave every
    /// output bit for bit the same and returns them.
    fn run_both_ways(model: &mut Model, label: &str) -> Vec<Vec<u32>> {
        let steps = std::mem::take(&mut model.steps);
        model.steps = (0..steps.last().map_or(0, |s| s.ops().end))
            .map(|i| Step::Op { i, mean: false })
            .collect();
        model.arrange().unwrap();
        fill(model);
        model.run();
        let one_by_one = results(model);
        model.steps = steps;
        model.arrange().unwrap();
        fill(model);
        model.run();
        assert_eq!(results(model), one_by_one, "{label}");
        one_by_one
    }

    /// Slot 0 is a 20x7 image of `ci` channels, three groups of rows; a 1x1
    /// convolution makes the residual in slot 2, another the pair's
    /// convolution in slot 1, and the add (its inputs `swapped` or not)
    /// writes `dst`. `outputs` are slots.
    fn pair_model(ci: usize, co: usize, dst: usize, swapped: bool, outputs: &[usize]) -> Model {
        let mut residual = op(Kind::Conv, 0, 2, ci, co);
        let mut conv = op(Kind::Conv, 0, 1, ci, co);
        window(&mut residual, (1, 1), 1, 0, 0);
        window(&mut conv, (1, 1), 1, 0, 0);
        let n = weigh(&mut residual, 0);
        let n = n + weigh(&mut conv, n);
        (conv.fused, residual.fused) = (Fused::Gelu, Fused::Relu);
        let mut add = op(Kind::Add, 1, dst, co, co);
        add.x2 = 2;
        if swapped {
            (add.x, add.x2) = (2, 1);
        }
        add.fused = Fused::Swish;
        model((0, 20, 7, ci), 4, outputs, &[residual, conv, add], n)
    }

    #[test]
    fn row_pairs_match_sequence() {
        for (ci, co) in [(3, 16), (8, 16), (16, 8), (16, 16), (24, 28), (32, 64)] {
            for dst in [0, 3] {
                for swapped in [false, true] {
                    for outputs in [&[dst, 1, 2][..], &[dst]] {
                        let label = format!("{ci}->{co} dst={dst} swapped={swapped} {outputs:?}");
                        let mut model = pair_model(ci, co, dst, swapped, outputs);
                        let Step::Pair(pair) = step_at(&model, 1) else {
                            panic!("{label}: no pair");
                        };
                        // Only an unobserved convolution output stays private.
                        assert_eq!(pair.private, outputs.len() == 1, "{label}");
                        let paired = run_both_ways(&mut model, &label);
                        let mut times = Vec::new();
                        // SAFETY: the test holds the only reference to the model.
                        unsafe { model.pool[1].get_mut() }.fill(f32::NAN);
                        fill(&mut model);
                        model.run_timed(&mut times, || 0.0);
                        assert_eq!(results(&model), paired, "{label}");
                        let steps: Vec<_> = times.into_iter().map(|(ops, _)| ops).collect();
                        assert_eq!(steps, [0..1, 1..3], "{label}");
                        // Never written, a private output has no memory.
                        assert_eq!(model.pool[1].len == 0, pair.private, "{label}");
                    }
                }
            }
        }
    }

    #[test]
    fn expanded_pairs_match_sequence() {
        // Widths of one run, several runs, and runs ending off a multiple of
        // four; one row per group and several.
        for (h, w) in [(2, 7), (3, 103), (2, 200), (5, 20)] {
            for (ci, cm) in [(8, 16), (16, 48), (12, 20)] {
                for observed in [false, true] {
                    let mut e = op(Kind::Conv, 0, 1, ci, cm);
                    let mut conv = op(Kind::Conv, 1, 2, cm, ci);
                    window(&mut e, (1, 1), 1, 0, 0);
                    window(&mut conv, (1, 1), 1, 0, 0);
                    e.fused = Fused::Gelu;
                    let n = weigh(&mut e, 0);
                    let n = n + weigh(&mut conv, n);
                    let mut add = op(Kind::Add, 2, 3, ci, ci);
                    add.x2 = 0;
                    // An output read afterwards keeps the expansion out.
                    let outputs: &[usize] = if observed { &[3, 1] } else { &[3] };
                    let mut model = model((0, h, w, ci), 4, outputs, &[e, conv, add], n);
                    let label = format!("{h}x{w} {ci}->{cm} observed={observed}");
                    let first = usize::from(observed);
                    let expanded = matches!(
                        step_at(&model, first),
                        Step::Pair(Pair { expand, conv: 1, .. }) if expand.is_some() != observed
                    );
                    assert!(expanded, "{label}");
                    run_both_ways(&mut model, &label);
                }
            }
        }
    }

    #[test]
    fn fed_layers_match_written_out_depthwise() {
        let c = 16;
        // What the depthwise output feeds: a pointwise convolution with its
        // mean, a row pair, an expanded pair whose add reads the depthwise
        // output, and one whose private narrowing names the depthwise
        // input's buffer, as the detector's do; observed, that narrowing
        // would overwrite input of later groups, which keeps the depthwise
        // convolution apart. At widths of one group and of several runs.
        for (h, w) in [(3, 7), (2, 103), (5, 20)] {
            for consumer in 0..5 {
                for k in [3, 7] {
                    let mut at = 0;
                    let mut dw = op(Kind::Dwconv, 0, 1, c, c);
                    window(&mut dw, (k, k), 1, k as isize / 2, k as isize / 2);
                    dw.fused = Fused::Relu;
                    at += weigh(&mut dw, at);
                    let mut pointwise = |x, y, ic, oc, fused| {
                        let mut op = op(Kind::Conv, x, y, ic, oc);
                        window(&mut op, (1, 1), 1, 0, 0);
                        op.fused = fused;
                        at += weigh(&mut op, at);
                        op
                    };
                    let mut ops = vec![dw];
                    let output = match consumer {
                        0 => {
                            ops.push(pointwise(1, 2, c, 8, Fused::Gelu));
                            ops.push(op(Kind::Mean, 2, 3, 8, 8));
                            3
                        }
                        1 => {
                            ops.push(pointwise(1, 2, c, c, Fused::None));
                            let mut add = op(Kind::Add, 2, 3, c, c);
                            add.x2 = 1;
                            ops.push(add);
                            3
                        }
                        2 => {
                            ops.push(pointwise(1, 2, c, 32, Fused::Gelu));
                            ops.push(pointwise(2, 3, 32, c, Fused::None));
                            let mut add = op(Kind::Add, 3, 4, c, c);
                            add.x2 = 1;
                            ops.push(add);
                            4
                        }
                        _ => {
                            ops.push(pointwise(1, 2, c, 32, Fused::Gelu));
                            ops.push(pointwise(2, 0, 32, c, Fused::None));
                            let mut add = op(Kind::Add, 1, 2, c, c);
                            add.x2 = 0;
                            ops.push(add);
                            2
                        }
                    };
                    let buffers = ops.len() + 1;
                    let outputs: &[usize] = if consumer == 4 {
                        &[output, 0]
                    } else {
                        &[output]
                    };
                    let mut model = model((0, h, w, c), buffers, outputs, &ops, at);
                    let label = format!("{h}x{w} consumer {consumer} {k}x{k}");
                    let fed = matches!(
                        model.steps[..],
                        [Step::Fed { dw: 0, .. }
                            | Step::Pair(Pair {
                                producer: Some(0),
                                ..
                            })]
                    );
                    assert_eq!(fed, consumer != 4, "{label}");
                    run_both_ways(&mut model, &label);
                }
            }
        }
    }

    #[test]
    fn chained_transposed_convs_match_sequence() {
        for (h, w) in [(1, 1), (3, 5), (4, 9)] {
            for observed in [false, true] {
                // The detector's last two layers: 2x2 windows, a ReLU then
                // the sigmoid, after a 1x1 convolution of the image.
                let mut at = 0;
                let mut conv = op(Kind::Conv, 0, 1, 3, 8);
                window(&mut conv, (1, 1), 1, 0, 0);
                at += weigh(&mut conv, at);
                let mut transposed = |x, y, ic, oc, fused| {
                    let mut op = op(Kind::Convtranspose, x, y, ic, oc);
                    (op.kh, op.kw, op.sh, op.sw, op.fused) = (2, 2, 2, 2, fused);
                    at += weigh(&mut op, at);
                    op
                };
                let first = transposed(1, 2, 8, 16, Fused::Relu);
                let second = transposed(2, 3, 16, 1, Fused::Sigmoid);
                let outputs: &[usize] = if observed { &[3, 2] } else { &[3] };
                let mut model = model((0, h, w, 3), 4, outputs, &[conv, first, second], at);
                let label = format!("{h}x{w} observed={observed}");
                let chained = matches!(step_at(&model, 1), Step::Transposed(1));
                assert_eq!(chained, !observed, "{label}");
                run_both_ways(&mut model, &label);
            }
        }
    }

    #[test]
    fn gated_expansions_match_sequence() {
        for (h, w) in [(2, 7), (3, 103), (5, 20)] {
            for (c, cm) in [(8, 16), (16, 48)] {
                for (residual_gated, wide) in [(false, 1), (true, 1), (true, 0)] {
                    // A gate from the image's mean, the gated image widened
                    // by `e` into buffer `wide` (0, the image's, holds
                    // fewer channels), narrowed by `conv` and added to the
                    // gated image or to the image.
                    let mean = op(Kind::Mean, 0, 1, c, c);
                    let mut squeeze = op(Kind::Conv, 1, 2, c, c);
                    window(&mut squeeze, (1, 1), 1, 0, 0);
                    squeeze.fused = Fused::Hardsigmoid;
                    let mut gate = op(Kind::Channelmul, 0, 3, c, c);
                    gate.x2 = 2;
                    let mut e = op(Kind::Conv, 3, wide, c, cm);
                    let mut conv = op(Kind::Conv, wide, 2, cm, c);
                    window(&mut e, (1, 1), 1, 0, 0);
                    window(&mut conv, (1, 1), 1, 0, 0);
                    e.fused = Fused::Gelu;
                    let n = weigh(&mut squeeze, 0);
                    let n = n + weigh(&mut e, n);
                    let n = n + weigh(&mut conv, n);
                    let mut add = op(Kind::Add, 2, 4, c, c);
                    add.x2 = if residual_gated { 3 } else { 0 };
                    let ops = [mean, squeeze, gate, e, conv, add];
                    // Observed, the narrowing writes the gate's buffer.
                    let outputs: &[usize] = if residual_gated { &[4, 2] } else { &[4] };
                    let mut model = model((0, h, w, c), 5, outputs, &ops, n);
                    let label = format!("{h}x{w} {c}->{cm} {residual_gated} {wide}");
                    let gated = matches!(
                        step_at(&model, 2),
                        Step::Pair(Pair {
                            gate: Some(2),
                            expand: Some(3),
                            conv: 4,
                            ..
                        })
                    );
                    assert!(gated, "{label}");
                    run_both_ways(&mut model, &label);
                }
            }
        }
    }

    #[test]
    fn convolutions_read_upsampled_concatenations_in_place() {
        // Maps at 1/4, 1/2 and full size, as the detector's head upsamples
        // and stacks them under its last 3x3 convolution with a fourth map
        // beside: `quarter` and `half` come from strided convolutions of
        // the image, `full` from a 1x1 one.
        let c = 4;
        for (h, w) in [(8, 8), (16, 24)] {
            for observed in [false, true] {
                let mut at = 0;
                let mut layer = |kind, (x, y), (ic, oc), k: usize, s| {
                    let mut op = op(kind, x, y, ic, oc);
                    window(&mut op, (k, k), s, 0, 0);
                    at += weigh(&mut op, at);
                    op
                };
                let half = layer(Kind::Conv, (0, 1), (3, c), 2, 2);
                let quarter = layer(Kind::Conv, (1, 2), (c, c), 2, 2);
                let full = layer(Kind::Conv, (0, 3), (3, c), 1, 1);
                let mut up4 = op(Kind::Upsample, 2, 4, c, c);
                (up4.sh, up4.sw) = (4, 4);
                let mut up2 = op(Kind::Upsample, 1, 2, c, c);
                (up2.sh, up2.sw) = (2, 2);
                let mut stack = op(Kind::Concatc, 4, 5, c, 2 * c);
                (stack.x2, stack.n2) = (2, c);
                let mut conv = op(Kind::Conv, 5, 6, 2 * c, 8);
                window(&mut conv, (3, 3), 1, 1, 1);
                (conv.x2, conv.n2, conv.fused) = (3, c, Fused::Relu);
                at += weigh(&mut conv, at);
                // Written into a buffer it would read in place, the
                // convolution must read the chain's output instead.
                let mut overwriting = conv.clone();
                overwriting.y = 1;
                let mut ops = [half, quarter, full, up4, up2, stack, overwriting];
                let chained = |model: &Model| {
                    matches!(
                        step_at(model, 3),
                        Step::Chained {
                            start: 3,
                            conv: 6,
                            ..
                        }
                    )
                };
                assert!(!chained(&model((0, h, w, 3), 7, &[1], &ops, at)));
                ops[6] = conv;
                let outputs: &[usize] = if observed { &[6, 5] } else { &[6] };
                let mut model = model((0, h, w, 3), 7, outputs, &ops, at);
                let label = format!("{h}x{w} observed={observed}");
                assert_eq!(chained(&model), !observed, "{label}");
                run_both_ways(&mut model, &label);
            }
        }
    }

    #[test]
    fn written_means_match_separate_mean() {
        let c = 8;
        let pointwise = |ic| {
            let mut op = op(Kind::Conv, 0, 1, ic, c);
            window(&mut op, (1, 1), 1, 0, 0);
            op.fused = Fused::Gelu;
            op
        };
        let windowed = |ic| {
            let mut op = op(Kind::Conv, 0, 1, ic, c);
            window(&mut op, (3, 3), 2, 1, 1);
            op.fused = Fused::Hardsigmoid;
            op
        };
        // The means are of the outputs after the activation.
        let depthwise = |ic| {
            let mut op = op(Kind::Dwconv, 0, 1, ic, c);
            window(&mut op, (3, 3), 1, 1, 1);
            op.fused = Fused::Swish;
            op
        };
        let layers = [
            ("pointwise", pointwise(5), 5),
            ("windowed", windowed(5), 5),
            ("RGB stem", windowed(3), 3),
            ("depthwise", depthwise(c), c),
        ];
        for (what, layer, ic) in layers {
            for (h, w) in [(1, 3), (7, 9), (2, 60)] {
                let mut layer = layer.clone();
                let n = weigh(&mut layer, 0);
                let mean = op(Kind::Mean, 1, 2, c, c);
                let mut model = model((0, h, w, ic), 3, &[2, 1], &[layer, mean], n);
                let label = format!("{what} {h}x{w}");
                let summed = matches!(model.steps[..], [Step::Op { i: 0, mean: true }]);
                assert!(summed, "{label}");
                run_both_ways(&mut model, &label);
            }
        }
    }

    #[test]
    fn gate_adds_match_sequence() {
        let c = 12;
        for residual_first in [false, true] {
            for (observed, fused) in [
                (false, Fused::None),
                (true, Fused::None),
                (false, Fused::Relu),
            ] {
                // A gate from the image's mean, then image * gate + image.
                let mean = op(Kind::Mean, 0, 1, c, c);
                let mut squeeze = op(Kind::Conv, 1, 2, c, c);
                window(&mut squeeze, (1, 1), 1, 0, 0);
                squeeze.fused = Fused::Hardsigmoidfifth;
                let n = weigh(&mut squeeze, 0);
                let mut gate = op(Kind::Channelmul, 0, 3, c, c);
                gate.x2 = 2;
                // Into the gate's buffer, as the detector's neck does.
                let mut add = op(Kind::Add, 0, 2, c, c);
                (add.x2, add.fused) = (3, fused);
                if !residual_first {
                    (add.x, add.x2) = (3, 0);
                }
                let outputs: &[usize] = if observed { &[2, 3] } else { &[2] };
                let mut model = model((0, 3, 9, c), 4, outputs, &[mean, squeeze, gate, add], n);
                let label =
                    format!("residual_first={residual_first} observed={observed} {fused:?}");
                let one_pass = matches!(step_at(&model, 2), Step::GateAdd(2));
                assert_eq!(one_pass, !observed, "{label}");
                run_both_ways(&mut model, &label);
            }
        }
    }

    #[test]
    fn in_place_outputs_match_definitions() {
        // Each written over the image, which `fill` sets to `input(i)`.
        let input = |i: usize| (i as f32 * 0.29).cos() * 2.0;
        for (h, w) in [(2, 2), (4, 6), (6, 20)] {
            for c in [5, 8] {
                let pixel = |y: usize, x: usize, k: usize| input((y * w + x) * c + k);
                let label = format!("{h}x{w}x{c}");

                let mut pool = op(Kind::Maxpool, 0, 1, c, c);
                (pool.kh, pool.kw, pool.sh, pool.sw, pool.pb, pool.pr) = (2, 2, 1, 1, 1, 1);
                let mut pooled = model((0, h, w, c), 2, &[1], &[pool], 0);
                assert_eq!(pooled.pool[1].ptr, pooled.pool[0].ptr, "{label}");
                let got = run_both_ways(&mut pooled, &label);
                let want: Vec<u32> = (0..h * w * c)
                    .map(|i| {
                        let (y, x, k) = (i / c / w, i / c % w, i % c);
                        let mut top = pixel(y, x, k);
                        for (dy, dx) in [(0, 1), (1, 0), (1, 1)] {
                            if y + dy < h && x + dx < w {
                                top = top.max(pixel(y + dy, x + dx, k));
                            }
                        }
                        top.to_bits()
                    })
                    .collect();
                assert_eq!(got[0], want, "max pool {label}");

                // The image plus a 2x2 stride-2 convolution of it, upsampled.
                let mut half = op(Kind::Conv, 0, 1, c, c);
                window(&mut half, (2, 2), 2, 0, 0);
                let n = weigh(&mut half, 0);
                let mut up = op(Kind::Upsampleadd, 1, 2, c, c);
                (up.x2, up.sh, up.sw) = (0, 2, 2);
                let mut summed = model((0, h, w, c), 3, &[2, 1], &[half, up], n);
                assert_eq!(summed.pool[2].ptr, summed.pool[0].ptr, "{label}");
                let got = run_both_ways(&mut summed, &label);
                let half = summed.output(1);
                let want: Vec<u32> = (0..h * w * c)
                    .map(|i| {
                        let (y, x, k) = (i / c / w, i / c % w, i % c);
                        (input(i) + half[((y / 2) * (w / 2) + x / 2) * c + k]).to_bits()
                    })
                    .collect();
                assert_eq!(got[0], want, "upsampled sum {label}");

                // The image plus a 3x3 convolution of it, either way round,
                // then a ReLU; a 1x1 one would run paired with the add.
                for swapped in [false, true] {
                    let mut conv = op(Kind::Conv, 0, 1, c, c);
                    window(&mut conv, (3, 3), 1, 1, 1);
                    let n = weigh(&mut conv, 0);
                    let mut add = op(Kind::Add, 0, 2, c, c);
                    (add.x2, add.fused) = (1, Fused::Relu);
                    if swapped {
                        (add.x, add.x2) = (1, 0);
                    }
                    let mut added = model((0, h, w, c), 3, &[2, 1], &[conv, add], n);
                    assert_eq!(added.pool[2].ptr, added.pool[0].ptr, "{label}");
                    let got = run_both_ways(&mut added, &label);
                    let conv = added.output(1);
                    let want: Vec<u32> = (0..h * w * c)
                        .map(|i| (input(i) + conv[i]).clamp(0.0, f32::INFINITY).to_bits())
                        .collect();
                    assert_eq!(got[0], want, "add swapped={swapped} {label}");
                }

                // The image times a gate from its mean, plus the image.
                let mean = op(Kind::Mean, 0, 1, c, c);
                let mut squeeze = op(Kind::Conv, 1, 2, c, c);
                window(&mut squeeze, (1, 1), 1, 0, 0);
                squeeze.fused = Fused::Hardsigmoidfifth;
                let n = weigh(&mut squeeze, 0);
                let mut gate = op(Kind::Channelmul, 0, 3, c, c);
                gate.x2 = 2;
                let mut add = op(Kind::Add, 0, 4, c, c);
                add.x2 = 3;
                let ops = [mean, squeeze, gate, add];
                let mut gated = model((0, h, w, c), 5, &[4, 2], &ops, n);
                assert!(matches!(step_at(&gated, 2), Step::GateAdd(2)), "{label}");
                assert_eq!(gated.pool[4].ptr, gated.pool[0].ptr, "{label}");
                let got = run_both_ways(&mut gated, &label);
                let g = gated.output(1);
                let want: Vec<u32> = (0..h * w * c)
                    .map(|i| (input(i) * g[i % c] + input(i)).to_bits())
                    .collect();
                assert_eq!(got[0], want, "gated sum {label}");
            }
        }
    }

    #[test]
    fn folded_activations_match_applied_after() {
        let (c, co) = (8, 16);
        let pointwise = || {
            let mut op = op(Kind::Conv, 0, 2, c, co);
            window(&mut op, (1, 1), 1, 0, 0);
            op
        };
        let stem = |ic| {
            let mut op = op(Kind::Conv, 0, 2, ic, co);
            window(&mut op, (3, 3), 2, 1, 1);
            op
        };
        let depthwise = |k| {
            let mut op = op(Kind::Dwconv, 0, 2, c, c);
            window(&mut op, k, 1, 3, 3);
            (op.pt, op.pb) = (0, 0);
            op
        };
        let transposed = || {
            let mut op = op(Kind::Convtranspose, 0, 2, c, 4);
            (op.kh, op.kw, op.sh, op.sw) = (2, 2, 2, 2);
            op
        };
        let add = || {
            let mut op = op(Kind::Add, 0, 2, c, c);
            op.x2 = 1;
            op
        };
        let shapes = [
            ("pointwise", pointwise(), c),
            ("gathered 3x3 stride 2", stem(c), c),
            ("RGB stem", stem(3), 3),
            ("depthwise 1x7", depthwise((1, 7)), c),
            ("transposed", transposed(), c),
            ("add", add(), c),
        ];
        // Slot 1 holds a second input for the addition: a 1x1 convolution of
        // the image, written first.
        for (what, shape, ic) in shapes {
            let mut second = op(Kind::Conv, 0, 1, ic, c);
            window(&mut second, (1, 1), 1, 0, 0);
            let mut shape = shape;
            let n = weigh(&mut second, 0);
            let n = n + if shape.kind == Kind::Add {
                0
            } else {
                weigh(&mut shape, n)
            };
            let build = |fused| {
                let mut last = shape.clone();
                last.fused = fused;
                model((0, 6, 9, ic), 3, &[2], &[second.clone(), last], n)
            };
            let mut plain = build(Fused::None);
            fill(&mut plain);
            plain.run();
            for fused in &plan::FUSED[1..] {
                let mut want = plain.output(0).to_vec();
                Tail::of(&Op {
                    fused: *fused,
                    ..shape.clone()
                })
                .apply(&mut want);
                let mut folded = build(*fused);
                fill(&mut folded);
                folded.run();
                let got: Vec<u32> = folded.output(0).iter().map(|v| v.to_bits()).collect();
                let want: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
                assert_eq!(got, want, "{what} with {fused:?}");
            }
        }
    }

    /// A recogniser-shaped model with every width rule: a padded stride-2
    /// stem, a right-padded max pool, a squeeze-and-excite gate from a mean,
    /// a width-halving average pool, attention and the CTC head, stored at an
    /// input `stored` pixels wide.
    fn line_model(stored: usize) -> Model {
        let mut at = 0;
        let mut stem = op(Kind::Conv, 0, 1, 3, 4);
        window(&mut stem, (3, 3), 2, 1, 1);
        stem.fused = Fused::Relu;
        at += weigh(&mut stem, at);
        let mut pool = op(Kind::Maxpool, 1, 2, 4, 4);
        (pool.kh, pool.kw, pool.sh, pool.sw, pool.pr) = (1, 2, 1, 1, 1);
        let mean = op(Kind::Mean, 2, 1, 4, 4);
        let mut gate = op(Kind::Conv, 1, 3, 4, 4);
        window(&mut gate, (1, 1), 1, 0, 0);
        gate.fused = Fused::Hardsigmoid;
        at += weigh(&mut gate, at);
        let mut scale = op(Kind::Channelmul, 2, 1, 4, 4);
        scale.x2 = 3;
        let mut average = op(Kind::Avgpool, 1, 2, 4, 4);
        (average.kh, average.kw, average.sh, average.sw) = (1, 2, 1, 2);
        let mut qkv = op(Kind::Conv, 2, 1, 4, 12);
        window(&mut qkv, (1, 1), 1, 0, 0);
        qkv.fused = Fused::Gelu;
        at += weigh(&mut qkv, at);
        let mut attention = op(Kind::Attention, 1, 2, 12, 4);
        (attention.n2, attention.w) = (2, Block(at, 1));
        at += 1;
        let mut head = op(Kind::Ctcargmax, 2, 3, 4, 2);
        (head.n2, head.w, head.b) = (37, Block(at, 4 * 48), Block(at + 4 * 48, 48));
        at += 5 * 48;
        let ops = [stem, pool, mean, gate, scale, average, qkv, attention, head];
        let values: Vec<f32> = (0..at).map(|i| (i as f32 * 0.37).sin() * 0.5).collect();
        let mut values = values;
        // Columns past the vocabulary never win.
        values[at - 48 + 37..].fill(-1e30);
        let bytes = encode((0, 2, stored, 3), 4, &[3], &ops, &values);
        let (plan, values) = Plan::parse(&bytes).unwrap();
        Model::new(plan, values).unwrap()
    }

    /// A detector-shaped model with every size rule along both axes: a
    /// padded stride-2 stem, a max pool and a 2x2 convolution padded below
    /// and to the right, a 7x7 depthwise layer, a gate from a mean, a
    /// stride-2 convolution upsampled back onto the pool, a convolution of
    /// two inputs, a channel concatenation and a transposed convolution to
    /// the full size, stored at `h` by `w`.
    fn page_model(h: usize, w: usize) -> Model {
        let mut at = 0;
        let mut ops = Vec::new();
        let mut layer = |kind, (x, y), (ic, oc), k: usize, s, (before, after), fused| {
            let mut op = op(kind, x, y, ic, oc);
            window(&mut op, (k, k), s, before, after);
            op.fused = fused;
            if matches!(kind, Kind::Conv | Kind::Dwconv) {
                at += weigh(&mut op, at);
            }
            ops.push(op);
        };
        layer(Kind::Conv, (0, 1), (3, 8), 3, 2, (1, 1), Fused::Relu);
        layer(Kind::Maxpool, (1, 2), (8, 8), 2, 1, (0, 1), Fused::None);
        layer(Kind::Conv, (1, 3), (8, 8), 2, 1, (0, 1), Fused::Gelu);
        layer(Kind::Dwconv, (3, 4), (8, 8), 7, 1, (3, 3), Fused::None);
        layer(Kind::Mean, (4, 5), (8, 8), 0, 0, (0, 0), Fused::None);
        layer(
            Kind::Conv,
            (5, 6),
            (8, 8),
            1,
            1,
            (0, 0),
            Fused::Hardsigmoidfifth,
        );
        layer(Kind::Channelmul, (4, 5), (8, 8), 0, 0, (0, 0), Fused::None);
        layer(Kind::Conv, (5, 4), (8, 8), 3, 2, (1, 1), Fused::Swish);
        layer(Kind::Upsampleadd, (4, 6), (8, 8), 1, 2, (0, 0), Fused::None);
        // The gate, the pool under the upsampled map, the 2x2 convolution
        // beside the sum and the gated map beside that.
        ops[6].x2 = 6;
        ops[8].x2 = 2;
        let mut two = op(Kind::Conv, 6, 4, 8, 8);
        window(&mut two, (3, 3), 1, 1, 1);
        (two.x2, two.n2) = (3, 8);
        at += weigh(&mut two, at);
        let mut concat = op(Kind::Concatc, 4, 6, 8, 16);
        (concat.x2, concat.n2) = (5, 8);
        let mut out = op(Kind::Convtranspose, 6, 1, 16, 1);
        (out.kh, out.kw, out.sh, out.sw, out.fused) = (2, 2, 2, 2, Fused::Sigmoid);
        at += weigh(&mut out, at);
        ops.extend([two, concat, out]);
        let values: Vec<f32> = (0..at).map(|i| (i as f32 * 0.37).sin() * 0.4).collect();
        let bytes = encode((0, h, w, 3), 7, &[1], &ops, &values);
        let (plan, values) = Plan::parse(&bytes).unwrap();
        Model::new(plan, values).unwrap()
    }

    fn run(model: &mut Model) -> Vec<u32> {
        for (i, v) in model.input_mut().iter_mut().enumerate() {
            *v = (i as f32 * 0.013).cos();
        }
        model.run();
        model.output(0).iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn page_size_matches_fresh_load() {
        let mut model = page_model(16, 16);
        // Narrower, taller, both, the smallest and back.
        for (h, w) in [(16, 24), (40, 8), (8, 36), (4, 4), (16, 16), (64, 48)] {
            model.set_input_size(h, w).unwrap();
            assert_eq!(model.input_shape(), (h, w, 3));
            assert_eq!(model.output(0).len(), h * w, "{h}x{w}");
            let got = run(&mut model);
            assert_eq!(got, run(&mut page_model(h, w)), "stored at {h}x{w}");
        }
    }

    #[test]
    fn bad_page_size_refused() {
        let mut model = page_model(16, 16);
        let before = run(&mut model);
        // An odd half size does not match its upsampled quarter; the
        // largest overflow the shapes, the next the buffer ceiling.
        for (h, w) in [
            (10, 16),
            (16, 22),
            (0, 16),
            (usize::MAX, 16),
            (1 << 20, 1 << 20),
        ] {
            assert!(model.set_input_size(h, w).is_err(), "{h}x{w}");
            assert_eq!(model.input_shape(), (16, 16, 3));
            assert_eq!(run(&mut model), before, "after {h}x{w}");
        }
    }

    #[test]
    fn width_matches_fresh_load() {
        let mut model = line_model(40);
        // The narrowest width that leaves one time step, wider and narrower
        // than the stored one, odd and even, and back.
        for width in [3, 40, 1001, 17, 3, 64] {
            model.set_input_size(2, width).unwrap();
            let t = ((width - 1) / 2 + 1 - 2) / 2 + 1;
            assert_eq!(model.input_shape(), (2, width, 3));
            assert_eq!(model.output(0).len(), 2 * t, "width {width}");
            let got = run(&mut model);
            assert_eq!(got, run(&mut line_model(width)), "stored at width {width}");
        }
    }

    #[test]
    fn bad_width_keeps_model_size() {
        let mut model = line_model(40);
        let before = run(&mut model);
        // Two columns leave one for the two-wide average pool; the widest
        // input overflows the shapes, the next one the buffer ceiling.
        for width in [2, 1, 0, usize::MAX, 1 << 40] {
            assert!(model.set_input_size(2, width).is_err(), "width {width}");
            assert_eq!(model.input_shape(), (2, 40, 3));
            assert_eq!(run(&mut model), before, "after width {width}");
        }
    }
}
