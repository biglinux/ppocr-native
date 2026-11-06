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
//! One image at a time, 32-bit floats throughout: no batching or
//! quantisation. [`Model::set_threads`] shares each layer's rows between
//! threads, giving the bits one thread gives.
//!
//! `plan.rs` reads and checks a plan; `schedule.rs` decides which operations
//! run together and where each tensor lives; this file owns the memory and
//! runs the steps; `conv.rs` (convolution, pooling) and `layers.rs` (the
//! rest) compute whole layers with the tiered SIMD code in `kernels/`.

#[macro_use]
mod kernels;
mod conv;
mod layers;
mod plan;
mod schedule;

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::any::Any;
use std::ops::Range;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex, PoisonError};
use std::thread;

use conv::{
    add_with_tail, convolve, depthwise, maxpool, maxpool_in_place, opt, reads_windows,
    rows_per_group, Tail,
};
pub use kernels::simd_tier;
pub use plan::{Block, Error, Fused, Kind, Op};
use plan::{Plan, MAX_SCRATCH_FLOATS};
use schedule::{expand_run, lay_out, next_step, Inputs, Pair, Placement, Step};

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

    /// Floats `start..start + len` alone, so that threads can hold views of
    /// different parts of one buffer.
    ///
    /// # Safety
    /// No mutable view of these floats may exist for the returned lifetime.
    unsafe fn slice(&self, start: usize, len: usize) -> &[f32] {
        assert!(start + len <= self.len, "view past its buffer");
        // SAFETY: inside the view, as asserted; as for `get` otherwise.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(start), len) }
    }

    /// [`View::slice`], mutable.
    ///
    /// # Safety
    /// No other view of these floats may exist for the returned lifetime.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut(&self, start: usize, len: usize) -> &mut [f32] {
        assert!(start + len <= self.len, "view past its buffer");
        // SAFETY: inside the view, as asserted; as for `get_mut` otherwise.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(start), len) }
    }

    fn overlaps(&self, other: &View) -> bool {
        let (a, b) = (self.ptr.as_ptr() as usize, other.ptr.as_ptr() as usize);
        self.len > 0 && other.len > 0 && a < b + other.len * 4 && b < a + self.len * 4
    }
}

/// The buffers one thread works in during a run.
struct Work {
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

impl Work {
    fn new() -> Self {
        Self {
            rows: Buffer::new(0),
            mid: Buffer::new(0),
            fed: Buffer::new(0),
            scratch: Buffer::new(0),
        }
    }
}

/// A model the threads of one of its runs share.
struct Shared<'a>(&'a Model);

// SAFETY: while the units of a step run, each thread writes only its own
// `Work`, and the units write disjoint parts of the step's outputs through
// `View::slice_mut`, never a part another unit reads (`Model::units`). What
// thread 0 does alone, a whole step or the gate copied into every thread's
// `Work`, runs while the others wait at a barrier, and a barrier orders every
// step after the one before it.
unsafe impl Sync for Shared<'_> {}

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
/// Running on one thread allocates nothing, and on several only starts the
/// run's threads; changing the input size allocates only when a buffer has
/// to grow.
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
    /// The working buffers of each thread a run uses, one or more.
    work: Vec<Work>,
    /// Per step, the next unit a thread of the run takes; one per operation,
    /// as many as steps there can be.
    claims: Vec<AtomicUsize>,
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
            claims: (0..plan.ops.len()).map(|_| AtomicUsize::new(0)).collect(),
            plan,
            weights,
            arena: Buffer::new(0),
            panels,
            steps,
            work: vec![Work::new()],
        };
        model.schedule()?;
        Ok(model)
    }

    /// Another model sharing this one's weights, with scratch memory of its
    /// own, at the same input size and on as many threads: a model per thread
    /// costs the weights once.
    pub fn try_clone(&self) -> Result<Self, Error> {
        let mut model = Self {
            plan: self.plan.clone(),
            weights: Arc::clone(&self.weights),
            arena: Buffer::new(0),
            pool: vec![View::EMPTY; self.plan.pool.len()],
            layout: Placement::new(self.plan.pool.len()),
            panels: self.panels.clone(),
            steps: Vec::with_capacity(self.plan.ops.len()),
            work: self.work.iter().map(|_| Work::new()).collect(),
            claims: (0..self.plan.ops.len())
                .map(|_| AtomicUsize::new(0))
                .collect(),
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
        let total = arena + self.work.len() * (rows + mid + fed + scratch);
        if total > MAX_SCRATCH_FLOATS {
            return Err(Error::Invalid(format!(
                "scratch wants {total} floats, ceiling is {MAX_SCRATCH_FLOATS}"
            )));
        }
        for work in &mut self.work {
            work.rows.reserve(rows)?;
            work.mid.reserve(mid)?;
            work.fed.reserve(fed)?;
            work.scratch.reserve(scratch)?;
        }
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

    /// Split each following run between `threads` threads (one when 0),
    /// each with working buffers of its own. A layer's rows are shared out
    /// in the pieces one thread computes them in, so the result has the same
    /// bits on any number of threads. On error the model keeps its threads.
    pub fn set_threads(&mut self, threads: usize) -> Result<(), Error> {
        let old = self.work.len();
        self.work.resize_with(threads.max(1), Work::new);
        if let Err(e) = self.arrange() {
            // The threads kept still hold buffers of their old size.
            self.work.truncate(old);
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

    /// Run the network once over the image in [`Model::input_mut`], on the
    /// threads [`Model::set_threads`] gave it, or on one thread when the
    /// system refuses to start them; the result is the same.
    pub fn run(&mut self) {
        if self.work.len() == 1 {
            for step in &self.steps {
                self.run_step(step, &self.work[0]);
            }
            return;
        }
        for claim in &mut self.claims {
            *claim.get_mut() = 0;
        }
        let this: &Model = self;
        let barrier = Barrier::new(this.work.len());
        let failed = Mutex::new(None);
        let shared = Shared(this);
        // Whether every helper started, once known: no helper reaches the
        // barrier, sized for all of them, before that.
        let start = (Mutex::new(None), Condvar::new());
        let (shared, barrier, failed, start) = (&shared, &barrier, &failed, &start);
        thread::scope(|scope| {
            let mut started = true;
            for t in 1..this.work.len() {
                let helper = thread::Builder::new().spawn_scoped(scope, move || {
                    let (go, ready) = start;
                    let mut go = go.lock().unwrap_or_else(PoisonError::into_inner);
                    while go.is_none() {
                        go = ready.wait(go).unwrap_or_else(PoisonError::into_inner);
                    }
                    if *go == Some(true) {
                        drop(go);
                        shared.0.share(t, barrier, failed);
                    }
                });
                if helper.is_err() {
                    started = false;
                    break;
                }
            }
            *start.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(started);
            start.1.notify_all();
            if started {
                this.share(0, barrier, failed);
            } else {
                for step in &this.steps {
                    this.run_step(step, &this.work[0]);
                }
            }
        });
        let panic = failed.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(panic) = panic {
            resume_unwind(panic);
        }
    }

    /// Thread `t`'s part of a run: every step in order, the ones
    /// [`Model::units`] divides taken a unit at a time by whichever thread
    /// is free, the others run by thread 0 alone, all threads meeting after
    /// each step. A panic is kept for [`Model::run`] to raise once every
    /// thread is through the steps, so that no thread waits at a meeting for
    /// one that stopped.
    fn share(&self, t: usize, barrier: &Barrier, failed: &Mutex<Option<Box<dyn Any + Send>>>) {
        let work = &self.work[t];
        let attempt = |job: &dyn Fn()| {
            if failed
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_some()
            {
                return;
            }
            if let Err(panic) = catch_unwind(AssertUnwindSafe(job)) {
                *failed.lock().unwrap_or_else(PoisonError::into_inner) = Some(panic);
            }
        };
        for (step, claim) in self.steps.iter().zip(&self.claims) {
            match self.units(step) {
                Some(units) => {
                    // A gate goes to every thread before any unit can write
                    // over it.
                    match *step {
                        Step::Pair(ref pair) if pair.gate.is_some() => {
                            if t == 0 {
                                attempt(&|| self.work.iter().for_each(|w| self.copy_gate(pair, w)));
                            }
                            barrier.wait();
                        }
                        Step::GateAdd(i) => {
                            if t == 0 {
                                let gate = &self.plan.ops[i];
                                attempt(&|| {
                                    self.work.iter().for_each(|w| self.copy_gate_add(gate, w))
                                });
                            }
                            barrier.wait();
                        }
                        _ => {}
                    }
                    attempt(&|| loop {
                        let unit = claim.fetch_add(1, Ordering::Relaxed);
                        if unit >= units {
                            break;
                        }
                        self.run_units(step, unit..unit + 1, work);
                    });
                    if step.mean().is_some() {
                        barrier.wait();
                        if t == 0 {
                            attempt(&|| self.mean_after(step));
                        }
                    }
                }
                None if t == 0 => attempt(&|| self.run_step(step, work)),
                None => {}
            }
            barrier.wait();
        }
    }

    /// The units a step divides into, which may run on any thread in any
    /// order and give the bits of the whole step: the rows or groups of rows
    /// it computes one at a time. `None` for a step that runs whole on one
    /// thread: one whose units would write what another unit reads, and the
    /// pooling and element-wise steps, a small share of a run.
    fn units(&self, step: &Step) -> Option<usize> {
        let ops = &self.plan.ops;
        match *step {
            Step::Op { i, .. } | Step::Chained { conv: i, .. } => self.op_units(i),
            Step::Pair(ref pair) => self
                .pair_divides(pair)
                .then(|| ops[pair.conv].oh.div_ceil(rows_per_group(&ops[pair.conv]))),
            // `can_feed` keeps the convolution's output apart from the
            // depthwise input its groups read through windows.
            Step::Fed { dw, .. } => {
                let conv = &ops[dw + 1];
                Some(conv.oh.div_ceil(rows_per_group(conv)))
            }
            // A row of the channel multiply and add, or an input row of the
            // first transposed convolution: each writes its own pixels.
            Step::GateAdd(i) => Some(ops[i].oh),
            Step::Transposed(i) => Some(ops[i].ih),
        }
    }

    /// [`Model::units`] of operation `i`: the rows of a convolution, or the
    /// groups of rows of a pointwise one. `Plan::check` and `lay_out` keep a
    /// convolution's output apart from all it reads.
    fn op_units(&self, i: usize) -> Option<usize> {
        let op = &self.plan.ops[i];
        match op.kind {
            Kind::Conv if op.is_pointwise() => Some(op.oh.div_ceil(rows_per_group(op))),
            Kind::Conv | Kind::Dwconv => Some(op.oh),
            _ => None,
        }
    }

    /// Whether `pair`'s groups can run apart: what one group writes is
    /// read by no other. The pair may write over an input it reads pixel
    /// by pixel when a pixel takes as many floats there as in the output,
    /// so that each group writes over its own pixels alone.
    fn pair_divides(&self, pair: &Pair) -> bool {
        let ops = &self.plan.ops;
        let (conv, add) = (&ops[pair.conv], &ops[pair.conv + 1]);
        let in_work = |slot: usize| {
            pair.gate.is_some_and(|g| slot == ops[g].y)
                || pair.producer.is_some_and(|d| slot == ops[d].y)
                || (pair.private && slot == conv.y)
        };
        // Inputs read pixel by pixel, with the floats a pixel takes there.
        let pixels = [
            pair.gate.map(|g| (ops[g].x, ops[g].oc)),
            Some(
                pair.expand
                    .map_or((conv.x, conv.ic), |e| (ops[e].x, ops[e].ic)),
            ),
            Some((add.x, conv.oc)),
            Some((add.x2, conv.oc)),
        ];
        // An input read through windows of neighbouring rows; the gate is
        // copied before any group runs.
        let whole = [pair.producer.map(|d| ops[d].x)];
        let written = [Some(add.y), (!pair.private).then_some(conv.y)];
        written.into_iter().flatten().all(|w| {
            let out = &self.pool[w];
            let by_pixel = pixels.into_iter().flatten().all(|(slot, c)| {
                let input = &self.pool[slot];
                in_work(slot) || !out.overlaps(input) || (out.ptr == input.ptr && c == conv.oc)
            });
            by_pixel
                && whole
                    .into_iter()
                    .flatten()
                    .all(|slot| !out.overlaps(&self.pool[slot]))
        })
    }

    /// Units `units` of a step [`Model::units`] divides, on `work`.
    fn run_units(&self, step: &Step, units: Range<usize>, work: &Work) {
        match *step {
            Step::Op { i, .. } => self.step(i, &Inputs::default(), units, work, &mut |_| {}),
            Step::Chained {
                conv, ref inputs, ..
            } => self.step(conv, inputs, units, work, &mut |_| {}),
            Step::Pair(ref pair) => self.row_pair(pair, units, work),
            Step::Fed { dw, .. } => self.fed_pointwise(dw, units, work, None),
            Step::GateAdd(i) => {
                self.gate_add(&self.plan.ops[i], &self.plan.ops[i + 1], units, work)
            }
            Step::Transposed(i) => self.transposed_pair(i, units, work),
        }
    }

    /// The channel mean of a divided step, from its finished output: the
    /// sums its units would have made one after another, pixel by pixel.
    fn mean_after(&self, step: &Step) {
        let Some(mean) = step.mean() else {
            return;
        };
        let (op, mean) = (&self.plan.ops[mean - 1], &self.plan.ops[mean]);
        let mut sums = layers::MeanSums::new(op.oc);
        let pixels = op.oh * op.ow;
        // SAFETY: the step is finished; the mean writes a buffer of its own.
        unsafe {
            sums.add(self.pool[op.y].slice(0, pixels * op.oc));
            sums.finish(self.pool[mean.y].get_mut(), pixels);
        }
    }

    /// [`Model::run`], recording the time of each step in milliseconds of
    /// `clock`, with the operations it ran: one, or several that run
    /// together a part of the image at a time. A CPU-time clock keeps a
    /// loaded host's descheduling out. It runs on one thread.
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
            self.run_step(step, &self.work[0]);
            times.push((step.ops(), clock() - started));
        }
    }

    /// A whole step on one thread.
    fn run_step(&self, step: &Step, work: &Work) {
        match *step {
            Step::Op { i, mean } => self.alone(i, &Inputs::default(), mean, work),
            Step::Chained {
                conv,
                ref inputs,
                mean,
                ..
            } => self.alone(conv, inputs, mean, work),
            Step::Pair(ref pair) => {
                let conv = &self.plan.ops[pair.conv];
                self.copy_gate(pair, work);
                self.row_pair(pair, 0..conv.oh.div_ceil(rows_per_group(conv)), work);
            }
            Step::Fed { dw, mean } => {
                let conv = &self.plan.ops[dw + 1];
                let mut sums = layers::MeanSums::new(if mean { conv.oc } else { 0 });
                let groups = conv.oh.div_ceil(rows_per_group(conv));
                self.fed_pointwise(dw, 0..groups, work, mean.then_some(&mut sums));
                if mean {
                    // SAFETY: the mean writes a buffer of its own.
                    sums.finish(
                        unsafe { self.pool[self.plan.ops[dw + 2].y].get_mut() },
                        conv.oh * conv.ow,
                    );
                }
            }
            Step::GateAdd(i) => {
                let gate = &self.plan.ops[i];
                self.copy_gate_add(gate, work);
                self.gate_add(gate, &self.plan.ops[i + 1], 0..gate.oh, work);
            }
            Step::Transposed(i) => self.transposed_pair(i, 0..self.plan.ops[i].ih, work),
        }
    }

    /// Operation `i`, a windowed convolution reading `inputs` if they name
    /// any; with `mean`, the channel mean after it summed as `i` writes.
    fn alone(&self, i: usize, inputs: &Inputs, mean: bool, work: &Work) {
        // Every unit of a convolution; `step` runs any other operation whole
        // and reads no units for it.
        let units = 0..self.op_units(i).unwrap_or(1);
        if !mean {
            self.step(i, inputs, units, work, &mut |_| {});
            return;
        }
        let (op, mean) = (&self.plan.ops[i], &self.plan.ops[i + 1]);
        let mut sums = layers::MeanSums::new(op.oc);
        self.step(i, inputs, units, work, &mut |rows| sums.add(rows));
        // SAFETY: the mean writes a buffer of its own; its input, the
        // layer's output, is complete once `step` returns.
        sums.finish(unsafe { self.pool[mean.y].get_mut() }, op.oh * op.ow);
    }

    /// Copy the gate of `pair`, if it has one, into `work`'s scratch after
    /// the gated run ([`gated_run`]), for [`Model::row_pair`] to read: a layer
    /// of the pair may write over the gate, which the multiply reads whole.
    fn copy_gate(&self, pair: &Pair, work: &Work) {
        if let Some(g) = pair.gate {
            let (g, at) = (&self.plan.ops[g], gated_run(pair, &self.plan.ops));
            // SAFETY: no other thread uses `work` meanwhile: it is the
            // caller's own, or every thread waits at the barrier that follows
            // thread 0's copy (`Model::share`). The pool is only read here.
            unsafe {
                work.scratch.get_mut()[at..at + g.oc]
                    .copy_from_slice(self.pool[g.x2].slice(0, g.oc));
            }
        }
    }

    /// The pointwise convolution and residual add of `pair`, a group of rows
    /// at a time. With an expansion the convolution before them runs too,
    /// its output only ever in `mid`, a run of pixels at a time; with a gate
    /// the channel multiply before that, its product only in `scratch`; with
    /// a producer the depthwise convolution before the first of them, a
    /// group of rows at a time into `fed`. Every other output is written to
    /// its tensor, except a private convolution's.
    fn row_pair(&self, pair: &Pair, units: Range<usize>, work: &Work) {
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
        let gated = gated_run(pair, ops);
        for unit in units.clone() {
            let group = if backwards {
                units.end - 1 - (unit - units.start)
            } else {
                unit
            };
            let first = group * rows;
            let pixels = rows.min(conv.oh - first) * conv.ow;
            if let Some(d) = producer {
                let dw = &ops[d];
                // SAFETY: `fed` is an allocation of its own, written only
                // here; the pool is only read.
                let (x, y) = unsafe {
                    (
                        self.pool[dw.x].get(),
                        &mut work.fed.get_mut()[..pixels * dw.oc],
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
                        &work.scratch.get()[..count * c]
                    } else if producer.is_some_and(|d| slot == ops[d].y) {
                        &work.fed.get()[start * c..][..count * c]
                    } else {
                        self.pool[slot].slice(pixel * c, count * c)
                    }
                };
                if let Some(g) = gate {
                    let g = &ops[g];
                    let x = read(g.x, g.oc);
                    // SAFETY: `scratch` is an allocation of its own; the gate
                    // reads the pool or `fed`.
                    let (y, gates) = unsafe { work.scratch.get_mut().split_at_mut(gated) };
                    kernels::channel_mul(y, x, &gates[..g.oc], count, g.oc);
                }
                if let Some(i) = expand {
                    let e = &ops[i];
                    let x = read(e.x, e.ic);
                    // SAFETY: `mid` is an allocation of its own, written only
                    // here.
                    let y = unsafe { &mut work.mid.get_mut()[..count * e.oc] };
                    let (clip, tail) = Tail::of(e).fold_clip();
                    let (w, b) = (block(e.w), opt(e.b, weights));
                    kernels::pointwise_with(y, w, x, b, count, e.ic, e.oc, self.panels[i], clip);
                    tail.apply(y);
                }
                let (dst, n) = (pixel * conv.oc, count * conv.oc);
                {
                    let x = if expand.is_some() {
                        // SAFETY: `mid` is only read from here on.
                        unsafe { &work.mid.get()[..count * conv.ic] }
                    } else {
                        read(conv.x, conv.ic)
                    };
                    // SAFETY: the validated convolution has distinct x/y
                    // buffers, and `rows` is an allocation of its own. Drop
                    // both views BEFORE the add can overwrite conv.x.
                    let y = unsafe {
                        if private {
                            &mut work.rows.get_mut()[..n]
                        } else {
                            self.pool[conv.y].slice_mut(dst, n)
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
                            unsafe { &work.rows.get()[..n] }
                        } else {
                            read(slot, conv.oc)
                        }
                    };
                    let (x, x2) = (input(add.x), input(add.x2));
                    // SAFETY: as above.
                    let y = unsafe { self.pool[add.y].slice_mut(dst, n) };
                    add_with_tail(y, x, x2, n, Tail::of(add));
                }
            }
        }
    }

    /// Depthwise convolution `i` a group of rows at a time into `fed`, each
    /// group straight through the pointwise convolution after it, for the
    /// groups `units`; with `sums`, that convolution's output summed for its
    /// channel mean as it goes.
    fn fed_pointwise(
        &self,
        i: usize,
        units: Range<usize>,
        work: &Work,
        mut sums: Option<&mut layers::MeanSums>,
    ) {
        let ops = &self.plan.ops;
        let (dw, conv) = (&ops[i], &ops[i + 1]);
        let weights = self.weights.get();
        let block = |b: Block| &weights[b.0..b.0 + b.1];
        let rows = rows_per_group(conv);
        let (clip, tail) = Tail::of(conv).fold_clip();
        for group in units {
            let first = group * rows;
            let pixels = rows.min(conv.oh - first) * conv.ow;
            // SAFETY: `fed` is an allocation of its own; `can_feed` keeps the
            // convolution's output apart from the depthwise input.
            let (x, fed, y) = unsafe {
                (
                    self.pool[dw.x].get(),
                    &mut work.fed.get_mut()[..pixels * dw.oc],
                    self.pool[conv.y].slice_mut(first * conv.ow * conv.oc, pixels * conv.oc),
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
            if let Some(sums) = sums.as_deref_mut() {
                sums.add(y);
            }
        }
    }

    /// Transposed convolutions `i` and `i + 1` over the first one's input
    /// rows `inputs`, row by row.
    fn transposed_pair(&self, i: usize, inputs: Range<usize>, work: &Work) {
        let (first, second) = (&self.plan.ops[i], &self.plan.ops[i + 1]);
        let per_input = first.kh * second.kh * second.ow * second.oc;
        // SAFETY: the first layer's input, the second's output and the
        // scratch are three buffers apart (`can_chain_transposed`); the
        // output rows of `inputs` are theirs alone (`Model::units`).
        let (x, y, scratch) = unsafe {
            (
                self.pool[first.x].get(),
                self.pool[second.y].slice_mut(inputs.start * per_input, inputs.len() * per_input),
                work.scratch.get_mut(),
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
        a.run_into(
            &b,
            x,
            y,
            scratch,
            inputs,
            &|out| tail_a.apply(out),
            &|out| tail_b.apply(out),
        );
    }

    /// Copy the gate of the channel multiply `gate` into `work`'s scratch,
    /// for [`Model::gate_add`]: the add may write over the gate, which every
    /// row of the multiply reads.
    fn copy_gate_add(&self, gate: &Op, work: &Work) {
        // SAFETY: no other thread uses `work` meanwhile: it is the caller's
        // own, or every thread waits at the barrier that follows thread 0's
        // copy (`Model::share`). The pool is only read here.
        unsafe {
            work.scratch.get_mut()[..gate.oc].copy_from_slice(&self.pool[gate.x2].get()[..gate.oc]);
        }
    }

    /// Rows `rows` of a channel multiply and the add of its product to its
    /// input, with the gate [`Model::copy_gate_add`] left in `work`.
    fn gate_add(&self, gate: &Op, add: &Op, rows: Range<usize>, work: &Work) {
        // SAFETY: the copy is only read from here on.
        let g = unsafe { &work.scratch.get()[..gate.oc] };
        let (clip, tail) = Tail::of(add).fold_clip();
        let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
        let pixels = rows.len() * gate.ow;
        let (start, n) = (rows.start * gate.ow * gate.oc, pixels * gate.oc);
        // SAFETY: the add's output has the multiply's input's memory, or
        // memory apart from both its inputs, live at the same step; each
        // pixel reads only itself, so the rows of `rows` are theirs alone.
        let y = unsafe { self.pool[add.y].slice_mut(start, n) };
        if self.pool[add.y].ptr == self.pool[gate.x].ptr {
            kernels::gate_add_in_place(y, g, pixels, gate.oc, bounds);
        } else {
            // SAFETY: as above.
            let x = unsafe { self.pool[gate.x].slice(start, n) };
            kernels::gate_add(y, x, g, pixels, gate.oc, bounds);
        }
        tail.apply(y);
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
    /// name any: the units `units` of a convolution ([`Model::op_units`]),
    /// any other operation whole. A convolution hands `finished` each run of
    /// whole output pixels once it is final, in order.
    fn step(
        &self,
        i: usize,
        inputs: &Inputs,
        units: Range<usize>,
        work: &Work,
        finished: &mut dyn FnMut(&[f32]),
    ) {
        let (op, panels) = (&self.plan.ops[i], self.panels[i]);
        if let Some(over) = self.written_over(op) {
            self.in_place(op, over);
            return;
        }
        // SAFETY: `Plan::check` proved that no operation writes a buffer it
        // also reads, and `lay_out` keeps the buffers apart in memory but
        // for the output written over an input, handled above; so the output
        // view never overlaps an input view. A convolution takes the rows of
        // its units alone, which no other unit touches (`Model::units`); any
        // other operation runs whole on one thread.
        let x = unsafe { self.pool[op.x].get() };
        let out = &self.pool[op.y];
        let y = || unsafe { out.get_mut() };
        let len = op.ow * op.oc;
        let weights = self.weights.get();
        // SAFETY: as for `x`; `Plan::check` proved `x2` distinct from `y`.
        let x2 = || unsafe { self.pool[op.x2].get() };
        // SAFETY: the scratch belongs to this thread, used by no other view
        // while one operation runs.
        let scratch = || unsafe { work.scratch.get_mut() };
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
                for oy in units {
                    // SAFETY: this unit's row alone, as above.
                    let row = unsafe { out.slice_mut(oy * len, len) };
                    kernels::conv_row(row, &sources[..count], zeros, w, b, op, oy, clip);
                    tail.apply(row);
                    finished(row);
                }
            }
            Kind::Conv => {
                let (w, b) = (block(op.w), opt(op.b, weights));
                let rows = if op.is_pointwise() {
                    rows_per_group(op)
                } else {
                    1
                };
                let first = units.start * rows;
                let end = (units.end * rows).min(op.oh);
                // SAFETY: the rows of this step's units alone, as above.
                let y = unsafe { out.slice_mut(first * len, (end - first) * len) };
                convolve(op, x, y, w, b, tail, panels, finished, units);
            }
            Kind::Dwconv => {
                let (w, b) = (block(op.w), opt(op.b, weights));
                // SAFETY: as above.
                let y = unsafe { out.slice_mut(units.start * len, units.len() * len) };
                depthwise(op, x, y, w, b, tail, finished, units);
            }
            Kind::Add => add_with_tail(y(), x, x2(), op.out_len(), tail),
            Kind::Channelmul => {
                kernels::channel_mul(y(), x, &x2()[..op.oc], op.oh * op.ow, op.oc);
            }
            Kind::Maxpool => maxpool(op, x, y()),
            Kind::Avgpool => layers::average_pool(op, x, y()),
            Kind::Concatc => layers::concat_channels(op, x, x2(), y()),
            Kind::Layernorm => layers::layer_norm(op, x, y(), block(op.w)),
            Kind::Mean => layers::mean(op, x, y()),
            Kind::Attention => layers::attention(op, x, y(), block(op.w)[0], scratch()),
            Kind::Upsample => layers::upsample(op, x, y()),
            Kind::Upsampleadd => layers::upsample_add(op, x, x2(), y()),
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
                transposed.run(x, y(), scratch(), &|out| tail.apply(out));
            }
            Kind::Ctchead | Kind::Ctcargmax => {
                let probs = op.kind == Kind::Ctchead;
                let vocab = if probs { op.oc } else { op.n2 };
                let (w, b) = (block(op.w), block(op.b));
                layers::ctc_head(y(), x, w, b, op.iw, op.ic, vocab, probs, scratch());
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
        let work: usize = self
            .work
            .iter()
            .map(|w| w.rows.len + w.mid.len + w.fed.len + w.scratch.len)
            .sum();
        (self.arena.len + work) * 4
    }

    /// The operations at the current input size, for callers that time or
    /// report them.
    pub fn ops(&self) -> &[Op] {
        &self.plan.ops
    }
}

/// Where the gate of a gated pair goes in a thread's scratch: after the run
/// of gated pixels the multiply writes there.
fn gated_run(pair: &Pair, ops: &[Op]) -> usize {
    match (pair.gate, pair.expand) {
        (Some(g), Some(e)) => expand_run(&ops[e]) * ops[g].oc,
        _ => 0,
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

#[cfg(test)]
mod tests {
    use super::plan::{encode, op};
    use super::*;

    /// A model of `ops` over an input `(slot, h, w, c)`, with `weights`
    /// floats of a smooth ramp.
    pub(super) fn model(
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

    pub(super) fn window(op: &mut Op, k: (usize, usize), s: usize, before: isize, after: isize) {
        (op.kh, op.kw, op.sh, op.sw) = (k.0, k.1, s, s);
        (op.pt, op.pl, op.pb, op.pr) = (before, before, after, after);
    }

    /// A convolution or transposed convolution with weights and bias at
    /// `at` in the weights; returns the floats it takes.
    pub(super) fn weigh(op: &mut Op, at: usize) -> usize {
        let taps = op.kh * op.kw;
        let (w, b) = match op.kind {
            Kind::Dwconv => (taps * op.oc, op.oc),
            Kind::Convtranspose => (taps * op.ic * op.oc, taps * op.oc),
            _ => (taps * (op.ic + op.n2) * op.oc, op.oc),
        };
        (op.w, op.b) = (Block(at, w), Block(at + w, b));
        w + b
    }

    pub(super) fn fill(model: &mut Model) {
        for (i, v) in model.input_mut().iter_mut().enumerate() {
            *v = (i as f32 * 0.29).cos() * 2.0;
        }
    }

    pub(super) fn results(model: &Model) -> Vec<Vec<u32>> {
        (0..model.output_count())
            .map(|o| model.output(o).iter().map(|v| v.to_bits()).collect())
            .collect()
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
    fn threads_match_one_thread() {
        let check = |label: &str, make: &dyn Fn() -> Model| {
            let model = make();
            let divided = model.steps.iter().filter(|s| model.units(s).is_some());
            assert!(divided.count() > 0, "{label}: no step is shared out");
            let want = run(&mut make());
            for threads in [2, 3, 8] {
                let mut model = make();
                model.set_threads(threads).unwrap();
                assert_eq!(run(&mut model), want, "{label} on {threads} threads");
            }
        };
        check("page 64x48", &|| page_model(64, 48));
        check("page 40x8", &|| page_model(40, 8));
        check("line 1001", &|| line_model(1001));
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
