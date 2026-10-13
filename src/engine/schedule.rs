//! How a run goes through a plan's operations, and where each tensor lives.
//!
//! [`next_step`] groups consecutive operations into one [`Step`] when what
//! passes between them can stay in a scratch buffer a group of rows or a run
//! of pixels at a time; each `can_*` test states what makes that safe.
//! [`lay_out`] then places every tensor a step writes in the model's arena,
//! sharing memory between tensors whose lives do not overlap.
//! `Model::schedule` in `mod.rs` calls both at load and on each change of
//! input size, and runs the steps.

use std::ops::Range;

use super::conv::{reads_windows, rows_per_group};
use super::plan::{Kind, Op, Plan};
use super::{kernels, layers};

/// What a run does next: one operation, or consecutive operations that
/// run together a part of the image at a time, so that what passes between
/// them stays in cache and, when nothing else reads it, never reaches its
/// tensor.
#[derive(Clone, Copy)]
pub(super) enum Step {
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
    pub(super) fn ops(&self) -> Range<usize> {
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
pub(super) struct Pair {
    /// A depthwise convolution computing each group's input rows into `fed`
    /// (see [`can_feed`]).
    pub(super) producer: Option<usize>,
    /// A channel multiply gating the expansion's input, a run of pixels at a
    /// time into `scratch` (see [`can_gate_expand`]).
    pub(super) gate: Option<usize>,
    /// A pointwise convolution widening the pair's input, a run of pixels at
    /// a time into `mid` (see [`can_expand`]).
    pub(super) expand: Option<usize>,
    pub(super) conv: usize,
    /// Only the add reads the convolution's output: it goes through `rows`
    /// and never reaches its tensor, which nothing would observe.
    pub(super) private: bool,
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

pub(super) fn expand_run(op: &Op) -> usize {
    EXPAND_RUN.min(rows_per_group(op) * op.ow)
}

/// The step a run takes at operation `i`. Each kind of step needs its own
/// kinds of operation at `i` and after it, so at most one applies.
pub(super) fn next_step(plan: &Plan, i: usize) -> Step {
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
pub(super) struct Inputs {
    pub(super) slots: [(usize, usize, u32); kernels::SOURCES],
    pub(super) count: usize,
}

/// The buffers the windowed convolution `i` can read in place of what the
/// unbroken run of upsamplings (by powers of two) and channel
/// concatenations just before it builds, and where that chain starts.
///
/// Every layer of the chain must feed the convolution, and its output must
/// go unread afterwards. The chain layers being the only ones between, a
/// buffer the chain reads still holds at the convolution what it held when
/// the chain would have read it, the chain's own writes being skipped. The
/// convolution must not write any buffer it reads in the chain's place:
/// `Plan::check` proved that only of the buffers it read through the chain.
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
pub(super) struct Placement {
    /// The first and last time a step names the buffer: 0 is before the
    /// run, step `k` is `k + 1`, and the outputs are read after the last.
    life: Vec<Option<(usize, usize)>>,
    /// Some step writes the buffer, so it needs memory.
    written: Vec<bool>,
    /// The buffer whose memory this one takes over, itself if none.
    with: Vec<usize>,
    /// Where the buffer's memory starts in the arena, in floats.
    pub(super) offset: Vec<Option<usize>>,
    /// Floats from `offset` taken while the buffer is live, its group's.
    size: Vec<usize>,
    /// Buffers heading a group, to place largest first.
    order: Vec<usize>,
    /// Ranges of the arena taken by groups placed and live at the same
    /// time as the one being placed.
    taken: Vec<(usize, usize)>,
}

impl Placement {
    pub(super) fn new(buffers: usize) -> Self {
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
pub(super) fn lay_out(plan: &Plan, steps: &[Step], place: &mut Placement) -> usize {
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
    use super::super::conv::Tail;
    use super::super::plan::{op, Fused};
    use super::super::tests::{fill, model, results, weigh, window};
    use super::super::{plan, Model};
    use super::*;

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
}
