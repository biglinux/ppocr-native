//! The execution plan: what `tools/export_ocr.py` wrote, and how it is checked.
//!
//! A plan is one file: a short binary description, then the weights. The
//! description lists the operations in the order they run and where each
//! one's numbers live. Spatial sizes are not stored: they follow from the
//! input size by the rules of `Plan::set_size`, so one plan serves every page
//! and every line width.
//!
//! Layout, every number little-endian:
//!
//! - the magic `PPOCRPL1`;
//! - seven `u32`: the input buffer, the input height, width and channels the
//!   model starts at, the buffer count, the output count, the operation count;
//! - one `u32` per output: the buffer it is read from;
//! - [`OP_FIELDS`] `u32` per operation: kind (index into [`KINDS`]), folded
//!   activation (index into [`FUSED`]), x, x2, y, ic, oc, kh, kw, sh, sw, pt,
//!   pb, pl, pr, n2, weights start and length, bias start and length, the
//!   blocks counted in floats from the start of the weights;
//! - zero padding up to the next 64-byte boundary, then the weights as `f32`
//!   to the end of the file.
//!
//! Nothing in a plan is trusted. It arrives from disk, and a description that
//! disagreed with its numbers could otherwise read past the end of them. So
//! loading checks, before anything runs: that every buffer index exists, that
//! every tensor fits the buffer it was assigned, that every block of numbers
//! lies inside the weights and is the size the operation's shape implies, and
//! that no number is infinite or not-a-number. After that the executor can
//! index without re-checking.

use std::fmt;
use std::path::Path;

/// Anything that stops a plan from being loaded or resized.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The file is not a plan this build reads.
    Format(String),
    /// The plan is self-inconsistent, disagrees with its weights, or cannot
    /// take the requested input size.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "reading the plan: {e}"),
            Self::Format(m) => write!(f, "unsupported plan: {m}"),
            Self::Invalid(m) => write!(f, "invalid plan: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

type Result<T> = std::result::Result<T, Error>;

/// First bytes of a plan; a later layout gets a new one.
const MAGIC: &[u8; 8] = b"PPOCRPL1";

/// `u32` fields per operation record.
pub const OP_FIELDS: usize = 20;

/// Ceiling on scratch memory, in floats: the detector needs 1.6 GiB for a
/// 4096-by-4096 page, and anything far past that means a corrupt or hostile
/// file or an absurd input size, which is refused rather than allocated.
pub(crate) const MAX_SCRATCH_FLOATS: usize = 1 << 30; // 4 GiB

/// Largest kernel, stride or padding accepted, far above any PP-OCR layer;
/// it keeps every window coordinate well inside `isize`.
const MAX_WINDOW: usize = 1 << 16;

/// What an operation does. Convolution covers the dense and depthwise cases
/// because they differ only in how the weights are indexed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Dense convolution: every output channel reads every input channel.
    /// One whose kernel is not 1x1 may read the channel concatenation of its
    /// input and a second one of `n2` channels.
    Conv,
    /// Depthwise convolution: each channel is filtered on its own.
    Dwconv,
    /// Elementwise sum of two tensors of the same shape.
    Add,
    /// Sliding-window maximum.
    Maxpool,
    /// Valid-window average, summing taps in row-major order then dividing.
    Avgpool,
    /// Multiply every pixel's channels by the `oc` values of the second
    /// input: a squeeze-and-excite gate.
    Channelmul,
    /// Each pixel's `ic` channels, then the second input's `n2`.
    Concatc,
    /// `(x - mean) / sqrt(var + eps) * gamma + beta` over each pixel's
    /// channels; `w` holds gamma, beta (`oc` each) then epsilon.
    Layernorm,
    /// Multi-head self-attention over the `iw` pixels of a one-row tensor:
    /// each pixel holds q, k and v (`oc` each, `n2` heads), `w` the score
    /// scale; the output holds each head's `oc / n2` values in head order.
    Attention,
    /// Each channel's mean over the whole map, a 1x1 output.
    Mean,
    /// CTC head: `x W + b` over `oc` classes, then softmax probabilities.
    /// `w` holds `ceil(oc / 16)` panels of `[ic][16]` columns, `b` as many
    /// values.
    Ctchead,
    /// [`Kind::Ctchead`] over `n2` classes keeping, per pixel, only the best
    /// class (as a float) and its probability: `oc` is 2.
    Ctcargmax,
    /// Nearest-neighbour upsampling by the whole factors `sh` and `sw`:
    /// `y[r][c] = x[r / sh][c / sw]`.
    Upsample,
    /// [`Kind::Upsample`] added to the second input, of the output's shape:
    /// `y[r][c] = x2[r][c] + x[r / sh][c / sw]`.
    Upsampleadd,
    /// Transposed convolution whose windows do not overlap (`kh = sh`,
    /// `kw = sw`, no padding): each input pixel's channels times `w`,
    /// `[ic][kh][kw][oc]`, give its `kh`-by-`kw` block of output pixels. `b`
    /// holds the bias once per tap.
    Convtranspose,
}

/// Kinds by their code in the file; `tools/export_ocr.py` keeps the same order.
pub const KINDS: [Kind; 15] = [
    Kind::Conv,
    Kind::Dwconv,
    Kind::Add,
    Kind::Maxpool,
    Kind::Avgpool,
    Kind::Channelmul,
    Kind::Concatc,
    Kind::Layernorm,
    Kind::Attention,
    Kind::Mean,
    Kind::Ctchead,
    Kind::Ctcargmax,
    Kind::Upsample,
    Kind::Upsampleadd,
    Kind::Convtranspose,
];

/// An activation folded into a convolution's or an addition's tail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fused {
    None,
    Relu,
    /// The logistic function as XNNPACK computes it: see
    /// `kernels::sigmoid_inplace`.
    Sigmoid,
    /// `x * Phi(x)` with the error function.
    Gelu,
    /// `x * sigmoid(x)`.
    Swish,
    /// `clamp(x * 0x1.55555Ap-3 + 0.5, 0, 1)`, PaddleOCR's hard sigmoid.
    Hardsigmoid,
    /// `clamp(x * 0.2 + 0.5, 0, 1)`, the hard sigmoid of the detector's neck.
    Hardsigmoidfifth,
}

/// Activations by their code in the file, in `tools/export_ocr.py`'s order.
pub const FUSED: [Fused; 7] = [
    Fused::None,
    Fused::Relu,
    Fused::Sigmoid,
    Fused::Gelu,
    Fused::Swish,
    Fused::Hardsigmoid,
    Fused::Hardsigmoidfifth,
];

/// A position and length inside the weights, in floats. `[0, 0]` means the
/// operation has none.
#[derive(Clone, Copy, Default, Debug)]
pub struct Block(pub usize, pub usize);

impl Block {
    fn is_empty(self) -> bool {
        self.1 == 0
    }
}

/// One step of the graph.
///
/// Buffers are referred to by index, not by tensor name: one per tensor, a
/// reshape sharing its input's. Which of them share memory is the model's
/// choice.
#[derive(Clone, Debug)]
pub struct Op {
    pub kind: Kind,
    /// Activation folded into this operation's tail.
    pub fused: Fused,
    /// Scratch buffer holding the first input.
    pub x: usize,
    /// Scratch buffer holding the second input of `Add`, `Channelmul`,
    /// `Concatc`, `Upsampleadd` and a two-input `Conv`.
    pub x2: usize,
    /// Scratch buffer the result is written to.
    pub y: usize,
    /// Input height, width and channel count; the sizes follow the input.
    pub ih: usize,
    pub iw: usize,
    pub ic: usize,
    /// Output height, width and channel count.
    pub oh: usize,
    pub ow: usize,
    pub oc: usize,
    /// Kernel height and width.
    pub kh: usize,
    pub kw: usize,
    /// Pixels skipped between kernel placements, vertically and horizontally.
    pub sh: usize,
    pub sw: usize,
    /// Rows added above and below, columns added left and right of the input.
    pub pt: isize,
    pub pb: isize,
    pub pl: isize,
    pub pr: isize,
    /// Channels of the second input of `Concatc` and a two-input `Conv`, the
    /// classes of `Ctcargmax` and the heads of `Attention`.
    pub n2: usize,
    /// Filter weights.
    pub w: Block,
    /// Per-output-channel offset added after the filter.
    pub b: Block,
}

impl Op {
    /// Elements written by this operation.
    pub fn out_len(&self) -> usize {
        self.oh
            .checked_mul(self.ow)
            .and_then(|n| n.checked_mul(self.oc))
            .expect("output shape overflows")
    }

    /// Elements read from the first input.
    pub fn in_len(&self) -> usize {
        self.ih
            .checked_mul(self.iw)
            .and_then(|n| n.checked_mul(self.ic))
            .expect("input shape overflows")
    }

    /// The buffer of the second input, for the operations that read one.
    pub(crate) fn second(&self) -> Option<usize> {
        let binary = matches!(
            self.kind,
            Kind::Add | Kind::Channelmul | Kind::Concatc | Kind::Upsampleadd
        ) || (self.kind == Kind::Conv && self.n2 > 0);
        binary.then_some(self.x2)
    }

    /// True when the whole layer is a single matrix multiply over pixels.
    pub fn is_pointwise(&self) -> bool {
        self.kind == Kind::Conv
            && (self.kh, self.kw, self.sh, self.sw) == (1, 1, 1, 1)
            && [self.pt, self.pb, self.pl, self.pr] == [0; 4]
    }
}

/// Where the image goes in.
#[derive(Clone, Debug)]
pub(crate) struct Input {
    pub(crate) slot: usize,
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) c: usize,
}

/// Where a result comes out.
#[derive(Clone, Debug)]
pub(crate) struct Output {
    pub(crate) slot: usize,
    /// Element count at the current input size.
    pub(crate) n: usize,
}

/// A whole model, ready to be turned into a runnable one.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub(crate) input: Input,
    pub(crate) outputs: Vec<Output>,
    /// Floats each buffer holds at the current size, the most any
    /// operation writes there. Where they lie in memory is the model's
    /// choice (`Model::schedule`).
    pub(crate) pool: Vec<usize>,
    pub(crate) ops: Vec<Op>,
}

impl Plan {
    /// Read and check a plan file.
    pub(crate) fn load(path: &Path) -> Result<(Self, Vec<f32>)> {
        Self::parse(&std::fs::read(path)?)
    }

    /// Split a plan file into its checked description and its weights.
    pub(crate) fn parse(bytes: &[u8]) -> Result<(Self, Vec<f32>)> {
        if bytes.get(..MAGIC.len()) != Some(MAGIC) {
            return Err(Error::Format("not a ppocr-native plan".into()));
        }
        let mut at = MAGIC.len();
        let mut next = || -> Result<usize> {
            let word: [u8; 4] = bytes
                .get(at..at + 4)
                .and_then(|w| w.try_into().ok())
                .ok_or_else(|| Error::Invalid("the description is truncated".into()))?;
            at += 4;
            usize::try_from(u32::from_le_bytes(word))
                .map_err(|_| Error::Invalid("a count does not fit this machine".into()))
        };
        let input = Input {
            slot: next()?,
            h: next()?,
            w: next()?,
            c: next()?,
        };
        let (buffers, outputs, ops) = (next()?, next()?, next()?);
        // Every list is read item by item, so a count larger than the file
        // fails on the first missing word instead of reserving memory.
        let outputs = (0..outputs)
            .map(|_| {
                Ok(Output {
                    slot: next()?,
                    n: 0,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let ops = (0..ops)
            .map(|_| {
                let mut field = [0usize; OP_FIELDS];
                for f in &mut field {
                    *f = next()?;
                }
                op_from_fields(field)
            })
            .collect::<Result<Vec<_>>>()?;
        if buffers == 0 || buffers > ops.len() + 1 {
            return Err(Error::Invalid(format!(
                "{buffers} buffers for {} operations",
                ops.len()
            )));
        }
        let start = at.next_multiple_of(64);
        let raw = bytes
            .get(start..)
            .ok_or_else(|| Error::Invalid("the plan ends before its weights".into()))?;
        let weights = decode_weights(raw)?;
        let (h, w) = (input.h, input.w);
        let mut plan = Self {
            input,
            outputs,
            pool: vec![0; buffers],
            ops,
        };
        plan.derive_shapes(h, w)?;
        plan.check(weights.len())?;
        plan.check_constants(&weights)?;
        Ok((plan, weights))
    }

    fn check_constants(&self, weights: &[f32]) -> Result<()> {
        // Block bounds and lengths have already been validated by check().
        for op in &self.ops {
            if op.kind == Kind::Layernorm && weights[op.w.0 + 2 * op.oc] <= 0.0 {
                return Err(Error::Invalid(
                    "layer normalization epsilon is not positive".into(),
                ));
            }
        }
        Ok(())
    }

    /// Re-derive the plan for an input `height` by `width` pixels: every
    /// operation's shape, the buffer sizes and the output lengths, then
    /// check the result as a whole. On error the plan keeps its size.
    /// Allocates nothing unless it fails.
    pub(crate) fn set_size(
        &mut self,
        height: usize,
        width: usize,
        weight_floats: usize,
    ) -> Result<()> {
        let old = (self.input.h, self.input.w);
        let result = self
            .derive_shapes(height, width)
            .and_then(|()| self.check(weight_floats));
        if result.is_err() {
            self.derive_shapes(old.0, old.1)
                .expect("a loaded plan derives its own size again");
        }
        result
    }

    /// Height, width and channels of what `slot` holds when operation
    /// `before` runs: the output of the last operation before it that wrote
    /// the slot, else the image.
    fn produced(&self, before: usize, slot: usize) -> Option<(usize, usize, usize)> {
        match self.ops[..before].iter().rev().find(|op| op.y == slot) {
            Some(op) => Some((op.oh, op.ow, op.oc)),
            None => (slot == self.input.slot).then_some((self.input.h, self.input.w, self.input.c)),
        }
    }

    /// The size rules, the same along both axes: a window's output follows
    /// from the input, its padding and its stride; upsampling and the
    /// transposed convolution multiply by their factors; a mean is one
    /// pixel; every other operation keeps the size. Channels never change.
    fn derive_shapes(&mut self, height: usize, width: usize) -> Result<()> {
        let at = |i: usize, what: &str| Error::Invalid(format!("op {i} {what}"));
        (self.input.h, self.input.w) = (height, width);
        for i in 0..self.ops.len() {
            let (h, w, c) = self
                .produced(i, self.ops[i].x)
                .ok_or_else(|| at(i, "reads a buffer nothing wrote"))?;
            let op = &self.ops[i];
            if c != op.ic {
                return Err(at(i, "reads a tensor of another shape"));
            }
            // Output length of a window `k` wide sliding by `s` over `n`
            // values padded by `before` and `after`.
            let window = |n: usize, before: isize, after: isize, k: usize, s: usize| {
                usize::try_from(before)
                    .ok()
                    .zip(usize::try_from(after).ok())
                    .and_then(|(before, after)| n.checked_add(before)?.checked_add(after))
                    .filter(|&span| span >= k && s > 0)
                    .map(|span| (span - k) / s + 1)
                    .ok_or_else(|| at(i, "has a window larger than its padded input"))
            };
            let scaled = |n: usize, factor: usize| {
                n.checked_mul(factor)
                    .ok_or_else(|| at(i, "output size overflows"))
            };
            let (oh, ow) = match op.kind {
                Kind::Conv | Kind::Dwconv | Kind::Maxpool | Kind::Avgpool => (
                    window(h, op.pt, op.pb, op.kh, op.sh)?,
                    window(w, op.pl, op.pr, op.kw, op.sw)?,
                ),
                Kind::Upsample | Kind::Upsampleadd => (scaled(h, op.sh)?, scaled(w, op.sw)?),
                Kind::Convtranspose => (scaled(h, op.kh)?, scaled(w, op.kw)?),
                Kind::Mean => (1, 1),
                Kind::Add
                | Kind::Channelmul
                | Kind::Concatc
                | Kind::Layernorm
                | Kind::Attention
                | Kind::Ctchead
                | Kind::Ctcargmax => (h, w),
            };
            let second = match op.kind {
                Kind::Add | Kind::Upsampleadd => Some((oh, ow, op.oc)),
                Kind::Concatc => Some((oh, ow, op.n2)),
                Kind::Channelmul => Some((1, 1, op.oc)),
                Kind::Conv if op.n2 > 0 => Some((h, w, op.n2)),
                _ => None,
            };
            if let Some(want) = second {
                if self.produced(i, op.x2) != Some(want) {
                    return Err(at(i, "reads a second tensor of another shape"));
                }
            }
            let op = &mut self.ops[i];
            (op.ih, op.iw, op.oh, op.ow) = (h, w, oh, ow);
        }
        self.pool.fill(0);
        let image = shape_len(&[height, width, self.input.c])
            .ok_or_else(|| Error::Invalid("input shape is zero or overflows".into()))?;
        let mut grow = |slot: usize, n: usize| match self.pool.get_mut(slot) {
            Some(size) => {
                *size = (*size).max(n);
                Ok(())
            }
            None => Err(Error::Invalid(format!("buffer {slot} does not exist"))),
        };
        grow(self.input.slot, image)?;
        for (i, op) in self.ops.iter().enumerate() {
            let n = shape_len(&[op.oh, op.ow, op.oc])
                .ok_or_else(|| at(i, "output shape is zero or overflows"))?;
            grow(op.y, n)?;
        }
        for k in 0..self.outputs.len() {
            let (h, w, c) = self
                .produced(self.ops.len(), self.outputs[k].slot)
                .ok_or_else(|| Error::Invalid(format!("output {k} is never written")))?;
            self.outputs[k].n = h * w * c;
        }
        Ok(())
    }

    /// Every consistency rule, in one place. `weight_floats` is the length of
    /// the weights.
    pub(crate) fn check(&self, weight_floats: usize) -> Result<()> {
        // The total is checked where the model lays the buffers out.
        if let Some(&n) = self.pool.iter().find(|&&n| n > MAX_SCRATCH_FLOATS) {
            return Err(Error::Invalid(format!(
                "a buffer wants {n} floats, ceiling is {MAX_SCRATCH_FLOATS}"
            )));
        }

        // `what` is a closure so a valid plan is checked without allocating.
        let slot = |i: usize, what: &dyn Fn() -> String| -> Result<usize> {
            self.pool.get(i).copied().ok_or_else(|| {
                Error::Invalid(format!(
                    "{} names buffer {i}, only {} exist",
                    what(),
                    self.pool.len()
                ))
            })
        };

        let input_len = shape_len(&[self.input.h, self.input.w, self.input.c])
            .ok_or_else(|| Error::Invalid("input shape is zero or overflows".into()))?;
        if slot(self.input.slot, &|| "the input".into())? < input_len {
            return Err(Error::Invalid(
                "the input buffer is smaller than the image".into(),
            ));
        }
        for (k, out) in self.outputs.iter().enumerate() {
            if slot(out.slot, &|| "an output".into())? < out.n {
                return Err(Error::Invalid(format!(
                    "output {k} claims {} elements, its buffer holds fewer",
                    out.n
                )));
            }
        }

        for (i, op) in self.ops.iter().enumerate() {
            let at = |what: &str| format!("op {i} ({:?}) {what}", op.kind);
            let input_len = shape_len(&[op.ih, op.iw, op.ic])
                .ok_or_else(|| Error::Invalid(at("input shape is zero or overflows")))?;
            shape_len(&[op.oh, op.ow, op.oc])
                .ok_or_else(|| Error::Invalid(at("output shape is zero or overflows")))?;
            match op.kind {
                Kind::Conv | Kind::Dwconv | Kind::Avgpool | Kind::Maxpool => {
                    // The execution loops use signed coordinates. Bound every
                    // coordinate and checked product before reaching any SIMD
                    // path.
                    if [op.kh, op.kw, op.sh, op.sw]
                        .iter()
                        .any(|&n| n == 0 || n > MAX_WINDOW)
                        || [op.pt, op.pb, op.pl, op.pr]
                            .iter()
                            .any(|&n| n < 0 || n as usize > MAX_WINDOW)
                    {
                        return Err(Error::Invalid(at(
                            "has an unsupported kernel, stride or padding",
                        )));
                    }
                    for (out, stride, kernel) in [(op.oh, op.sh, op.kh), (op.ow, op.sw, op.kw)] {
                        let extent = out.checked_mul(stride).and_then(|n| n.checked_add(kernel));
                        if extent.is_none_or(|n| n > isize::MAX as usize) {
                            return Err(Error::Invalid(at("window coordinates overflow")));
                        }
                    }
                    if op.sw.checked_mul(op.ic).is_none() {
                        return Err(Error::Invalid(at("pixel stride overflows")));
                    }
                }
                Kind::Mean => {
                    if (op.oh, op.ow) != (1, 1) || op.ic != op.oc {
                        return Err(Error::Invalid(at("invalid channel mean")));
                    }
                }
                Kind::Concatc => {
                    if (op.ih, op.iw) != (op.oh, op.ow) || op.ic.checked_add(op.n2) != Some(op.oc) {
                        return Err(Error::Invalid(at("invalid channel concatenation")));
                    }
                }
                Kind::Ctchead | Kind::Ctcargmax => {
                    let vocab = if op.kind == Kind::Ctchead {
                        op.oc
                    } else {
                        op.n2
                    };
                    let columns = vocab.div_ceil(16) * 16;
                    if op.ih != 1
                        || op.oh != 1
                        || op.iw != op.ow
                        || vocab == 0
                        || (op.kind == Kind::Ctcargmax && op.oc != 2)
                        || columns.checked_mul(op.ic) != Some(op.w.1)
                        || op.b.1 != columns
                    {
                        return Err(Error::Invalid(at("invalid CTC head")));
                    }
                }
                Kind::Layernorm => {
                    if op.oc.checked_mul(2).and_then(|n| n.checked_add(1)) != Some(op.w.1) {
                        return Err(Error::Invalid(at("needs gamma, beta and epsilon")));
                    }
                }
                Kind::Attention => {
                    if op.ih != 1
                        || op.oh != 1
                        || op.iw != op.ow
                        || op.oc.checked_mul(3) != Some(op.ic)
                        || op.n2 == 0
                        || !op.oc.is_multiple_of(op.n2)
                        || op.w.1 != 1
                        || op
                            .iw
                            .checked_mul(op.iw)
                            .is_none_or(|n| n > MAX_SCRATCH_FLOATS)
                    {
                        return Err(Error::Invalid(at("invalid attention shape")));
                    }
                }
                Kind::Upsample | Kind::Upsampleadd => {
                    if op.ic != op.oc
                        || op.sh == 0
                        || op.sw == 0
                        || op.ih.checked_mul(op.sh) != Some(op.oh)
                        || op.iw.checked_mul(op.sw) != Some(op.ow)
                    {
                        return Err(Error::Invalid(at("invalid upsampling")));
                    }
                }
                Kind::Convtranspose => {
                    let columns = op.kh.checked_mul(op.kw).and_then(|n| n.checked_mul(op.oc));
                    if op.kh == 0
                        || op.kw == 0
                        || (op.sh, op.sw) != (op.kh, op.kw)
                        || [op.pt, op.pb, op.pl, op.pr] != [0; 4]
                        || op.ih.checked_mul(op.kh) != Some(op.oh)
                        || op.iw.checked_mul(op.kw) != Some(op.ow)
                        || columns.and_then(|n| n.checked_mul(op.ic)) != Some(op.w.1)
                        || (!op.b.is_empty() && Some(op.b.1) != columns)
                    {
                        return Err(Error::Invalid(at("invalid transposed convolution")));
                    }
                }
                Kind::Add | Kind::Channelmul => {}
            }
            if matches!(op.kind, Kind::Add | Kind::Channelmul | Kind::Layernorm)
                && (op.ih, op.iw, op.ic) != (op.oh, op.ow, op.oc)
            {
                return Err(Error::Invalid(at("elementwise input/output shapes differ")));
            }
            if op.kind == Kind::Avgpool
                && (op.ic != op.oc
                    || [op.pt, op.pb, op.pl, op.pr] != [0; 4]
                    || op.kh > op.ih
                    || op.kw > op.iw
                    || op.oh != (op.ih - op.kh) / op.sh + 1
                    || op.ow != (op.iw - op.kw) / op.sw + 1)
            {
                return Err(Error::Invalid(at(
                    "average pool requires complete valid windows",
                )));
            }
            if op.kind == Kind::Maxpool {
                if op.ic != op.oc || op.pt != 0 || op.pl != 0 {
                    return Err(Error::Invalid(at(
                        "maxpool changes channels or requests unsupported padding",
                    )));
                }
                if (op.oh - 1) * op.sh >= op.ih || (op.ow - 1) * op.sw >= op.iw {
                    return Err(Error::Invalid(at(
                        "maxpool contains an empty output window",
                    )));
                }
            }
            if op.is_pointwise() && (op.ih != op.oh || op.iw != op.ow) {
                return Err(Error::Invalid(at(
                    "unit-stride pointwise changes spatial dimensions",
                )));
            }
            if slot(op.x, &|| at("input"))? < input_len {
                return Err(Error::Invalid(at("reads past its input buffer")));
            }
            if slot(op.y, &|| at("output"))? < op.out_len() {
                return Err(Error::Invalid(at("writes past its output buffer")));
            }
            // The executor holds a mutable view of the output buffer and a
            // shared view of the inputs at the same time. That is only sound
            // because no operation writes a buffer it is also reading.
            if op.y == op.x || op.second() == Some(op.y) {
                return Err(Error::Invalid(at("writes the buffer it reads")));
            }
            if op.second().is_some() {
                let need = match op.kind {
                    Kind::Add | Kind::Upsampleadd => op.out_len(),
                    Kind::Concatc => op.oh * op.ow * op.n2,
                    Kind::Conv => shape_len(&[op.ih, op.iw, op.n2])
                        .ok_or_else(|| Error::Invalid(at("second input overflows")))?,
                    _ => op.oc,
                };
                if slot(op.x2, &|| at("second input"))? < need {
                    return Err(Error::Invalid(at("reads past its second input buffer")));
                }
            }
            if matches!(op.kind, Kind::Conv | Kind::Dwconv) {
                let taps = op
                    .kh
                    .checked_mul(op.kw)
                    .ok_or_else(|| Error::Invalid(at("kernel area overflows")))?;
                if op.n2 > 0 && (op.kind == Kind::Dwconv || op.is_pointwise()) {
                    return Err(Error::Invalid(at("reads a second input it cannot")));
                }
                let want = if op.kind == Kind::Dwconv {
                    if op.ic != op.oc {
                        return Err(Error::Invalid(at("is depthwise but changes channel count")));
                    }
                    taps.checked_mul(op.oc)
                        .ok_or_else(|| Error::Invalid(at("weight shape overflows")))?
                } else {
                    op.ic
                        .checked_add(op.n2)
                        .and_then(|n| n.checked_mul(taps))
                        .and_then(|n| n.checked_mul(op.oc))
                        .ok_or_else(|| Error::Invalid(at("weight shape overflows")))?
                };
                if op.w.1 != want {
                    return Err(Error::Invalid(format!(
                        "{}: {} weights for a shape needing {want}",
                        at("weights"),
                        op.w.1
                    )));
                }
                if !op.b.is_empty() && op.b.1 != op.oc {
                    return Err(Error::Invalid(at("has a bias of the wrong length")));
                }
            }
            let weighted = matches!(
                op.kind,
                Kind::Conv
                    | Kind::Dwconv
                    | Kind::Layernorm
                    | Kind::Attention
                    | Kind::Ctchead
                    | Kind::Ctcargmax
                    | Kind::Convtranspose
            );
            if !weighted && !(op.w.is_empty() && op.b.is_empty()) {
                return Err(Error::Invalid(at("has weights it does not use")));
            }
            if op.fused != Fused::None
                && !matches!(
                    op.kind,
                    Kind::Conv | Kind::Dwconv | Kind::Add | Kind::Convtranspose
                )
            {
                return Err(Error::Invalid(at("has an invalid folded activation")));
            }
            for (block, what) in [(op.w, "weights"), (op.b, "bias")] {
                if block.is_empty() {
                    continue;
                }
                let end = block.0.checked_add(block.1).ok_or_else(|| {
                    Error::Invalid(at(&format!("{what} block overflows an address")))
                })?;
                if end > weight_floats {
                    return Err(Error::Invalid(format!(
                        "{}: reaches float {end} of {weight_floats}",
                        at(what)
                    )));
                }
            }
        }
        Ok(())
    }
}

/// One operation record, in the field order of the module documentation.
/// Spatial sizes stay zero until `derive_shapes` sets them.
fn op_from_fields(f: [usize; OP_FIELDS]) -> Result<Op> {
    let kind = *KINDS
        .get(f[0])
        .ok_or_else(|| Error::Format(format!("operation kind {} is unknown", f[0])))?;
    let fused = *FUSED
        .get(f[1])
        .ok_or_else(|| Error::Format(format!("activation {} is unknown", f[1])))?;
    // `check` bounds the padding; here it only has to fit.
    let pad =
        |v: usize| isize::try_from(v).map_err(|_| Error::Invalid("padding does not fit".into()));
    Ok(Op {
        kind,
        fused,
        x: f[2],
        x2: f[3],
        y: f[4],
        ih: 0,
        iw: 0,
        ic: f[5],
        oh: 0,
        ow: 0,
        oc: f[6],
        kh: f[7],
        kw: f[8],
        sh: f[9],
        sw: f[10],
        pt: pad(f[11])?,
        pb: pad(f[12])?,
        pl: pad(f[13])?,
        pr: pad(f[14])?,
        n2: f[15],
        w: Block(f[16], f[17]),
        b: Block(f[18], f[19]),
    })
}

fn shape_len(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(
        1usize,
        |n, &dim| if dim == 0 { None } else { n.checked_mul(dim) },
    )
}

/// Turn the raw weights into floats, rejecting a truncated block and any
/// value that would poison every later result.
fn decode_weights(raw: &[u8]) -> Result<Vec<f32>> {
    if !raw.len().is_multiple_of(4) {
        return Err(Error::Invalid(format!(
            "the weights are {} bytes, not a whole number of floats",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*chunk));
    }
    if let Some(pos) = out.iter().position(|v| !v.is_finite()) {
        return Err(Error::Invalid(format!(
            "the weights hold {} at float {pos}",
            out[pos]
        )));
    }
    Ok(out)
}

/// A plan file for the tests, laid out as `tools/export_ocr.py` writes one:
/// `input` is the buffer, height, width and channels; sizes in `ops` are
/// ignored and derived again on load.
#[cfg(test)]
pub(crate) fn encode(
    input: (usize, usize, usize, usize),
    buffers: usize,
    outputs: &[usize],
    ops: &[Op],
    weights: &[f32],
) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let mut put = |v: usize| out.extend_from_slice(&u32::try_from(v).unwrap().to_le_bytes());
    let (slot, h, w, c) = input;
    for v in [slot, h, w, c, buffers, outputs.len(), ops.len()] {
        put(v);
    }
    for &slot in outputs {
        put(slot);
    }
    for op in ops {
        let kind = KINDS.iter().position(|&k| k == op.kind).unwrap();
        let fused = FUSED.iter().position(|&f| f == op.fused).unwrap();
        let pad = |p: isize| usize::try_from(p).unwrap();
        for v in [
            kind,
            fused,
            op.x,
            op.x2,
            op.y,
            op.ic,
            op.oc,
            op.kh,
            op.kw,
            op.sh,
            op.sw,
            pad(op.pt),
            pad(op.pb),
            pad(op.pl),
            pad(op.pr),
            op.n2,
            op.w.0,
            op.w.1,
            op.b.0,
            op.b.1,
        ] {
            put(v);
        }
    }
    out.resize(out.len().next_multiple_of(64), 0);
    for v in weights {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// An operation of `kind` from buffer `x` to `y`, `ic` to `oc` channels,
/// with no window, weights or activation: tests set what they need.
#[cfg(test)]
pub(crate) fn op(kind: Kind, x: usize, y: usize, ic: usize, oc: usize) -> Op {
    Op {
        kind,
        fused: Fused::None,
        x,
        x2: 0,
        y,
        ih: 0,
        iw: 0,
        ic,
        oh: 0,
        ow: 0,
        oc,
        kh: 0,
        kw: 0,
        sh: 0,
        sw: 0,
        pt: 0,
        pb: 0,
        pl: 0,
        pr: 0,
        n2: 0,
        w: Block::default(),
        b: Block::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1x1 convolution, 4 channels to 4, over a 2x2 image.
    fn pointwise() -> Op {
        let mut conv = op(Kind::Conv, 0, 1, 4, 4);
        (conv.kh, conv.kw, conv.sh, conv.sw) = (1, 1, 1, 1);
        (conv.w, conv.b) = (Block(0, 16), Block(16, 4));
        conv
    }

    fn file(ops: &[Op], weights: usize) -> Vec<u8> {
        let values: Vec<f32> = (0..weights).map(|i| i as f32).collect();
        encode((0, 2, 2, 4), 2, &[1], ops, &values)
    }

    #[test]
    fn plan_file_round_trips() {
        let (plan, weights) = Plan::parse(&file(&[pointwise()], 20)).unwrap();
        let op = &plan.ops[0];
        assert_eq!((op.ih, op.iw, op.oh, op.ow), (2, 2, 2, 2));
        assert_eq!(
            (plan.pool.as_slice(), plan.outputs[0].n),
            (&[16, 16][..], 16)
        );
        assert_eq!((weights.len(), weights[3]), (20, 3.0));
    }

    #[test]
    fn malformed_files_refused() {
        let good = file(&[pointwise()], 20);
        // Seven header words and one output, then the operation's fields.
        let header = MAGIC.len() + 4 * 8;
        let mut cases: Vec<(&str, Vec<u8>)> = vec![
            ("not a plan", b"not a plan at all, really".to_vec()),
            ("truncated description", good[..header + 10].to_vec()),
            ("no weights", good[..128].to_vec()),
            ("truncated weights", good[..good.len() - 2].to_vec()),
        ];
        let mut edit = |what, at: usize, v: u32| {
            let mut bytes = good.clone();
            bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
            cases.push((what, bytes));
        };
        let field = |k: usize| header + 4 * k;
        edit("unknown kind", field(0), 99);
        edit("unknown activation", field(1), 99);
        edit("missing buffer", field(4), 7);
        edit("weights past the end", field(16), 8);
        edit("weights the shape contradicts", field(17), 12);
        edit("a zero-sized image", MAGIC.len() + 4, 0);
        edit("no buffers", MAGIC.len() + 16, 0);
        edit(
            "an operation count past the file",
            MAGIC.len() + 24,
            u32::MAX,
        );
        let mut nan = good.clone();
        let end = nan.len();
        nan[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        cases.push(("a weight that is not a number", nan));
        for (what, bytes) in cases {
            assert!(Plan::parse(&bytes).is_err(), "{what} was accepted");
        }
    }

    #[test]
    fn malformed_operations_refused() {
        type Edit = dyn Fn(&mut Op);
        let bad: [(&str, &Edit); 9] = [
            ("zero input channels", &|op| op.ic = 0),
            ("zero stride", &|op| op.sw = 0),
            ("padding past the window limit", &|op| op.pl = 1 << 20),
            ("bias of the wrong length", &|op| op.b = Block(16, 3)),
            ("output over its input", &|op| op.y = 0),
            ("folded activation on a mean", &|op| {
                (op.kind, op.fused) = (Kind::Mean, Fused::Relu);
                (op.w, op.b) = (Block::default(), Block::default());
            }),
            ("unused weights on an addition", &|op| op.kind = Kind::Add),
            ("second input on a pointwise layer", &|op| op.n2 = 4),
            ("overlapping transposed windows", &|op| {
                (op.kind, op.kh, op.kw, op.sh, op.sw) = (Kind::Convtranspose, 2, 2, 1, 1);
                (op.w, op.b) = (Block(0, 64), Block::default());
            }),
        ];
        for (what, edit) in bad {
            let mut op = pointwise();
            edit(&mut op);
            assert!(
                Plan::parse(&file(&[op], 80)).is_err(),
                "{what} was accepted"
            );
        }
    }

    #[test]
    fn sizes_follow_the_input() {
        // A padded stride-2 3x3 convolution, then a 2x upsampling.
        let mut stem = op(Kind::Conv, 0, 1, 3, 4);
        (stem.kh, stem.kw, stem.sh, stem.sw) = (3, 3, 2, 2);
        (stem.pt, stem.pb, stem.pl, stem.pr) = (1, 1, 1, 1);
        (stem.w, stem.b) = (Block(0, 108), Block(108, 4));
        let mut up = op(Kind::Upsample, 1, 0, 4, 4);
        (up.sh, up.sw) = (2, 2);
        let bytes = encode((0, 8, 8, 3), 2, &[0], &[stem, up], &[0.5; 112]);
        let (mut plan, weights) = Plan::parse(&bytes).unwrap();
        plan.set_size(6, 10, weights.len()).unwrap();
        let shapes: Vec<_> = plan.ops.iter().map(|op| (op.oh, op.ow)).collect();
        assert_eq!(shapes, [(3, 5), (6, 10)]);
        assert_eq!(
            (plan.outputs[0].n, plan.pool.as_slice()),
            (240, &[240, 60][..])
        );
        // Past the scratch ceiling: refused, and the size stays.
        assert!(plan.set_size(1 << 20, 1 << 20, weights.len()).is_err());
        assert_eq!((plan.input.h, plan.input.w, plan.ops[1].ow), (6, 10, 10));
    }
}
