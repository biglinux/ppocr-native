//! Convolution and pooling over whole tensors: each driver hands groups of
//! rows to the kernels in `super::kernels`, takes the borders a tap at a
//! time, and applies the layer's folded activation while a group is still
//! in cache.

use super::kernels;
use super::layers;
use super::plan::{Block, Fused, Op};

/// Whether a convolution goes through `kernels::conv_row`: every one with a
/// window, except RGB stems, whose register tiles in [`convolve`] keep three
/// channels per tap.
pub(super) fn reads_windows(op: &Op) -> bool {
    !op.is_pointwise() && (op.ic != 3 || op.n2 > 0)
}

pub(super) fn opt(b: Block, weights: &[f32]) -> Option<&[f32]> {
    (b.1 > 0).then(|| &weights[b.0..b.0 + b.1])
}

/// Smallest pixel run a pointwise call is given.
const GROUP_PIXELS: usize = 48;

/// Output rows per pointwise call: enough for [`GROUP_PIXELS`]. Short rows
/// are too short for the 12-pixel tiles, while whole large images would fall
/// out of cache before the folded activation or the paired addition reads
/// them.
pub(super) fn rows_per_group(op: &Op) -> usize {
    GROUP_PIXELS.div_ceil(op.ow).min(op.oh)
}

/// The activation folded into an operation's tail, applied to each finished
/// run of outputs while it is still in cache. A ReLU is the clamp to
/// `[0, inf]`, which the convolution kernels can apply as they store.
#[derive(Clone, Copy)]
pub(super) struct Tail {
    clip: Option<(f32, f32)>,
    /// What runs after the clamp; never `Relu`.
    fused: Fused,
}

impl Tail {
    pub(super) fn of(op: &Op) -> Self {
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
    pub(super) fn fold_clip(self) -> (Option<(f32, f32)>, Self) {
        (self.clip, Self { clip: None, ..self })
    }

    pub(super) fn apply(self, y: &mut [f32]) {
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
pub(super) fn add_with_tail(y: &mut [f32], x1: &[f32], x2: &[f32], n: usize, tail: Tail) {
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
pub(super) fn convolve(
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
pub(super) fn depthwise(
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
pub(super) fn maxpool(op: &Op, x: &[f32], y: &mut [f32]) {
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

/// [`maxpool`] writing over its input, at stride 1 with the input's size:
/// every window starts at its own output pixel and reaches only down and
/// right, so going forwards each pixel is written after the last window that
/// reads it. The taps are taken in the order [`maxpool`] takes them.
pub(super) fn maxpool_in_place(op: &Op, y: &mut [f32]) {
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
