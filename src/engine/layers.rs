//! The layers around the convolutions: channel concatenation, the channel
//! mean, average pooling, layer normalization, multi-head attention, the CTC
//! head, nearest upsampling, the transposed convolution, and the GELU, swish
//! and hard-sigmoid tails.
//!
//! The tiers follow `kernels`: tier 3 has a scalar definition with fused
//! multiply-adds, tiers 0-2 share one without, and every vector kernel gives
//! its tier's bits. Reductions run in [`LANES`] lanes on every tier, element
//! `j` of a run in lane `j % 8`, as one AVX register or two SSE registers
//! hold them. `exp` is `kernels::exp_neg`; GELU's normal distribution
//! function is a rational fit (absolute error 7e-9 before rounding).

use super::kernels::{self, exp_neg, madd};
#[cfg(target_arch = "x86_64")]
use super::kernels::{exp2_bits_avx, exp2_bits_avx2, exp2_bits_sse, simd_tier};
use super::plan::Op;

/// Where the normal distribution function is 0 or 1 to `f32` precision.
const PHI_BOUND: f32 = 5.5;
/// `Phi(x) - 1/2 = x P(x^2) / Q(x^2)` on `[-5.5, 5.5]`: a minimax rational
/// fit of degree 5 over 5 in `x^2`, from `P[0]` up, `Q` from its `x^2`
/// term up after a constant 1. Every coefficient is positive, so Horner's
/// rule adds without cancelling.
const PHI_P: [f32; 6] = [
    f32::from_bits(0x3ecc_4228), // 0.39894223
    f32::from_bits(0x3d0a_a93c), // 0.0338528
    f32::from_bits(0x3b99_43c9), // 0.00467727
    f32::from_bits(0x3931_0738), // 1.6882725e-4
    f32::from_bits(0x36d5_2462), // 6.35213e-6
    f32::from_bits(0x32c3_6a3c), // 2.2749298e-8
];
const PHI_Q: [f32; 5] = [
    f32::from_bits(0x3e80_c788), // 0.2515223
    f32::from_bits(0x3cea_aaee), // 0.028645959
    f32::from_bits(0x3af6_ffc6), // 0.0018844537
    f32::from_bits(0x389a_737a), // 7.3648014e-5
    f32::from_bits(0x35a0_b9e3), // 1.1975029e-6
];
/// PaddleOCR's hard-sigmoid slope as the ONNX export stores it.
const HARDSIGMOID_ALPHA: f32 = f32::from_bits(0x3e2a_aaad);
/// The slope of the detector neck's hard sigmoid, 0.2.
const HARDSIGMOID_FIFTH: f32 = 0.2;

/// Lanes of every reduction, whatever the tier's register width.
const LANES: usize = 8;

/// `x Phi(x)`, the normal distribution function from [`PHI_P`] and
/// [`PHI_Q`] at `x` clamped to [`PHI_BOUND`], itself clamped to `[0, 1]`.
/// Eleven multiply-adds and one division; the error-function form took an
/// exponential besides, and twice the arithmetic.
#[inline(always)]
fn gelu_one<const FMA: bool>(x: f32) -> f32 {
    let c = x.clamp(-PHI_BOUND, PHI_BOUND);
    let s = c * c;
    let mut p = PHI_P[5];
    for &a in PHI_P[..5].iter().rev() {
        p = madd::<FMA>(p, s, a);
    }
    let mut q = PHI_Q[4];
    for &a in PHI_Q[..4].iter().rev() {
        q = madd::<FMA>(q, s, a);
    }
    q = madd::<FMA>(q, s, 1.0);
    x * madd::<FMA>(c, p / q, 0.5).clamp(0.0, 1.0)
}

#[inline(always)]
fn swish_one<const FMA: bool>(x: f32) -> f32 {
    x * kernels::sigmoid_one::<FMA>(x)
}

pub(crate) fn gelu_inplace(y: &mut [f32]) {
    activation_inplace::<true>(y);
}

pub(crate) fn swish_inplace(y: &mut [f32]) {
    activation_inplace::<false>(y);
}

/// GELU, or swish when `GELU` is false.
fn activation_inplace<const GELU: bool>(y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let (p, n) = (y.as_mut_ptr(), y.len());
        // SAFETY: the kernels read and write `y`'s `n` floats, each value
        // before its own store; each instruction set was detected.
        unsafe {
            match simd_tier() {
                3 => return activation_avx2::<GELU>(p, n),
                2 => return activation_avx::<GELU>(p, n),
                1 => return activation_sse::<GELU>(p, n),
                _ => {}
            }
        }
    }
    for v in y {
        *v = if GELU {
            gelu_one::<false>(*v)
        } else {
            swish_one::<false>(*v)
        };
    }
}

/// [`gelu_one`] or [`swish_one`] on an array of vectors, each step for all
/// of them before the next, as `vexp_neg_each!` does.
#[cfg(target_arch = "x86_64")]
macro_rules! activation_each {
    ($x:expr, $gelu:expr, $lanes:tt, $fma:tt, $exp2:ident) => {{
        let x = $x;
        let (sign, one) = (vsplat!($lanes, -0.0), vsplat!($lanes, 1.0));
        let mut y = x;
        if $gelu {
            let (lo, hi) = (vsplat!($lanes, -PHI_BOUND), vsplat!($lanes, PHI_BOUND));
            let (mut c, mut s, mut p, mut q) = (x, x, x, x);
            for j in 0..x.len() {
                c[j] = vclip!($lanes, x[j], lo, hi);
                s[j] = vmul!($lanes, c[j], c[j]);
                p[j] = vsplat!($lanes, PHI_P[5]);
                q[j] = vsplat!($lanes, PHI_Q[4]);
            }
            for &a in PHI_P[..5].iter().rev() {
                for j in 0..x.len() {
                    p[j] = vfma!($lanes, $fma, p[j], s[j], vsplat!($lanes, a));
                }
            }
            for &a in PHI_Q[..4].iter().rev().chain(&[1.0]) {
                for j in 0..x.len() {
                    q[j] = vfma!($lanes, $fma, q[j], s[j], vsplat!($lanes, a));
                }
            }
            let (zero, half) = (vsplat!($lanes, 0.0), vsplat!($lanes, 0.5));
            for j in 0..x.len() {
                let phi = vfma!($lanes, $fma, c[j], vdiv!($lanes, p[j], q[j]), half);
                y[j] = vmul!($lanes, x[j], vclip!($lanes, phi, zero, one));
            }
        } else {
            for j in 0..x.len() {
                y[j] = vor!($lanes, x[j], sign);
            }
            let e = vexp_neg_each!($lanes, $fma, $exp2, y);
            for j in 0..x.len() {
                let f = vdiv!($lanes, e[j], vadd!($lanes, e[j], one));
                y[j] = vmul!(
                    $lanes,
                    x[j],
                    vblendv!($lanes, vsub!($lanes, one, f), f, x[j])
                );
            }
        }
        y
    }};
}

/// [`gelu_one`] or [`swish_one`] on three vectors at a time, then one, then
/// one value at a time. A GELU vector is two Horner chains and a division:
/// three in flight hide more of their latency than two, despite three
/// spills; four spilled and measured slower.
#[cfg(target_arch = "x86_64")]
macro_rules! activation_body {
    ($p:expr, $n:expr, $gelu:expr, $lanes:tt, $fma:tt, $exp2:ident) => {{
        // SAFETY: the caller's contract.
        unsafe {
            let (p, n) = ($p, $n);
            let mut i = 0;
            while i + 3 * $lanes <= n {
                let x: [_; 3] = std::array::from_fn(|j| vld!($lanes, p.add(i + j * $lanes)));
                let y = activation_each!(x, $gelu, $lanes, $fma, $exp2);
                for (j, y) in y.into_iter().enumerate() {
                    vst!($lanes, p.add(i + j * $lanes), y);
                }
                i += 3 * $lanes;
            }
            while i + $lanes <= n {
                let y = activation_each!([vld!($lanes, p.add(i))], $gelu, $lanes, $fma, $exp2);
                vst!($lanes, p.add(i), y[0]);
                i += $lanes;
            }
            while i < n {
                let v = *p.add(i);
                *p.add(i) = if $gelu {
                    gelu_one::<$fma>(v)
                } else {
                    swish_one::<$fma>(v)
                };
                i += 1;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn activation_avx2<const GELU: bool>(p: *mut f32, n: usize) {
    activation_body!(p, n, GELU, 8, true, exp2_bits_avx2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn activation_avx<const GELU: bool>(p: *mut f32, n: usize) {
    activation_body!(p, n, GELU, 8, false, exp2_bits_avx)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn activation_sse<const GELU: bool>(p: *mut f32, n: usize) {
    activation_body!(p, n, GELU, 4, false, exp2_bits_sse)
}

/// `clamp(x * alpha + 0.5, 0, 1)`: one product, one sum and a clamp, the
/// same bits on every tier. It runs on squeeze-and-excite vectors of a few
/// hundred values.
pub(crate) fn hardsigmoid_inplace(y: &mut [f32]) {
    for v in y {
        *v = (*v * HARDSIGMOID_ALPHA + 0.5).clamp(0.0, 1.0);
    }
}

/// [`hardsigmoid_inplace`] with the slope 0.2.
pub(crate) fn hardsigmoid_fifth_inplace(y: &mut [f32]) {
    for v in y {
        *v = (*v * HARDSIGMOID_FIFTH + 0.5).clamp(0.0, 1.0);
    }
}

/// What MAXPS keeps: `a` only when it is greater, `b` for NaN or equal zeros.
#[inline(always)]
fn max_lane(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// The lanes, then the values past the last whole lane group.
fn largest(lanes: &[f32; LANES], tail: &[f32]) -> f32 {
    lanes
        .iter()
        .chain(tail)
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
}

fn lane_sum(lanes: &[f32; LANES]) -> f32 {
    lanes.iter().fold(0.0, |sum, &v| sum + v)
}

/// Softmax of one row in place: subtract the maximum, exponentiate, then
/// multiply by the reciprocal of the sum. Maximum and sum run in lanes over
/// the whole groups of eight, then over the rest one value at a time.
pub(crate) fn softmax_row(y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let (p, n) = (y.as_mut_ptr(), y.len());
        // SAFETY: the kernels stay inside `y`'s `n` floats; each instruction
        // set was detected.
        unsafe {
            match simd_tier() {
                3 => return softmax_avx2(p, n),
                2 => return softmax_avx(p, n),
                1 => return softmax_sse(p, n),
                _ => {}
            }
        }
    }
    softmax_scalar::<false>(y);
}

/// [`softmax_row`]'s definition for one tier, lane by lane.
fn softmax_scalar<const FMA: bool>(y: &mut [f32]) {
    let body = y.len() / LANES * LANES;
    let mut lanes = [f32::NEG_INFINITY; LANES];
    for (j, &v) in y[..body].iter().enumerate() {
        lanes[j % LANES] = max_lane(lanes[j % LANES], v);
    }
    let max = largest(&lanes, &y[body..]);
    let mut sums = [0.0; LANES];
    for (j, v) in y[..body].iter_mut().enumerate() {
        *v = exp_neg::<FMA>(*v - max);
        sums[j % LANES] += *v;
    }
    let mut sum = lane_sum(&sums);
    for v in &mut y[body..] {
        *v = exp_neg::<FMA>(*v - max);
        sum += *v;
    }
    let r = 1.0 / sum;
    for v in y {
        *v *= r;
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! softmax_body {
    ($p:expr, $n:expr, $lanes:tt, $fma:tt, $exp2:ident) => {{
        // SAFETY: the caller's contract; every access is below `n`.
        unsafe {
            let (p, n) = ($p, $n);
            let body = n / LANES * LANES;
            let mut m = [vsplat!($lanes, f32::NEG_INFINITY); LANES / $lanes];
            let mut i = 0;
            while i < body {
                for (h, m) in m.iter_mut().enumerate() {
                    *m = vmax!($lanes, *m, vld!($lanes, p.add(i + h * $lanes)));
                }
                i += LANES;
            }
            let mut lanes = [0.0f32; LANES];
            for (h, &m) in m.iter().enumerate() {
                vst!($lanes, lanes.as_mut_ptr().add(h * $lanes), m);
            }
            let max = largest(&lanes, std::slice::from_raw_parts(p.add(body), n - body));
            let top = vsplat!($lanes, max);
            let mut s = [vsplat!($lanes, 0.0); LANES / $lanes];
            i = 0;
            while i < body {
                for (h, s) in s.iter_mut().enumerate() {
                    let at = p.add(i + h * $lanes);
                    let e = vexp_neg!($lanes, $fma, $exp2, vsub!($lanes, vld!($lanes, at), top));
                    vst!($lanes, at, e);
                    *s = vadd!($lanes, *s, e);
                }
                i += LANES;
            }
            for (h, &s) in s.iter().enumerate() {
                vst!($lanes, lanes.as_mut_ptr().add(h * $lanes), s);
            }
            let mut sum = lane_sum(&lanes);
            for j in body..n {
                let e = exp_neg::<$fma>(*p.add(j) - max);
                *p.add(j) = e;
                sum += e;
            }
            let r = 1.0 / sum;
            let scale = vsplat!($lanes, r);
            i = 0;
            while i + $lanes <= n {
                vst!(
                    $lanes,
                    p.add(i),
                    vmul!($lanes, vld!($lanes, p.add(i)), scale)
                );
                i += $lanes;
            }
            while i < n {
                *p.add(i) *= r;
                i += 1;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_avx2(p: *mut f32, n: usize) {
    softmax_body!(p, n, 8, true, exp2_bits_avx2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn softmax_avx(p: *mut f32, n: usize) {
    softmax_body!(p, n, 8, false, exp2_bits_avx)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn softmax_sse(p: *mut f32, n: usize) {
    softmax_body!(p, n, 4, false, exp2_bits_sse)
}

/// Each pixel's channels, then the second input's.
pub(crate) fn concat_channels(op: &Op, x: &[f32], x2: &[f32], y: &mut [f32]) {
    let (a, b) = (op.ic, op.n2);
    for p in 0..op.oh * op.ow {
        y[p * op.oc..][..a].copy_from_slice(&x[p * a..][..a]);
        y[p * op.oc + a..][..b].copy_from_slice(&x2[p * b..][..b]);
    }
}

/// Channels whose sums [`mean`] keeps at once, and the most a producer can
/// sum as it writes, see [`MeanSums`].
pub(crate) const MEAN_CHANNELS: usize = 512;

/// Each channel's mean over every pixel: the pixels summed in order in
/// `f64`, then divided by their count. A page's detector maps hold over a
/// hundred thousand pixels, and an `f32` running sum there drifts by parts
/// in a thousand. One sum per channel, so every tier agrees.
pub(crate) fn mean(op: &Op, x: &[f32], y: &mut [f32]) {
    let (c, pixels) = (op.ic, op.ih * op.iw);
    let mut sums = [0.0f64; MEAN_CHANNELS];
    for first in (0..c).step_by(MEAN_CHANNELS) {
        let sums = &mut sums[..MEAN_CHANNELS.min(c - first)];
        sums.fill(0.0);
        for pixel in x[..pixels * c].chunks_exact(c) {
            for (sum, &v) in sums.iter_mut().zip(&pixel[first..]) {
                *sum += f64::from(v);
            }
        }
        for (out, &sum) in y[first..].iter_mut().zip(sums.iter()) {
            *out = (sum / pixels as f64) as f32;
        }
    }
}

/// [`mean`] summed by the layer that writes its input, run by run of whole
/// pixels as they are finished, while they are still in cache: the same
/// sums in the same order, for at most [`MEAN_CHANNELS`] channels.
pub(crate) struct MeanSums {
    sums: [f64; MEAN_CHANNELS],
    c: usize,
}

impl MeanSums {
    pub(crate) fn new(c: usize) -> Self {
        assert!(c <= MEAN_CHANNELS);
        Self {
            sums: [0.0; MEAN_CHANNELS],
            c,
        }
    }

    pub(crate) fn add(&mut self, pixels: &[f32]) {
        for pixel in pixels.chunks_exact(self.c) {
            for (sum, &v) in self.sums.iter_mut().zip(pixel) {
                *sum += f64::from(v);
            }
        }
    }

    /// The means of `pixels` pixels, into `y`.
    pub(crate) fn finish(&self, y: &mut [f32], pixels: usize) {
        for (out, &sum) in y[..self.c].iter_mut().zip(&self.sums) {
            *out = (sum / pixels as f64) as f32;
        }
    }
}

/// Each valid window's taps summed in row-major order, then divided by
/// their count. `Plan::check` allows only whole windows without padding.
pub(crate) fn average_pool(op: &Op, x: &[f32], y: &mut [f32]) {
    let divisor = (op.kh * op.kw) as f32;
    for oy in 0..op.oh {
        for ox in 0..op.ow {
            let start = (oy * op.ow + ox) * op.oc;
            let out = &mut y[start..start + op.oc];
            out.fill(0.0);
            for ky in 0..op.kh {
                for kx in 0..op.kw {
                    let src = ((oy * op.sh + ky) * op.iw + ox * op.sw + kx) * op.ic;
                    for (dst, &v) in out.iter_mut().zip(&x[src..src + op.ic]) {
                        *dst += v;
                    }
                }
            }
            for dst in out {
                *dst /= divisor;
            }
        }
    }
}

/// `(x - mean) / sqrt(var + eps) * gamma + beta` per pixel, in the order of
/// the ONNX graph it replaces.
pub(crate) fn layer_norm(op: &Op, x: &[f32], y: &mut [f32], w: &[f32]) {
    let c = op.oc;
    let (gamma, beta, eps) = (&w[..c], &w[c..2 * c], w[2 * c]);
    for p in 0..op.oh * op.ow {
        let v = &x[p * c..][..c];
        let mean = v.iter().sum::<f32>() / c as f32;
        let var = v.iter().map(|&a| (a - mean) * (a - mean)).sum::<f32>() / c as f32;
        let r = (var + eps).sqrt();
        for (((out, &a), &g), &b) in y[p * c..][..c].iter_mut().zip(v).zip(gamma).zip(beta) {
            *out = (a - mean) / r * g + b;
        }
    }
}

/// Floats of scratch [`attention`] needs for `op`.
pub(crate) fn attention_scratch(op: &Op) -> usize {
    let (t, hd) = (op.iw, op.oc / op.n2);
    let lanes = hd.next_multiple_of(8);
    t * (2 * hd + 2 * lanes + t)
}

/// Multi-head self-attention. Pixel `s` of `x` holds `[q | k | v]`, each `c`
/// floats split into `heads` runs of `hd`. For each head:
/// scores `S = (Q * scale) K^T` over all `t` queries at once, softmax over
/// each row, output `S V`. Both products are 1x1-convolution shaped and go
/// through `kernels::pointwise_with`; V and the output rows are padded to whole
/// vectors.
pub(crate) fn attention(op: &Op, x: &[f32], y: &mut [f32], scale: f32, scratch: &mut [f32]) {
    let (t, c, heads) = (op.iw, op.oc, op.n2);
    let hd = c / heads;
    let lanes = hd.next_multiple_of(8);
    let (q, rest) = scratch.split_at_mut(t * hd);
    let (kt, rest) = rest.split_at_mut(hd * t);
    let (v, rest) = rest.split_at_mut(t * lanes);
    let (out, rest) = rest.split_at_mut(t * lanes);
    let scores = &mut rest[..t * t];
    for h in 0..heads {
        for s in 0..t {
            let row = &x[s * 3 * c..][..3 * c];
            for d in 0..hd {
                q[s * hd + d] = row[h * hd + d] * scale;
                kt[d * t + s] = row[c + h * hd + d];
            }
            v[s * lanes..][..hd].copy_from_slice(&row[2 * c + h * hd..][..hd]);
            v[s * lanes + hd..][..lanes - hd].fill(0.0);
        }
        kernels::pointwise_with(scores, kt, q, None, t, hd, t, false, None);
        for row in scores.chunks_exact_mut(t) {
            softmax_row(row);
        }
        kernels::pointwise_with(out, v, scores, None, t, t, lanes, false, None);
        for i in 0..t {
            y[i * c + h * hd..][..hd].copy_from_slice(&out[i * lanes..][..hd]);
        }
    }
}

/// Each input row repeated `sh` times, each pixel `sw` times: copies only,
/// so every tier gives the same bits.
pub(crate) fn upsample(op: &Op, x: &[f32], y: &mut [f32]) {
    let (c, row) = (op.oc, op.ow * op.oc);
    for iy in 0..op.ih {
        let (first, rest) = y[iy * op.sh * row..][..op.sh * row].split_at_mut(row);
        let src = &x[iy * op.iw * c..][..op.iw * c];
        for (out, pixel) in first.chunks_exact_mut(op.sw * c).zip(src.chunks_exact(c)) {
            for copy in out.chunks_exact_mut(c) {
                copy.copy_from_slice(pixel);
            }
        }
        for again in rest.chunks_exact_mut(row) {
            again.copy_from_slice(first);
        }
    }
}

/// [`upsample`] plus the second input, `x2 + x`, one output row at a time.
pub(crate) fn upsample_add(op: &Op, x: &[f32], x2: &[f32], y: &mut [f32]) {
    let (c, row, run) = (op.oc, op.ow * op.oc, op.sw * op.oc);
    let streamed = kernels::streamed(&y[..op.oh * row], c);
    for oy in 0..op.oh {
        let src = &x[oy / op.sh * op.iw * c..][..op.iw * c];
        let (out, addend) = (&mut y[oy * row..][..row], &x2[oy * row..][..row]);
        if streamed {
            kernels::upsample_add_row_streamed(out, addend, src, op.sw, c);
            continue;
        }
        for ((out, addend), pixel) in out
            .chunks_exact_mut(run)
            .zip(addend.chunks_exact(run))
            .zip(src.chunks_exact(c))
        {
            for (out, addend) in out.chunks_exact_mut(c).zip(addend.chunks_exact(c)) {
                for ((o, &a), &v) in out.iter_mut().zip(addend).zip(pixel) {
                    *o = a + v;
                }
            }
        }
    }
}

/// [`upsample_add`] writing over the second input: `y += x` upsampled.
pub(crate) fn upsample_add_in_place(op: &Op, x: &[f32], y: &mut [f32]) {
    let (c, row, run) = (op.oc, op.ow * op.oc, op.sw * op.oc);
    for oy in 0..op.oh {
        let src = &x[oy / op.sh * op.iw * c..][..op.iw * c];
        let out = &mut y[oy * row..][..row];
        for (out, pixel) in out.chunks_exact_mut(run).zip(src.chunks_exact(c)) {
            for out in out.chunks_exact_mut(c) {
                for (o, &v) in out.iter_mut().zip(pixel) {
                    *o += v;
                }
            }
        }
    }
}

/// Floats of scratch [`Transposed::run`] needs for `op`: one input row's
/// output blocks.
pub(crate) fn transpose_scratch(op: &Op) -> usize {
    op.iw * op.kh * op.kw * op.oc
}

/// Floats of scratch [`Transposed::run_into`] needs for `first` feeding
/// `second`: each one's blocks and the rows `first` hands `second`.
pub(crate) fn transpose_pair_scratch(first: &Op, second: &Op) -> usize {
    transpose_scratch(first) + first.kh * first.ow * first.oc + transpose_scratch(second)
}

/// [`super::plan::Kind::Convtranspose`], one input row at a time: the row
/// goes through `kernels::pointwise_with` into blocks, `[iw][kh][kw][oc]`,
/// `finish` gets them, and output row `kh * iy + ky` is then each pixel's
/// `[ky]` run of `kw * oc` floats, end to end. The clamp is applied as it
/// stores.
pub(crate) struct Transposed<'a> {
    pub(crate) op: &'a Op,
    pub(crate) w: &'a [f32],
    pub(crate) bias: Option<&'a [f32]>,
    pub(crate) clip: Option<(f32, f32)>,
    pub(crate) panels: bool,
}

impl Transposed<'_> {
    /// The whole layer.
    pub(crate) fn run(
        &self,
        x: &[f32],
        y: &mut [f32],
        scratch: &mut [f32],
        finish: &dyn Fn(&mut [f32]),
    ) {
        let (op, blocks) = (self.op, &mut scratch[..transpose_scratch(self.op)]);
        let rows = op.kh * op.ow * op.oc;
        for iy in 0..op.ih {
            let src = &x[iy * op.iw * op.ic..][..op.iw * op.ic];
            self.row(src, &mut y[iy * rows..][..rows], blocks, finish);
        }
    }

    /// The layer, each pair of its output rows handed straight to `next`,
    /// which writes `y`: the map between never leaves the cache.
    pub(crate) fn run_into(
        &self,
        next: &Self,
        x: &[f32],
        y: &mut [f32],
        scratch: &mut [f32],
        finish: &dyn Fn(&mut [f32]),
        finish_next: &dyn Fn(&mut [f32]),
    ) {
        let (op, after) = (self.op, next.op);
        let (blocks, rest) = scratch.split_at_mut(transpose_scratch(op));
        let (rows, rest) = rest.split_at_mut(op.kh * op.ow * op.oc);
        let next_blocks = &mut rest[..transpose_scratch(after)];
        let next_rows = after.kh * after.ow * after.oc;
        for iy in 0..op.ih {
            let src = &x[iy * op.iw * op.ic..][..op.iw * op.ic];
            self.row(src, rows, blocks, finish);
            for (r, row) in rows.chunks_exact(op.ow * op.oc).enumerate() {
                let out = &mut y[(iy * op.kh + r) * next_rows..][..next_rows];
                next.row(row, out, next_blocks, finish_next);
            }
        }
    }

    /// One input row into its `kh` output rows, `out`.
    fn row(&self, src: &[f32], out: &mut [f32], blocks: &mut [f32], finish: &dyn Fn(&mut [f32])) {
        let op = self.op;
        let (ci, run) = (op.ic, op.kw * op.oc);
        let block = op.kh * run;
        kernels::pointwise_with(
            blocks,
            self.w,
            src,
            self.bias,
            op.iw,
            ci,
            block,
            self.panels,
            self.clip,
        );
        finish(blocks);
        for (ky, out) in out.chunks_exact_mut(op.ow * op.oc).enumerate() {
            for (dst, pixel) in out.chunks_exact_mut(run).zip(blocks.chunks_exact(block)) {
                dst.copy_from_slice(&pixel[ky * run..][..run]);
            }
        }
    }
}

/// Columns per weight panel of the CTC head.
pub(crate) const PANEL: usize = 16;
/// Floats of running softmax per pixel: lane maxima, sums and best columns.
const STATE: usize = 3 * LANES;

/// Floats of per-pixel state [`ctc_head`] needs for `t` pixels.
pub(crate) fn head_state(t: usize) -> usize {
    t * STATE
}

/// The CTC head: logits `x W + b` over `vocab` classes for each of `t`
/// pixels, then either the softmax probabilities (`probs`, `[t][vocab]`) or
/// only each pixel's best class and its probability (`[t][2]`, the index as a
/// float). `w` holds `ceil(vocab / 16)` panels of `[ci][16]` columns, zero past
/// `vocab`; `b` holds `ceil(vocab / 16) * 16` values, -1e30 past `vocab`, so
/// padding never wins the maximum and adds nothing to the sum.
///
/// Each 16-column panel stays in L1 while every pixel passes through it, so
/// the 9 MB of weights are read once per line instead of once per pixel tile.
/// Each logit starts from its bias and adds the inputs in order. The softmax
/// runs online in [`LANES`] lanes, lane `l` holding columns `16 j + l` and
/// `16 j + 8 + l` of panel `j`: the largest logit so far, the sum of
/// `exp(logit - largest)` (rescaled when the largest grows) and the first
/// column that reached it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ctc_head(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: &[f32],
    t: usize,
    ci: usize,
    vocab: usize,
    probs: bool,
    state: &mut [f32],
) {
    let columns = vocab.div_ceil(PANEL) * PANEL;
    assert!(x.len() >= t * ci && w.len() >= columns * ci && b.len() >= columns);
    assert!(y.len() >= t * if probs { vocab } else { 2 } && state.len() >= head_state(t));
    #[cfg(target_arch = "x86_64")]
    {
        let (yp, xp, wp, bp, sp) = (
            y.as_mut_ptr(),
            x.as_ptr(),
            w.as_ptr(),
            b.as_ptr(),
            state.as_mut_ptr(),
        );
        // SAFETY: the asserts bound every access the kernels make; each
        // instruction set was detected.
        unsafe {
            match (simd_tier(), probs) {
                (3, true) => return head_avx2::<true>(yp, xp, wp, bp, t, ci, vocab, sp),
                (3, false) => return head_avx2::<false>(yp, xp, wp, bp, t, ci, vocab, sp),
                (2, true) => return head_avx::<true>(yp, xp, wp, bp, t, ci, vocab, sp),
                (2, false) => return head_avx::<false>(yp, xp, wp, bp, t, ci, vocab, sp),
                (1, true) => return head_sse::<true>(yp, xp, wp, bp, t, ci, vocab, sp),
                (1, false) => return head_sse::<false>(yp, xp, wp, bp, t, ci, vocab, sp),
                _ => {}
            }
        }
    }
    if probs {
        head_scalar::<false, true>(y, x, w, b, t, ci, vocab, state);
    } else {
        head_scalar::<false, false>(y, x, w, b, t, ci, vocab, state);
    }
}

/// [`ctc_head`]'s definition for one tier, lane by lane.
#[allow(clippy::too_many_arguments)]
fn head_scalar<const FMA: bool, const PROBS: bool>(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: &[f32],
    t: usize,
    ci: usize,
    vocab: usize,
    state: &mut [f32],
) {
    for s in state[..t * STATE].as_chunks_mut::<STATE>().0 {
        s[..LANES].fill(f32::NEG_INFINITY);
        s[LANES..].fill(0.0);
    }
    for j in 0..vocab.div_ceil(PANEL) {
        let panel = &w[j * ci * PANEL..][..ci * PANEL];
        for p in 0..t {
            let mut logits = [0.0f32; PANEL];
            for (col, logit) in logits.iter_mut().enumerate() {
                let mut acc = b[j * PANEL + col];
                for (i, &v) in x[p * ci..][..ci].iter().enumerate() {
                    acc = madd::<FMA>(panel[i * PANEL + col], v, acc);
                }
                *logit = acc;
            }
            let s = &mut state[p * STATE..][..STATE];
            for l in 0..LANES {
                let (a0, a1, m) = (logits[l], logits[LANES + l], s[l]);
                let top = max_lane(a0, a1);
                let grown = max_lane(m, top);
                let mut sum = s[LANES + l] * exp_neg::<FMA>(m - grown);
                sum += exp_neg::<FMA>(a0 - grown);
                sum += exp_neg::<FMA>(a1 - grown);
                (s[l], s[LANES + l]) = (grown, sum);
                // The second column only when strictly greater, the panel
                // only when strictly above the lane's maximum so far.
                if top > m {
                    s[2 * LANES + l] = (j * PANEL + l + if a1 > a0 { LANES } else { 0 }) as f32;
                }
            }
            if PROBS {
                let n = PANEL.min(vocab - j * PANEL);
                y[p * vocab + j * PANEL..][..n].copy_from_slice(&logits[..n]);
            }
        }
    }
    for p in 0..t {
        let s = &state[p * STATE..][..STATE];
        let (max, sum) = head_total::<FMA>(s);
        let r = 1.0 / sum;
        if PROBS {
            for v in &mut y[p * vocab..][..vocab] {
                *v = exp_neg::<FMA>(*v - max) * r;
            }
        } else {
            (y[2 * p], y[2 * p + 1]) = (head_best(s, max), r);
        }
    }
}

/// A pixel's largest logit and its softmax denominator, from its lanes.
#[inline(always)]
fn head_total<const FMA: bool>(s: &[f32]) -> (f32, f32) {
    let max = s[..LANES].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for l in 0..LANES {
        sum += s[LANES + l] * exp_neg::<FMA>(s[l] - max);
    }
    (max, sum)
}

/// The first column holding the largest logit: the smallest best column
/// among the lanes that reached it.
#[inline(always)]
fn head_best(s: &[f32], max: f32) -> f32 {
    (0..LANES)
        .filter(|&l| s[l] == max)
        .map(|l| s[2 * LANES + l])
        .fold(f32::INFINITY, f32::min)
}

/// [`head_scalar`] with `$tile` pixels per pass through a panel's columns,
/// `$lanes` columns per register. A partial tile repeats its last pixel and
/// keeps only the real ones.
#[cfg(target_arch = "x86_64")]
macro_rules! head_body {
    ($y:expr, $x:expr, $w:expr, $b:expr, $t:expr, $ci:expr, $vocab:expr, $state:expr,
     $probs:expr, $lanes:tt, $fma:tt, $exp2:ident, $tile:literal, $ids:expr) => {{
        // SAFETY: the caller's contract: the pointers hold what `ctc_head`
        // asserted, no pixel index reaches `t`, and stores into a row stop
        // at `vocab` columns.
        unsafe {
            let (yp, xp, wp, bp, sp) = ($y, $x, $w, $b, $state);
            let (t, ci, vocab) = ($t, $ci, $vocab);
            for p in 0..t {
                let s = sp.add(p * STATE);
                for l in 0..LANES {
                    *s.add(l) = f32::NEG_INFINITY;
                    *s.add(LANES + l) = 0.0;
                    *s.add(2 * LANES + l) = 0.0;
                }
            }
            for j in 0..vocab.div_ceil(PANEL) {
                let panel = wp.add(j * ci * PANEL);
                let left = vocab - j * PANEL;
                for h in 0..LANES / $lanes {
                    let (c0, c1) = (h * $lanes, LANES + h * $lanes);
                    let (b0, b1) = (
                        vld!($lanes, bp.add(j * PANEL + c0)),
                        vld!($lanes, bp.add(j * PANEL + c1)),
                    );
                    let idx0 = vadd!($lanes, $ids, vsplat!($lanes, (j * PANEL + c0) as f32));
                    let idx1 = vadd!($lanes, idx0, vsplat!($lanes, LANES as f32));
                    let mut p0 = 0;
                    while p0 < t {
                        let count = (t - p0).min($tile);
                        let mut a0 = [b0; $tile];
                        let mut a1 = [b1; $tile];
                        // Opaque per-pixel pointers, as in `kernels::pointwise_with`.
                        let px: [*const f32; $tile] =
                            std::hint::black_box(std::array::from_fn(|k| {
                                xp.add((p0 + k.min(count - 1)) * ci)
                            }));
                        let mut row = panel;
                        for i in 0..ci {
                            let (r0, r1) = (vld!($lanes, row.add(c0)), vld!($lanes, row.add(c1)));
                            for k in 0..$tile {
                                let v = vsplat!($lanes, *px[k].add(i));
                                a0[k] = vfma!($lanes, $fma, r0, v, a0[k]);
                                a1[k] = vfma!($lanes, $fma, r1, v, a1[k]);
                            }
                            row = row.add(PANEL);
                        }
                        for k in 0..count {
                            let s = sp.add((p0 + k) * STATE + c0);
                            let (m, sum, best) = (
                                vld!($lanes, s),
                                vld!($lanes, s.add(LANES)),
                                vld!($lanes, s.add(2 * LANES)),
                            );
                            let top = vmax!($lanes, a0[k], a1[k]);
                            let grown = vmax!($lanes, m, top);
                            // Where no lane's largest logit grew, the rescale
                            // is exp(0), exactly one.
                            let held = vmask!($lanes, veq!($lanes, grown, m)) == (1 << $lanes) - 1;
                            let mut total = if held {
                                sum
                            } else {
                                let rescale =
                                    vexp_neg!($lanes, $fma, $exp2, vsub!($lanes, m, grown));
                                vmul!($lanes, sum, rescale)
                            };
                            let e0 = vexp_neg!($lanes, $fma, $exp2, vsub!($lanes, a0[k], grown));
                            total = vadd!($lanes, total, e0);
                            let e1 = vexp_neg!($lanes, $fma, $exp2, vsub!($lanes, a1[k], grown));
                            total = vadd!($lanes, total, e1);
                            let column = vblendv!($lanes, idx0, idx1, vgt!($lanes, a1[k], a0[k]));
                            vst!($lanes, s, grown);
                            vst!($lanes, s.add(LANES), total);
                            let best = vblendv!($lanes, best, column, vgt!($lanes, top, m));
                            vst!($lanes, s.add(2 * LANES), best);
                            if $probs {
                                let out = yp.add((p0 + k) * vocab + j * PANEL);
                                for (c, v) in [(c0, a0[k]), (c1, a1[k])] {
                                    if c + $lanes <= left {
                                        vst!($lanes, out.add(c), v);
                                    } else if c < left {
                                        let mut tmp = [0.0f32; $lanes];
                                        vst!($lanes, tmp.as_mut_ptr(), v);
                                        std::ptr::copy_nonoverlapping(
                                            tmp.as_ptr(),
                                            out.add(c),
                                            left - c,
                                        );
                                    }
                                }
                            }
                        }
                        p0 += $tile;
                    }
                }
            }
            for p in 0..t {
                let s = std::slice::from_raw_parts(sp.add(p * STATE), STATE);
                let (max, sum) = head_total::<$fma>(s);
                let r = 1.0 / sum;
                if $probs {
                    let out = yp.add(p * vocab);
                    let (top, scale) = (vsplat!($lanes, max), vsplat!($lanes, r));
                    let mut c = 0;
                    while c + $lanes <= vocab {
                        let v = vsub!($lanes, vld!($lanes, out.add(c)), top);
                        let e = vexp_neg!($lanes, $fma, $exp2, v);
                        vst!($lanes, out.add(c), vmul!($lanes, e, scale));
                        c += $lanes;
                    }
                    while c < vocab {
                        *out.add(c) = exp_neg::<$fma>(*out.add(c) - max) * r;
                        c += 1;
                    }
                } else {
                    *yp.add(2 * p) = head_best(s, max);
                    *yp.add(2 * p + 1) = r;
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn head_avx2<const PROBS: bool>(
    y: *mut f32,
    x: *const f32,
    w: *const f32,
    b: *const f32,
    t: usize,
    ci: usize,
    vocab: usize,
    state: *mut f32,
) {
    use std::arch::x86_64::_mm256_setr_ps;
    let ids = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    head_body!(
        y,
        x,
        w,
        b,
        t,
        ci,
        vocab,
        state,
        PROBS,
        8,
        true,
        exp2_bits_avx2,
        6,
        ids
    )
}

/// Four pixels: without fused multiply-adds each product needs a register
/// of its own, and six would spill.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn head_avx<const PROBS: bool>(
    y: *mut f32,
    x: *const f32,
    w: *const f32,
    b: *const f32,
    t: usize,
    ci: usize,
    vocab: usize,
    state: *mut f32,
) {
    use std::arch::x86_64::_mm256_setr_ps;
    let ids = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    head_body!(
        y,
        x,
        w,
        b,
        t,
        ci,
        vocab,
        state,
        PROBS,
        8,
        false,
        exp2_bits_avx,
        4,
        ids
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn head_sse<const PROBS: bool>(
    y: *mut f32,
    x: *const f32,
    w: *const f32,
    b: *const f32,
    t: usize,
    ci: usize,
    vocab: usize,
    state: *mut f32,
) {
    use std::arch::x86_64::_mm_setr_ps;
    let ids = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
    head_body!(
        y,
        x,
        w,
        b,
        t,
        ci,
        vocab,
        state,
        PROBS,
        4,
        false,
        exp2_bits_sse,
        4,
        ids
    )
}

#[cfg(test)]
mod tests {
    use super::super::kernels::simd_tier;
    use super::super::plan::Kind;
    use super::*;

    fn erf64(x: f64) -> f64 {
        // Simpson's rule on 20 000 panels is exact to 1e-12 here.
        let n = 20_000;
        let h = x / n as f64;
        let f = |t: f64| (-t * t).exp();
        let mut s = f(0.0) + f(x);
        for i in 1..n {
            s += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        s * h / 3.0 * 2.0 / std::f64::consts::PI.sqrt()
    }

    /// Deterministic values in `[-scale, scale)`.
    fn noise(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state >> 8) as f32 / 8_388_608.0 - 1.0) * scale
            })
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn activation_tiers_match_scalar() {
        let fma = simd_tier() == 3;
        let xs: Vec<f32> = (-4000..4000)
            .map(|i| i as f32 * 0.0025)
            .chain([-90.0, 90.0, 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY])
            .collect();
        // Odd lengths exercise every vector tail.
        for n in [xs.len(), xs.len() - 3, 5] {
            let mut gelu = xs[..n].to_vec();
            gelu_inplace(&mut gelu);
            let mut swish = xs[..n].to_vec();
            swish_inplace(&mut swish);
            for (i, &x) in xs[..n].iter().enumerate() {
                let (g, s) = if fma {
                    (gelu_one::<true>(x), swish_one::<true>(x))
                } else {
                    (gelu_one::<false>(x), swish_one::<false>(x))
                };
                assert_eq!(gelu[i].to_bits(), g.to_bits(), "gelu({x})");
                assert_eq!(swish[i].to_bits(), s.to_bits(), "swish({x})");
                if x.is_finite() {
                    let xd = f64::from(x);
                    let want = 0.5 * xd * (1.0 + erf64(xd / 2f64.sqrt()));
                    let bound = 2e-6 * xd.abs().max(1.0);
                    // The rational fit's own error is 7e-9; this is rounding.
                    assert!((f64::from(g) - want).abs() < bound / 4.0, "gelu({x}) = {g}");
                    let want = xd / (1.0 + (-xd).exp());
                    assert!((f64::from(s) - want).abs() < bound, "swish({x}) = {s}");
                }
            }
        }
    }

    /// An operation of `kind` with these input and output shapes.
    fn op(
        kind: Kind,
        (ih, iw, ic): (usize, usize, usize),
        (oh, ow, oc): (usize, usize, usize),
    ) -> Op {
        let mut op = super::super::plan::op(kind, 0, 1, ic, oc);
        (op.ih, op.iw, op.oh, op.ow) = (ih, iw, oh, ow);
        op
    }

    #[test]
    fn upsampling_repeats_rows_and_pixels() {
        let (ih, iw, c, sh, sw) = (3, 4, 5, 2, 3);
        let mut up = op(Kind::Upsample, (ih, iw, c), (ih * sh, iw * sw, c));
        (up.sh, up.sw) = (sh, sw);
        let x = noise(ih * iw * c, 3, 1.0);
        let n = ih * sh * iw * sw * c;
        let addend = noise(n, 4, 1.0);
        let (mut y, mut sum) = (vec![0.0; n], vec![0.0; n]);
        upsample(&up, &x, &mut y);
        upsample_add(&up, &x, &addend, &mut sum);
        for (i, (&v, &total)) in y.iter().zip(&sum).enumerate() {
            let (pixel, ch) = (i / c, i % c);
            let (r, col) = (pixel / (iw * sw), pixel % (iw * sw));
            let want = x[((r / sh) * iw + col / sw) * c + ch];
            assert_eq!(v, want, "at {i}");
            assert_eq!(total, addend[i] + want, "sum at {i}");
        }
    }

    #[test]
    fn transposed_convolution_matches_float64() {
        let (ih, iw, ic, kh, kw, oc) = (3, 4, 5, 2, 3, 3);
        let (oh, ow) = (ih * kh, iw * kw);
        let mut layer = op(Kind::Convtranspose, (ih, iw, ic), (oh, ow, oc));
        (layer.kh, layer.kw, layer.sh, layer.sw) = (kh, kw, kh, kw);
        let (x, w) = (
            noise(ih * iw * ic, 5, 1.0),
            noise(ic * kh * kw * oc, 7, 1.0),
        );
        let b = noise(kh * kw * oc, 9, 0.5);
        let clip = (-0.25, 0.75);
        let mut y = vec![0.0; oh * ow * oc];
        let mut scratch = vec![0.0; transpose_scratch(&layer)];
        let finished = std::cell::Cell::new(0);
        let transposed = Transposed {
            op: &layer,
            w: &w,
            bias: Some(&b),
            clip: Some(clip),
            panels: false,
        };
        transposed.run(&x, &mut y, &mut scratch, &|rows| {
            finished.set(finished.get() + rows.len())
        });
        assert_eq!(finished.get(), ih * iw * kh * kw * oc);
        for (i, &got) in y.iter().enumerate() {
            let (pixel, o) = (i / oc, i % oc);
            let (r, col) = (pixel / ow, pixel % ow);
            let tap = (r % kh * kw + col % kw) * oc + o;
            let src = &x[((r / kh) * iw + col / kw) * ic..][..ic];
            let sum: f64 = src
                .iter()
                .enumerate()
                .map(|(c, &v)| f64::from(v) * f64::from(w[c * kh * kw * oc + tap]))
                .sum::<f64>()
                + f64::from(b[tap]);
            let want = sum.clamp(f64::from(clip.0), f64::from(clip.1));
            assert!(
                (f64::from(got) - want).abs() < 1e-5,
                "at {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn channel_mean_holds_on_pages() {
        // A large constant offset plus small noise: an `f32` running sum
        // loses the noise's mean entirely.
        let (pixels, c) = (300_000, 3);
        let mut x = noise(pixels * c, 13, 0.01);
        for (i, v) in x.iter_mut().enumerate() {
            *v += 1000.0 + (i % c) as f32;
        }
        let layer = op(Kind::Mean, (600, 500, c), (1, 1, c));
        let mut y = [0.0; 3];
        mean(&layer, &x, &mut y);
        for (ch, &got) in y.iter().enumerate() {
            let want = x
                .iter()
                .skip(ch)
                .step_by(c)
                .map(|&v| f64::from(v))
                .sum::<f64>()
                / pixels as f64;
            assert!(
                (f64::from(got) - want).abs() < 1e-4,
                "channel {ch}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn softmax_tiers_match_scalar() {
        for n in [1usize, 7, 8, 15, 120, 653] {
            let x = noise(n, n as u32, 12.0);
            let mut y = x.clone();
            softmax_row(&mut y);
            let mut want = x.clone();
            if simd_tier() == 3 {
                softmax_scalar::<true>(&mut want);
            } else {
                softmax_scalar::<false>(&mut want);
            }
            assert_eq!(bits(&y), bits(&want), "n={n}");
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = x.iter().map(|&v| (v as f64 - max).exp()).sum();
            for (&a, &b) in x.iter().zip(&y) {
                let exact = (a as f64 - max).exp() / sum;
                assert!((b as f64 - exact).abs() < 1e-6 * exact.max(1e-3), "n={n}");
            }
        }
    }

    #[test]
    fn ctc_head_tiers_match_scalar() {
        let ci = 19;
        // 37 classes leave a partial last panel; 64 fill four.
        for vocab in [37usize, 64] {
            let columns = vocab.div_ceil(PANEL) * PANEL;
            let logical = noise(ci * vocab, 7, 0.8);
            let mut w = vec![0.0; ci * columns];
            for i in 0..ci {
                for c in 0..vocab {
                    w[(c / PANEL * ci + i) * PANEL + c % PANEL] = logical[i * vocab + c];
                }
            }
            let mut b = vec![-1e30; columns];
            b[..vocab].copy_from_slice(&noise(vocab, 11, 0.5));
            // Every tile remainder of the four- and six-pixel kernels.
            for t in [1usize, 4, 6, 7, 13] {
                let x = noise(t * ci, t as u32 + 3, 2.0);
                for probs in [true, false] {
                    let width = if probs { vocab } else { 2 };
                    let (mut got, mut want) = (vec![0.0; t * width], vec![0.0; t * width]);
                    let mut state = vec![0.0; head_state(t)];
                    ctc_head(&mut got, &x, &w, &b, t, ci, vocab, probs, &mut state);
                    let definition = match (simd_tier() == 3, probs) {
                        (true, true) => head_scalar::<true, true>,
                        (true, false) => head_scalar::<true, false>,
                        (false, true) => head_scalar::<false, true>,
                        (false, false) => head_scalar::<false, false>,
                    };
                    definition(&mut want, &x, &w, &b, t, ci, vocab, &mut state);
                    assert_eq!(bits(&got), bits(&want), "vocab={vocab} t={t} probs={probs}");
                    for p in 0..t {
                        let logits: Vec<f64> = (0..vocab)
                            .map(|c| {
                                let dot: f64 = (0..ci)
                                    .map(|i| {
                                        f64::from(x[p * ci + i]) * f64::from(logical[i * vocab + c])
                                    })
                                    .sum();
                                dot + f64::from(b[c])
                            })
                            .collect();
                        let (best, &max) = logits
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap();
                        let sum: f64 = logits.iter().map(|v| (v - max).exp()).sum();
                        if probs {
                            for (c, v) in logits.iter().enumerate() {
                                let exact = (v - max).exp() / sum;
                                let gap = (f64::from(got[p * vocab + c]) - exact).abs();
                                assert!(gap < 1e-5, "t={t} p={p} c={c}");
                            }
                        } else {
                            assert_eq!(got[2 * p], best as f32, "t={t} p={p}");
                            assert!((f64::from(got[2 * p + 1]) - 1.0 / sum).abs() < 1e-5);
                        }
                    }
                }
            }
        }
    }
}
