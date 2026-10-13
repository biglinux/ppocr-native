//! Dense convolution with a window, read in place from up to four
//! upsampled sources, and the three-channel first layer.

use super::super::plan::Op;
use super::{simd_tier, span};

/// One input of [`conv_row`]: `c` channels per pixel of an image `2^shift`
/// times smaller than the layer's input along both axes, read as if
/// upsampled by repeating each pixel, which is how nearest upsampling and
/// channel concatenation feeding a convolution need never be written out.
#[derive(Clone, Copy)]
pub(crate) struct Source<'a> {
    pub(crate) data: &'a [f32],
    pub(crate) c: usize,
    pub(crate) shift: u32,
}

/// Most inputs [`conv_row`] reads side by side.
pub(crate) const SOURCES: usize = 4;

/// One output row `oy` of a dense convolution with a window over the
/// channel concatenation of `sources`, read where they lie: `y` holds the
/// row's `ow * co` outputs, `w` is `[kh][kw][sum of c][co]`.
///
/// Each output is what [`pointwise_with`](super::pointwise_with) gives over the window gathered
/// into one vector, tap by tap, each source's channels in order, zero
/// outside the image: the same products summed in the same order, a padded
/// tap multiplying `zeros` (as many zeros as the widest source) as the
/// gathered zeros were. That order is the tier's: fused multiply-adds on
/// tier 3 except the last `co % 8` columns; the bias added last, or first
/// where the gathered call's tiles took it first (`co % 16 == 0` on AVX
/// tiers, every pixel but the row's last `ow % 4`); then the clamp.
///
/// Gathering cost more than the arithmetic at the detector's 12 to 48
/// channels: a copy call per tap per pixel. Read in place, each tap is one
/// pointer per pixel of a register tile.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_row(
    y: &mut [f32],
    sources: &[Source],
    zeros: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    op: &Op,
    oy: usize,
    clip: Option<(f32, f32)>,
) {
    let co = op.oc;
    let channels: usize = sources.iter().map(|s| s.c).sum();
    let taps = span(op.kh, op.kw);
    assert!(y.len() >= span(op.ow, co) && oy < op.oh);
    assert!(channels == op.ic + op.n2 && (1..=SOURCES).contains(&sources.len()));
    assert!(w.len() >= span(span(taps, channels), co) && bias.is_none_or(|b| b.len() >= co));
    for s in sources {
        let (h, w) = (op.ih >> s.shift, op.iw >> s.shift);
        assert!(s.shift < 16 && (h << s.shift, w << s.shift) == (op.ih, op.iw));
        assert!(s.data.len() >= span(span(h, w), s.c) && zeros.len() >= s.c);
    }
    debug_assert!(zeros.iter().all(|&v| v == 0.0));
    let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
    let tier = simd_tier();
    // Pixels whose gathered tiles took the bias first.
    let first = if tier >= 2 && co.is_multiple_of(16) {
        op.ow - op.ow % 4
    } else {
        0
    };
    let top = (oy * op.sh) as isize - op.pt;
    #[cfg(target_arch = "x86_64")]
    if tier >= 1 && co.is_multiple_of(4) {
        let mut inputs = [(std::ptr::null(), 0, 0, 0); SOURCES];
        for (input, s) in inputs.iter_mut().zip(sources) {
            *input = (s.data.as_ptr(), s.c, s.shift, op.iw >> s.shift);
        }
        let t = ConvRow {
            y: y.as_mut_ptr(),
            sources: inputs,
            shifted: sources.len() > 2 || sources.iter().any(|s| s.shift > 0),
            zeros: zeros.as_ptr(),
            w: w.as_ptr(),
            b: bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
            co,
            ih: op.ih,
            iw: op.iw,
            kh: op.kh,
            kw: op.kw,
            sw: op.sw,
            top,
            left: -op.pl,
            bounds,
        };
        // SAFETY: the asserts cover every view; `Plan::check` bounds every
        // window coordinate; each tier was detected. A tap outside the
        // image reads `zeros`, which holds as many floats as a pixel.
        unsafe {
            match tier {
                3 => conv_row_avx2(&t, op.ow, first),
                2 => conv_row_avx(&t, op.ow, first),
                _ => conv_row_sse(&t, op.ow, first),
            }
        }
        return;
    }
    // The definition, for tier 0 and widths without a vector kernel.
    let fused = |j: usize| tier == 3 && j < co / 8 * 8;
    for ox in 0..op.ow {
        for j in 0..co {
            let early = ox < first;
            let mut acc = match bias {
                Some(b) if early => b[j],
                _ => 0.0,
            };
            for ky in 0..op.kh {
                for kx in 0..op.kw {
                    let iy = top + ky as isize;
                    let ix = (ox * op.sw + kx) as isize - op.pl;
                    let inside =
                        iy >= 0 && ix >= 0 && (iy as usize) < op.ih && (ix as usize) < op.iw;
                    let (iy, ix) = (iy.max(0) as usize, ix.max(0) as usize);
                    let mut row = &w[(ky * op.kw + kx) * channels * co..];
                    for s in sources {
                        let pixel = (iy >> s.shift) * (op.iw >> s.shift) + (ix >> s.shift);
                        for i in 0..s.c {
                            let v = if inside { s.data[pixel * s.c + i] } else { 0.0 };
                            let a = row[i * co + j];
                            acc = if fused(j) {
                                a.mul_add(v, acc)
                            } else {
                                acc + a * v
                            };
                        }
                        row = &row[s.c * co..];
                    }
                }
            }
            if let Some(b) = bias.filter(|_| !early) {
                acc += b[j];
            }
            y[ox * co + j] = acc.clamp(bounds.0, bounds.1);
        }
    }
}

/// Raw views of one layer at one output row, checked by [`conv_row`].
#[cfg(target_arch = "x86_64")]
struct ConvRow {
    y: *mut f32,
    /// Data, channels, shift and width of each source; unused ones have no
    /// channels. Unless `shifted`, only the first two are used, unshifted.
    sources: [(*const f32, usize, u32, usize); SOURCES],
    shifted: bool,
    zeros: *const f32,
    w: *const f32,
    b: *const f32,
    co: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    sw: usize,
    /// Input row under kernel row 0, input column under kernel column 0 of
    /// output pixel 0; either may lie in the padding.
    top: isize,
    left: isize,
    bounds: (f32, f32),
}

/// `$p` output pixels from `$ox` by `$nb` blocks of `$lanes` columns from
/// `$col`, then, with `$half`, four columns more summed unfused, as the
/// masked tail of `pw_any8_body` sums them. Bias first when `$first`.
/// `$shifted` reads every source slot with its shift; without, only the
/// first two, at full size: the setup of each tap shows at a dozen
/// channels.
#[cfg(target_arch = "x86_64")]
macro_rules! conv_tile {
    ($t:expr, $ox:expr, $col:expr, $first:expr, $lanes:tt, $fma:tt, $p:expr, $nb:expr, $half:expr, $shifted:tt) => {{
        #[allow(unused_imports)]
        use std::arch::x86_64::*;
        let (t, ox, col, first): (&ConvRow, usize, usize, bool) = ($t, $ox, $col, $first);
        let mut acc = [[vsplat!($lanes, 0.0); $nb]; $p];
        let mut tail = [_mm_setzero_ps(); $p];
        if first && !t.b.is_null() {
            for a in &mut acc {
                for (m, v) in a.iter_mut().enumerate() {
                    *v = vld!($lanes, t.b.add(col + $lanes * m));
                }
            }
            if $half {
                tail = [_mm_loadu_ps(t.b.add(col + $lanes * $nb)); $p];
            }
        }
        // The weights in their order: tap by tap, each source's channels.
        let mut row = t.w.add(col);
        for ky in 0..t.kh {
            let iy = t.top + ky as isize;
            let live = iy >= 0 && (iy as usize) < t.ih;
            for kx in 0..t.kw {
                let sources = if $shifted {
                    &t.sources[..]
                } else {
                    &t.sources[..2]
                };
                for &(src, c, shift, width) in sources {
                    let px: [*const f32; $p] = std::array::from_fn(|k| {
                        let ix = t.left + ((ox + k) * t.sw + kx) as isize;
                        if live && ix >= 0 && (ix as usize) < t.iw {
                            let pixel = if $shifted {
                                (iy as usize >> shift) * width + (ix as usize >> shift)
                            } else {
                                iy as usize * t.iw + ix as usize
                            };
                            src.add(pixel * c)
                        } else {
                            t.zeros
                        }
                    });
                    for i in 0..c {
                        let r: [_; $nb] =
                            std::array::from_fn(|m| vld!($lanes, row.add($lanes * m)));
                        let h = if $half {
                            _mm_loadu_ps(row.add($lanes * $nb))
                        } else {
                            _mm_setzero_ps()
                        };
                        for k in 0..$p {
                            let v = *px[k].add(i);
                            let s = vsplat!($lanes, v);
                            for m in 0..$nb {
                                acc[k][m] = vfma!($lanes, $fma, r[m], s, acc[k][m]);
                            }
                            if $half {
                                tail[k] = _mm_add_ps(_mm_mul_ps(h, _mm_set1_ps(v)), tail[k]);
                            }
                        }
                        row = row.add(t.co);
                    }
                }
            }
        }
        let (lo, hi) = (vsplat!($lanes, t.bounds.0), vsplat!($lanes, t.bounds.1));
        for k in 0..$p {
            let out = t.y.add((ox + k) * t.co + col);
            for m in 0..$nb {
                let mut v = acc[k][m];
                if !first && !t.b.is_null() {
                    v = vadd!($lanes, v, vld!($lanes, t.b.add(col + $lanes * m)));
                }
                vst!($lanes, out.add($lanes * m), vclip!($lanes, v, lo, hi));
            }
            if $half {
                let mut v = tail[k];
                if !first && !t.b.is_null() {
                    v = _mm_add_ps(v, _mm_loadu_ps(t.b.add(col + $lanes * $nb)));
                }
                let (lo, hi) = (_mm_set1_ps(t.bounds.0), _mm_set1_ps(t.bounds.1));
                _mm_storeu_ps(out.add($lanes * $nb), vclip!(4, v, lo, hi));
            }
        }
    }};
}

/// Every pixel of the row for the columns `$col..` that one tile covers:
/// bias-first pixels, then the rest, `$p` at a time, then one at a time.
#[cfg(target_arch = "x86_64")]
macro_rules! conv_pixels {
    ($t:expr, $ow:expr, $first:expr, $col:expr, $lanes:tt, $fma:tt, $p:expr, $nb:expr, $half:expr, $shifted:tt) => {{
        let (t, ow, first, col) = ($t, $ow, $first, $col);
        let mut ox = 0;
        while ox + $p <= first {
            conv_tile!(t, ox, col, true, $lanes, $fma, $p, $nb, $half, $shifted);
            ox += $p;
        }
        while ox < first {
            conv_tile!(t, ox, col, true, $lanes, $fma, 1, $nb, $half, $shifted);
            ox += 1;
        }
        while ox + $p <= ow {
            conv_tile!(t, ox, col, false, $lanes, $fma, $p, $nb, $half, $shifted);
            ox += $p;
        }
        while ox < ow {
            conv_tile!(t, ox, col, false, $lanes, $fma, 1, $nb, $half, $shifted);
            ox += 1;
        }
    }};
}

/// Twelve accumulators, or nine: columns 16 at a time by six pixels, 24 by
/// three, twelve (8 + 4) by six; other widths eight columns, or the last
/// four, at a time by four pixels. Four pixels of 24 columns need a
/// seventeenth register, and the accumulator LLVM then spills in the inner
/// loop halved the rate. Taps are not blocked to keep their weights in the
/// first-level cache: that gains on some layers and loses on others.
#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::reversed_empty_ranges,
    reason = "a tile of the last four columns has no full vector"
)]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_row_avx2(t: &ConvRow, ow: usize, first: usize) {
    macro_rules! groups {
        ($shifted:tt) => {
            match t.co {
                12 => conv_pixels!(t, ow, first, 0, 8, true, 6, 1, true, $shifted),
                co if co.is_multiple_of(16) => {
                    for col in (0..co).step_by(16) {
                        conv_pixels!(t, ow, first, col, 8, true, 6, 2, false, $shifted);
                    }
                }
                co if co.is_multiple_of(24) => {
                    for col in (0..co).step_by(24) {
                        conv_pixels!(t, ow, first, col, 8, true, 3, 3, false, $shifted);
                    }
                }
                co => {
                    for col in (0..co / 8 * 8).step_by(8) {
                        conv_pixels!(t, ow, first, col, 8, true, 4, 1, false, $shifted);
                    }
                    if co % 8 == 4 {
                        conv_pixels!(t, ow, first, co - 4, 8, true, 4, 0, true, $shifted);
                    }
                }
            }
        };
    }
    // SAFETY: the caller's contract, checked by `conv_row`.
    unsafe {
        if t.shifted {
            groups!(true)
        } else {
            groups!(false)
        }
    }
}

/// [`conv_row_avx2`] without fused multiply-adds: a product needs a register
/// of its own, so 16 and twelve columns go four pixels at a time.
#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::reversed_empty_ranges,
    reason = "a tile of the last four columns has no full vector"
)]
#[target_feature(enable = "avx")]
unsafe fn conv_row_avx(t: &ConvRow, ow: usize, first: usize) {
    macro_rules! groups {
        ($shifted:tt) => {
            match t.co {
                12 => conv_pixels!(t, ow, first, 0, 8, false, 4, 1, true, $shifted),
                co if co.is_multiple_of(16) => {
                    for col in (0..co).step_by(16) {
                        conv_pixels!(t, ow, first, col, 8, false, 4, 2, false, $shifted);
                    }
                }
                co if co.is_multiple_of(24) => {
                    for col in (0..co).step_by(24) {
                        conv_pixels!(t, ow, first, col, 8, false, 3, 3, false, $shifted);
                    }
                }
                co => {
                    for col in (0..co / 8 * 8).step_by(8) {
                        conv_pixels!(t, ow, first, col, 8, false, 4, 1, false, $shifted);
                    }
                    if co % 8 == 4 {
                        conv_pixels!(t, ow, first, co - 4, 8, false, 4, 0, true, $shifted);
                    }
                }
            }
        };
    }
    // SAFETY: the caller's contract, checked by `conv_row`.
    unsafe {
        if t.shifted {
            groups!(true)
        } else {
            groups!(false)
        }
    }
}

/// [`conv_row_avx2`] in four-column registers, every sum unfused.
#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::int_plus_one,
    reason = "tiles of one pixel step like wider ones"
)]
#[target_feature(enable = "sse4.1")]
unsafe fn conv_row_sse(t: &ConvRow, ow: usize, first: usize) {
    macro_rules! groups {
        ($shifted:tt) => {
            match t.co {
                24 => conv_pixels!(t, ow, first, 0, 4, false, 1, 6, false, $shifted),
                48 => {
                    for col in [0, 24] {
                        conv_pixels!(t, ow, first, col, 4, false, 1, 6, false, $shifted);
                    }
                }
                12 => conv_pixels!(t, ow, first, 0, 4, false, 3, 3, false, $shifted),
                co => {
                    for col in (0..co).step_by(4) {
                        conv_pixels!(t, ow, first, col, 4, false, 4, 1, false, $shifted);
                    }
                }
            }
        };
    }
    // SAFETY: the caller's contract, checked by `conv_row`.
    unsafe {
        if t.shifted {
            groups!(true)
        } else {
            groups!(false)
        }
    }
}

/// Interior windows of an RGB stem with 24 output channels, the detector's,
/// or 48, the recogniser's, in register tiles of three or two pixels: the
/// spatial reduction stays in registers and each output keeps the tap
/// fallback's bias-first, tap-ordered sum. Returns the pixels done: none
/// for other shapes and below AVX.
pub(crate) fn rgb_interior(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    op: &Op,
    count: usize,
) -> usize {
    // Pixels that leave room for the RGB broadcasts and the weights without
    // spilling accumulators on AVX2.
    let pixels = match op.oc {
        24 => 3,
        48 => 2,
        _ => return 0,
    };
    if count < pixels || op.ic != 3 {
        return 0;
    }
    let count = count / pixels * pixels;
    assert!(y.len() >= count * op.oc && w.len() >= op.kh * op.kw * 3 * op.oc);
    assert!(x.len() >= ((op.kh - 1) * op.iw + (op.kw - 1) + (count - 1) * op.sw) * 3 + 3);
    if let Some(b) = bias {
        assert!(b.len() >= op.oc);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // SAFETY: complete windows and output sizes checked above, and each
        // specialization is reached only after the matching ISA detection.
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        match (simd_tier(), op.oc) {
            (3, 24) => rgb24_avx2(y, x, w, b, op, count),
            (2, 24) => rgb24_avx(y, x, w, b, op, count),
            (3, _) => rgb48_avx2(y, x, w, b, op, count),
            (2, _) => rgb48_avx(y, x, w, b, op, count),
            _ => return 0,
        }
        count
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

#[cfg(target_arch = "x86_64")]
macro_rules! rgb_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$op:expr,$count:expr,$blocks:expr,$pixels:expr,$fma:tt) => {{
        use std::arch::x86_64::*;
        unsafe {
            let (yp, xp, wp, b, op, count) =
                ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b, $op, $count);
            let co = $blocks * 8;
            let mut p = 0;
            while p < count {
                let mut acc = [_mm256_setzero_ps(); $pixels * $blocks];
                if !b.is_null() {
                    for j in 0..$blocks {
                        let seed = _mm256_loadu_ps(b.add(j * 8));
                        for k in 0..$pixels {
                            acc[k * $blocks + j] = seed;
                        }
                    }
                }
                for ky in 0..op.kh {
                    for kx in 0..op.kw {
                        for ic in 0..3 {
                            let src = ((ky * op.iw + kx) + p * op.sw) * 3 + ic;
                            let weight = ((ky * op.kw + kx) * 3 + ic) * co;
                            for j in 0..$blocks {
                                let v = _mm256_loadu_ps(wp.add(weight + j * 8));
                                for k in 0..$pixels {
                                    acc[k * $blocks + j] = fma8!(
                                        $fma,
                                        v,
                                        _mm256_set1_ps(*xp.add(src + k * op.sw * 3)),
                                        acc[k * $blocks + j]
                                    );
                                }
                            }
                        }
                    }
                }
                for k in 0..$pixels {
                    for j in 0..$blocks {
                        _mm256_storeu_ps(yp.add((p + k) * co + j * 8), acc[k * $blocks + j]);
                    }
                }
                p += $pixels;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb24_avx2(y: &mut [f32], x: &[f32], w: &[f32], b: *const f32, op: &Op, count: usize) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 3, 3, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rgb24_avx(y: &mut [f32], x: &[f32], w: &[f32], b: *const f32, op: &Op, count: usize) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 3, 3, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb48_avx2(y: &mut [f32], x: &[f32], w: &[f32], b: *const f32, op: &Op, count: usize) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 6, 2, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rgb48_avx(y: &mut [f32], x: &[f32], w: &[f32], b: *const f32, op: &Op, count: usize) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 6, 2, false)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{bits, ramp};
    use super::super::{fill_bias, matvec_t_acc, pointwise_with};
    use super::*;

    #[test]
    fn rgb_windows_match_taps() {
        for (oc, pixels) in [(24, 3), (48, 2)] {
            for (k, stride) in [(3, 2), (5, 2), (3, 1)] {
                for count in [1, 3, 4, 7, 8, 13, 17] {
                    let iw = (count - 1) * stride + k;
                    let mut op = crate::engine::plan::op(crate::engine::Kind::Conv, 0, 1, 3, oc);
                    (op.ih, op.iw, op.oh, op.ow) = (k, iw, 1, count);
                    (op.kh, op.kw, op.sh, op.sw) = (k, k, stride, stride);
                    let x = ramp(k * iw * 3, 0.8);
                    let w = ramp(k * k * 3 * oc, 2.1);
                    let b = ramp(oc, 0.2);
                    for bias in [None, Some(b.as_slice())] {
                        let mut got = vec![7.0; count * oc + 1];
                        let done = rgb_interior(&mut got, &x, &w, bias, &op, count);
                        let tiled = if simd_tier() >= 2 {
                            count / pixels * pixels
                        } else {
                            0
                        };
                        assert_eq!(done, tiled, "{k}x{k} count={count}");
                        for p in 0..done {
                            let mut want = vec![0.0; oc];
                            fill_bias(&mut want, bias, 1, oc);
                            for ky in 0..k {
                                for kx in 0..k {
                                    let src = (ky * iw + p * stride + kx) * 3;
                                    let tap = &w[(ky * k + kx) * 3 * oc..][..3 * oc];
                                    matvec_t_acc(&mut want, tap, &x[src..src + 3], 3, oc);
                                }
                            }
                            assert_eq!(bits(&got[p * oc..][..oc]), bits(&want), "{k}x{k} p={p}");
                        }
                        assert!(got[done * oc..].iter().all(|&v| v == 7.0));
                    }
                }
            }
        }
    }

    /// Output row `oy` the gathered way: each pixel's window copied out, tap
    /// by tap, `x`'s channels then `x2`'s, zero outside the image, then
    /// `pointwise_with` over the row.
    #[allow(clippy::too_many_arguments)]
    fn gathered_row(
        op: &crate::engine::plan::Op,
        x: &[f32],
        x2: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        oy: usize,
        clip: Option<(f32, f32)>,
    ) -> Vec<f32> {
        let tap = op.ic + op.n2;
        let k = op.kh * op.kw * tap;
        let mut windows = vec![0.0; op.ow * k];
        for (ox, window) in windows.chunks_exact_mut(k).enumerate() {
            for ky in 0..op.kh {
                for kx in 0..op.kw {
                    let iy = (oy * op.sh + ky) as isize - op.pt;
                    let ix = (ox * op.sw + kx) as isize - op.pl;
                    if iy < 0 || ix < 0 || iy as usize >= op.ih || ix as usize >= op.iw {
                        continue;
                    }
                    let pixel = iy as usize * op.iw + ix as usize;
                    let dst = &mut window[(ky * op.kw + kx) * tap..][..tap];
                    dst[..op.ic].copy_from_slice(&x[pixel * op.ic..][..op.ic]);
                    dst[op.ic..].copy_from_slice(&x2[pixel * op.n2..][..op.n2]);
                }
            }
        }
        let mut y = vec![0.0; op.ow * op.oc];
        pointwise_with(&mut y, w, &windows, bias, op.ow, k, op.oc, false, clip);
        y
    }

    #[test]
    fn in_place_windows_match_gathered() {
        // The 2x2 padded below and to the right, 3x3 at strides 1 and 2,
        // and a 1x3; one or two inputs; widths with every tile remainder.
        let windows = [
            ((2, 2), 1, (0, 1, 0, 1)),
            ((3, 3), 1, (1, 1, 1, 1)),
            ((3, 3), 2, (1, 1, 1, 1)),
            ((1, 3), 1, (0, 0, 1, 1)),
        ];
        for co in [12, 24, 48, 4, 8, 16, 20, 28, 40, 7] {
            for (k, s, (pt, pb, pl, pr)) in windows {
                for (ic, n2) in [(5, 0), (8, 4), (12, 0), (96, 24)] {
                    for (ih, iw) in [(3, 1), (4, 10), (5, 15)] {
                        let mut op =
                            crate::engine::plan::op(crate::engine::Kind::Conv, 0, 1, ic, co);
                        (op.kh, op.kw, op.sh, op.sw, op.n2) = (k.0, k.1, s, s, n2);
                        (op.pt, op.pb, op.pl, op.pr) = (pt, pb, pl, pr);
                        op.oh = (ih + (pt + pb) as usize - k.0) / s + 1;
                        op.ow = (iw + (pl + pr) as usize - k.1) / s + 1;
                        (op.ih, op.iw) = (ih, iw);
                        let x = ramp(ih * iw * ic, 0.4);
                        let x2 = ramp(ih * iw * n2, 1.7);
                        let w = ramp(k.0 * k.1 * (ic + n2) * co, 2.3);
                        let b = ramp(co, 0.9);
                        let zeros = vec![0.0; ic.max(n2)];
                        let clips = [None, Some((0.0, f32::INFINITY))];
                        for (bias, clip) in [None, Some(b.as_slice())].into_iter().zip(clips) {
                            for oy in 0..op.oh {
                                let want = gathered_row(&op, &x, &x2, &w, bias, oy, clip);
                                let mut got = vec![123.0; op.ow * co + 1];
                                let sources = [
                                    Source {
                                        data: &x,
                                        c: ic,
                                        shift: 0,
                                    },
                                    Source {
                                        data: &x2,
                                        c: n2,
                                        shift: 0,
                                    },
                                ];
                                let count = if n2 > 0 { 2 } else { 1 };
                                conv_row(
                                    &mut got,
                                    &sources[..count],
                                    &zeros,
                                    &w,
                                    bias,
                                    &op,
                                    oy,
                                    clip,
                                );
                                assert_eq!(
                                    bits(&got[..op.ow * co]),
                                    bits(&want),
                                    "co={co} {k:?}/{s} ic={ic}+{n2} {ih}x{iw} row {oy}"
                                );
                                assert_eq!(got[op.ow * co], 123.0);
                            }
                        }
                    }
                }
            }
        }
    }
}
