//! Depthwise convolution: single taps, and whole 3x3 or wider windows whose
//! accumulator is stored once per pixel.

use super::{simd_tier, span};

/// One depthwise kernel tap: `y[p*c + ch] += x[p*step + ch] * w[ch]`.
///
/// `step` is the distance in floats between neighbouring source pixels, which
/// is `c` for a unit stride and a multiple of it when the convolution skips
/// pixels. The weight is constant per channel, so a whole channel vector is
/// held in a register and reused across every pixel in the run.
pub(crate) fn depthwise_tap(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    count: usize,
    c: usize,
    step: usize,
) {
    assert!(y.len() >= span(count, c) && w.len() >= c);
    assert!(
        count == 0
            || x.len()
                >= span(count - 1, step)
                    .checked_add(c)
                    .expect("input span overflows")
    );
    #[cfg(target_arch = "x86_64")]
    if c.is_multiple_of(8) {
        // SAFETY: the asserts above cover every element the bodies touch.
        unsafe {
            match simd_tier() {
                3 => return dw_avx2(y, x, w, count, c, step),
                2 => return dw_avx(y, x, w, count, c, step),
                1 => return dw_sse(y, x, w, count, c, step),
                _ => {}
            }
        }
    }
    for p in 0..count {
        for ch in 0..c {
            y[p * c + ch] += x[p * step + ch] * w[ch];
        }
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! dw_body {
    ($y:expr, $x:expr, $w:expr, $count:expr, $c:expr, $step:expr, $lanes:tt, $fma:tt) => {{
        // SAFETY: bounds are the caller's contract, checked by `depthwise_tap`.
        unsafe {
            let (y, x, w, count, c, step) = ($y, $x, $w, $count, $c, $step);
            let (yp, xp, wp) = (y.as_mut_ptr(), x.as_ptr(), w.as_ptr());
            for p in 0..count {
                let (dst, src) = (p * c, p * step);
                let mut ch = 0usize;
                while ch + $lanes <= c {
                    let acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, xp.add(src + ch)),
                        vld!($lanes, wp.add(ch)),
                        vld!($lanes, yp.add(dst + ch))
                    );
                    vst!($lanes, yp.add(dst + ch), acc);
                    ch += $lanes;
                }
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dw_avx2(y: &mut [f32], x: &[f32], w: &[f32], count: usize, c: usize, step: usize) {
    dw_body!(y, x, w, count, c, step, 8, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dw_avx(y: &mut [f32], x: &[f32], w: &[f32], count: usize, c: usize, step: usize) {
    dw_body!(y, x, w, count, c, step, 8, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dw_sse(y: &mut [f32], x: &[f32], w: &[f32], count: usize, c: usize, step: usize) {
    dw_body!(y, x, w, count, c, step, 4, false)
}

// Whole 3x3 depthwise windows. A pixel's accumulator is stored only once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn depthwise_3x3_interior(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    count: usize,
    c: usize,
    row_stride: usize,
    step: usize,
    clip: Option<(f32, f32)>,
) -> bool {
    if count == 0 || c == 0 {
        return false;
    }
    // MAXPS/MINPS against infinities return every value unchanged.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(unused_variables, reason = "only the x86 kernels clip here")
    )]
    let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
    assert!(y.len() >= count.checked_mul(c).expect("depthwise output overflow"));
    let last = (count - 1)
        .checked_mul(step)
        .and_then(|n| n.checked_add(span(2, row_stride)))
        .and_then(|n| n.checked_add(span(3, c)))
        .expect("depthwise input overflow");
    assert!(x.len() >= last && w.len() >= span(9, c));
    if let Some(b) = bias {
        assert!(b.len() >= c);
    }
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 1 && c.is_multiple_of(4) {
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        // SAFETY: bounds above cover the run, lane divisibility is matched
        // below, and the tier was detected.
        unsafe {
            match simd_tier() {
                3 if c.is_multiple_of(8) => {
                    dw3_avx2(y, x, w, b, count, c, row_stride, step, bounds)
                }
                2 if c.is_multiple_of(8) => dw3_avx(y, x, w, b, count, c, row_stride, step, bounds),
                _ => dw3_sse(y, x, w, b, count, c, row_stride, step, bounds),
            }
        }
        return true;
    }
    false
}

/// The live part of `kh`-by-`kw` depthwise windows, on AVX tiers with
/// `c % 8 == 0`. Returns false when it did nothing.
///
/// Every pixel of the run reads kernel rows `rows.0..rows.1` and columns
/// `cols.0..cols.1` of a kernel `kw` wide; `x` starts under tap
/// (`rows.0`, `cols.0`) of the first pixel. A border pixel simply has fewer
/// live taps, the ones the fallback does not skip as padding. Each output
/// keeps exactly the fallback's arithmetic: seeded with the bias, then one
/// `x * w + acc` per live tap in row-major order, fused on tier 3 and separate
/// on tier 2, as [`depthwise_tap`] does for these widths. It is stored once,
/// instead of read and written back once per tap.
#[allow(clippy::too_many_arguments)]
pub(crate) fn depthwise_window(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    count: usize,
    c: usize,
    row_stride: usize,
    step: usize,
    kw: usize,
    rows: (usize, usize),
    cols: (usize, usize),
    clip: Option<(f32, f32)>,
) -> bool {
    if count == 0 || c == 0 || !c.is_multiple_of(8) {
        return false;
    }
    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(unused_variables, reason = "only the x86 kernels clip here")
    )]
    let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
    let (rows, cols) = if rows.0 < rows.1 && cols.0 < cols.1 {
        (rows, cols)
    } else {
        ((0, 0), (0, 0)) // no live tap: every output is its bias
    };
    assert!(cols.1 <= kw && y.len() >= count.checked_mul(c).expect("depthwise output overflow"));
    if rows.0 < rows.1 {
        let last = (count - 1)
            .checked_mul(step)
            .and_then(|n| n.checked_add(span(rows.1 - rows.0 - 1, row_stride)))
            .and_then(|n| n.checked_add(span(cols.1 - cols.0, c)))
            .expect("depthwise input overflow");
        assert!(x.len() >= last && w.len() >= span((rows.1 - 1) * kw + cols.1, c));
    }
    if let Some(b) = bias {
        assert!(b.len() >= c);
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: bounds above, lane divisibility, and feature dispatch.
    unsafe {
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        match simd_tier() {
            3 => {
                dwk_avx2(
                    y, x, w, b, count, c, row_stride, step, kw, rows, cols, bounds,
                );
                return true;
            }
            2 => {
                dwk_avx(
                    y, x, w, b, count, c, row_stride, step, kw, rows, cols, bounds,
                );
                return true;
            }
            _ => {}
        }
    }
    false
}

/// `$pixels` neighbouring pixels by `$blocks` eight-channel blocks at `ch`:
/// eight independent chains however the run is shaped.
#[cfg(target_arch = "x86_64")]
macro_rules! dwk_tile {
    ($yp:expr,$xp:expr,$wp:expr,$b:expr,$p:expr,$ch:expr,$c:expr,$row:expr,$step:expr,$kw:expr,$rows:expr,$cols:expr,$clip:expr,$pixels:literal,$blocks:literal,$fma:tt) => {{
        let (yp, xp, wp, b, p, ch, c, row, step, kw) =
            ($yp, $xp, $wp, $b, $p, $ch, $c, $row, $step, $kw);
        let ((ky0, ky1), (kx0, kx1)) = ($rows, $cols);
        let mut acc = [[_mm256_setzero_ps(); $blocks]; $pixels];
        if !b.is_null() {
            for a in acc.iter_mut() {
                for (m, v) in a.iter_mut().enumerate() {
                    *v = _mm256_loadu_ps(b.add(ch + 8 * m));
                }
            }
        }
        for ky in ky0..ky1 {
            for kx in kx0..kx1 {
                let tap = wp.add((ky * kw + kx) * c + ch);
                let src = xp.add(p * step + (ky - ky0) * row + (kx - kx0) * c + ch);
                for m in 0..$blocks {
                    let weight = _mm256_loadu_ps(tap.add(8 * m));
                    for (k, a) in acc.iter_mut().enumerate() {
                        let v = _mm256_loadu_ps(src.add(k * step + 8 * m));
                        a[m] = fma8!($fma, v, weight, a[m]);
                    }
                }
            }
        }
        for (k, a) in acc.iter().enumerate() {
            for (m, v) in a.iter().enumerate() {
                let (lo, hi, clamps) = $clip;
                let v = if clamps { vclip!(8, *v, lo, hi) } else { *v };
                _mm256_storeu_ps(yp.add((p + k) * c + ch + 8 * m), v);
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! dwk_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$count:expr,$c:expr,$row:expr,$step:expr,$kw:expr,$rows:expr,$cols:expr,$bounds:expr,$fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: checked by depthwise_window; every live tap lies inside it.
        unsafe {
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let clip = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1), clamps);
            let (yp, xp, wp, b) = ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b);
            let (count, c, row, step, kw, rows, cols) =
                ($count, $c, $row, $step, $kw, $rows, $cols);
            // Fewer pixels left (the end of a run, or a lone border column)
            // trade pixels for channel blocks, keeping eight chains.
            macro_rules! run {
                ($pixels:literal, $blocks:literal, $p:expr, $ch:expr) => {
                    dwk_tile!(
                        yp, xp, wp, b, $p, $ch, c, row, step, kw, rows, cols, clip, $pixels,
                        $blocks, $fma
                    )
                };
            }
            macro_rules! pixels {
                ($pixels:literal, $blocks:literal, $p:expr) => {{
                    let mut ch = 0;
                    while ch + 8 * $blocks <= c {
                        run!($pixels, $blocks, $p, ch);
                        ch += 8 * $blocks;
                    }
                    while ch < c {
                        run!($pixels, 1, $p, ch);
                        ch += 8;
                    }
                }};
            }
            let mut p = 0;
            // 5- and 7-wide windows with every column live, at stride one:
            // eight neighbouring pixels read 12 or 14 input columns per
            // kernel row, each loaded once instead of once per tap (17 loads
            // per 40 products instead of 45 for five columns, 21 per 56
            // instead of 63 for seven). Every accumulator still takes bias,
            // then its live rows in order, each left to right: the generic
            // tile's sums. Ten pixels, all 16 registers on AVX2, gain nothing
            // for five columns; seven fill them with eight.
            macro_rules! whole_rows {
                ($k:literal) => {
                    if (kw, cols, step) == ($k, (0, $k), c) && rows.0 < rows.1 {
                        while p + 8 <= count {
                            let mut ch = 0;
                            while ch < c {
                                let mut acc = [_mm256_setzero_ps(); 8];
                                if !b.is_null() {
                                    acc = [_mm256_loadu_ps(b.add(ch)); 8];
                                }
                                for ky in rows.0..rows.1 {
                                    let taps = wp.add(ky * $k * c + ch);
                                    let mut k = [_mm256_setzero_ps(); $k];
                                    for (kx, weight) in k.iter_mut().enumerate() {
                                        *weight = _mm256_loadu_ps(taps.add(kx * c));
                                    }
                                    let src = xp.add(p * c + (ky - rows.0) * row + ch);
                                    for j in 0..8 + $k - 1 {
                                        let v = _mm256_loadu_ps(src.add(j * c));
                                        for (kx, weight) in k.iter().enumerate() {
                                            if kx <= j && j - kx < 8 {
                                                acc[j - kx] = fma8!($fma, v, *weight, acc[j - kx]);
                                            }
                                        }
                                    }
                                }
                                for (i, v) in acc.iter().enumerate() {
                                    let (lo, hi, clamps) = clip;
                                    let v = if clamps { vclip!(8, *v, lo, hi) } else { *v };
                                    _mm256_storeu_ps(yp.add((p + i) * c + ch), v);
                                }
                                ch += 8;
                            }
                            p += 8;
                        }
                    }
                };
            }
            whole_rows!(5);
            whole_rows!(7);
            while p + 8 <= count {
                pixels!(8, 1, p);
                p += 8;
            }
            if p + 4 <= count {
                pixels!(4, 2, p);
                p += 4;
            }
            if p + 2 <= count {
                pixels!(2, 4, p);
                p += 2;
            }
            if p < count {
                pixels!(1, 8, p);
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dwk_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    kw: usize,
    rows: (usize, usize),
    cols: (usize, usize),
    bounds: (f32, f32),
) {
    dwk_body!(y, x, w, b, count, c, row, step, kw, rows, cols, bounds, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn dwk_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    kw: usize,
    rows: (usize, usize),
    cols: (usize, usize),
    bounds: (f32, f32),
) {
    dwk_body!(y, x, w, b, count, c, row, step, kw, rows, cols, bounds, false)
}

#[cfg(target_arch = "x86_64")]
macro_rules! dw3_body {
    ($y:expr,$x:expr,$w:expr,$b:expr,$count:expr,$c:expr,$row:expr,$step:expr,$bounds:expr,$lanes:tt,$fma:tt) => {{
        // SAFETY: checked by depthwise_3x3_interior. Each lane belongs to one
        // pixel, and each of the nine source vectors is in the validated window.
        unsafe {
            let (lo, hi) = (vsplat!($lanes, $bounds.0), vsplat!($lanes, $bounds.1));
            // Only a real clamp pays for it: nine taps share each store.
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let (yp, xp, wp, b) = ($y.as_mut_ptr(), $x.as_ptr(), $w.as_ptr(), $b);
            let (count, c, row, step) = ($count, $c, $row, $step);
            // Pixels outer, channels inner: a tile reads its three rows as
            // short runs of whole pixels, in address order, which the
            // prefetcher follows. Channels outer read every pixel's few
            // bytes of one block, a cache line apart, which it does not.
            macro_rules! tile {
                ($pixels:literal, $p:expr) => {{
                    let mut ch = 0;
                    while ch < c {
                        let seed = if b.is_null() {
                            vsplat!($lanes, 0.0)
                        } else {
                            vld!($lanes, b.add(ch))
                        };
                        let mut acc = [seed; $pixels];
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let k = vld!($lanes, wp.add((ky * 3 + kx) * c + ch));
                                for (i, a) in acc.iter_mut().enumerate() {
                                    let at = ($p + i) * step + ky * row + kx * c + ch;
                                    *a = vfma!($lanes, $fma, vld!($lanes, xp.add(at)), k, *a);
                                }
                            }
                        }
                        for (i, &a) in acc.iter().enumerate() {
                            let v = if clamps { vclip!($lanes, a, lo, hi) } else { a };
                            vst!($lanes, yp.add(($p + i) * c + ch), v);
                        }
                        ch += $lanes;
                    }
                }};
            }
            let mut p = 0;
            while p + 4 <= count {
                tile!(4, p);
                p += 4;
            }
            while p < count {
                tile!(1, p);
                p += 1;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw3_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    bounds: (f32, f32),
) {
    if step == c {
        // Stride one at any width: naming the step `c` lets LLVM see that a
        // pixel's middle column is its neighbour's left one and load it once.
        dw3_body!(y, x, w, b, count, c, row, c, bounds, 8, true)
    } else {
        dw3_body!(y, x, w, b, count, c, row, step, bounds, 8, true)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw3_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    bounds: (f32, f32),
) {
    if step == c {
        // Stride one at any width: naming the step `c` lets LLVM see that a
        // pixel's middle column is its neighbour's left one and load it once.
        dw3_body!(y, x, w, b, count, c, row, c, bounds, 8, false)
    } else {
        dw3_body!(y, x, w, b, count, c, row, step, bounds, 8, false)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn dw3_sse(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    count: usize,
    c: usize,
    row: usize,
    step: usize,
    bounds: (f32, f32),
) {
    if step == c {
        // Stride one at any width: naming the step `c` lets LLVM see that a
        // pixel's middle column is its neighbour's left one and load it once.
        dw3_body!(y, x, w, b, count, c, row, c, bounds, 4, false)
    } else {
        dw3_body!(y, x, w, b, count, c, row, step, bounds, 4, false)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{bits, ramp};
    use super::super::{clip_inplace, fill_bias};
    use super::*;

    #[test]
    fn depthwise_tap_honours_the_pixel_step() {
        let (c, count, step) = (8, 5, 16);
        let w = ramp(c, 0.9);
        let x = ramp((count - 1) * step + c, 2.5);
        let mut y = vec![1.0; count * c];
        depthwise_tap(&mut y, &x, &w, count, c, step);
        for p in 0..count {
            for ch in 0..c {
                let want = 1.0 + x[p * step + ch] * w[ch];
                assert!((y[p * c + ch] - want).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn depthwise_windows_match_nine_taps() {
        for c in [4, 8, 16, 24, 28, 32, 36, 64, 128] {
            for count in [1, 2, 3, 4, 5, 7, 8, 9, 17, 103] {
                for stride in [1, 2] {
                    let row = ((count - 1) * stride + 3) * c;
                    let x = ramp(row * 3, 0.7);
                    let w = ramp(9 * c, 1.3);
                    let b = ramp(c, 3.2);
                    for bias in [None, Some(b.as_slice())] {
                        let mut want = vec![0.0; count * c];
                        fill_bias(&mut want, bias, count, c);
                        for ky in 0..3 {
                            for kx in 0..3 {
                                depthwise_tap(
                                    &mut want,
                                    &x[ky * row + kx * c..],
                                    &w[(ky * 3 + kx) * c..],
                                    count,
                                    c,
                                    stride * c,
                                );
                            }
                        }
                        let mut got = vec![123.0; count * c + 8];
                        if depthwise_3x3_interior(
                            &mut got,
                            &x,
                            &w,
                            bias,
                            count,
                            c,
                            row,
                            stride * c,
                            None,
                        ) {
                            assert_eq!(
                                bits(&got[..count * c]),
                                bits(&want),
                                "c={c} count={count} stride={stride}"
                            );
                            assert!(got[count * c..].iter().all(|&v| v == 123.0));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn any_depthwise_window_matches_taps() {
        for (kh, kw) in [(5, 5), (3, 5), (1, 1), (2, 2), (7, 7)] {
            for c in [8, 16, 24, 32, 64] {
                for count in [1, 3, 7, 8, 9, 16, 17] {
                    for stride in [1, 2] {
                        let row = ((count - 1) * stride + kw) * c;
                        let x = ramp(row * kh, 0.7);
                        let w = ramp(kh * kw * c, 1.3);
                        let b = ramp(c, 3.2);
                        for bias in [None, Some(b.as_slice())] {
                            let mut want = vec![0.0; count * c];
                            fill_bias(&mut want, bias, count, c);
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    depthwise_tap(
                                        &mut want,
                                        &x[ky * row + kx * c..],
                                        &w[(ky * kw + kx) * c..],
                                        count,
                                        c,
                                        stride * c,
                                    );
                                }
                            }
                            let mut got = vec![123.0; count * c + 8];
                            let done = depthwise_window(
                                &mut got,
                                &x,
                                &w,
                                bias,
                                count,
                                c,
                                row,
                                stride * c,
                                kw,
                                (0, kh),
                                (0, kw),
                                None,
                            );
                            assert_eq!(done, simd_tier() >= 2, "{kh}x{kw} c={c}");
                            if done {
                                assert_eq!(
                                    bits(&got[..count * c]),
                                    bits(&want),
                                    "{kh}x{kw} c={c} count={count} stride={stride}"
                                );
                                assert!(got[count * c..].iter().all(|&v| v == 123.0));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn depthwise_borders_match_taps() {
        let (kh, kw) = (5, 5);
        // Every live sub-window a padded border can leave, including none.
        for rows in [(0, 5), (2, 5), (0, 3), (1, 4), (3, 3)] {
            for cols in [(0, 5), (2, 5), (0, 2), (1, 3), (4, 5), (0, 0)] {
                for c in [8, 24, 32, 40, 64] {
                    for (count, step) in [1, 2, 9, 12].into_iter().flat_map(|n| [(n, 1), (n, 2)]) {
                        let (nr, nc) = (rows.1 - rows.0, cols.1 - cols.0);
                        let row = ((count - 1) * step + kw) * c;
                        let mut x = ramp(row * kh, 0.4);
                        x[row + 3 * c + 5] = f32::NAN;
                        let w = ramp(kh * kw * c, 1.9);
                        let b = ramp(c, 2.6);
                        let clips = [None, Some((-0.1, 0.2))];
                        for (bias, clip) in [None, Some(b.as_slice())].into_iter().zip(clips) {
                            let mut want = vec![0.0; count * c];
                            fill_bias(&mut want, bias, count, c);
                            for ky in 0..nr {
                                for kx in 0..nc {
                                    depthwise_tap(
                                        &mut want,
                                        &x[ky * row + kx * c..],
                                        &w[((rows.0 + ky) * kw + cols.0 + kx) * c..],
                                        count,
                                        c,
                                        step * c,
                                    );
                                }
                            }
                            if let Some((low, high)) = clip {
                                clip_inplace(&mut want, low, high);
                            }
                            let mut got = vec![123.0; count * c + 8];
                            let src = if nr > 0 && nc > 0 { &x[..] } else { &x[..0] };
                            let done = depthwise_window(
                                &mut got,
                                src,
                                &w,
                                bias,
                                count,
                                c,
                                row,
                                step * c,
                                kw,
                                rows,
                                cols,
                                clip,
                            );
                            assert_eq!(done, simd_tier() >= 2);
                            if done {
                                assert_eq!(
                                    bits(&got[..count * c]),
                                    bits(&want),
                                    "rows={rows:?} cols={cols:?} c={c} count={count} step={step}"
                                );
                                assert!(got[count * c..].iter().all(|&v| v == 123.0));
                            }
                        }
                    }
                }
            }
        }
    }
}
