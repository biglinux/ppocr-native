//! Transposed matrix-vector products, plain and four pixels at a time: the
//! per-pixel tail of the pointwise kernels and the tap-at-a-time borders of
//! the convolutions in `engine/conv.rs`.

use super::{simd_tier, span};

/// `y[0..n] = sum_i x[i] * a[i*n ..]`, with `a` laid out `[m][n]`.
///
/// Each output block is held in a register across the whole reduction, so the
/// result is stored once instead of being read back `m` times, and there is no
/// horizontal sum at the end.
pub(crate) fn matvec_t(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    assert!(y.len() >= n && x.len() >= m && a.len() >= span(m, n));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the assert above covers every element the bodies touch.
    unsafe {
        match simd_tier() {
            3 => return mvt_avx2(y, a, x, m, n),
            2 => return mvt_avx(y, a, x, m, n),
            1 => return mvt_sse(y, a, x, m, n),
            _ => {}
        }
    }
    for j in 0..n {
        let mut s = 0.0f32;
        for i in 0..m {
            s += x[i] * a[i * n + j];
        }
        y[j] = s;
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! mvt_body {
    ($y:expr, $a:expr, $x:expr, $m:expr, $n:expr, $fma:tt) => {{
        #[allow(unused_imports)]
        use std::arch::x86_64::*;
        // SAFETY: bounds are the caller's contract, checked by `matvec_t`.
        unsafe {
            let (y, a, x, m, n) = ($y, $a, $x, $m, $n);
            let (yp, ap, xp) = (y.as_mut_ptr(), a.as_ptr(), x.as_ptr());
            let mut j = 0usize;
            // Eight blocks keep eight independent chains in flight; four left
            // the reduction waiting on multiply-add latency. Each output still
            // sums over `i` in order.
            while j + 64 <= n {
                let mut acc = [_mm256_setzero_ps(); 8];
                for i in 0..m {
                    let s = _mm256_set1_ps(*xp.add(i));
                    let row = ap.add(i * n + j);
                    for (k, a) in acc.iter_mut().enumerate() {
                        *a = fma8!($fma, _mm256_loadu_ps(row.add(8 * k)), s, *a);
                    }
                }
                for (k, a) in acc.into_iter().enumerate() {
                    _mm256_storeu_ps(yp.add(j + 8 * k), a);
                }
                j += 64;
            }
            while j + 32 <= n {
                let mut a0 = _mm256_setzero_ps();
                let mut a1 = _mm256_setzero_ps();
                let mut a2 = _mm256_setzero_ps();
                let mut a3 = _mm256_setzero_ps();
                for i in 0..m {
                    let s = _mm256_set1_ps(*xp.add(i));
                    let row = ap.add(i * n + j);
                    a0 = fma8!($fma, _mm256_loadu_ps(row), s, a0);
                    a1 = fma8!($fma, _mm256_loadu_ps(row.add(8)), s, a1);
                    a2 = fma8!($fma, _mm256_loadu_ps(row.add(16)), s, a2);
                    a3 = fma8!($fma, _mm256_loadu_ps(row.add(24)), s, a3);
                }
                _mm256_storeu_ps(yp.add(j), a0);
                _mm256_storeu_ps(yp.add(j + 8), a1);
                _mm256_storeu_ps(yp.add(j + 16), a2);
                _mm256_storeu_ps(yp.add(j + 24), a3);
                j += 32;
            }
            while j + 8 <= n {
                let mut acc = _mm256_setzero_ps();
                for i in 0..m {
                    acc = fma8!(
                        $fma,
                        _mm256_loadu_ps(ap.add(i * n + j)),
                        _mm256_set1_ps(*xp.add(i)),
                        acc
                    );
                }
                _mm256_storeu_ps(yp.add(j), acc);
                j += 8;
            }
            while j < n {
                let mut s = 0.0f32;
                for i in 0..m {
                    s += *xp.add(i) * *ap.add(i * n + j);
                }
                *yp.add(j) = s;
                j += 1;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mvt_avx2(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    mvt_body!(y, a, x, m, n, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn mvt_avx(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    mvt_body!(y, a, x, m, n, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn mvt_sse(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    use std::arch::x86_64::*;
    // SAFETY: bounds are the caller's contract, checked by `matvec_t`.
    unsafe {
        let (yp, ap, xp) = (y.as_mut_ptr(), a.as_ptr(), x.as_ptr());
        let mut j = 0usize;
        while j + 4 <= n {
            let mut acc = _mm_setzero_ps();
            for i in 0..m {
                acc = fma4!(
                    _mm_loadu_ps(ap.add(i * n + j)),
                    _mm_set1_ps(*xp.add(i)),
                    acc
                );
            }
            _mm_storeu_ps(yp.add(j), acc);
            j += 4;
        }
        while j < n {
            let mut s = 0.0f32;
            for i in 0..m {
                s += *xp.add(i) * *ap.add(i * n + j);
            }
            *yp.add(j) = s;
            j += 1;
        }
    }
}

/// `y[0..n] += sum_i x[i] * a[i*n ..]`, one pixel at a time.
///
/// The accumulating twin of [`matvec_t`], for the leftover pixels at the end
/// of a row that the four-at-a-time kernel cannot cover, and for layers whose
/// output is a single pixel.
pub(crate) fn matvec_t_acc(y: &mut [f32], a: &[f32], x: &[f32], m: usize, n: usize) {
    assert!(y.len() >= n && x.len() >= m && a.len() >= span(m, n));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the assert above covers every element the bodies touch.
    unsafe {
        let (yp, ap, xp) = (y.as_mut_ptr(), a.as_ptr(), x.as_ptr());
        match simd_tier() {
            3 => return mvt_acc_avx2(yp, ap, xp, m, n),
            2 => return mvt_acc_avx(yp, ap, xp, m, n),
            1 => return mvt_acc_sse(yp, ap, xp, m, n),
            _ => {}
        }
    }
    for j in 0..n {
        let mut s = y[j];
        for i in 0..m {
            s += x[i] * a[i * n + j];
        }
        y[j] = s;
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! mvt_acc_body {
    ($y:expr, $a:expr, $x:expr, $m:expr, $n:expr, $lanes:tt, $fma:tt) => {{
        // SAFETY: bounds are the caller's contract, checked by `matvec_t_acc`.
        unsafe {
            let (yp, ap, xp) = ($y, $a, $x);
            let (m, n) = ($m, $n);
            let mut j = 0usize;
            // Eight independent chains, as in `mvt_body`; same order per output.
            while j + 8 * $lanes <= n {
                let mut acc = [vld!($lanes, yp.add(j)); 8];
                for (k, a) in acc.iter_mut().enumerate().skip(1) {
                    *a = vld!($lanes, yp.add(j + k * $lanes));
                }
                for i in 0..m {
                    let s = vsplat!($lanes, *xp.add(i));
                    let row = ap.add(i * n + j);
                    for (k, a) in acc.iter_mut().enumerate() {
                        *a = vfma!($lanes, $fma, vld!($lanes, row.add(k * $lanes)), s, *a);
                    }
                }
                for (k, a) in acc.into_iter().enumerate() {
                    vst!($lanes, yp.add(j + k * $lanes), a);
                }
                j += 8 * $lanes;
            }
            while j + 4 * $lanes <= n {
                let mut a0 = vld!($lanes, yp.add(j));
                let mut a1 = vld!($lanes, yp.add(j + $lanes));
                let mut a2 = vld!($lanes, yp.add(j + 2 * $lanes));
                let mut a3 = vld!($lanes, yp.add(j + 3 * $lanes));
                for i in 0..m {
                    let s = vsplat!($lanes, *xp.add(i));
                    let row = ap.add(i * n + j);
                    a0 = vfma!($lanes, $fma, vld!($lanes, row), s, a0);
                    a1 = vfma!($lanes, $fma, vld!($lanes, row.add($lanes)), s, a1);
                    a2 = vfma!($lanes, $fma, vld!($lanes, row.add(2 * $lanes)), s, a2);
                    a3 = vfma!($lanes, $fma, vld!($lanes, row.add(3 * $lanes)), s, a3);
                }
                vst!($lanes, yp.add(j), a0);
                vst!($lanes, yp.add(j + $lanes), a1);
                vst!($lanes, yp.add(j + 2 * $lanes), a2);
                vst!($lanes, yp.add(j + 3 * $lanes), a3);
                j += 4 * $lanes;
            }
            while j + $lanes <= n {
                let mut acc = vld!($lanes, yp.add(j));
                for i in 0..m {
                    acc = vfma!(
                        $lanes,
                        $fma,
                        vld!($lanes, ap.add(i * n + j)),
                        vsplat!($lanes, *xp.add(i)),
                        acc
                    );
                }
                vst!($lanes, yp.add(j), acc);
                j += $lanes;
            }
            while j < n {
                let mut s = *yp.add(j);
                for i in 0..m {
                    s += *xp.add(i) * *ap.add(i * n + j);
                }
                *yp.add(j) = s;
                j += 1;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mvt_acc_avx2(y: *mut f32, a: *const f32, x: *const f32, m: usize, n: usize) {
    mvt_acc_body!(y, a, x, m, n, 8, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn mvt_acc_avx(y: *mut f32, a: *const f32, x: *const f32, m: usize, n: usize) {
    mvt_acc_body!(y, a, x, m, n, 8, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn mvt_acc_sse(y: *mut f32, a: *const f32, x: *const f32, m: usize, n: usize) {
    mvt_acc_body!(y, a, x, m, n, 4, false)
}

/// Four pixels sharing one weight matrix, accumulating into `ys`.
///
/// Used by convolutions with a kernel bigger than one pixel, where each tap is
/// a separate pass and the running sum has to stay in memory between taps.
/// Four independent accumulators keep the multiply pipeline busy even when the
/// output is only 8 channels wide, where a single accumulator would stall
/// waiting on its own previous result.
///
/// # Safety
/// `a` must hold `m*n` floats, each `xs[k]` at least `m`, each `ys[k]` at
/// least `n`, and the four output ranges must not overlap each other or
/// either input/weight range. Address products must fit `usize`.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn matvec_t_acc_x4(
    ys: [*mut f32; 4],
    a: *const f32,
    xs: [*const f32; 4],
    m: usize,
    n: usize,
) {
    // SAFETY: forwarded from this function's own contract.
    unsafe {
        match simd_tier() {
            3 => mvt_x4_avx2(ys, a, xs, m, n),
            2 => mvt_x4_avx(ys, a, xs, m, n),
            1 => mvt_x4_sse(ys, a, xs, m, n),
            _ => {
                for p in 0..4 {
                    for j in 0..n {
                        let mut acc = *ys[p].add(j);
                        for i in 0..m {
                            acc += *xs[p].add(i) * *a.add(i * n + j);
                        }
                        *ys[p].add(j) = acc;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! mvt_x4_body {
    ($ys:expr, $a:expr, $xs:expr, $m:expr, $n:expr, $lanes:tt, $fma:tt) => {{
        // SAFETY: bounds and non-overlap are the caller's contract.
        unsafe {
            let (ys, a, xs, m, n) = ($ys, $a, $xs, $m, $n);
            let mut j = 0usize;
            while j + $lanes <= n {
                let mut a0 = vld!($lanes, ys[0].add(j));
                let mut a1 = vld!($lanes, ys[1].add(j));
                let mut a2 = vld!($lanes, ys[2].add(j));
                let mut a3 = vld!($lanes, ys[3].add(j));
                for i in 0..m {
                    let row = vld!($lanes, a.add(i * n + j));
                    a0 = vfma!($lanes, $fma, row, vsplat!($lanes, *xs[0].add(i)), a0);
                    a1 = vfma!($lanes, $fma, row, vsplat!($lanes, *xs[1].add(i)), a1);
                    a2 = vfma!($lanes, $fma, row, vsplat!($lanes, *xs[2].add(i)), a2);
                    a3 = vfma!($lanes, $fma, row, vsplat!($lanes, *xs[3].add(i)), a3);
                }
                vst!($lanes, ys[0].add(j), a0);
                vst!($lanes, ys[1].add(j), a1);
                vst!($lanes, ys[2].add(j), a2);
                vst!($lanes, ys[3].add(j), a3);
                j += $lanes;
            }
            while j < n {
                for k in 0..4 {
                    let mut s = *ys[k].add(j);
                    for i in 0..m {
                        s += *xs[k].add(i) * *a.add(i * n + j);
                    }
                    *ys[k].add(j) = s;
                }
                j += 1;
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mvt_x4_avx2(ys: [*mut f32; 4], a: *const f32, xs: [*const f32; 4], m: usize, n: usize) {
    mvt_x4_body!(ys, a, xs, m, n, 8, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn mvt_x4_avx(ys: [*mut f32; 4], a: *const f32, xs: [*const f32; 4], m: usize, n: usize) {
    mvt_x4_body!(ys, a, xs, m, n, 8, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn mvt_x4_sse(ys: [*mut f32; 4], a: *const f32, xs: [*const f32; 4], m: usize, n: usize) {
    mvt_x4_body!(ys, a, xs, m, n, 4, false)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{ramp, reference_matvec};
    use super::*;

    #[test]
    fn matvec_matches_the_definition_across_widths() {
        for (m, n) in [(3, 8), (16, 8), (16, 16), (8, 32), (7, 13), (64, 128)] {
            let a = ramp(m * n, 0.1);
            let x = ramp(m, 2.0);
            let mut y = vec![0.0; n];
            matvec_t(&mut y, &a, &x, m, n);
            for (got, want) in y.iter().zip(reference_matvec(&a, &x, m, n)) {
                assert!((got - want).abs() < 1e-4, "{got} vs {want} for {m}x{n}");
            }
        }
    }
}
