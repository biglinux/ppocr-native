//! Elementwise passes: bias fill, add, max, clamp, channel gates, and the
//! streaming stores that keep large outputs out of the cache.

use std::sync::OnceLock;

use super::{simd_tier, span};

/// Copy a per-channel bias across `count` neighbouring pixels.
pub(crate) fn fill_bias(y: &mut [f32], bias: Option<&[f32]>, count: usize, c: usize) {
    let n = span(count, c);
    assert!(y.len() >= n);
    if n == 0 {
        return;
    }
    if let Some(b) = bias {
        assert!(b.len() >= c);
    }
    match bias {
        None => y[..n].fill(0.0),
        Some(b) => {
            for p in 0..count {
                y[p * c..p * c + c].copy_from_slice(&b[..c]);
            }
        }
    }
}

/// [`add_with`] without a clamp, for the tests.
#[cfg(test)]
fn add(y: &mut [f32], x1: &[f32], x2: &[f32], n: usize) {
    add_with(y, x1, x2, n, None);
}

/// `y = x1 + x2`, then optionally [`clip_inplace`], in one pass: both split vector
/// and scalar elements at the same index, so each element gets the same
/// operations in the same order.
pub(crate) fn add_with(y: &mut [f32], x1: &[f32], x2: &[f32], n: usize, clip: Option<(f32, f32)>) {
    assert!(y.len() >= n && x1.len() >= n && x2.len() >= n);
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
        // SAFETY: the assert above covers every element read and written.
        unsafe {
            return add_avx(y.as_mut_ptr(), x1.as_ptr(), x2.as_ptr(), n, bounds);
        }
    }
    for i in 0..n {
        y[i] = x1[i] + x2[i];
    }
    if let Some((low, high)) = clip {
        clip_inplace(&mut y[..n], low, high);
    }
}

/// [`add_with`] writing over its first input: `y = y + x`, a rounded sum
/// being the same whichever input comes first.
pub(crate) fn add_in_place(y: &mut [f32], x: &[f32], n: usize, clip: Option<(f32, f32)>) {
    assert!(y.len() >= n && x.len() >= n);
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        let bounds = clip.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
        let y = y.as_mut_ptr();
        // SAFETY: the assert above covers every element read and written.
        unsafe {
            return add_avx(y, y, x.as_ptr(), n, bounds);
        }
    }
    for i in 0..n {
        y[i] += x[i];
    }
    if let Some((low, high)) = clip {
        clip_inplace(&mut y[..n], low, high);
    }
}

/// # Safety
/// AVX; each pointer valid for `n` floats. `y` may be `x1` or `x2`: each
/// element is read before it is written.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_avx(yp: *mut f32, ap: *const f32, bp: *const f32, n: usize, bounds: (f32, f32)) {
    use std::arch::x86_64::*;
    // SAFETY: the caller's contract.
    unsafe {
        let (lo, hi) = (_mm256_set1_ps(bounds.0), _mm256_set1_ps(bounds.1));
        let clamps = bounds != (f32::NEG_INFINITY, f32::INFINITY);
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_add_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)));
            _mm256_storeu_ps(yp.add(i), vclip_if!(clamps, 8, v, lo, hi));
            i += 8;
        }
        while i < n {
            *yp.add(i) = (*ap.add(i) + *bp.add(i)).clamp(bounds.0, bounds.1);
            i += 1;
        }
    }
}

/// `y = max(y, x)`.
pub(crate) fn max_into(y: &mut [f32], x: &[f32], n: usize) {
    assert!(y.len() >= n && x.len() >= n);
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        // SAFETY: the assert above covers every element read and written.
        unsafe {
            return max_avx(y, x, n);
        }
    }
    for i in 0..n {
        if x[i] > y[i] {
            y[i] = x[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn max_avx(y: &mut [f32], x: &[f32], n: usize) {
    use std::arch::x86_64::*;
    // SAFETY: `max_into` asserted both slices hold at least `n` floats.
    unsafe {
        let (yp, xp) = (y.as_mut_ptr(), x.as_ptr());
        let mut i = 0usize;
        while i + 8 <= n {
            _mm256_storeu_ps(
                yp.add(i),
                _mm256_max_ps(_mm256_loadu_ps(yp.add(i)), _mm256_loadu_ps(xp.add(i))),
            );
            i += 8;
        }
        while i < n {
            if *xp.add(i) > *yp.add(i) {
                *yp.add(i) = *xp.add(i);
            }
            i += 1;
        }
    }
}

/// Clamp every value to `[low, high]` in place, exactly as [`f32::clamp`]
/// (and the standalone `Clip` operation) does.
pub(crate) fn clip_inplace(y: &mut [f32], low: f32, high: f32) {
    #[cfg(target_arch = "x86_64")]
    if simd_tier() >= 2 {
        // SAFETY: AVX detected; the body stays inside `y`.
        return unsafe { clip_avx(y, low, high) };
    }
    for v in y {
        *v = v.clamp(low, high);
    }
}

/// `min(high, max(low, v))` with these operand orders is `f32::clamp` bit for
/// bit: MAXPS/MINPS return their second operand for a NaN or a tie, so NaN
/// and -0 pass through as `clamp` leaves them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn clip_avx(y: &mut [f32], low: f32, high: f32) {
    use std::arch::x86_64::*;
    let (lo, hi) = (_mm256_set1_ps(low), _mm256_set1_ps(high));
    let (chunks, rest) = y.as_chunks_mut::<8>();
    for chunk in chunks {
        // SAFETY: each chunk is exactly eight floats.
        unsafe {
            let v = _mm256_loadu_ps(chunk.as_ptr());
            _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_min_ps(hi, _mm256_max_ps(lo, v)));
        }
    }
    for v in rest {
        *v = v.clamp(low, high);
    }
}

/// Squeeze-and-excite gating: `y[p][k] = x[p][k] * gate[k]` for `pixels`
/// pixels of `c` channels. One rounded product per value, so every tier
/// gives the same bits; the vector tiers only widen the loop.
pub(crate) fn channel_mul(y: &mut [f32], x: &[f32], gate: &[f32], pixels: usize, c: usize) {
    let n = span(pixels, c);
    assert!(y.len() >= n && x.len() >= n && gate.len() >= c && c > 0);
    let (y, x, gate) = (&mut y[..n], &x[..n], &gate[..c]);
    #[cfg(target_arch = "x86_64")]
    match simd_tier() {
        // SAFETY: AVX, or SSE4.1, was detected.
        2.. => return unsafe { channel_mul_avx(y, x, gate) },
        1 => return unsafe { channel_mul_sse(y, x, gate) },
        _ => {}
    }
    channel_mul_rows(y, x, gate);
}

#[inline(always)]
fn channel_mul_rows(y: &mut [f32], x: &[f32], gate: &[f32]) {
    let c = gate.len();
    for (out, pixel) in y.chunks_exact_mut(c).zip(x.chunks_exact(c)) {
        for ((o, &v), &g) in out.iter_mut().zip(pixel).zip(gate) {
            *o = v * g;
        }
    }
}

/// [`channel_mul_rows`] compiled for AVX: the compiler widens the product loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn channel_mul_avx(y: &mut [f32], x: &[f32], gate: &[f32]) {
    channel_mul_rows(y, x, gate);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn channel_mul_sse(y: &mut [f32], x: &[f32], gate: &[f32]) {
    channel_mul_rows(y, x, gate);
}

/// `y = x * gate + x`, the product rounded first, then clamped to `bounds`:
/// a squeeze-and-excite gate and the residual add that alone reads its
/// product, in one pass, whichever input the add names first (a rounded sum
/// does not depend on the order of its terms). Two rounded operations and a
/// clamp per value, as [`channel_mul`] then [`add_with`] give them, the same
/// bits on every tier.
pub(crate) fn gate_add(
    y: &mut [f32],
    x: &[f32],
    gate: &[f32],
    pixels: usize,
    c: usize,
    bounds: (f32, f32),
) {
    let n = span(pixels, c);
    assert!(y.len() >= n && x.len() >= n && gate.len() >= c && c > 0);
    let (y, x, gate) = (&mut y[..n], &x[..n], &gate[..c]);
    #[cfg(target_arch = "x86_64")]
    match simd_tier() {
        // SAFETY: `streamed` checked the contract.
        2.. if streamed(y, c) => return unsafe { gate_add_stream_avx(y, x, gate, bounds) },
        // SAFETY: AVX, or SSE4.1, was detected.
        2.. => return unsafe { gate_add_avx(y, x, gate, bounds) },
        1 => return unsafe { gate_add_sse(y, x, gate, bounds) },
        _ => {}
    }
    gate_add_rows(y, x, gate, bounds);
}

#[inline(always)]
fn gate_add_rows(y: &mut [f32], x: &[f32], gate: &[f32], bounds: (f32, f32)) {
    let c = gate.len();
    for (out, pixel) in y.chunks_exact_mut(c).zip(x.chunks_exact(c)) {
        for ((o, &v), &g) in out.iter_mut().zip(pixel).zip(gate) {
            *o = (v * g + v).clamp(bounds.0, bounds.1);
        }
    }
}

/// [`gate_add_rows`] compiled for AVX: the compiler widens the loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gate_add_avx(y: &mut [f32], x: &[f32], gate: &[f32], bounds: (f32, f32)) {
    gate_add_rows(y, x, gate, bounds);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn gate_add_sse(y: &mut [f32], x: &[f32], gate: &[f32], bounds: (f32, f32)) {
    gate_add_rows(y, x, gate, bounds);
}

/// [`gate_add`] writing over its input: `y = y * gate + y`.
pub(crate) fn gate_add_in_place(
    y: &mut [f32],
    gate: &[f32],
    pixels: usize,
    c: usize,
    bounds: (f32, f32),
) {
    let n = span(pixels, c);
    assert!(y.len() >= n && gate.len() >= c && c > 0);
    let (y, gate) = (&mut y[..n], &gate[..c]);
    #[cfg(target_arch = "x86_64")]
    match simd_tier() {
        // SAFETY: AVX, or SSE4.1, was detected.
        2.. => return unsafe { gate_add_in_place_avx(y, gate, bounds) },
        1 => return unsafe { gate_add_in_place_sse(y, gate, bounds) },
        _ => {}
    }
    gate_add_in_place_rows(y, gate, bounds);
}

#[inline(always)]
fn gate_add_in_place_rows(y: &mut [f32], gate: &[f32], bounds: (f32, f32)) {
    for out in y.chunks_exact_mut(gate.len()) {
        for (o, &g) in out.iter_mut().zip(gate) {
            *o = (*o * g + *o).clamp(bounds.0, bounds.1);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gate_add_in_place_avx(y: &mut [f32], gate: &[f32], bounds: (f32, f32)) {
    gate_add_in_place_rows(y, gate, bounds);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn gate_add_in_place_sse(y: &mut [f32], gate: &[f32], bounds: (f32, f32)) {
    gate_add_in_place_rows(y, gate, bounds);
}

/// [`gate_add_rows`] with streaming stores, eight channels at a time.
///
/// # Safety
/// AVX; `y` 32-byte aligned; `gate.len()` a multiple of eight that divides
/// the lengths of `y` and `x`, which are equal.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gate_add_stream_avx(y: &mut [f32], x: &[f32], gate: &[f32], bounds: (f32, f32)) {
    use std::arch::x86_64::*;
    let c = gate.len();
    // SAFETY: the caller's contract keeps every access inside whole pixels
    // and every store aligned.
    unsafe {
        let (lo, hi) = (_mm256_set1_ps(bounds.0), _mm256_set1_ps(bounds.1));
        for (out, pixel) in y.chunks_exact_mut(c).zip(x.chunks_exact(c)) {
            for k in (0..c).step_by(8) {
                let v = _mm256_loadu_ps(pixel.as_ptr().add(k));
                let sum = _mm256_add_ps(_mm256_mul_ps(v, _mm256_loadu_ps(gate.as_ptr().add(k))), v);
                _mm256_stream_ps(out.as_mut_ptr().add(k), vclip!(8, sum, lo, hi));
            }
        }
        _mm_sfence();
    }
}

/// Whether `y`, whole pixels of `c` floats, is written with streaming
/// stores, which skip reading each cache line in before overwriting it:
/// past twice the last-level cache a tensor has left it before anything
/// reads it back, so that read only costs bandwidth. Linux reports the
/// cache size; 16 MiB is assumed where it does not.
pub(crate) fn streamed(y: &[f32], c: usize) -> bool {
    static CACHE_BYTES: OnceLock<usize> = OnceLock::new();
    let cache = *CACHE_BYTES.get_or_init(|| {
        let size = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size");
        let kib = size
            .ok()
            .and_then(|s| s.trim().strip_suffix('K')?.parse::<usize>().ok());
        kib.and_then(|kib| kib.checked_mul(1024))
            .unwrap_or(16 << 20)
    });
    streamable(y, c) && y.len().saturating_mul(4) >= cache.saturating_mul(2)
}

/// What the streaming kernels need: AVX, and `y` 32-byte aligned whole
/// pixels of a multiple of eight channels.
fn streamable(y: &[f32], c: usize) -> bool {
    cfg!(target_arch = "x86_64")
        && simd_tier() >= 2
        && c.is_multiple_of(8)
        && y.len().is_multiple_of(c)
        && (y.as_ptr() as usize).is_multiple_of(32)
}

/// One output row of a max pool with streaming stores: `x` holds the live
/// input rows, `iw` pixels each, and pixel `ox` is the maximum over columns
/// `ox * sw` onwards, `kw` of them cut at the row's end, of every row, the
/// first tap copied and each later one taken in order as [`max_into`] does.
pub(crate) fn max_row_streamed(
    y: &mut [f32],
    x: &[f32],
    iw: usize,
    kw: usize,
    sw: usize,
    c: usize,
) {
    assert!(streamable(y, c) && !y.is_empty() && kw > 0 && iw > 0);
    let (ow, rows) = (y.len() / c, x.len() / (iw * c));
    assert!(rows > 0 && x.len() == rows * iw * c && (ow - 1) * sw < iw);
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `streamable` covers AVX and aligned stores of whole pixels;
    // every window starts inside the rows and stops at their end.
    unsafe {
        max_row_avx::<true>(y.as_mut_ptr(), x.as_ptr(), ow, rows, iw, kw, sw, c);
    }
}

/// One output row of a max pool at stride 1 written over its input:
/// `rows`, whole rows of `iw` pixels from the output row's own, becomes in
/// its first row each pixel's maximum over the `kw` columns from it, of
/// every row, as [`max_row_streamed`] computes it. A window reaches only
/// down and right, so each pixel is stored after the last window reading
/// it. False, with nothing done, below the AVX tier or for a channel count
/// not a multiple of eight.
pub(crate) fn max_row_in_place(rows: &mut [f32], iw: usize, kw: usize, c: usize) -> bool {
    if !(cfg!(target_arch = "x86_64") && simd_tier() >= 2 && c.is_multiple_of(8)) {
        return false;
    }
    assert!(kw > 0 && iw > 0 && !rows.is_empty() && rows.len().is_multiple_of(iw * c));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: AVX was detected; every window starts inside the rows and
    // stops at their end, and every store is to the first row.
    unsafe {
        let p = rows.as_mut_ptr();
        max_row_avx::<false>(p, p, iw, rows.len() / (iw * c), iw, kw, 1, c);
    }
    true
}

/// `ow` pixels of a max pool row into `y`, from `rows` input rows of `iw`
/// pixels at `x`, the first tap loaded and each later one taken with
/// `MAXPS` in row-major order, as [`max_into`] takes them.
///
/// # Safety
/// AVX; `c` a multiple of eight; `x` valid for `rows * iw * c` floats and
/// `y` for `ow * c`, 32-byte aligned when `STREAM`; `(ow - 1) * sw < iw`.
/// `y` may be `x` at stride 1.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn max_row_avx<const STREAM: bool>(
    y: *mut f32,
    x: *const f32,
    ow: usize,
    rows: usize,
    iw: usize,
    kw: usize,
    sw: usize,
    c: usize,
) {
    use std::arch::x86_64::*;
    // SAFETY: the caller's contract.
    unsafe {
        for ox in 0..ow {
            let (first, end) = (ox * sw, (ox * sw + kw).min(iw));
            for k in (0..c).step_by(8) {
                let tap = |row: usize, ix: usize| x.add((row * iw + ix) * c + k);
                let mut top = _mm256_loadu_ps(tap(0, first));
                for ix in first + 1..end {
                    top = _mm256_max_ps(top, _mm256_loadu_ps(tap(0, ix)));
                }
                for row in 1..rows {
                    for ix in first..end {
                        top = _mm256_max_ps(top, _mm256_loadu_ps(tap(row, ix)));
                    }
                }
                if STREAM {
                    _mm256_stream_ps(y.add(ox * c + k), top);
                } else {
                    _mm256_storeu_ps(y.add(ox * c + k), top);
                }
            }
        }
        if STREAM {
            _mm_sfence();
        }
    }
}

/// One output row of an upsampled sum with streaming stores: each `c`-float
/// pixel of `x` repeated `times` times, added to `addend`.
pub(crate) fn upsample_add_row_streamed(
    y: &mut [f32],
    addend: &[f32],
    x: &[f32],
    times: usize,
    c: usize,
) {
    assert!(streamable(y, c) && addend.len() == y.len() && x.len() * times == y.len());
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `streamable` covers AVX and aligned stores of whole pixels;
    // the lengths match.
    unsafe {
        upsample_add_row_stream_avx(y, addend, x, times, c);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn upsample_add_row_stream_avx(
    y: &mut [f32],
    addend: &[f32],
    x: &[f32],
    times: usize,
    c: usize,
) {
    use std::arch::x86_64::*;
    // SAFETY: `upsample_add_row_streamed`'s asserts.
    unsafe {
        let pixels = y.chunks_exact_mut(c).zip(addend.chunks_exact(c));
        for (i, (out, add)) in pixels.enumerate() {
            let pixel = x.as_ptr().add(i / times * c);
            for k in (0..c).step_by(8) {
                let v = _mm256_loadu_ps(pixel.add(k));
                let sum = _mm256_add_ps(_mm256_loadu_ps(add.as_ptr().add(k)), v);
                _mm256_stream_ps(out.as_mut_ptr().add(k), sum);
            }
        }
        _mm_sfence();
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{bits, ramp};
    use super::*;

    #[test]
    fn fused_add_clip_matches_two_passes() {
        let special = [
            f32::NAN,
            -0.0,
            0.0,
            6.0,
            -1.0,
            7.5,
            f32::INFINITY,
            3.25,
            -f32::INFINITY,
        ];
        for n in [1, 7, 8, 13, 45] {
            let a: Vec<f32> = (0..n).map(|i| special[i % special.len()]).collect();
            let b: Vec<f32> = (0..n)
                .map(|i| special[(i * 5 + 2) % special.len()] * 0.5)
                .collect();
            let mut want = vec![0.0; n];
            add(&mut want, &a, &b, n);
            clip_inplace(&mut want, 0.0, 6.0);
            let mut got = vec![0.0; n];
            add_with(&mut got, &a, &b, n, Some((0.0, 6.0)));
            assert_eq!(bits(&got), bits(&want), "n={n}");
        }
    }

    #[test]
    fn vector_clip_matches_clamp() {
        let special = [
            f32::NAN,
            -f32::NAN,
            -0.0,
            0.0,
            6.0,
            -1.0,
            7.5,
            f32::INFINITY,
            3.25,
        ];
        let values: Vec<f32> = (0..45).map(|i| special[i % special.len()]).collect();
        let mut got = values.clone();
        clip_inplace(&mut got, 0.0, 6.0);
        for (&g, &v) in got.iter().zip(&values) {
            assert_eq!(g.to_bits(), v.clamp(0.0, 6.0).to_bits(), "{v}");
        }
    }

    #[test]
    fn channel_mul_tiers_match_products() {
        for c in [16, 24, 7, 1] {
            let pixels = 13;
            let (x, gate) = (ramp(pixels * c, 0.3), ramp(c, 1.1));
            let mut y = vec![f32::NAN; pixels * c + 1];
            channel_mul(&mut y, &x, &gate, pixels, c);
            for (i, &v) in y[..pixels * c].iter().enumerate() {
                assert_eq!(v.to_bits(), (x[i] * gate[i % c]).to_bits(), "c={c} at {i}");
            }
            assert!(y[pixels * c].is_nan(), "c={c}: wrote past the tensor");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn streamed_stores_match_cached_stores() {
        if simd_tier() < 2 {
            return;
        }
        // `n` floats of `buffer` from a 32-byte boundary.
        fn aligned(buffer: &mut Vec<f32>, n: usize) -> &mut [f32] {
            buffer.resize(n + 8, f32::NAN);
            let skip = buffer.as_ptr().align_offset(32);
            &mut buffer[skip..skip + n]
        }
        let mut buffer = Vec::new();
        for c in [8, 24, 96] {
            let mut x = ramp(7 * 3 * c, 0.4);
            x[c + 5] = f32::NAN;
            let gate = ramp(c, 1.1);
            for bounds in [(f32::NEG_INFINITY, f32::INFINITY), (-0.2, 0.3)] {
                let mut want = vec![0.0; 7 * c];
                gate_add_rows(&mut want, &x[..7 * c], &gate, bounds);
                let got = aligned(&mut buffer, 7 * c);
                // SAFETY: AVX, an aligned destination of whole pixels.
                unsafe { gate_add_stream_avx(got, &x[..7 * c], &gate, bounds) };
                assert_eq!(bits(got), bits(&want), "gate c={c} {bounds:?}");
            }
            // Max pools over one row or several, windows cut at the end.
            for (rows, kw, sw) in [(1, 2, 1), (2, 2, 1), (3, 3, 2)] {
                let (iw, x) = (7, &x[..rows * 7 * c]);
                let ow = (iw - 1) / sw + 1;
                let mut want = vec![0.0; ow * c];
                for (ox, out) in want.chunks_exact_mut(c).enumerate() {
                    let mut seeded = false;
                    for row in 0..rows {
                        for ix in ox * sw..(ox * sw + kw).min(iw) {
                            let tap = &x[(row * iw + ix) * c..][..c];
                            if seeded {
                                max_into(out, tap, c);
                            } else {
                                out.copy_from_slice(tap);
                                seeded = true;
                            }
                        }
                    }
                }
                let got = aligned(&mut buffer, ow * c);
                max_row_streamed(got, x, iw, kw, sw, c);
                assert_eq!(bits(got), bits(&want), "pool c={c} {rows}x{kw}/{sw}");
            }
            let (addend, x) = (ramp(6 * c, 2.3), &x[..3 * c]);
            let want: Vec<f32> = (0..6 * c)
                .map(|i| addend[i] + x[i / (2 * c) * c + i % c])
                .collect();
            let got = aligned(&mut buffer, 6 * c);
            upsample_add_row_streamed(got, &addend, x, 2, c);
            assert_eq!(bits(got), bits(&want), "upsample add c={c}");
        }
    }
}
