//! Pointwise (1x1) convolution: a matrix multiply over a run of pixels,
//! tiled per tier, with optional 16-column weight panels and a fused clamp.

use super::{clip_inplace, matvec_t, simd_tier, span};

/// [`pointwise_with`] with row-major weights and no clamp, for the tests.
#[cfg(test)]
pub(super) fn pointwise(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    count: usize,
    ci: usize,
    co: usize,
) {
    pointwise_with(y, w, x, bias, count, ci, co, false, None);
}

/// 1x1 convolution over `count` neighbouring pixels: `y[p] = bias + W^T x[p]`,
/// with `W` laid out `[in_channel][out_channel]`.
///
/// Most convolutions of both PP-OCR networks are this shape, so it gets a
/// dedicated kernel rather than one matrix-vector call per pixel.
///
/// What limits it is not multiply throughput but the load ports. Every input
/// value has to be splashed across a whole vector register before it can be
/// multiplied, and on x86 that splash is itself a load. Holding two 8-wide
/// output blocks per tile lets two weight-row loads and four splashes feed
/// eight vector multiply-adds: six loads per eight products, where the plain
/// one-pixel-at-a-time form needs five loads per four.
///
/// Multiples of sixteen output channels use six pixels on AVX2 and four on
/// AVX, with a four-pixel remainder before the scalar tail. Irregular widths
/// use four pixels with 8/4/scalar output tails and add bias last.
///
/// Weights may be stored as 16-column panels, `[co/16][ci][16]`, by
/// `Model::load`, and optionally clamping each output
/// to `clip` as it is stored. Panels are the same products and sums in the
/// same order; only the weight addresses change, so a panel is contiguous in
/// cache instead of `co` floats apart per input channel (with `co` = 128, 256
/// or 1152 that stride maps every row of a panel to a few L1 sets and evicts
/// it). They need `co % 16 == 0` and an AVX tier: the only kernels that read
/// them. The clamp is [`clip_inplace`] bit for bit, without a second pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pointwise_with(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    clip: Option<(f32, f32)>,
) {
    assert!(y.len() >= span(count, co) && x.len() >= span(count, ci) && w.len() >= span(ci, co));
    assert!(!panels || (co.is_multiple_of(16) && simd_tier() >= 2));
    // MAXPS/MINPS against infinities return every value unchanged, NaN and
    // -0 included, so the vector kernels clamp unconditionally.
    let vector_clip = simd_tier() >= 2;
    let bounds = clip
        .filter(|_| vector_clip)
        .unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
    if let Some(b) = bias {
        assert!(b.len() >= co);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let tiled = 0;
    #[cfg(target_arch = "x86_64")]
    let tiled = {
        let bias_ptr = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        let tier = simd_tier();
        // SAFETY: the asserts above cover every element the bodies touch, and a
        // null bias pointer is only read when `bias` was `Some`.
        unsafe {
            if co.is_multiple_of(16) {
                // Exactly 16 outputs gets its own copy of each tile: a constant
                // `co` drops the panel loop and the strided store addressing.
                let exact = co == 16;
                match tier {
                    3 => {
                        // Twelve preserves the four-pixel bias-first boundary.
                        let prefix = count / 12 * 12;
                        let (wide, narrow) = if exact {
                            (pw_p6o16_avx2::<16> as Tile16, pw_p4o16_avx2::<16> as Tile16)
                        } else {
                            (pw_p6o16_avx2::<0> as Tile16, pw_p4o16_avx2::<0> as Tile16)
                        };
                        let done = wide(y, w, x, bias_ptr, prefix, ci, co, panels, bounds);
                        done + narrow(
                            &mut y[done * co..],
                            w,
                            &x[done * ci..],
                            bias_ptr,
                            count - done,
                            ci,
                            co,
                            panels,
                            bounds,
                        )
                    }
                    2 if exact => {
                        pw_p4o16_avx::<16>(y, w, x, bias_ptr, count, ci, co, panels, bounds)
                    }
                    // Only while the weights fit a 256 KB L2: past it the
                    // 4x16 tile is faster.
                    2 if co.is_multiple_of(32) && ci * co <= 65536 => {
                        pw_p2o32_avx(y, w, x, bias_ptr, count, ci, co, panels, bounds)
                    }
                    2 => pw_p4o16_avx::<0>(y, w, x, bias_ptr, count, ci, co, panels, bounds),
                    1 => pw_any_sse(y, w, x, bias_ptr, count, ci, co, bounds),
                    _ => 0,
                }
            } else {
                match tier {
                    3 => pw_any_avx2(y, w, x, bias_ptr, count, ci, co, bounds),
                    2 => pw_any_avx(y, w, x, bias_ptr, count, ci, co, bounds),
                    1 => pw_any_sse(y, w, x, bias_ptr, count, ci, co, bounds),
                    _ => 0,
                }
            }
        }
    };
    pointwise_rest(y, w, x, bias, tiled, count, ci, co, panels, bounds);
    if let Some((low, high)) = clip.filter(|_| !vector_clip) {
        clip_inplace(&mut y[..count * co], low, high);
    }
}

/// Pixels the tiled kernels did not cover, one matrix-vector call each.
// Shape, weights and bias are eight separate facts about one convolution;
// bundling them into a struct would only move the argument list.
#[allow(clippy::too_many_arguments)]
fn pointwise_rest(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    from: usize,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) {
    // Up to four leftover pixels share one pass over the weights instead of
    // one pass each, with the arithmetic of `matvec_t` plus a trailing bias.
    #[cfg(target_arch = "x86_64")]
    if co.is_multiple_of(8) && simd_tier() >= 2 {
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        let mut from = from;
        while from < count {
            let pixels = (count - from).min(4);
            // SAFETY: `pointwise` checked every pixel below `count` and the
            // bias length; co is a multiple of eight; AVX was detected.
            unsafe {
                let (yr, xr) = (&mut y[from * co..], &x[from * ci..]);
                if simd_tier() == 3 {
                    pw_rest_avx2(yr, w, xr, b, pixels, ci, co, panels, bounds);
                } else {
                    pw_rest_avx(yr, w, xr, b, pixels, ci, co, panels, bounds);
                }
            }
            from += pixels;
        }
        return;
    }
    assert!(!panels, "panel weights reach only the AVX kernels");
    for p in from..count {
        let out = &mut y[p * co..p * co + co];
        matvec_t(out, w, &x[p * ci..p * ci + ci], ci, co);
        if let Some(b) = bias {
            for (v, &bv) in out.iter_mut().zip(b) {
                *v += bv;
            }
        }
        for v in out {
            *v = v.clamp(bounds.0, bounds.1);
        }
    }
}

/// Leftover pixels (at most four) for `co % 8 == 0`: each output is zero,
/// then one multiply-add per input channel in order, then the bias.
#[cfg(target_arch = "x86_64")]
macro_rules! pw_rest_body {
    ($y:expr, $w:expr, $x:expr, $b:expr, $pixels:expr, $ci:expr, $co:expr, $panels:expr, $bounds:expr, $fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: the caller's contract, checked by `pointwise`.
        unsafe {
            let (yp, wp, xp, b) = ($y.as_mut_ptr(), $w.as_ptr(), $x.as_ptr(), $b);
            let (lo, hi) = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1));
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let (pixels, ci, co, panels) = ($pixels, $ci, $co, $panels);
            let mut j = 0;
            // One or two pixels would leave two or four chains waiting on
            // multiply-add latency: take several 16-column blocks at once.
            macro_rules! blocks {
                ($p:literal, $blocks:literal) => {
                    while j + 16 * $blocks <= co {
                        let mut acc = [[_mm256_setzero_ps(); 2 * $blocks]; $p];
                        for i in 0..ci {
                            for m in 0..$blocks {
                                let jb = j + 16 * m;
                                let row = if panels {
                                    wp.add(jb * ci + i * 16)
                                } else {
                                    wp.add(i * co + jb)
                                };
                                let (r0, r1) = (_mm256_loadu_ps(row), _mm256_loadu_ps(row.add(8)));
                                for (k, a) in acc.iter_mut().enumerate() {
                                    let v = _mm256_set1_ps(*xp.add(k * ci + i));
                                    a[2 * m] = fma8!($fma, r0, v, a[2 * m]);
                                    a[2 * m + 1] = fma8!($fma, r1, v, a[2 * m + 1]);
                                }
                            }
                        }
                        for (k, a) in acc.iter().enumerate() {
                            for (m, &v) in a.iter().enumerate() {
                                let v = if b.is_null() {
                                    v
                                } else {
                                    _mm256_add_ps(v, _mm256_loadu_ps(b.add(j + 8 * m)))
                                };
                                _mm256_storeu_ps(
                                    yp.add(k * co + j + 8 * m),
                                    vclip_if!(clamps, 8, v, lo, hi),
                                );
                            }
                        }
                        j += 16 * $blocks;
                    }
                };
            }
            match pixels {
                1 => blocks!(1, 4),
                2 => blocks!(2, 2),
                _ => {}
            }
            while j < co {
                let wide = j + 16 <= co;
                // Row `i` of this 16-column block: in place, or in its panel.
                let (base, stride) = if panels { (j * ci, 16) } else { (j, co) };
                let mut low = [_mm256_setzero_ps(); 4];
                let mut high = [_mm256_setzero_ps(); 4];
                for i in 0..ci {
                    let row = wp.add(base + i * stride);
                    let r0 = _mm256_loadu_ps(row);
                    let r1 = if wide {
                        _mm256_loadu_ps(row.add(8))
                    } else {
                        r0
                    };
                    for k in 0..pixels {
                        let v = _mm256_set1_ps(*xp.add(k * ci + i));
                        low[k] = fma8!($fma, r0, v, low[k]);
                        if wide {
                            high[k] = fma8!($fma, r1, v, high[k]);
                        }
                    }
                }
                for k in 0..pixels {
                    let (mut a, mut c) = (low[k], high[k]);
                    if !b.is_null() {
                        a = _mm256_add_ps(a, _mm256_loadu_ps(b.add(j)));
                        if wide {
                            c = _mm256_add_ps(c, _mm256_loadu_ps(b.add(j + 8)));
                        }
                    }
                    _mm256_storeu_ps(yp.add(k * co + j), vclip_if!(clamps, 8, a, lo, hi));
                    if wide {
                        _mm256_storeu_ps(yp.add(k * co + j + 8), vclip_if!(clamps, 8, c, lo, hi));
                    }
                }
                j += if wide { 16 } else { 8 };
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_rest_avx2(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    pixels: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) {
    pw_rest_body!(y, w, x, b, pixels, ci, co, panels, bounds, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_rest_avx(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    pixels: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) {
    pw_rest_body!(y, w, x, b, pixels, ci, co, panels, bounds, false)
}

/// Register tile by sixteen output channels. Returns the pixel count handled.
#[cfg(target_arch = "x86_64")]
macro_rules! pw_o16_body {
    ($y:expr, $w:expr, $x:expr, $b:expr, $count:expr, $ci:expr, $co:expr, $panels:expr, $bounds:expr, $pixels:expr, $fma:tt) => {
        pw_o16_body!($y, $w, $x, $b, $count, $ci, $co, $panels, $bounds, $pixels, $fma, false)
    };
    ($y:expr, $w:expr, $x:expr, $b:expr, $count:expr, $ci:expr, $co:expr, $panels:expr, $bounds:expr, $pixels:expr, $fma:tt, $skew:expr) => {{
        #[allow(unused_imports)]
        use std::arch::x86_64::*;
        // SAFETY: bounds are the caller's contract, checked by `pointwise`.
        unsafe {
            let (y, w, x, b, count, ci, co, panels) = ($y, $w, $x, $b, $count, $ci, $co, $panels);
            let (floor, ceil) = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1));
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let (yp, wp, xp) = (y.as_mut_ptr(), w.as_ptr(), x.as_ptr());
            // One tile: `$pixels` pixels by the 16 output channels at `jb`;
            // the tile's pixel `k` is read at `src + offsets[k]`. Opaque
            // offsets keep LLVM from seeing that the pixels are `step` apart
            // and rebuilding their addresses with per-channel arithmetic, five
            // extra micro-ops in every trip of the channel loop. Made once per
            // call, they cost a tile only additions, less than opaque pointers
            // made per tile.
            let offsets = |step: usize| -> [usize; $pixels] {
                std::hint::black_box(std::array::from_fn(|k| k * step))
            };
            let tile = |src: *const f32, offsets: [usize; $pixels], p: usize, jb: usize| {
                let (lo, hi) = if b.is_null() {
                    (_mm256_setzero_ps(), _mm256_setzero_ps())
                } else {
                    (_mm256_loadu_ps(b.add(jb)), _mm256_loadu_ps(b.add(jb + 8)))
                };
                let mut low = [lo; $pixels];
                let mut high = [hi; $pixels];
                let (base, stride) = if panels { (jb * ci, 16) } else { (jb, co) };
                let px: [*const f32; $pixels] = std::array::from_fn(|k| src.add(offsets[k]));
                let mut row = wp.add(base);
                for i in 0..ci {
                    let r0 = _mm256_loadu_ps(row);
                    let r1 = _mm256_loadu_ps(row.add(8));
                    for k in 0..$pixels {
                        let value = _mm256_set1_ps(*px[k].add(i));
                        low[k] = fma8!($fma, r0, value, low[k]);
                        high[k] = fma8!($fma, r1, value, high[k]);
                    }
                    row = row.add(stride);
                }
                for k in 0..$pixels {
                    let (a, c) = (
                        vclip_if!(clamps, 8, low[k], floor, ceil),
                        vclip_if!(clamps, 8, high[k], floor, ceil),
                    );
                    _mm256_storeu_ps(yp.add((p + k) * co + jb), a);
                    _mm256_storeu_ps(yp.add((p + k) * co + jb + 8), c);
                }
            };
            let done = count / $pixels * $pixels;
            // Sandy Bridge cannot serve two L1 reads in one cycle from the
            // same 16-byte bank of different lines (address bits 4-6). A pixel
            // stride of a multiple of 128 bytes puts every broadcast of a tile
            // in one bank; copied at a stride of ci + 4 floats, the tile's
            // pixels fall in different banks. Same values, so same sums. The
            // copy pays for itself from four 16-column panels per pixel tile;
            // repeated for every panel, as the order below for co > count
            // would need, it costs more than the conflicts it removes.
            let skew = $skew && ci.is_multiple_of(32) && ci <= SKEW_MAX_CI && co >= 64;
            let stride = ci + 4;
            let (direct, skewed) = (offsets(ci), offsets(stride));
            const COPY: usize = if $skew {
                $pixels * (SKEW_MAX_CI + 4)
            } else {
                0
            };
            let mut copy = [std::mem::MaybeUninit::<f32>::uninit(); COPY];
            let cp = copy.as_mut_ptr().cast::<f32>();
            // SAFETY (with `skew`): pixel p + k < count lies in x, and
            // k * stride + ci <= COPY because ci <= SKEW_MAX_CI. A tile reads
            // only the ci floats copied for each of its pixels.
            let gather = |p: usize| {
                for k in 0..$pixels {
                    std::ptr::copy_nonoverlapping(xp.add((p + k) * ci), cp.add(k * stride), ci);
                }
            };
            // Same tiles either way. When the weights outweigh the pixels,
            // finish each 16-column panel across every pixel while it is in
            // L1, instead of streaming all weights once per pixel tile.
            if co > count {
                for jb in (0..co).step_by(16) {
                    for p in (0..done).step_by($pixels) {
                        tile(xp.add(p * ci), direct, p, jb);
                    }
                }
            } else {
                for p in (0..done).step_by($pixels) {
                    if skew {
                        gather(p);
                    }
                    for jb in (0..co).step_by(16) {
                        if skew {
                            tile(cp.cast_const(), skewed, p, jb);
                        } else {
                            tile(xp.add(p * ci), direct, p, jb);
                        }
                    }
                }
            }
            done
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_p6o16_avx2<const CO: usize>(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) -> usize {
    let co = if CO > 0 { CO } else { co };
    pw_o16_body!(y, w, x, b, count, ci, co, panels, bounds, 6, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_p4o16_avx2<const CO: usize>(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) -> usize {
    let co = if CO > 0 { CO } else { co };
    pw_o16_body!(y, w, x, b, count, ci, co, panels, bounds, 4, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_p4o16_avx<const CO: usize>(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) -> usize {
    let co = if CO > 0 { CO } else { co };
    pw_o16_body!(y, w, x, b, count, ci, co, panels, bounds, 4, false, true)
}

/// Register tile of two pixels by 32 output channels, for `co % 32 == 0` on
/// AVX: each broadcast input feeds four registers instead of two, which
/// beats `pw_p4o16_avx` while the weights fit in L2. Each output keeps the
/// bias-first, input-ordered sum: the same bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_p2o32_avx(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    panels: bool,
    bounds: (f32, f32),
) -> usize {
    use std::arch::x86_64::*;
    // SAFETY: bounds are the caller's contract, checked by `pointwise`;
    // co % 32 == 0, so every 32-column tile lies inside a row of y and w.
    unsafe {
        let (floor, ceil) = (_mm256_set1_ps(bounds.0), _mm256_set1_ps(bounds.1));
        let clamps = bounds != (f32::NEG_INFINITY, f32::INFINITY);
        let (yp, wp, xp) = (y.as_mut_ptr(), w.as_ptr(), x.as_ptr());
        let tile = |p: usize, jb: usize| {
            let seed = |m: usize| {
                if b.is_null() {
                    _mm256_setzero_ps()
                } else {
                    _mm256_loadu_ps(b.add(jb + 8 * m))
                }
            };
            let mut acc = [[seed(0), seed(1), seed(2), seed(3)]; 2];
            // Two 16-column panels when the weights were repacked.
            let (left, right, stride) = if panels {
                (jb * ci, (jb + 16) * ci, 16)
            } else {
                (jb, jb + 16, co)
            };
            let (x0, x1) = (xp.add(p * ci), xp.add((p + 1) * ci));
            for i in 0..ci {
                let (l, r) = (wp.add(left + i * stride), wp.add(right + i * stride));
                let rows = [
                    _mm256_loadu_ps(l),
                    _mm256_loadu_ps(l.add(8)),
                    _mm256_loadu_ps(r),
                    _mm256_loadu_ps(r.add(8)),
                ];
                let values = [_mm256_set1_ps(*x0.add(i)), _mm256_set1_ps(*x1.add(i))];
                for (a, &value) in acc.iter_mut().zip(&values) {
                    for (v, &row) in a.iter_mut().zip(&rows) {
                        *v = fma8!(false, row, value, *v);
                    }
                }
            }
            for (k, a) in acc.iter().enumerate() {
                for (m, &v) in a.iter().enumerate() {
                    let v = vclip_if!(clamps, 8, v, floor, ceil);
                    _mm256_storeu_ps(yp.add((p + k) * co + jb + 8 * m), v);
                }
            }
        };
        // The pixels `pw_p4o16_avx` would cover: the rest adds its bias last.
        let done = count / 4 * 4;
        // As in `pw_o16_body`, keep a panel in L1 when weights outweigh pixels.
        if co > count {
            for jb in (0..co).step_by(32) {
                for p in (0..done).step_by(2) {
                    tile(p, jb);
                }
            }
        } else {
            for p in (0..done).step_by(2) {
                for jb in (0..co).step_by(32) {
                    tile(p, jb);
                }
            }
        }
        done
    }
}

#[cfg(target_arch = "x86_64")]
type Tile16 = unsafe fn(
    &mut [f32],
    &[f32],
    &[f32],
    *const f32,
    usize,
    usize,
    usize,
    bool,
    (f32, f32),
) -> usize;

/// Widest input copied into the bank-skewed tile buffer of `pw_o16_body`.
#[cfg(target_arch = "x86_64")]
const SKEW_MAX_CI: usize = 1152;

// Pixel batching also for widths 24, 28, 36, 42, 88, etc. on the SSE tier;
// AVX tiers use [`pw_any8_body`]. Only the last `count % 4` pixels fall back
// to matrix-vector calls.
#[cfg(target_arch = "x86_64")]
macro_rules! pw_any_body {
    ($y:expr,$w:expr,$x:expr,$b:expr,$count:expr,$ci:expr,$co:expr,$bounds:expr,$lanes:tt,$fma:tt) => {{
        unsafe {
            let (yp, wp, xp, b) = ($y.as_mut_ptr(), $w.as_ptr(), $x.as_ptr(), $b);
            let (low, high) = $bounds;
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let (lo, hi) = (vsplat!($lanes, low), vsplat!($lanes, high));
            let (count, ci, co) = ($count, $ci, $co);
            let mut p = 0;
            while p + 4 <= count {
                let mut j = 0;
                // Two column blocks give eight independent chains, enough to
                // hide multiply-add latency; four stalled on it. Each output
                // still sums over `i` in order and adds bias last.
                while j + 2 * $lanes <= co {
                    let z = vsplat!($lanes, 0.0);
                    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
                    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
                    for i in 0..ci {
                        let row = vld!($lanes, wp.add(i * co + j));
                        let next = vld!($lanes, wp.add(i * co + j + $lanes));
                        let s0 = vsplat!($lanes, *xp.add(p * ci + i));
                        let s1 = vsplat!($lanes, *xp.add((p + 1) * ci + i));
                        let s2 = vsplat!($lanes, *xp.add((p + 2) * ci + i));
                        let s3 = vsplat!($lanes, *xp.add((p + 3) * ci + i));
                        a0 = vfma!($lanes, $fma, row, s0, a0);
                        c0 = vfma!($lanes, $fma, next, s0, c0);
                        a1 = vfma!($lanes, $fma, row, s1, a1);
                        c1 = vfma!($lanes, $fma, next, s1, c1);
                        a2 = vfma!($lanes, $fma, row, s2, a2);
                        c2 = vfma!($lanes, $fma, next, s2, c2);
                        a3 = vfma!($lanes, $fma, row, s3, a3);
                        c3 = vfma!($lanes, $fma, next, s3, c3);
                    }
                    if !b.is_null() {
                        let bias = vld!($lanes, b.add(j));
                        let more = vld!($lanes, b.add(j + $lanes));
                        a0 = vadd!($lanes, a0, bias);
                        a1 = vadd!($lanes, a1, bias);
                        a2 = vadd!($lanes, a2, bias);
                        a3 = vadd!($lanes, a3, bias);
                        c0 = vadd!($lanes, c0, more);
                        c1 = vadd!($lanes, c1, more);
                        c2 = vadd!($lanes, c2, more);
                        c3 = vadd!($lanes, c3, more);
                    }
                    vst!(
                        $lanes,
                        yp.add(p * co + j),
                        vclip_if!(clamps, $lanes, a0, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 1) * co + j),
                        vclip_if!(clamps, $lanes, a1, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 2) * co + j),
                        vclip_if!(clamps, $lanes, a2, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 3) * co + j),
                        vclip_if!(clamps, $lanes, a3, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add(p * co + j + $lanes),
                        vclip_if!(clamps, $lanes, c0, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 1) * co + j + $lanes),
                        vclip_if!(clamps, $lanes, c1, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 2) * co + j + $lanes),
                        vclip_if!(clamps, $lanes, c2, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 3) * co + j + $lanes),
                        vclip_if!(clamps, $lanes, c3, lo, hi)
                    );
                    j += 2 * $lanes;
                }
                if j + $lanes <= co {
                    let z = vsplat!($lanes, 0.0);
                    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
                    for i in 0..ci {
                        let row = vld!($lanes, wp.add(i * co + j));
                        a0 = vfma!($lanes, $fma, row, vsplat!($lanes, *xp.add(p * ci + i)), a0);
                        a1 = vfma!(
                            $lanes,
                            $fma,
                            row,
                            vsplat!($lanes, *xp.add((p + 1) * ci + i)),
                            a1
                        );
                        a2 = vfma!(
                            $lanes,
                            $fma,
                            row,
                            vsplat!($lanes, *xp.add((p + 2) * ci + i)),
                            a2
                        );
                        a3 = vfma!(
                            $lanes,
                            $fma,
                            row,
                            vsplat!($lanes, *xp.add((p + 3) * ci + i)),
                            a3
                        );
                    }
                    // Bias is last, as in `matvec_t`: adding it
                    // first would change the f32 results of this tier.
                    if !b.is_null() {
                        let bias = vld!($lanes, b.add(j));
                        a0 = vadd!($lanes, a0, bias);
                        a1 = vadd!($lanes, a1, bias);
                        a2 = vadd!($lanes, a2, bias);
                        a3 = vadd!($lanes, a3, bias);
                    }
                    vst!(
                        $lanes,
                        yp.add(p * co + j),
                        vclip_if!(clamps, $lanes, a0, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 1) * co + j),
                        vclip_if!(clamps, $lanes, a1, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 2) * co + j),
                        vclip_if!(clamps, $lanes, a2, lo, hi)
                    );
                    vst!(
                        $lanes,
                        yp.add((p + 3) * co + j),
                        vclip_if!(clamps, $lanes, a3, lo, hi)
                    );
                    j += $lanes;
                }
                // Four pixels side by side, so four sums advance together;
                // each keeps its own in-order scalar reduction.
                while j < co {
                    let mut acc = [0.0f32; 4];
                    for i in 0..ci {
                        let weight = *wp.add(i * co + j);
                        for (k, sum) in acc.iter_mut().enumerate() {
                            *sum += *xp.add((p + k) * ci + i) * weight;
                        }
                    }
                    for (k, mut sum) in acc.into_iter().enumerate() {
                        if !b.is_null() {
                            sum += *b.add(j);
                        }
                        *yp.add((p + k) * co + j) = sum.clamp(low, high);
                    }
                    j += 1;
                }
                p += 4;
            }
            p
        }
    }};
}

/// [`pw_any_body`] on eight lanes: `$pixels` pixels per tile by sixteen, then
/// eight, output columns, and the last `co % 8` columns in masked lanes.
/// A four-lane or scalar tail would multiply and add separately in input
/// order too, so eight masked lanes doing the same are the same arithmetic,
/// for as many pixels at once as there are chains to hide.
#[cfg(target_arch = "x86_64")]
macro_rules! pw_any8_body {
    ($y:expr,$w:expr,$x:expr,$b:expr,$count:expr,$ci:expr,$co:expr,$bounds:expr,$pixels:literal,$fma:tt) => {
        pw_any8_body!(@tiles $y, $w, $x, $b, $count, $ci, $co, $bounds, $pixels, [], $fma)
    };
    // `$pixels24`: tile height of the optional 24-column block.
    ($y:expr,$w:expr,$x:expr,$b:expr,$count:expr,$ci:expr,$co:expr,$bounds:expr,$pixels:literal,$pixels24:literal,$fma:tt) => {
        pw_any8_body!(@tiles $y, $w, $x, $b, $count, $ci, $co, $bounds, $pixels, [$pixels24], $fma)
    };
    (@tiles $y:expr,$w:expr,$x:expr,$b:expr,$count:expr,$ci:expr,$co:expr,$bounds:expr,$pixels:literal,[$($pixels24:literal)?],$fma:tt) => {{
        use std::arch::x86_64::*;
        // SAFETY: pointwise checked all lengths; masked lanes read and write
        // only columns below `co`.
        unsafe {
            let (yp, wp, xp, b) = ($y.as_mut_ptr(), $w.as_ptr(), $x.as_ptr(), $b);
            let (count, ci, co) = ($count, $ci, $co);
            let (lo, hi) = (_mm256_set1_ps($bounds.0), _mm256_set1_ps($bounds.1));
            let clamps = $bounds != (f32::NEG_INFINITY, f32::INFINITY);
            let done = count / $pixels * $pixels;
            let wide = co / 8 * 8;
            let bias = |j: usize| {
                if b.is_null() {
                    _mm256_setzero_ps()
                } else {
                    _mm256_loadu_ps(b.add(j))
                }
            };
            // One tile: `$pixels` pixels by the 16 columns at `j`.
            let block16 = |p: usize, j: usize| {
                let mut low = [_mm256_setzero_ps(); $pixels];
                let mut high = [_mm256_setzero_ps(); $pixels];
                for i in 0..ci {
                    let r0 = _mm256_loadu_ps(wp.add(i * co + j));
                    let r1 = _mm256_loadu_ps(wp.add(i * co + j + 8));
                    for k in 0..$pixels {
                        let s = _mm256_set1_ps(*xp.add((p + k) * ci + i));
                        low[k] = fma8!($fma, r0, s, low[k]);
                        high[k] = fma8!($fma, r1, s, high[k]);
                    }
                }
                // Bias is last, as in `matvec_t`.
                let (b0, b1) = if b.is_null() {
                    (low, high)
                } else {
                    let (u, v) = (bias(j), bias(j + 8));
                    (
                        low.map(|a| _mm256_add_ps(a, u)),
                        high.map(|a| _mm256_add_ps(a, v)),
                    )
                };
                for k in 0..$pixels {
                    _mm256_storeu_ps(
                        yp.add((p + k) * co + j),
                        vclip_if!(clamps, 8, b0[k], lo, hi),
                    );
                    _mm256_storeu_ps(
                        yp.add((p + k) * co + j + 8),
                        vclip_if!(clamps, 8, b1[k], lo, hi),
                    );
                }
            };
            // The same for the last eight full columns, when `wide % 16 == 8`.
            let block8 = |p: usize, j: usize| {
                let mut acc = [_mm256_setzero_ps(); $pixels];
                for i in 0..ci {
                    let r = _mm256_loadu_ps(wp.add(i * co + j));
                    for k in 0..$pixels {
                        let s = _mm256_set1_ps(*xp.add((p + k) * ci + i));
                        acc[k] = fma8!($fma, r, s, acc[k]);
                    }
                }
                if !b.is_null() {
                    let u = bias(j);
                    acc = acc.map(|a| _mm256_add_ps(a, u));
                }
                for k in 0..$pixels {
                    _mm256_storeu_ps(
                        yp.add((p + k) * co + j),
                        vclip_if!(clamps, 8, acc[k], lo, hi),
                    );
                }
            };
            // Mutated only by the optional 24-column block below.
            #[allow(unused_mut)]
            let (mut pairs, mut single, mut covered) = (wide / 16 * 16, wide % 16 == 8, done);
            $(
                // Widths of 16k + 8 finish with one 24-column block: the 16-
                // and 8-column blocks it replaces each reloaded the same
                // broadcasts. Used only on AVX without FMA, where it is faster.
                if single && pairs >= 16 {
                    let j = pairs - 16;
                    (pairs, single) = (j, false);
                    covered = count / $pixels24 * $pixels24;
                    for p in (0..covered).step_by($pixels24) {
                        let mut c0 = [_mm256_setzero_ps(); $pixels24];
                        let mut c1 = [_mm256_setzero_ps(); $pixels24];
                        let mut c2 = [_mm256_setzero_ps(); $pixels24];
                        for i in 0..ci {
                            let r0 = _mm256_loadu_ps(wp.add(i * co + j));
                            let r1 = _mm256_loadu_ps(wp.add(i * co + j + 8));
                            let r2 = _mm256_loadu_ps(wp.add(i * co + j + 16));
                            for k in 0..$pixels24 {
                                let s = _mm256_set1_ps(*xp.add((p + k) * ci + i));
                                c0[k] = fma8!($fma, r0, s, c0[k]);
                                c1[k] = fma8!($fma, r1, s, c1[k]);
                                c2[k] = fma8!($fma, r2, s, c2[k]);
                            }
                        }
                        for k in 0..$pixels24 {
                            for (m, a) in [c0[k], c1[k], c2[k]].into_iter().enumerate() {
                                // Bias is last, as in `matvec_t`.
                                let a = if b.is_null() { a } else { _mm256_add_ps(a, bias(j + 8 * m)) };
                                _mm256_storeu_ps(
                                    yp.add((p + k) * co + j + 8 * m),
                                    vclip_if!(clamps, 8, a, lo, hi),
                                );
                            }
                        }
                    }
                }
            )?
            for p in (0..done).step_by($pixels) {
                for j in (0..pairs).step_by(16) {
                    block16(p, j);
                }
                if single {
                    block8(p, pairs);
                }
            }
            if wide < co {
                let left = (co - wide) as i32;
                let mask = _mm256_setr_epi32(
                    -i32::from(left > 0),
                    -i32::from(left > 1),
                    -i32::from(left > 2),
                    -i32::from(left > 3),
                    -i32::from(left > 4),
                    -i32::from(left > 5),
                    -i32::from(left > 6),
                    0,
                );
                let tail_bias = if b.is_null() {
                    None
                } else {
                    Some(_mm256_maskload_ps(b.add(wide), mask))
                };
                macro_rules! tail {
                    ($n:literal, $p:expr) => {{
                        let mut acc = [_mm256_setzero_ps(); $n];
                        for i in 0..ci {
                            let r = _mm256_maskload_ps(wp.add(i * co + wide), mask);
                            for (k, a) in acc.iter_mut().enumerate() {
                                let s = _mm256_set1_ps(*xp.add(($p + k) * ci + i));
                                *a = fma8!(false, r, s, *a);
                            }
                        }
                        for (k, a) in acc.into_iter().enumerate() {
                            let a = tail_bias.map_or(a, |u| _mm256_add_ps(a, u));
                            _mm256_maskstore_ps(
                                yp.add(($p + k) * co + wide),
                                mask,
                                vclip_if!(clamps, 8, a, lo, hi),
                            );
                        }
                    }};
                }
                let mut p = 0;
                while p + 8 <= done {
                    tail!(8, p);
                    p += 8;
                }
                if p + 4 <= done {
                    tail!(4, p);
                    p += 4;
                }
                while p < done {
                    tail!(1, p);
                    p += 1;
                }
            }
            // `pointwise_rest` recomputes every column of the pixels past
            // `covered` with the same arithmetic.
            done.min(covered)
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_any_avx2(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    bounds: (f32, f32),
) -> usize {
    pw_any8_body!(y, w, x, b, count, ci, co, bounds, 6, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_any_avx(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    bounds: (f32, f32),
) -> usize {
    pw_any8_body!(y, w, x, b, count, ci, co, bounds, 4, 3, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn pw_any_sse(
    y: &mut [f32],
    w: &[f32],
    x: &[f32],
    b: *const f32,
    count: usize,
    ci: usize,
    co: usize,
    bounds: (f32, f32),
) -> usize {
    // SAFETY: pointwise checked all lengths before dispatch.
    pw_any_body!(y, w, x, b, count, ci, co, bounds, 4, false)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{bits, ramp, reference_matvec};
    use super::*;

    #[test]
    fn pointwise_matches_per_pixel_matvec() {
        for (ci, co) in [(8, 16), (16, 8), (16, 32), (3, 16), (128, 64), (5, 7)] {
            let count = 11; // not a multiple of any tile width
            let w = ramp(ci * co, 0.3);
            let x = ramp(count * ci, 1.1);
            let bias = ramp(co, 4.0);
            let mut y = vec![0.0; count * co];
            pointwise(&mut y, &w, &x, Some(&bias), count, ci, co);
            for p in 0..count {
                let want = reference_matvec(&w, &x[p * ci..], ci, co);
                for j in 0..co {
                    let got = y[p * co + j];
                    let expect = want[j] + bias[j];
                    assert!(
                        (got - expect).abs() < 1e-4,
                        "pixel {p} channel {j}: {got} vs {expect} for {ci}->{co}"
                    );
                }
            }
        }
    }

    #[test]
    fn irregular_pointwise_matches_matvec() {
        for (ci, co) in [(12, 7), (144, 24), (32, 39), (240, 40), (384, 97)] {
            for count in [1, 4, 6, 13, 50] {
                let w = ramp(ci * co, 0.3);
                let x = ramp(count * ci, 1.1);
                let bias = ramp(co, 4.0);
                let mut got = vec![0.0; count * co];
                pointwise(&mut got, &w, &x, Some(&bias), count, ci, co);
                let mut want = vec![0.0; count * co];
                for p in 0..count {
                    let out = &mut want[p * co..][..co];
                    matvec_t(out, &w, &x[p * ci..][..ci], ci, co);
                    for (v, b) in out.iter_mut().zip(&bias) {
                        *v += b;
                    }
                }
                assert_eq!(bits(&got), bits(&want), "{ci}->{co} count={count}");
            }
        }
    }

    #[test]
    fn pointwise_clamp_matches_clip() {
        for (ci, co) in [
            (16, 8),
            (5, 16),
            (24, 32),
            (32, 64),
            (9, 24),
            (12, 7),
            (3, 40),
        ] {
            for count in [1, 5, 13, 30] {
                let w: Vec<f32> = ramp(ci * co, 0.3).iter().map(|v| v * 9.0).collect();
                let mut x = ramp(count * ci, 1.1);
                x[(count / 2) * ci + 1] = f32::NAN; // one pixel turns every output into NaN
                let bias = ramp(co, 4.0);
                let panels = co % 16 == 0 && simd_tier() >= 2;
                let packed: Vec<f32> = if panels {
                    (0..co / 16)
                        .flat_map(|j| {
                            (0..ci).flat_map(move |i| (0..16).map(move |k| (i, j * 16 + k)))
                        })
                        .map(|(i, j)| w[i * co + j])
                        .collect()
                } else {
                    w.clone()
                };
                let mut want = vec![0.0; count * co];
                pointwise(&mut want, &w, &x, Some(&bias), count, ci, co);
                clip_inplace(&mut want, 0.0, 6.0);
                let mut got = vec![0.0; count * co];
                let clip = Some((0.0, 6.0));
                pointwise_with(
                    &mut got,
                    &packed,
                    &x,
                    Some(&bias),
                    count,
                    ci,
                    co,
                    panels,
                    clip,
                );
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{ci}->{co} count={count} panels={panels}"
                );
            }
        }
    }

    #[test]
    fn pointwise_tiles_keep_sum_order() {
        let tier = simd_tier();
        for ci in [1, 3, 8, 13, 16, 32] {
            for co in [1, 4, 6, 8, 16, 24, 28, 32, 36, 42, 48, 88] {
                for count in [1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 17, 24, 32, 50] {
                    let x = ramp(count * ci, 0.6);
                    let w = ramp(ci * co, 0.9);
                    let b = ramp(co, 2.1);
                    for bias in [None, Some(b.as_slice())] {
                        let mut got = vec![123.0; count * co + 8];
                        pointwise(&mut got, &w, &x, bias, count, ci, co);
                        let tiled = if tier >= 2 && co % 16 == 0 {
                            count / 4 * 4
                        } else {
                            0
                        };
                        for p in 0..count {
                            for j in 0..co {
                                let before = p < tiled;
                                let mut acc = if before {
                                    bias.map_or(0.0, |b| b[j])
                                } else {
                                    0.0
                                };
                                for i in 0..ci {
                                    let (a, v) = (w[i * co + j], x[p * ci + i]);
                                    acc = if tier == 3 && j < co / 8 * 8 {
                                        a.mul_add(v, acc)
                                    } else {
                                        acc + a * v
                                    };
                                }
                                if !before {
                                    if let Some(b) = bias {
                                        acc += b[j];
                                    }
                                }
                                assert_eq!(
                                    got[p * co + j].to_bits(),
                                    acc.to_bits(),
                                    "ci={ci} co={co} n={count} p={p} ch={j}"
                                );
                            }
                        }
                        assert!(got[count * co..].iter().all(|&v| v == 123.0));
                    }
                }
            }
        }
    }
}
