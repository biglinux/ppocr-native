//! Numeric kernels of the PP-OCR networks.
//!
//! Every tensor is NHWC: height, then width, then channel, with channel the
//! fastest-moving axis. That one choice decides the shape of everything here:
//!
//! * a 1x1 convolution becomes a matrix multiply over pixels, with the channel
//!   vector contiguous, so no gather and no transpose is ever needed;
//! * a depthwise tap becomes an elementwise multiply-add over a contiguous run
//!   of channels, because the kernel weight is constant per channel;
//! * an activation reads the same bytes the convolution just wrote.
//!
//! Four instruction-set tiers are dispatched at runtime, decided once per
//! process: AVX2 with fused multiply-add, AVX without it, SSE4.1, and a plain
//! scalar fallback. Results are bit-identical across runs and machines within
//! one tier; they are not identical across tiers. The tiled pointwise kernels add bias before the dot product while the
//! irregular-width kernels add it afterwards, and tier 3 rounds once per fused
//! multiply-add, so every kernel keeps the summation order of the tier it
//! belongs to.
//!
//! The kernel family and the tier scheme come from the BigLinux
//! `dpdfnet-native` and `deepfilternet-quantized-ladspa` audio work (see
//! `NOTICE`). What is new here is the two-dimensional shape handling and the
//! pointwise micro-kernel, which are tuned for image-sized tensors rather than
//! single spectral frames.

#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
static TIER: AtomicU8 = AtomicU8::new(u8::MAX);

/// Widest usable instruction set: 3 = AVX2+FMA, 2 = AVX, 1 = SSE4.1, 0 = scalar.
///
/// Set `PPOCR_NATIVE_TIER` to cap it, which is how a machine with AVX2 can
/// be made to run an older machine's code path for comparison. Values above
/// what the CPU actually has are ignored.
#[cfg(target_arch = "x86_64")]
pub fn simd_tier() -> u8 {
    let cached = TIER.load(Ordering::Relaxed);
    if cached != u8::MAX {
        return cached;
    }
    let detected = if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    {
        3
    } else if std::is_x86_feature_detected!("avx") {
        2
    } else if std::is_x86_feature_detected!("sse4.1") {
        1
    } else {
        0
    };
    let tier = std::env::var("PPOCR_NATIVE_TIER")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .map_or(detected, |cap| cap.min(detected));
    TIER.store(tier, Ordering::Relaxed);
    tier
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_tier() -> u8 {
    0
}

#[cfg(target_arch = "x86_64")]
macro_rules! fma8 {
    (true, $a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm256_fmadd_ps($a, $b, $c)
    };
    (false, $a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm256_add_ps(std::arch::x86_64::_mm256_mul_ps($a, $b), $c)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! fma4 {
    ($a:expr, $b:expr, $c:expr) => {
        std::arch::x86_64::_mm_add_ps(std::arch::x86_64::_mm_mul_ps($a, $b), $c)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vld {
    (8, $p:expr) => {
        std::arch::x86_64::_mm256_loadu_ps($p)
    };
    (4, $p:expr) => {
        std::arch::x86_64::_mm_loadu_ps($p)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vst {
    (8, $p:expr, $v:expr) => {
        std::arch::x86_64::_mm256_storeu_ps($p, $v)
    };
    (4, $p:expr, $v:expr) => {
        std::arch::x86_64::_mm_storeu_ps($p, $v)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vsplat {
    (8, $s:expr) => {
        std::arch::x86_64::_mm256_set1_ps($s)
    };
    (4, $s:expr) => {
        std::arch::x86_64::_mm_set1_ps($s)
    };
}

/// `min(hi, max(lo, v))`: [`clip_inplace`]'s clamp, bit for bit.
#[cfg(target_arch = "x86_64")]
macro_rules! vclip {
    (8, $v:expr, $lo:expr, $hi:expr) => {
        std::arch::x86_64::_mm256_min_ps($hi, std::arch::x86_64::_mm256_max_ps($lo, $v))
    };
    (4, $v:expr, $lo:expr, $hi:expr) => {
        std::arch::x86_64::_mm_min_ps($hi, std::arch::x86_64::_mm_max_ps($lo, $v))
    };
}

/// [`vclip!`] only when `on`: a loop-invariant test is cheaper than two
/// vector operations per store when there is nothing to clamp.
#[cfg(target_arch = "x86_64")]
macro_rules! vclip_if {
    ($on:expr, $lanes:tt, $v:expr, $lo:expr, $hi:expr) => {
        if $on {
            vclip!($lanes, $v, $lo, $hi)
        } else {
            $v
        }
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vfma {
    (8, $fma:tt, $a:expr, $b:expr, $c:expr) => {
        fma8!($fma, $a, $b, $c)
    };
    (4, false, $a:expr, $b:expr, $c:expr) => {
        fma4!($a, $b, $c)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vmul {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_mul_ps($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_mul_ps($a, $b)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vadd {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_add_ps($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_add_ps($a, $b)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vsub {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_sub_ps($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_sub_ps($a, $b)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vdiv {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_div_ps($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_div_ps($a, $b)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vor {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_or_ps($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_or_ps($a, $b)
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! vandnot {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_andnot_ps($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_andnot_ps($a, $b)
    };
}

/// `a < b`, ordered and signalling, as an all-ones lane mask.
#[cfg(target_arch = "x86_64")]
macro_rules! vlt {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_cmp_ps::<{ std::arch::x86_64::_CMP_LT_OS }>($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_cmplt_ps($a, $b)
    };
}

/// `a > b ? a : b` per lane, which is `b` when either is NaN.
#[cfg(target_arch = "x86_64")]
macro_rules! vmax {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_max_ps($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_max_ps($a, $b)
    };
}

/// `a == b` as an all-ones lane mask, false for NaN.
#[cfg(target_arch = "x86_64")]
macro_rules! veq {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_cmp_ps::<{ std::arch::x86_64::_CMP_EQ_OQ }>($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_cmpeq_ps($a, $b)
    };
}

/// The sign bit of each lane, lane `i` as bit `i`.
#[cfg(target_arch = "x86_64")]
macro_rules! vmask {
    (8, $a:expr) => {
        std::arch::x86_64::_mm256_movemask_ps($a)
    };
    (4, $a:expr) => {
        std::arch::x86_64::_mm_movemask_ps($a)
    };
}

/// `a > b` as an all-ones lane mask.
#[cfg(target_arch = "x86_64")]
macro_rules! vgt {
    (8, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm256_cmp_ps::<{ std::arch::x86_64::_CMP_GT_OQ }>($a, $b)
    };
    (4, $a:expr, $b:expr) => {
        std::arch::x86_64::_mm_cmpgt_ps($a, $b)
    };
}

/// `b` where the sign bit of `mask` is set, else `a`.
#[cfg(target_arch = "x86_64")]
macro_rules! vblendv {
    (8, $a:expr, $b:expr, $mask:expr) => {
        std::arch::x86_64::_mm256_blendv_ps($a, $b, $mask)
    };
    (4, $a:expr, $b:expr, $mask:expr) => {
        std::arch::x86_64::_mm_blendv_ps($a, $b, $mask)
    };
}

#[inline]
fn span(a: usize, b: usize) -> usize {
    a.checked_mul(b).expect("kernel dimensions overflow")
}

// Pointwise (1x1) convolution.

/// [`pointwise_with`] with row-major weights and no clamp, for the tests.
#[cfg(test)]
fn pointwise(
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

// Transposed matrix-vector, plain and four-pixel.

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

// Depthwise.

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

// Elementwise.

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

// Constants of XNNPACK's f32-vsigmoid "p5-div" kernels: exp(-|x|) as 2^n
// times a degree-5 polynomial, n rounded by a magic bias that also leaves
// the exponent bits of 2^n in place.
pub(crate) const EXP_MAGIC_BIAS: f32 = f32::from_bits(0x4b40_007f); // 0x1.8000FEp23
pub(crate) const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b); // 0x1.715476p0
pub(crate) const MINUS_LN2: f32 = f32::from_bits(0xbf31_7218); // -0x1.62E430p-1
pub(crate) const MINUS_LN2_HI: f32 = f32::from_bits(0xbf31_7200); // -0x1.62E400p-1
pub(crate) const MINUS_LN2_LO: f32 = f32::from_bits(0xb5bf_be8e); // -0x1.7F7D1Cp-20
pub(crate) const EXP_C5: f32 = f32::from_bits(0x3c07_cfce); // 0x1.0F9F9Cp-7
pub(crate) const EXP_C4: f32 = f32::from_bits(0x3d2b_9d0d); // 0x1.573A1Ap-5
pub(crate) const EXP_C3: f32 = f32::from_bits(0x3e2a_ad40); // 0x1.555A80p-3
pub(crate) const EXP_C2: f32 = f32::from_bits(0x3eff_fee3); // 0x1.FFFDC6p-2
pub(crate) const EXP_C1: f32 = f32::from_bits(0x3f7f_fffb); // 0x1.FFFFF6p-1
/// Below this, exp(z) is denormal and taken as 0.
pub(crate) const EXP_CUTOFF: f32 = f32::from_bits(0xc2ae_ac4f); // -0x1.5D589Ep+6

/// `a * b + c`, fused when `FMA` (tier 3), rounded twice otherwise.
#[inline(always)]
pub(crate) fn madd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `exp(z)` for `z <= 0`, 0 below [`EXP_CUTOFF`]. Tier 3 (`FMA`) reduces
/// the argument with one fused multiply-add (`rr1`), the other tiers with
/// two-part `ln 2` (`rr2`), as XNNPACK does for each instruction set.
#[inline(always)]
pub(crate) fn exp_neg<const FMA: bool>(z: f32) -> f32 {
    let n = madd::<FMA>(z, LOG2E, EXP_MAGIC_BIAS);
    let s = f32::from_bits(n.to_bits() << 23);
    let n = n - EXP_MAGIC_BIAS;
    let t = if FMA {
        madd::<FMA>(n, MINUS_LN2, z)
    } else {
        madd::<FMA>(n, MINUS_LN2_LO, madd::<FMA>(n, MINUS_LN2_HI, z))
    };
    let p = madd::<FMA>(
        madd::<FMA>(
            madd::<FMA>(madd::<FMA>(EXP_C5, t, EXP_C4), t, EXP_C3),
            t,
            EXP_C2,
        ),
        t,
        EXP_C1,
    );
    if z < EXP_CUTOFF {
        0.0
    } else {
        madd::<FMA>(t * s, p, s)
    }
}

/// [`exp_neg`] on one vector; `$exp2` turns the biased `n` into `2^n`.
#[cfg(target_arch = "x86_64")]
macro_rules! vexp_neg {
    ($lanes:tt, $fma:tt, $exp2:ident, $z:expr) => {
        vexp_neg_each!($lanes, $fma, $exp2, [$z])[0]
    };
}

/// [`vexp_neg!`] on an array of vectors, each step for all of them before
/// the next: one exponential is a chain of a dozen dependent operations,
/// and side by side in program order several chains overlap where the
/// scheduler can see them.
#[cfg(target_arch = "x86_64")]
macro_rules! vexp_neg_each {
    ($lanes:tt, $fma:tt, $exp2:ident, $z:expr) => {{
        use $crate::engine::kernels::{
            EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_C5, EXP_CUTOFF, EXP_MAGIC_BIAS, LOG2E, MINUS_LN2,
            MINUS_LN2_HI, MINUS_LN2_LO,
        };
        let z = $z;
        let bias = vsplat!($lanes, EXP_MAGIC_BIAS);
        let (mut k, mut s, mut t, mut q) = (z, z, z, z);
        for j in 0..z.len() {
            k[j] = vfma!($lanes, $fma, z[j], vsplat!($lanes, LOG2E), bias);
        }
        for j in 0..z.len() {
            s[j] = $exp2(k[j]);
            k[j] = vsub!($lanes, k[j], bias);
        }
        for j in 0..z.len() {
            t[j] = if $fma {
                vfma!($lanes, $fma, k[j], vsplat!($lanes, MINUS_LN2), z[j])
            } else {
                vfma!($lanes, $fma, k[j], vsplat!($lanes, MINUS_LN2_HI), z[j])
            };
        }
        if !$fma {
            for j in 0..z.len() {
                t[j] = vfma!($lanes, $fma, k[j], vsplat!($lanes, MINUS_LN2_LO), t[j]);
            }
        }
        for j in 0..z.len() {
            q[j] = vfma!(
                $lanes,
                $fma,
                vsplat!($lanes, EXP_C5),
                t[j],
                vsplat!($lanes, EXP_C4)
            );
        }
        for c in [EXP_C3, EXP_C2, EXP_C1] {
            for j in 0..z.len() {
                q[j] = vfma!($lanes, $fma, q[j], t[j], vsplat!($lanes, c));
            }
        }
        for j in 0..z.len() {
            let e = vfma!($lanes, $fma, vmul!($lanes, t[j], s[j]), q[j], s[j]);
            q[j] = vandnot!($lanes, vlt!($lanes, z[j], vsplat!($lanes, EXP_CUTOFF)), e);
        }
        q
    }};
}

/// The logistic function in place, as XNNPACK's `f32-vsigmoid` computes it:
/// with `z = -|x|` and `e = exp(z)` from [`exp_neg`], `e / (e + 1)` for
/// negative `x` and `1 -` that otherwise, so neither tail cancels. Tiers 0-2
/// give the same bits.
pub(crate) fn sigmoid_inplace(y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let (p, n) = (y.as_mut_ptr(), y.len());
        // SAFETY: the kernels read and write `y`'s `n` floats, each value
        // before its own store; each instruction set was detected.
        unsafe {
            match simd_tier() {
                3 => return sigmoid_avx2(p, n),
                2 => return sigmoid_avx(p, n),
                1 => return sigmoid_sse(p, n),
                _ => {}
            }
        }
    }
    for v in y {
        *v = sigmoid_one::<false>(*v);
    }
}

/// One value of [`sigmoid_inplace`]; `FMA` selects tier 3's arithmetic.
#[inline(always)]
pub(crate) fn sigmoid_one<const FMA: bool>(x: f32) -> f32 {
    let e = exp_neg::<FMA>(f32::from_bits(x.to_bits() | 0x8000_0000));
    let f = e / (e + 1.0);
    if x.is_sign_negative() {
        f
    } else {
        1.0 - f
    }
}

/// [`sigmoid_one`] on `$lanes` values at a time, then one at a time.
#[cfg(target_arch = "x86_64")]
macro_rules! sigmoid_body {
    ($p:expr, $n:expr, $lanes:tt, $fma:tt, $exp2:ident) => {{
        // SAFETY: the caller's contract.
        unsafe {
            let (p, n) = ($p, $n);
            let (sign, one) = (vsplat!($lanes, -0.0), vsplat!($lanes, 1.0));
            let mut i = 0;
            while i + $lanes <= n {
                let x = vld!($lanes, p.add(i));
                let e = vexp_neg!($lanes, $fma, $exp2, vor!($lanes, x, sign));
                let f = vdiv!($lanes, e, vadd!($lanes, e, one));
                vst!(
                    $lanes,
                    p.add(i),
                    vblendv!($lanes, vsub!($lanes, one, f), f, x)
                );
                i += $lanes;
            }
            while i < n {
                *p.add(i) = sigmoid_one::<$fma>(*p.add(i));
                i += 1;
            }
        }
    }};
}

/// `2^n` from the magic-biased `k`: its low bits shifted into the exponent.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn exp2_bits_avx2(k: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(k), 23))
}

/// [`exp2_bits_avx2`] for AVX, which has no 256-bit integer shift: each half
/// is shifted with SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
pub(crate) fn exp2_bits_avx(k: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let low = exp2_bits_sse(_mm256_castps256_ps128(k));
    _mm256_insertf128_ps(
        _mm256_castps128_ps256(low),
        exp2_bits_sse(_mm256_extractf128_ps(k, 1)),
        1,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
pub(crate) fn exp2_bits_sse(k: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;
    _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(k), 23))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_avx2(p: *mut f32, n: usize) {
    sigmoid_body!(p, n, 8, true, exp2_bits_avx2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sigmoid_avx(p: *mut f32, n: usize) {
    sigmoid_body!(p, n, 8, false, exp2_bits_avx)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sigmoid_sse(p: *mut f32, n: usize) {
    sigmoid_body!(p, n, 4, false, exp2_bits_sse)
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

// Dense convolution with a window.

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
/// Each output is what [`pointwise_with`] gives over the window gathered
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
    op: &super::plan::Op,
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
    op: &super::plan::Op,
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
unsafe fn rgb24_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &super::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 3, 3, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rgb24_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &super::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 3, 3, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb48_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &super::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 6, 2, true)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rgb48_avx(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: *const f32,
    op: &super::plan::Op,
    count: usize,
) {
    // SAFETY: the caller validated full windows and the exact channel count.
    rgb_body!(y, x, w, b, op, count, 6, 2, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight textbook definition, used to check every dispatched kernel.
    fn reference_matvec(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|j| (0..m).map(|i| x[i] * a[i * n + j]).sum())
            .collect()
    }

    fn ramp(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect()
    }

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

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
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

    #[test]
    fn rgb_windows_match_taps() {
        for (oc, pixels) in [(24, 3), (48, 2)] {
            for (k, stride) in [(3, 2), (5, 2), (3, 1)] {
                for count in [1, 3, 4, 7, 8, 13, 17] {
                    let iw = (count - 1) * stride + k;
                    let mut op = crate::engine::plan::op(super::super::Kind::Conv, 0, 1, 3, oc);
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
                            crate::engine::plan::op(super::super::Kind::Conv, 0, 1, ic, co);
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

    /// Values around every branch of the sigmoid: NaN of either sign, signed
    /// zeros, infinities, the denormal cutoff near -87.3, and a smooth sweep.
    fn activation_inputs() -> Vec<f32> {
        let mut values = vec![
            f32::NAN,
            -f32::NAN,
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            3.0,
            -3.0,
            3.0f32.next_up(),
            (-3.0f32).next_down(),
            EXP_CUTOFF,
            EXP_CUTOFF.next_up(),
            -EXP_CUTOFF,
            1.0e-40,
            -1.0e-40,
            f32::MAX,
            f32::MIN,
        ];
        values.extend((0..4000).map(|i| (i as f32 - 2000.0) * 0.0517));
        values
    }

    #[test]
    fn sigmoid_tiers_match_scalar_and_libm() {
        let fma = simd_tier() == 3;
        let x = activation_inputs();
        for n in [x.len(), x.len() - 3, 7] {
            let mut got = x[..n].to_vec();
            sigmoid_inplace(&mut got);
            for (&g, &v) in got.iter().zip(&x) {
                // NaN stays NaN; its sign is not something LLVM preserves.
                if v.is_nan() {
                    assert!(g.is_nan());
                    continue;
                }
                let want = if fma {
                    sigmoid_one::<true>(v)
                } else {
                    sigmoid_one::<false>(v)
                };
                assert_eq!(g.to_bits(), want.to_bits(), "sigmoid({v})");
                // Within 1e-7 absolutely, and 4 units in the last place where
                // the result is a normal float, like libm's `1 / (1 + exp(-x))`.
                let exact = 1.0 / (1.0 + (-(v as f64)).exp());
                let gap = (g as f64 - exact).abs();
                let relative =
                    exact < f32::MIN_POSITIVE as f64 || gap <= 4.0 * f32::EPSILON as f64 * exact;
                assert!(
                    gap <= 1.0e-7 && relative,
                    "sigmoid({v}) = {g}, exactly {exact}"
                );
            }
        }
        // The tails saturate exactly; the middle is exactly one half.
        let mut ends = [f32::NEG_INFINITY, -100.0, 0.0, -0.0, 100.0, f32::INFINITY];
        sigmoid_inplace(&mut ends);
        assert_eq!(ends, [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
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

    #[test]
    fn span_overflow_panics_before_access() {
        assert!(std::panic::catch_unwind(|| {
            depthwise_tap(&mut [], &[], &[1.0; 8], usize::MAX / 8 + 1, 8, 8);
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| {
            pointwise(&mut [], &[], &[], None, usize::MAX / 8 + 1, 0, 8);
        })
        .is_err());
    }
}
