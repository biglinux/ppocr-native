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
//! one tier; they are not identical across tiers. The tiled pointwise kernels
//! add bias before the dot product while the irregular-width kernels add it
//! afterwards, and tier 3 rounds once per fused multiply-add, so every kernel
//! keeps the summation order of the tier it belongs to.
//!
//! This file holds the tier choice and the vector macros every family uses;
//! each family of kernels is a submodule, reached by the rest of the engine
//! through `kernels::`. `tools/check_isa.py` classes each function by its
//! `_avx2`, `_avx` or `_sse` suffix, so a tier's entry point must keep it.
//!
//! The kernel family and the tier scheme come from the BigLinux
//! `dpdfnet-native` and `deepfilternet-quantized-ladspa` audio work (see
//! `NOTICE`); the two-dimensional shape handling and the pointwise
//! micro-kernel are tuned here for image-sized tensors.

#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicU8, Ordering};

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

#[macro_use]
mod activation;
mod dense;
mod depthwise;
mod elementwise;
mod matvec;
mod pointwise;

pub(crate) use activation::*;
pub(crate) use dense::*;
pub(crate) use depthwise::*;
pub(crate) use elementwise::*;
pub(crate) use matvec::*;
pub(crate) use pointwise::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight textbook definition, used to check every dispatched kernel.
    pub(super) fn reference_matvec(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|j| (0..m).map(|i| x[i] * a[i * n + j]).sum())
            .collect()
    }

    pub(super) fn ramp(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect()
    }

    pub(super) fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
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
