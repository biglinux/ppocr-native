//! The exponential under the sigmoid, softmax, GELU and swish, computed as
//! XNNPACK's f32-vsigmoid "p5-div" kernels do (see `NOTICE`), and the
//! sigmoid itself.

use super::simd_tier;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
