//! NEON-vectorized polynomial kernels for aarch64, processing eight `i16`
//! coefficients per vector.
//!
//! All three kernels are vectorized. The transforms vectorize the stride-≥8
//! butterfly stages (eight butterflies per vector under one broadcast zeta) and
//! leave the narrow stride-4/2 stages scalar; `ntt`'s final Barrett reduction
//! is also scalar, while `ntt_inverse`'s per-stage reduction and final scale
//! are vectorized. The arithmetic matches the scalar backend exactly — the
//! `tests` module cross-checks every kernel — in the signed Montgomery
//! convention of [`crate::algebraic::field`].
//!
//! The only `unsafe` in the crate lives in this module (and the future `avx2`
//! sibling): NEON is baseline on aarch64, so each intrinsic call is sound, as
//! the per-block `SAFETY` notes record.
#![allow(unsafe_code)]

use core::arch::aarch64::{
    int16x4_t, int16x8_t, int16x8x2_t, int32x4_t, vaddq_s16, vcombine_s16, vdup_n_s16, vdupq_n_s16,
    vdupq_n_s32, vget_low_s16, vld1q_s16, vld2q_s16, vmlaq_n_s32, vmovl_high_s16, vmovl_s16,
    vmovn_s32, vmul_s16, vmull_high_s16, vmull_s16, vmulq_n_s32, vshrn_n_s32, vshrq_n_s32,
    vst1q_s16, vst2q_s16, vsubq_s16, vsubq_s32,
};

use crate::{algebraic::field::FieldElement, parameters};

#[cfg(test)]
mod tests;

/// The modulus q, mirroring `crate::algebraic::field`.
const Q: i16 = parameters::Q as i16;

/// `q^-1 mod 2^16`, signed, mirroring `crate::algebraic::field`.
const QINV: i16 = -3327;

/// `((2^26 + q/2) / q)`, the Barrett multiplier, mirroring `field`'s
/// `BARRETT_V`.
const BARRETT_V: i32 = ((1 << 26) + (parameters::Q as i32) / 2) / (parameters::Q as i32);

/// The final NTT⁻¹ scale `f = 128^-1 * R^2 mod q = 1441` in Montgomery form,
/// mirroring `generic`'s `NTT_INVERSE_SCALE`.
const NTT_INVERSE_SCALE: i16 = 1441;

/// Montgomery-reduces four `i32` products to `i16`, returning `a * R^-1 mod q`
/// in `(-q, q)` per lane: the vector form of `FieldElement::montgomery_reduce`.
#[inline]
fn montgomery_reduce(a: int32x4_t, q: int16x4_t, qinv: int16x4_t) -> int16x4_t {
    // SAFETY: NEON is baseline on aarch64; every intrinsic below is total.
    unsafe {
        let a_low = vmovn_s32(a); // a as i16 (low 16 bits)
        let m = vmul_s16(a_low, qinv); // m = (a as i16) * QINV
        let m_q = vmull_s16(m, q); // m * q, widened to i32
        let difference = vsubq_s32(a, m_q); // a - m*q

        vshrn_n_s32::<16>(difference) // (a - m*q) >> 16, narrowed to i16
    }
}

/// Montgomery-multiplies eight coefficient lanes, `a * b -> a*b*R^-1 mod q`:
/// the vector form of the scalar `*` on `FieldElement`.
#[inline]
fn fqmul(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is baseline on aarch64; every intrinsic below is total.
    unsafe {
        let q = vdup_n_s16(Q);
        let qinv = vdup_n_s16(QINV);

        let product_low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        let product_high = vmull_high_s16(a, b);

        vcombine_s16(
            montgomery_reduce(product_low, q, qinv),
            montgomery_reduce(product_high, q, qinv),
        )
    }
}

/// Barrett-reduces eight `i16` lanes to a representative in `(-q/2, q/2]`: the
/// vector form of `FieldElement::reduce`.
#[inline]
fn barrett_reduce(a: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is baseline on aarch64; every intrinsic below is total.
    unsafe {
        let bias = vdupq_n_s32(1 << 25);
        let q = Q as i32;

        // t = (BARRETT_V * a + 2^25) >> 26 ; result = a - t*q, per i32 lane.
        let low = vmovl_s16(vget_low_s16(a));
        let t_low = vshrq_n_s32::<26>(vmlaq_n_s32(bias, low, BARRETT_V));
        let result_low = vmovn_s32(vsubq_s32(low, vmulq_n_s32(t_low, q)));

        let high = vmovl_high_s16(a);
        let t_high = vshrq_n_s32::<26>(vmlaq_n_s32(bias, high, BARRETT_V));
        let result_high = vmovn_s32(vsubq_s32(high, vmulq_n_s32(t_high, q)));

        vcombine_s16(result_low, result_high)
    }
}

/// Pointwise base multiplication of two NTT representations: 128 degree-one
/// products modulo the quadratics `X^2 - gamma`, computed eight pairs at a
/// time.
///
/// Matches [`super::generic::multiply`] exactly. The result is scaled by `R^-1`
/// (Montgomery convention), which `ntt_inverse` later undoes.
pub(crate) fn multiply(
    f: &[FieldElement; parameters::N],
    g: &[FieldElement; parameters::N],
) -> [FieldElement; parameters::N] {
    let mut h = [FieldElement::ZERO; parameters::N];

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the three
    // length-256 arrays and the length-128 `GAMMA_MONT` reinterpret as `[i16]`.
    // Each iteration reads/writes a 16-`i16` window of `f`/`g`/`h` (16 windows
    // tile 256) and an 8-`i16` window of the gammas (16 windows tile 128), all
    // in bounds. `vld2q`/`vst2q` are unaligned. NEON is baseline on aarch64.
    unsafe {
        let f_ptr = f.as_ptr().cast::<i16>();
        let g_ptr = g.as_ptr().cast::<i16>();
        let h_ptr = h.as_mut_ptr().cast::<i16>();
        let gamma_ptr = super::GAMMA_MONT.as_ptr().cast::<i16>();

        let mut pair = 0;
        while pair < 128 {
            // Deinterleave eight pairs: `.0` is the even (degree-0) coefficient,
            // `.1` the odd (degree-1).
            let f_de = vld2q_s16(f_ptr.add(2 * pair));
            let g_de = vld2q_s16(g_ptr.add(2 * pair));
            let gamma = vld1q_s16(gamma_ptr.add(pair));

            let (a0, a1) = (f_de.0, f_de.1);
            let (b0, b1) = (g_de.0, g_de.1);

            // c0 = a0*b0 + a1*b1*gamma ; c1 = a0*b1 + a1*b0
            let c0 = vaddq_s16(fqmul(a0, b0), fqmul(fqmul(a1, b1), gamma));
            let c1 = vaddq_s16(fqmul(a0, b1), fqmul(a1, b0));

            vst2q_s16(h_ptr.add(2 * pair), int16x8x2_t(c0, c1));

            pair += 8;
        }
    }

    h
}

/// Forward NTT in place, then Barrett-reduce. Vectorizes the stride-≥8
/// butterfly stages (eight butterflies per vector under one broadcast zeta);
/// the stride-4 and stride-2 stages and the final reduction run scalar.
///
/// Matches [`super::generic::ntt`] exactly.
// reason: scalar-tail butterfly/zeta indices are provably in 0..256, as in the
// generic backend.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt(coefficients: &mut [FieldElement; parameters::N]) {
    let mut k = 1;

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the array and
    // `ZETA_MONT` reinterpret as `[i16]`. Each vector window `[j, j + 8)` and
    // `[j + len, j + len + 8)` stays within the 256-element array (the loops
    // bound `j + len + 8 <= start + 2*len <= 256`), and `k < 32 < 128` indexes
    // `ZETA_MONT`. `vld1q`/`vst1q` are unaligned. NEON is baseline on aarch64.
    unsafe {
        let ptr = coefficients.as_mut_ptr().cast::<i16>();
        let zeta_ptr = super::ZETA_MONT.as_ptr().cast::<i16>();

        for len in [128usize, 64, 32, 16, 8] {
            let mut start = 0;
            while start < 256 {
                let zeta = vdupq_n_s16(*zeta_ptr.add(k));
                k += 1;

                let mut j = start;
                while j < start + len {
                    let vj = vld1q_s16(ptr.add(j));
                    let vjl = vld1q_s16(ptr.add(j + len));
                    let t = fqmul(zeta, vjl);

                    vst1q_s16(ptr.add(j + len), vsubq_s16(vj, t));
                    vst1q_s16(ptr.add(j), vaddq_s16(vj, t));

                    j += 8;
                }

                start += 2 * len;
            }
        }
    }

    // Scalar tail: stride-4 and stride-2 groups are narrower than a vector.
    for len in [4usize, 2] {
        let mut start = 0;
        while start < 256 {
            let zeta = super::ZETA_MONT[k];
            k += 1;

            for j in start..start + len {
                let t = zeta * coefficients[j + len];
                coefficients[j + len] = coefficients[j] - t;
                coefficients[j] = coefficients[j] + t;
            }

            start += 2 * len;
        }
    }

    for coefficient in coefficients.iter_mut() {
        *coefficient = coefficient.reduce();
    }
}

/// Inverse NTT in place, followed by the scale back to the standard domain.
///
/// The stride-≥8 Gentleman-Sande stages run eight butterflies per vector, and
/// the final scale is vectorized; the narrower stride-4 and stride-2 stages run
/// scalar.
///
/// Matches [`super::generic::ntt_inverse`] exactly.
// reason: scalar-head butterfly/zeta indices are provably in 0..256, as in the
// generic backend.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt_inverse(coefficients: &mut [FieldElement; parameters::N]) {
    let mut k = 127;

    // Scalar head: stride-2 and stride-4 groups are narrower than a vector.
    for len in [2usize, 4] {
        let mut start = 0;
        while start < 256 {
            let zeta = super::ZETA_MONT[k];
            k -= 1;

            for j in start..start + len {
                let t = coefficients[j];
                coefficients[j] = (t + coefficients[j + len]).reduce();
                coefficients[j + len] = zeta * (coefficients[j + len] - t);
            }

            start += 2 * len;
        }
    }

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the array and
    // `ZETA_MONT` reinterpret as `[i16]`. Each vector window `[j, j + 8)` and
    // `[j + len, j + len + 8)` stays within the 256-element array, and the
    // descending `k` indexes `ZETA_MONT` (it reaches 0 only after the last use).
    // `vld1q`/`vst1q` are unaligned. NEON is baseline on aarch64.
    unsafe {
        let ptr = coefficients.as_mut_ptr().cast::<i16>();
        let zeta_ptr = super::ZETA_MONT.as_ptr().cast::<i16>();

        for len in [8usize, 16, 32, 64, 128] {
            let mut start = 0;
            while start < 256 {
                let zeta = vdupq_n_s16(*zeta_ptr.add(k));
                k -= 1;

                let mut j = start;
                while j < start + len {
                    let vj = vld1q_s16(ptr.add(j));
                    let vjl = vld1q_s16(ptr.add(j + len));

                    vst1q_s16(ptr.add(j), barrett_reduce(vaddq_s16(vj, vjl)));
                    vst1q_s16(ptr.add(j + len), fqmul(zeta, vsubq_s16(vjl, vj)));

                    j += 8;
                }

                start += 2 * len;
            }
        }

        // Final scale: every coefficient times `NTT_INVERSE_SCALE`.
        let scale = vdupq_n_s16(NTT_INVERSE_SCALE);
        let mut j = 0;
        while j < 256 {
            vst1q_s16(ptr.add(j), fqmul(scale, vld1q_s16(ptr.add(j))));
            j += 8;
        }
    }
}
