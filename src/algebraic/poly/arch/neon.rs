//! NEON-vectorized polynomial kernels for aarch64, processing eight `i16`
//! coefficients per vector.
//!
//! Only [`multiply`] is vectorized so far; [`ntt`] and [`ntt_inverse`] reuse
//! the portable [`super::generic`] kernels. The arithmetic matches the scalar
//! backend exactly — the `tests` module cross-checks it — in the signed
//! Montgomery convention of [`crate::algebraic::field`].
//!
//! The only `unsafe` in the crate lives in this module (and the future `avx2`
//! sibling): NEON is baseline on aarch64, so each intrinsic call is sound, as
//! the per-block `SAFETY` notes record.
#![allow(unsafe_code)]

use core::arch::aarch64::{
    int16x4_t, int16x8_t, int16x8x2_t, int32x4_t, vaddq_s16, vcombine_s16, vdup_n_s16,
    vget_low_s16, vld1q_s16, vld2q_s16, vmovn_s32, vmul_s16, vmull_high_s16, vmull_s16,
    vshrn_n_s32, vst2q_s16, vsubq_s32,
};

use crate::{algebraic::field::FieldElement, parameters};

#[cfg(test)]
mod tests;

pub(crate) use super::generic::{ntt, ntt_inverse};

/// The modulus q, mirroring `crate::algebraic::field`.
const Q: i16 = parameters::Q as i16;

/// `q^-1 mod 2^16`, signed, mirroring `crate::algebraic::field`.
const QINV: i16 = -3327;

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
