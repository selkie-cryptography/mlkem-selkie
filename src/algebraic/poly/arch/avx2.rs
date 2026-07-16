//! AVX2-vectorized polynomial kernels for x86_64, processing sixteen `i16`
//! coefficients per `__m256i`.
//!
//! Mirrors the verified NEON backend: the stride-≥16 butterfly stages run
//! sixteen butterflies per vector under one broadcast zeta, while the narrow
//! stride-8/4/2 stages (and `ntt`'s final reduce) run scalar. `multiply`
//! separates even/odd coefficients with shuffles (AVX2 has no deinterleaving
//! load). The arithmetic matches the scalar backend (the `tests` module
//! cross-checks it), in the signed Montgomery convention of
//! [`crate::algebraic::field`].
//!
//! `unsafe` is confined to this module's intrinsic blocks. The module compiles
//! only when `mlkem_selkie_arch = "avx2"` is set, i.e. when the build enables
//! AVX2, so each intrinsic call is sound, as the per-block `SAFETY` notes
//! record.
#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_loadu_si256, _mm256_mulhi_epi16, _mm256_mulhrs_epi16,
    _mm256_mullo_epi16, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi16,
    _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_srai_epi16, _mm256_storeu_si256,
    _mm256_sub_epi16,
};

use crate::{algebraic::field::FieldElement, parameters};

#[cfg(test)]
mod tests;

/// The modulus q, mirroring `crate::algebraic::field`.
const Q: i16 = parameters::Q as i16;

/// `q^-1 mod 2^16`, signed, mirroring `crate::algebraic::field`.
const QINV: i16 = -3327;

/// `((2^26 + q/2) / q) = 20159`, the Barrett multiplier, mirroring `field`.
const BARRETT_V: i16 = 20159;

/// The final NTT⁻¹ scale `f = 128^-1 * R^2 mod q = 1441` in Montgomery form,
/// mirroring `generic`'s `NTT_INVERSE_SCALE`.
const NTT_INVERSE_SCALE: i16 = 1441;

/// Montgomery-multiplies sixteen coefficient lanes, `a * b -> a*b*R^-1 mod q`:
/// the vector form of the scalar `*` on `FieldElement`.
///
/// Uses the `mullo`/`mulhi` identity: with `lo`/`hi` the low/high halves of
/// `a*b` and `m = lo * QINV`, the reduction `(a*b - m*q) >> 16` equals
/// `hi - mulhi(m, q)` because `a*b - m*q` is a multiple of `2^16`.
#[inline]
fn fqmul(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe {
        let q = _mm256_set1_epi16(Q);
        let qinv = _mm256_set1_epi16(QINV);

        let lo = _mm256_mullo_epi16(a, b);
        let hi = _mm256_mulhi_epi16(a, b);
        let m = _mm256_mullo_epi16(lo, qinv);
        let t = _mm256_mulhi_epi16(m, q);

        _mm256_sub_epi16(hi, t)
    }
}

/// Multiplies `a` by a compile-time constant `b` using Barrett reduction with
/// the precomputed multiplier `b_bar = round(b * 2^15 / q)`, returning
/// `a * b mod q` per lane in four vector instructions.
///
/// `b` must be a canonical (non-Montgomery) representative for the result to
/// preserve the Montgomery-domain convention: given `a = a_true * R`,
/// `a * b mod q = (a_true * b) * R`. For NTT butterflies with the shared
/// [`super::ZETA_BARRETT`] table paired with [`super::ZETA_RAW`], this
/// replaces the [`fqmul`] path (`mullo` + `mulhi` + `mullo` + `mulhi` + `sub`,
/// five intrinsics) with four.
#[inline]
fn barrett_const_mul(a: __m256i, b: __m256i, b_bar: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe {
        let c_low = _mm256_mullo_epi16(a, b);
        let t = _mm256_mulhrs_epi16(a, b_bar);
        let tq = _mm256_mullo_epi16(t, _mm256_set1_epi16(Q));

        _mm256_sub_epi16(c_low, tq)
    }
}

/// Barrett-reduces sixteen `i16` lanes to a representative in `(-q/2, q/2]`:
/// the vector form of `FieldElement::reduce`.
///
/// `t = (a*BARRETT_V + 2^25) >> 26` is computed as `(mulhi(a, BARRETT_V) + 2^9)
/// >> 10`, since `mulhi` is the `>> 16`.
#[inline]
fn barrett_reduce(a: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe {
        let mut t = _mm256_mulhi_epi16(a, _mm256_set1_epi16(BARRETT_V));
        t = _mm256_add_epi16(t, _mm256_set1_epi16(1 << 9));
        t = _mm256_srai_epi16::<10>(t);
        t = _mm256_mullo_epi16(t, _mm256_set1_epi16(Q));

        _mm256_sub_epi16(a, t)
    }
}

/// Pointwise base multiplication of two NTT representations: 128 degree-one
/// products modulo the quadratics `X^2 - gamma`, computed sixteen pairs at a
/// time.
///
/// Matches [`super::generic::multiply`]. AVX2 has no deinterleaving
/// load, so even (degree-0) and odd (degree-1) coefficients are separated with
/// a `shuffle_epi8` + `permute4x64`/`permute2x128` sequence and re-interleaved
/// on store. The result is scaled by `R^-1`, which `ntt_inverse` later undoes.
pub(crate) fn multiply(
    f: &[FieldElement; parameters::N],
    g: &[FieldElement; parameters::N],
) -> [FieldElement; parameters::N] {
    let mut h = [FieldElement::ZERO; parameters::N];

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the three
    // length-256 arrays and the length-128 `GAMMA_MONT` reinterpret as `[i16]`.
    // Each iteration reads/writes a 32-`i16` window of `f`/`g`/`h` (8 windows
    // tile 256) and a 16-`i16` window of the gammas (8 windows tile 128), all in
    // bounds. `loadu`/`storeu` are unaligned; the module is AVX2.
    unsafe {
        let f_ptr = f.as_ptr().cast::<i16>();
        let g_ptr = g.as_ptr().cast::<i16>();
        let h_ptr = h.as_mut_ptr().cast::<i16>();
        let gamma_ptr = super::GAMMA_MONT.as_ptr().cast::<i16>();

        // Per-128-bit-lane byte masks (repeated across both lanes): gather the
        // even `i16`s into the low half and odds into the high half, and the
        // inverse that re-interleaves them.
        let deinterleave = _mm256_setr_epi8(
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, //
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
        );
        let interleave = _mm256_setr_epi8(
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, //
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
        );

        let mut pair = 0;
        while pair < 128 {
            let f0 = _mm256_loadu_si256(f_ptr.add(2 * pair).cast::<__m256i>());
            let f1 = _mm256_loadu_si256(f_ptr.add(2 * pair + 16).cast::<__m256i>());
            let g0 = _mm256_loadu_si256(g_ptr.add(2 * pair).cast::<__m256i>());
            let g1 = _mm256_loadu_si256(g_ptr.add(2 * pair + 16).cast::<__m256i>());

            // Each load holds 8 pairs; deinterleave to `[8 even | 8 odd]`.
            let f0d = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(f0, deinterleave));
            let f1d = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(f1, deinterleave));
            let g0d = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(g0, deinterleave));
            let g1d = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(g1, deinterleave));

            // Gather 16 pairs' even (a0/b0) and odd (a1/b1) coefficients.
            let a0 = _mm256_permute2x128_si256::<0x20>(f0d, f1d);
            let a1 = _mm256_permute2x128_si256::<0x31>(f0d, f1d);
            let b0 = _mm256_permute2x128_si256::<0x20>(g0d, g1d);
            let b1 = _mm256_permute2x128_si256::<0x31>(g0d, g1d);

            let gamma = _mm256_loadu_si256(gamma_ptr.add(pair).cast::<__m256i>());

            // c0 = a0*b0 + a1*b1*gamma ; c1 = a0*b1 + a1*b0
            let c0 = _mm256_add_epi16(fqmul(a0, b0), fqmul(fqmul(a1, b1), gamma));
            let c1 = _mm256_add_epi16(fqmul(a0, b1), fqmul(a1, b0));

            // Re-interleave even (c0) and odd (c1) coefficients back to pairs.
            let lo = _mm256_permute2x128_si256::<0x20>(c0, c1);
            let hi = _mm256_permute2x128_si256::<0x31>(c0, c1);
            let h0 = _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0xD8>(lo), interleave);
            let h1 = _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0xD8>(hi), interleave);

            _mm256_storeu_si256(h_ptr.add(2 * pair).cast::<__m256i>(), h0);
            _mm256_storeu_si256(h_ptr.add(2 * pair + 16).cast::<__m256i>(), h1);

            pair += 16;
        }
    }

    h
}

/// Forward NTT in place, then Barrett-reduce. Vectorizes the stride-≥16
/// butterfly stages sixteen lanes wide; the narrow stride-8/4/2 stages and the
/// final reduction run scalar.
///
/// Matches [`super::generic::ntt`].
// reason: scalar-tail butterfly/zeta indices are provably in 0..256, as in the
// generic backend.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt(coefficients: &mut [FieldElement; parameters::N]) {
    let mut k = 1;

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the array
    // reinterprets as `[i16]`. Each `__m256i` window `[j, j + 16)` and
    // `[j + len, j + len + 16)` stays within the 256-element array, and `k`
    // indexes `ZETA_RAW` / `ZETA_BARRETT`. `loadu`/`storeu` are unaligned;
    // the module is AVX2.
    unsafe {
        let ptr = coefficients.as_mut_ptr().cast::<i16>();
        let zeta_raw_ptr = super::ZETA_RAW.as_ptr().cast::<i16>();
        let zeta_bar_ptr = super::ZETA_BARRETT.as_ptr();

        for len in [128usize, 64, 32, 16] {
            let mut start = 0;
            while start < 256 {
                let zeta = _mm256_set1_epi16(*zeta_raw_ptr.add(k));
                let zeta_bar = _mm256_set1_epi16(*zeta_bar_ptr.add(k));
                k += 1;

                let mut j = start;
                while j < start + len {
                    let vj = _mm256_loadu_si256(ptr.add(j).cast::<__m256i>());
                    let vjl = _mm256_loadu_si256(ptr.add(j + len).cast::<__m256i>());
                    let t = barrett_const_mul(vjl, zeta, zeta_bar);

                    _mm256_storeu_si256(
                        ptr.add(j + len).cast::<__m256i>(),
                        _mm256_sub_epi16(vj, t),
                    );
                    _mm256_storeu_si256(ptr.add(j).cast::<__m256i>(), _mm256_add_epi16(vj, t));

                    j += 16;
                }

                start += 2 * len;
            }
        }
    }

    // Scalar tail: stride-8, stride-4, and stride-2 groups are narrower than a
    // vector.
    for len in [8usize, 4, 2] {
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
/// Vectorizes the stride-≥16 Gentleman-Sande stages, their Barrett reduction,
/// and the final scale; the narrow stride-2/4/8 stages run scalar.
///
/// Matches [`super::generic::ntt_inverse`].
// reason: scalar-head butterfly/zeta indices are provably in 0..256, as in the
// generic backend.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt_inverse(coefficients: &mut [FieldElement; parameters::N]) {
    let mut k = 127;

    // Scalar head: stride-2, stride-4, and stride-8 groups are narrower than a
    // vector.
    for len in [2usize, 4, 8] {
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

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the array
    // reinterprets as `[i16]`. Each `__m256i` window stays within the
    // 256-element array, and the descending `k` indexes `ZETA_RAW` /
    // `ZETA_BARRETT` (it reaches 0 only after the last use). `loadu`/`storeu`
    // are unaligned; the module is AVX2.
    unsafe {
        let ptr = coefficients.as_mut_ptr().cast::<i16>();
        let zeta_raw_ptr = super::ZETA_RAW.as_ptr().cast::<i16>();
        let zeta_bar_ptr = super::ZETA_BARRETT.as_ptr();

        for len in [16usize, 32, 64, 128] {
            let mut start = 0;
            while start < 256 {
                let zeta = _mm256_set1_epi16(*zeta_raw_ptr.add(k));
                let zeta_bar = _mm256_set1_epi16(*zeta_bar_ptr.add(k));
                k -= 1;

                let mut j = start;
                while j < start + len {
                    let vj = _mm256_loadu_si256(ptr.add(j).cast::<__m256i>());
                    let vjl = _mm256_loadu_si256(ptr.add(j + len).cast::<__m256i>());

                    _mm256_storeu_si256(
                        ptr.add(j).cast::<__m256i>(),
                        barrett_reduce(_mm256_add_epi16(vj, vjl)),
                    );
                    _mm256_storeu_si256(
                        ptr.add(j + len).cast::<__m256i>(),
                        barrett_const_mul(_mm256_sub_epi16(vjl, vj), zeta, zeta_bar),
                    );

                    j += 16;
                }

                start += 2 * len;
            }
        }

        // Final scale: every coefficient times `NTT_INVERSE_SCALE`.
        let scale = _mm256_set1_epi16(NTT_INVERSE_SCALE);
        let mut j = 0;
        while j < 256 {
            let scaled = fqmul(scale, _mm256_loadu_si256(ptr.add(j).cast::<__m256i>()));
            _mm256_storeu_si256(ptr.add(j).cast::<__m256i>(), scaled);
            j += 16;
        }
    }
}
