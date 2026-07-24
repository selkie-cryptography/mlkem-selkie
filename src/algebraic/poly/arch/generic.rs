//! Portable scalar NTT and base-multiplication kernels: the always-available
//! fallback backend.
//!
//! Arithmetic follows the CRYSTALS-Kyber signed Montgomery convention (see
//! [`crate::algebraic::field`]): the zeta tables are stored in Montgomery form,
//! so a butterfly's `zeta * x` yields the true product, base multiplication
//! leaves products scaled by `R^-1`, and [`ntt_inverse`] folds the `1/128` and
//! the compensating `R` into a single final scale. Implements [Algorithms
//! 9-12] of FIPS 203.
//!
//! [Algorithms 9-12]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.9
//!
//! Always compiled as the portable reference. When a vector backend supersedes
//! a kernel for the active target, this version stays as the fallback for other
//! architectures and as the differential-test oracle, so some kernels here are
//! unused in any single build.
#![allow(dead_code)]

use super::{GAMMA_MONT, ProductAccumulator, ZETA_BARRETT, ZETA_RAW};
use crate::{algebraic::field::FieldElement, parameters};

/// The loop strides taken by the NTT (and NTT⁻¹) butterfly stages, in [NTT]
/// direction; NTT⁻¹ walks them in reverse.
///
/// [NTT]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.9
const NTT_SERIES: [usize; 7] = [128, 64, 32, 16, 8, 4, 2];

/// Final NTT⁻¹ scale `f = 128^-1 * R^2 mod q = 1441` (Montgomery form), folding
/// the `1/128` of the inverse transform with the `R` that undoes base
/// multiplication's `R^-1`.
const NTT_INVERSE_SCALE: FieldElement = FieldElement::from_montgomery_table(1441);

/// Forward NTT in place: maps `f` in Rq to `f_hat` in Tq, then Barrett-reduces
/// so the coefficients stay small for base multiplication.
///
/// Implements [Algorithm 9, `NTT(f)`] from FIPS 203.
///
/// [Algorithm 9, `NTT(f)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.9
// reason: butterfly indices j and j+len are provably in 0..256 for every stage;
// the iterator rewrite (split_at_mut across the stride) obscures the transform.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt(coefficients: &mut [FieldElement; parameters::N]) {
    let mut k = 1;

    for len in NTT_SERIES {
        for start in (0..256).step_by(2 * len) {
            let zeta = ZETA_RAW[k] as i16;
            let zeta_bar = ZETA_BARRETT[k];
            k += 1;

            for j in start..(start + len) {
                let t = coefficients[j + len].barrett_const_mul(zeta, zeta_bar);
                coefficients[j + len] = coefficients[j] - t;
                coefficients[j] = coefficients[j] + t;
            }
        }
    }

    for coefficient in coefficients.iter_mut() {
        *coefficient = coefficient.reduce();
    }
}

/// Inverse NTT in place: maps `f_hat` in Tq back to `f` in Rq, applying the
/// `NTT_INVERSE_SCALE` factor that restores the standard domain.
///
/// Implements [Algorithm 10, `NTT⁻¹(f_hat)`] from FIPS 203.
///
/// [Algorithm 10, `NTT⁻¹(f_hat)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.10
// reason: butterfly indices j and j+len are provably in 0..256 for every stage.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt_inverse(coefficients: &mut [FieldElement; parameters::N]) {
    let mut k = 127;

    for len in NTT_SERIES.into_iter().rev() {
        for start in (0..256).step_by(2 * len) {
            let zeta = ZETA_RAW[k] as i16;
            let zeta_bar = ZETA_BARRETT[k];
            k -= 1;

            for j in start..(start + len) {
                let t = coefficients[j];
                let sum = t + coefficients[j + len];
                // Lazy reduction: only len-2 and len-16 reduce; representatives
                // stay within 8q < i16::MAX between reductions.
                coefficients[j] = if len == 2 || len == 16 {
                    sum.reduce()
                } else {
                    sum
                };
                coefficients[j + len] =
                    (coefficients[j + len] - t).barrett_const_mul(zeta, zeta_bar);
            }
        }
    }

    for coefficient in coefficients.iter_mut() {
        *coefficient = *coefficient * NTT_INVERSE_SCALE;
    }
}

/// `Compress_d` of every coefficient: canonicalizes and maps each to
/// `round((2^d / q) * x) mod 2^d`, one output value per coefficient.
///
/// Constant-time on secret-derived inputs, as `FieldElement::compress` is.
pub(crate) fn compress(
    coefficients: &[FieldElement; parameters::N],
    d: usize,
) -> [u16; parameters::N] {
    coefficients.map(|c| c.compress(d))
}

/// `Decompress_d` of every value: maps each `d`-bit value back into Zq via
/// `round((q / 2^d) * y)`.
pub(crate) fn decompress(values: &[u16; parameters::N], d: usize) -> [FieldElement; parameters::N] {
    values.map(|v| FieldElement::decompress(v, d))
}

/// The canonical representative in `[0, q)` of every coefficient
/// (`FieldElement::value`), re-interleaved to natural order for
/// serialization.
// reason: indices 2i, 2i + 1, i, and 128 + i are provably in 0..256 for i in
// 0..128.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
pub(crate) fn canonical(coefficients: &[FieldElement; parameters::N]) -> [u16; parameters::N] {
    const HALF: usize = parameters::N / 2;

    let mut natural = [0u16; parameters::N];

    for i in 0..HALF {
        natural[2 * i] = coefficients[i].value();
        natural[2 * i + 1] = coefficients[HALF + i].value();
    }

    natural
}

/// Pointwise base multiplication of two NTT representations, reducing to 128
/// independent degree-one products modulo the quadratics `X^2 - gamma`. The
/// result is scaled by `R^-1` (Montgomery convention), which [`ntt_inverse`]
/// later undoes.
///
/// Implements [Algorithm 11, `MultiplyNTTs`] (and [Algorithm 12,
/// `BaseCaseMultiply`]) from FIPS 203.
///
/// [Algorithm 11, `MultiplyNTTs`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.11
/// [Algorithm 12, `BaseCaseMultiply`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.12
// reason: indices i and 128 + i are provably in 0..256 for i in 0..128, and
// the even/odd-half indexing is clearer than a split-iterator rewrite.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
pub(crate) fn multiply(
    f: &[FieldElement; parameters::N],
    g: &[FieldElement; parameters::N],
) -> [FieldElement; parameters::N] {
    const HALF: usize = parameters::N / 2;

    let mut h = [FieldElement::ZERO; parameters::N];

    for i in 0..HALF {
        let gamma = GAMMA_MONT[i];

        h[i] = (f[i] * g[i]) + (f[HALF + i] * g[HALF + i] * gamma);
        h[HALF + i] = (f[i] * g[HALF + i]) + (f[HALF + i] * g[i]);
    }

    h
}

/// Splits natural coefficient order into Tq's evens-then-odds storage.
// reason: indices 2i, 2i + 1, i, and 128 + i are provably in 0..256 for i in
// 0..128.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
pub(crate) fn pack(natural: &[FieldElement; parameters::N]) -> [FieldElement; parameters::N] {
    const HALF: usize = parameters::N / 2;

    let mut halves = [FieldElement::ZERO; parameters::N];

    for i in 0..HALF {
        halves[i] = natural[2 * i];
        halves[HALF + i] = natural[2 * i + 1];
    }

    halves
}

/// Re-interleaves Tq's evens-then-odds storage back to natural order: the
/// inverse of [`pack`].
// reason: as [`pack`] — every index is provably in 0..256.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
pub(crate) fn unpack(halves: &[FieldElement; parameters::N]) -> [FieldElement; parameters::N] {
    const HALF: usize = parameters::N / 2;

    let mut natural = [FieldElement::ZERO; parameters::N];

    for i in 0..HALF {
        natural[2 * i] = halves[i];
        natural[2 * i + 1] = halves[HALF + i];
    }

    natural
}

/// Accumulates one component of an asymmetric base-multiplication dot product
/// into `acc`: raw `i32` products, no reduction.
///
/// Per base pair, the degree-0 plane gains `f_e * g_e + f_o * cache_i` and
/// the degree-1 plane `f_e * g_o + f_o * g_e`, with the pair halves read
/// straight from the evens-then-odds storage halves and `cache` holding the
/// precomputed
/// `gamma_i * g_o` products. [`basemul_reduce`] later performs the single
/// Montgomery reduction per coefficient; the deferred sum matches
/// per-component reduction mod q.
// reason: indices 2i+1 and i are provably in bounds for i in 0..128, and the
// pairwise indexing matches `multiply`.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
pub(crate) fn basemul_accumulate(
    acc: &mut ProductAccumulator,
    f: &[FieldElement; parameters::N],
    g: &[FieldElement; parameters::N],
    cache: &[FieldElement; parameters::N / 2],
) {
    const HALF: usize = parameters::N / 2;

    for i in 0..HALF {
        acc.even[i] += f[i].widening_mul(g[i]) + f[HALF + i].widening_mul(cache[i]);
        acc.odd[i] += f[i].widening_mul(g[HALF + i]) + f[HALF + i].widening_mul(g[i]);
    }
}

/// Montgomery-reduces the accumulated product sums into the evens-then-odds
/// storage halves, one reduction per coefficient. The result is scaled by
/// `R^-1` (Montgomery convention), as [`multiply`]'s is.
// reason: indices 2i+1 and i are provably in bounds for i in 0..128, and the
// pairwise indexing matches `multiply`.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
pub(crate) fn basemul_reduce(acc: &ProductAccumulator) -> [FieldElement; parameters::N] {
    let mut h = [FieldElement::ZERO; parameters::N];

    const HALF: usize = parameters::N / 2;

    for i in 0..HALF {
        h[i] = FieldElement::from_product_sum(acc.even[i]);
        h[HALF + i] = FieldElement::from_product_sum(acc.odd[i]);
    }

    h
}
