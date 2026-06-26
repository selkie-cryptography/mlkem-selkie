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

use super::{GAMMA_MONT, ZETA_MONT};
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
            let zeta = ZETA_MONT[k];
            k += 1;

            for j in start..(start + len) {
                let t = zeta * coefficients[j + len];
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
            let zeta = ZETA_MONT[k];
            k -= 1;

            for j in start..(start + len) {
                let t = coefficients[j];
                coefficients[j] = (t + coefficients[j + len]).reduce();
                coefficients[j + len] = zeta * (coefficients[j + len] - t);
            }
        }
    }

    for coefficient in coefficients.iter_mut() {
        *coefficient = *coefficient * NTT_INVERSE_SCALE;
    }
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
// reason: indices 2i and 2i+1 are provably in 0..256 for i in 0..128, and the
// pairwise indexing is clearer than a chunked-iterator rewrite over the arrays.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
pub(crate) fn multiply(
    f: &[FieldElement; parameters::N],
    g: &[FieldElement; parameters::N],
) -> [FieldElement; parameters::N] {
    let mut h = [FieldElement::ZERO; parameters::N];

    for i in 0..128 {
        let (even, odd) = (2 * i, 2 * i + 1);
        let gamma = GAMMA_MONT[i];

        h[even] = (f[even] * g[even]) + (f[odd] * g[odd] * gamma);
        h[odd] = (f[even] * g[odd]) + (f[odd] * g[even]);
    }

    h
}
