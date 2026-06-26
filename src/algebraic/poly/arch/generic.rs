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

/// The Montgomery-form values `ζ^BitRev7(i) * R mod q` for `i` in `{0, ...,
/// 127}`, derived from the canonical [FIPS 203 Appendix A] table.
///
/// [FIPS 203 Appendix A]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#appendix.A
const ZETA_MONT: [FieldElement; 128] = FieldElement::montgomery_table([
    1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 3260, 569, 1746, 296,
    2447, 1339, 1476, 3046, 56, 2240, 1333, 1426, 2094, 535, 2882, 2393, 2879, 1974, 821, 289, 331,
    3253, 1756, 1197, 2304, 2277, 2055, 650, 1977, 2513, 632, 2865, 33, 1320, 1915, 2319, 1435,
    807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648, 2474, 3110, 1227, 910, 17, 2761, 583,
    2649, 1637, 723, 2288, 1100, 1409, 2662, 3281, 233, 756, 2156, 3015, 3050, 1703, 1651, 2789,
    1789, 1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641, 1584, 2298, 2037,
    3220, 375, 2549, 2090, 1645, 1063, 319, 2773, 757, 2099, 561, 2466, 2594, 2804, 1092, 403,
    1026, 1143, 2150, 2775, 886, 1722, 1212, 1874, 1029, 2110, 2935, 885, 2154,
]);

/// The Montgomery-form values `ζ^(2 BitRev7(i) + 1) * R mod q` for `i` in
/// `{0, ..., 127}`, derived from the canonical [FIPS 203 Appendix A] table (the
/// modular reduction applied, matching BoringSSL).
///
/// [FIPS 203 Appendix A]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#appendix.A
const GAMMA_MONT: [FieldElement; 128] = FieldElement::montgomery_table([
    17, 3312, 2761, 568, 583, 2746, 2649, 680, 1637, 1692, 723, 2606, 2288, 1041, 1100, 2229, 1409,
    1920, 2662, 667, 3281, 48, 233, 3096, 756, 2573, 2156, 1173, 3015, 314, 3050, 279, 1703, 1626,
    1651, 1678, 2789, 540, 1789, 1540, 1847, 1482, 952, 2377, 1461, 1868, 2687, 642, 939, 2390,
    2308, 1021, 2437, 892, 2388, 941, 733, 2596, 2337, 992, 268, 3061, 641, 2688, 1584, 1745, 2298,
    1031, 2037, 1292, 3220, 109, 375, 2954, 2549, 780, 2090, 1239, 1645, 1684, 1063, 2266, 319,
    3010, 2773, 556, 757, 2572, 2099, 1230, 561, 2768, 2466, 863, 2594, 735, 2804, 525, 1092, 2237,
    403, 2926, 1026, 2303, 1143, 2186, 2150, 1179, 2775, 554, 886, 2443, 1722, 1607, 1212, 2117,
    1874, 1455, 1029, 2300, 2110, 1219, 2935, 394, 885, 2444, 2154, 1175,
]);
