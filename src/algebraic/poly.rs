//! The polynomial rings Rq and Tq and the NTT between them.
//!
//! [`RqElement`] is the standard domain; [`TqElement`] is the NTT
//! representation. Keeping them distinct types makes the domain of every value
//! visible in the type system: an [`RqElement`] cannot be multiplied as though
//! it were already transformed.
//!
//! Arithmetic follows the CRYSTALS-Kyber signed Montgomery convention (see
//! [`crate::algebraic::field`]): the zeta tables are stored in Montgomery form
//! so a butterfly's `fqmul(zeta, x)` yields the true product, base
//! multiplication leaves products scaled by `R^-1`, and `ntt_inverse` folds the
//! `1/128` and the compensating `R` into a single final scale (`f = 1441`).
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf

use core::ops::{Add, AddAssign, Index, Mul, Sub};

use crate::{algebraic::field::FieldElement, parameters};

#[cfg(test)]
mod tests;

/// The loop strides taken by the NTT (and NTT⁻¹) butterfly stages.
///
/// Listed in [NTT] direction; NTT⁻¹ walks them in reverse.
///
/// [NTT]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.9
const NTT_SERIES: [usize; 7] = [128, 64, 32, 16, 8, 4, 2];

/// Final NTT⁻¹ scale `f = 128^-1 * R^2 mod q = 1441` (Montgomery form), folding
/// the `1/128` of the inverse transform with the `R` that undoes base
/// multiplication's `R^-1`.
const NTT_INVERSE_SCALE: FieldElement = FieldElement::from_montgomery_table(1441);

/// Returns the integer represented by bit-reversing the unsigned 7-bit value
/// that corresponds to the input integer `i` in `{0, ..., 127}`.
///
/// If `r = r_0 + 2 r_1 + ... + 64 r_6` with `r_i` in `{0, 1}`, then
/// `BitRev7(r) = r_6 + 2 r_5 + ... + 64 r_0`. Described in [section 4.3] of
/// FIPS 203.
///
/// [section 4.3]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.3
#[allow(dead_code)] // used only by the unit tests; the zeta tables are precomputed.
fn bit_rev_7(i: u8) -> u8 {
    let mut reversed: u8 = 0;

    for bit in 0..7 {
        reversed <<= 1;
        reversed |= (i >> bit) & 1;
    }

    reversed
}

/// A polynomial of the ring Rq or Tq: 256 coefficients in Zq.
///
/// Both rings have exactly n = 256 coefficients over Zq, so the standard-domain
/// [`RqElement`] and the NTT-domain [`TqElement`] share this trait for their
/// shared structure while remaining distinct types.
pub trait PolynomialRingElement: Copy + Index<usize, Output = FieldElement> {
    /// The polynomial all of whose coefficients are zero.
    const ZERO: Self;

    /// Constructs a polynomial from its coefficients.
    fn new(coefficients: [FieldElement; parameters::N]) -> Self;

    /// Returns this polynomial's coefficients.
    fn coefficients(&self) -> [FieldElement; parameters::N];
}

/// Elements of the polynomial ring Rq over Zq.
///
/// This is the standard domain of ML-KEM values, as opposed to the NTT
/// representation [`TqElement`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RqElement([FieldElement; parameters::N]);

impl RqElement {
    /// Transforms a polynomial `f` in Rq into its NTT representation `f_hat` in
    /// Tq, then Barrett-reduces the result so the coefficients stay small for
    /// base multiplication.
    ///
    /// Implements [Algorithm 9, `NTT(f)`] from FIPS 203.
    ///
    /// [Algorithm 9, `NTT(f)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.9
    #[allow(clippy::indexing_slicing)] // reason: butterfly indices j and j+len
    // are provably in 0..256 for every stage; the iterator rewrite (split_at_mut
    // across the stride) obscures the transform and is materially worse.
    pub fn ntt(self) -> TqElement {
        let mut f_hat = self.0;
        let mut k = 1;

        for len in NTT_SERIES {
            for start in (0..256).step_by(2 * len) {
                let zeta = ZETA_MONT[k];
                k += 1;

                for j in start..(start + len) {
                    let t = zeta * f_hat[j + len];
                    f_hat[j + len] = f_hat[j] - t;
                    f_hat[j] = f_hat[j] + t;
                }
            }
        }

        for coefficient in &mut f_hat {
            *coefficient = coefficient.reduce();
        }

        TqElement::new(f_hat)
    }
}

impl PolynomialRingElement for RqElement {
    const ZERO: Self = Self([FieldElement::ZERO; parameters::N]);

    fn new(coefficients: [FieldElement; parameters::N]) -> Self {
        Self(coefficients)
    }

    fn coefficients(&self) -> [FieldElement; parameters::N] {
        self.0
    }
}

impl Index<usize> for RqElement {
    type Output = FieldElement;

    // reason: the `Index` contract is to index and panic on out-of-bounds; an
    // `Output`-returning method cannot forward a `.get` failure.
    #[allow(clippy::indexing_slicing)]
    fn index(&self, index: usize) -> &FieldElement {
        &self.0[index]
    }
}

impl Add for RqElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut result = self.0;

        for (r, b) in result.iter_mut().zip(rhs.0) {
            *r = *r + b;
        }

        Self(result)
    }
}

impl Sub for RqElement {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut result = self.0;

        for (r, b) in result.iter_mut().zip(rhs.0) {
            *r = *r - b;
        }

        Self(result)
    }
}

impl From<TqElement> for RqElement {
    fn from(value: TqElement) -> Self {
        value.ntt_inverse()
    }
}

/// Elements of the polynomial ring Tq over Zq: the NTT representation.
///
/// The ring Rq is isomorphic to Tq, a direct sum of 128 quadratic extensions of
/// Zq. The NTT is the computationally efficient isomorphism between them. See
/// [section 4.3] of FIPS 203.
///
/// [section 4.3]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.3
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TqElement([FieldElement; parameters::N]);

impl TqElement {
    /// Transforms a polynomial `f_hat` in Tq from its NTT representation back
    /// into `f` in Rq.
    ///
    /// Because base multiplication leaves products scaled by `R^-1`, this also
    /// applies the [`NTT_INVERSE_SCALE`] factor that restores the standard
    /// domain — so `(f_hat * g_hat).ntt_inverse()` is the true product `f * g`,
    /// while `f.ntt().ntt_inverse()` is `f` scaled by `R` (not the identity).
    ///
    /// Implements [Algorithm 10, `NTT⁻¹(f_hat)`] from FIPS 203.
    ///
    /// [Algorithm 10, `NTT⁻¹(f_hat)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.10
    #[allow(clippy::indexing_slicing)] // reason: butterfly indices j and j+len
    // are provably in 0..256 for every stage; see `RqElement::ntt`.
    pub fn ntt_inverse(self) -> RqElement {
        let mut f = self.0;
        let mut k = 127;

        for len in NTT_SERIES.into_iter().rev() {
            for start in (0..256).step_by(2 * len) {
                let zeta = ZETA_MONT[k];
                k -= 1;

                for j in start..(start + len) {
                    let t = f[j];
                    f[j] = (t + f[j + len]).reduce();
                    f[j + len] = zeta * (f[j + len] - t);
                }
            }
        }

        for coefficient in &mut f {
            *coefficient = *coefficient * NTT_INVERSE_SCALE;
        }

        RqElement::new(f)
    }

    /// Scales every coefficient by `R`, undoing the `R^-1` left by base
    /// multiplication so an NTT-domain product can be added to true NTT values
    /// (`K-PKE.KeyGen`'s `t_hat = A . s_hat + e_hat`).
    pub fn to_montgomery(self) -> Self {
        Self(self.0.map(FieldElement::to_montgomery))
    }
}

impl PolynomialRingElement for TqElement {
    const ZERO: Self = Self([FieldElement::ZERO; parameters::N]);

    fn new(coefficients: [FieldElement; parameters::N]) -> Self {
        Self(coefficients)
    }

    fn coefficients(&self) -> [FieldElement; parameters::N] {
        self.0
    }
}

impl Index<usize> for TqElement {
    type Output = FieldElement;

    // reason: see `RqElement`'s `Index` impl — indexing is the trait's contract.
    #[allow(clippy::indexing_slicing)]
    fn index(&self, index: usize) -> &FieldElement {
        &self.0[index]
    }
}

impl Add for TqElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut result = self.0;

        for (r, b) in result.iter_mut().zip(rhs.0) {
            *r = *r + b;
        }

        Self(result)
    }
}

impl AddAssign for TqElement {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for TqElement {
    type Output = Self;

    /// Computes the product of two NTT representations (polynomials in Tq).
    ///
    /// Implements [Algorithm 11, `MultiplyNTTs(f_hat, g_hat)`] from FIPS 203,
    /// which reduces to 128 independent degree-one products via
    /// `base_case_multiply`. The result is scaled by `R^-1` (Montgomery
    /// convention), which `ntt_inverse` later undoes.
    ///
    /// [Algorithm 11, `MultiplyNTTs(f_hat, g_hat)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.11
    // reason: indices 2i and 2i+1 are provably in 0..256 for i in 0..128, and the
    // loop reads `GAMMA_MONT[i]` while writing `h_hat[2i]`/`h_hat[2i+1]` from
    // `f_hat`/`g_hat`; the pairwise indexing is clearer than a chunked-iterator
    // rewrite over four parallel arrays.
    #[allow(clippy::indexing_slicing, clippy::needless_range_loop)]
    fn mul(self, rhs: Self) -> Self {
        /// Computes the product of two degree-one polynomials modulo the
        /// quadratic `X^2 - gamma`, with `gamma` in Montgomery form.
        ///
        /// Implements [Algorithm 12, `BaseCaseMultiply(a0, a1, b0, b1, gamma)`]
        /// from FIPS 203.
        ///
        /// [Algorithm 12, `BaseCaseMultiply(a0, a1, b0, b1, gamma)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.12
        fn base_case_multiply(
            a0: FieldElement,
            a1: FieldElement,
            b0: FieldElement,
            b1: FieldElement,
            gamma: FieldElement,
        ) -> (FieldElement, FieldElement) {
            let c0 = (a0 * b0) + (a1 * b1 * gamma);
            let c1 = (a0 * b1) + (a1 * b0);

            (c0, c1)
        }

        let (f_hat, g_hat) = (self.0, rhs.0);
        let mut h_hat = [FieldElement::ZERO; parameters::N];

        for i in 0..128 {
            let (even, odd) = (2 * i, 2 * i + 1);

            (h_hat[even], h_hat[odd]) = base_case_multiply(
                f_hat[even],
                f_hat[odd],
                g_hat[even],
                g_hat[odd],
                GAMMA_MONT[i],
            );
        }

        Self(h_hat)
    }
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
