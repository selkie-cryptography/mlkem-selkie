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

mod arch;

#[cfg(test)]
mod tests;

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
/// Both rings have n = 256 coefficients over Zq, so the standard-domain
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
    pub fn ntt(self) -> TqElement {
        let mut coefficients = self.0;
        arch::ntt(&mut coefficients);

        TqElement::new(coefficients)
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
    /// applies the `NTT_INVERSE_SCALE` factor that restores the standard
    /// domain — so `(f_hat * g_hat).ntt_inverse()` is the true product `f * g`,
    /// while `f.ntt().ntt_inverse()` is `f` scaled by `R` (not the identity).
    ///
    /// Implements [Algorithm 10, `NTT⁻¹(f_hat)`] from FIPS 203.
    ///
    /// [Algorithm 10, `NTT⁻¹(f_hat)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.10
    pub fn ntt_inverse(self) -> RqElement {
        let mut coefficients = self.0;
        arch::ntt_inverse(&mut coefficients);

        RqElement::new(coefficients)
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
    /// which reduces to 128 independent degree-one base-case products
    /// (delegated to the architecture backend's `multiply`). The result is
    /// scaled by `R^-1` (Montgomery convention), which `ntt_inverse` later
    /// undoes.
    ///
    /// [Algorithm 11, `MultiplyNTTs(f_hat, g_hat)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.11
    fn mul(self, rhs: Self) -> Self {
        Self(arch::multiply(&self.0, &rhs.0))
    }
}
