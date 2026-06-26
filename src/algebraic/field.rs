//! The prime field Zq for ML-KEM, where q = 3329 for every parameter set.
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf

use core::ops::{Add, Mul, Neg, Sub};

use crate::parameters;

#[cfg(test)]
mod tests;

/// Elements of `Z` mod q, where q = 3329 for all ML-KEM parameter sets.
///
/// Stored in canonical form, `0 <= value < Q`.
// TODO: OPTIMIZE. This is not production-grade by performance or by security:
// the `%` reductions are variable-time. Replace with signed Montgomery /
// Barrett reduction and constant-time conditional subtraction before any
// production use. `ML-KEM.Decaps` computes over secret-derived field elements
// (the decrypted message and re-encryption), so this is a constant-time gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldElement(u16);

impl FieldElement {
    /// The additive identity.
    pub const ZERO: Self = Self(0);

    /// Instantiates a new `FieldElement` from an integer, reduced modulo q.
    pub const fn new(value: u16) -> Self {
        Self(value % parameters::Q)
    }

    /// Returns the canonical representative in `0 <= value < Q`.
    pub const fn value(self) -> u16 {
        self.0
    }

    /// Reduces a wider intermediate modulo q.
    const fn reduce(value: u32) -> Self {
        Self((value % parameters::Q as u32) as u16)
    }

    /// `Compress_d`: maps this field element to a `d`-bit value in
    /// `{0, ..., 2^d - 1}` via `round((2^d / q) * x) mod 2^d`.
    ///
    /// The rounding is exact: `2^d x / q` is never a half-integer because q is
    /// odd, so the half-up/half-down choice is immaterial.
    ///
    /// Defined by [equation 4.7] of FIPS 203.
    ///
    /// [equation 4.7]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.2.1
    pub fn compress(self, d: usize) -> u16 {
        let numerator = (u32::from(self.0) << d) + u32::from(parameters::Q / 2);
        let quotient = numerator / u32::from(parameters::Q);

        (quotient & ((1 << d) - 1)) as u16
    }

    /// `Decompress_d`: maps a `d`-bit value back into Zq via
    /// `round((q / 2^d) * y)`.
    ///
    /// Defined by [equation 4.8] of FIPS 203.
    ///
    /// [equation 4.8]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.2.1
    pub fn decompress(value: u16, d: usize) -> Self {
        let numerator = u32::from(parameters::Q) * u32::from(value) + (1 << (d - 1));

        Self::reduce(numerator >> d)
    }
}

impl From<u8> for FieldElement {
    fn from(value: u8) -> Self {
        Self::new(u16::from(value))
    }
}

impl From<u16> for FieldElement {
    fn from(value: u16) -> Self {
        Self::new(value)
    }
}

impl From<FieldElement> for u16 {
    fn from(value: FieldElement) -> Self {
        value.0
    }
}

impl Add for FieldElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // Both operands are < Q, so the sum is < 2Q and fits in a u16.
        Self::new(self.0 + rhs.0)
    }
}

impl Sub for FieldElement {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // Add Q before subtracting so the intermediate stays non-negative;
        // self.0 + Q - rhs.0 is in (0, 2Q) and fits in a u16.
        Self::new(self.0 + parameters::Q - rhs.0)
    }
}

impl Mul for FieldElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::reduce(u32::from(self.0) * u32::from(rhs.0))
    }
}

impl Neg for FieldElement {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(parameters::Q - self.0)
    }
}
