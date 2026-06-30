//! The prime field Zq for ML-KEM, where q = 3329 for every parameter set.
//!
//! Arithmetic uses the signed Montgomery convention of the CRYSTALS-Kyber
//! reference: coefficients are `i16` representatives of `Z/q`, not necessarily
//! canonical, and multiplication is Montgomery multiplication
//! (`a * b -> a*b*R^-1 mod q`, with `R = 2^16`). The NTT zeta tables are stored
//! in Montgomery form so the `R^-1` cancels (see [`crate::algebraic::poly`]).
//! Every operation is branch-free and division-free, so it is constant-time.
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf

use core::ops::{Add, Mul, Neg, Sub};

use zeroize::Zeroize;

use crate::parameters;

#[cfg(test)]
mod tests;

/// The modulus q as a signed integer.
const Q: i16 = parameters::Q as i16;

/// `q^-1 mod 2^16`, signed, so `q * QINV ≡ 1 (mod 2^16)`.
const QINV: i16 = -3327;

/// `((2^26 + q/2) / q)`, the Barrett multiplier for reduction by q.
const BARRETT_V: i32 = ((1 << 26) + (Q as i32) / 2) / (Q as i32);

/// `R^2 mod q` with `R = 2^16`; multiplying by this via Montgomery reduction
/// scales a value by `R` (used to undo the `R^-1` left by base multiplication).
const MONT_R_SQUARED: i16 = 1353;

/// `ceil(2^33 / q) = 2580335`, the Barrett multiplier for division by `q`.
/// Verified to give `floor(n / q)` via `(n * BARRETT_DIV_Q) >> 33` for every
/// `n` in `[0, (q - 1) << 12 + q/2]` (max ~1.36e7, at `compress`'s `d = 12`),
/// which contains the full `u16` range used by [`FieldElement::new`].
const BARRETT_DIV_Q: u64 = 2_580_335;

/// An element of `Z` mod q, in signed Montgomery representation.
///
/// The stored value is a representative of the residue class, not necessarily
/// in `0..q`; [`Self::value`] returns the canonical representative.
// `repr(transparent)` guarantees the same layout as `i16`, so the SIMD backends
// (`crate::algebraic::poly::arch`) can reinterpret a `[FieldElement; N]` as
// `[i16; N]` for vector loads/stores. `derive(Zeroize)` only adds a method, so
// the transparent layout stands.
#[derive(Debug, Clone, Copy, Zeroize)]
#[repr(transparent)]
pub struct FieldElement(i16);

impl FieldElement {
    /// The additive identity.
    pub const ZERO: Self = Self(0);

    /// Instantiates a canonical `FieldElement` from an integer, reduced mod q.
    ///
    /// Division-free: the reduction is the explicit Barrett mul-shift
    /// `(value * BARRETT_DIV_Q) >> 33`, exact for every `u16` input, then
    /// `value - quotient * q` gives the canonical representative in `[0, q)`.
    /// This keeps `new` constant-time on the secret-derived `ByteDecode_12`
    /// inputs (the recovered ciphertext coefficients in `K-PKE.Decrypt`).
    #[inline]
    pub const fn new(value: u16) -> Self {
        let v = value as u64;
        let quotient = (v * BARRETT_DIV_Q) >> 33;
        Self((v - quotient * Q as u64) as i16)
    }

    /// Returns the canonical representative in `0 <= value < Q`.
    #[inline]
    pub const fn value(self) -> u16 {
        let r = Self::barrett(self.0);

        // Branch-free conditional add of q when r is negative: r >> 15 is all-1s
        // (i.e. -1) for r < 0 and 0 otherwise.
        (r + ((r >> 15) & Q)) as u16
    }

    /// Returns a Barrett-reduced representative, congruent mod q and bounded by
    /// `(-q/2, q/2]`. Used to keep NTT coefficients small.
    #[inline]
    pub const fn reduce(self) -> Self {
        Self(Self::barrett(self.0))
    }

    /// Scales by `R = 2^16` mod q via Montgomery multiplication by `R^2`.
    ///
    /// Base multiplication leaves products scaled by `R^-1`; applying this
    /// restores the standard domain (`K-PKE.KeyGen`'s `t_hat`).
    #[inline]
    pub const fn to_montgomery(self) -> Self {
        Self(Self::montgomery_reduce(
            self.0 as i32 * MONT_R_SQUARED as i32,
        ))
    }

    /// Wraps a raw Montgomery-form representative (e.g. the inverse-NTT scale).
    #[inline]
    pub(super) const fn from_montgomery_table(value: i16) -> Self {
        Self(value)
    }

    /// Builds a 128-entry table of Montgomery-form field elements from
    /// canonical zeta values at compile time (the NTT zeta tables).
    pub(super) const fn montgomery_table(raw: [u16; 128]) -> [Self; 128] {
        let mut table = [Self::ZERO; 128];

        #[allow(clippy::indexing_slicing)] // reason: i < 128 by the loop guard.
        let mut i = 0;
        while i < 128 {
            #[allow(clippy::indexing_slicing)] // reason: i < 128 by the loop guard.
            {
                table[i] = Self::new(raw[i]).to_montgomery();
            }
            i += 1;
        }

        table
    }

    /// Barrett reduction of `a` to a representative in `(-q/2, q/2]`.
    #[inline]
    const fn barrett(a: i16) -> i16 {
        let t = ((BARRETT_V * a as i32 + (1 << 25)) >> 26) as i16;

        a - t.wrapping_mul(Q)
    }

    /// Montgomery reduction: returns `a * R^-1 mod q` in `(-q, q)`, with
    /// `R = 2^16`.
    #[inline]
    const fn montgomery_reduce(a: i32) -> i16 {
        let m = (a as i16).wrapping_mul(QINV);

        ((a - (m as i32) * (Q as i32)) >> 16) as i16
    }

    /// `Compress_d`: maps this element to a `d`-bit value in `{0, ..., 2^d -
    /// 1}` via `round((2^d / q) * x) mod 2^d`.
    ///
    /// Defined by [equation 4.7] of FIPS 203. Division by `q` is computed via
    /// the explicit Barrett mul-shift `(num * BARRETT_DIV_Q) >> 33`, which is
    /// constant-time on secret-derived inputs (the recovered message bits in
    /// `K-PKE.Decrypt`'s `compress_message`).
    ///
    /// [equation 4.7]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.2.1
    #[inline]
    pub fn compress(self, d: usize) -> u16 {
        let numerator = (u32::from(self.value()) << d) + u32::from(parameters::Q / 2);
        let quotient = ((u64::from(numerator) * BARRETT_DIV_Q) >> 33) as u32;

        (quotient & ((1 << d) - 1)) as u16
    }

    /// `Decompress_d`: maps a `d`-bit value back into Zq via
    /// `round((q / 2^d) * y)`.
    ///
    /// Defined by [equation 4.8] of FIPS 203.
    ///
    /// [equation 4.8]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.2.1
    #[inline]
    pub fn decompress(value: u16, d: usize) -> Self {
        let numerator = u32::from(parameters::Q) * u32::from(value) + (1 << (d - 1));

        Self::new((numerator >> d) as u16)
    }
}

#[cfg(any(test, feature = "expose-internals"))]
impl FieldElement {
    /// Reference multiplication in the canonical value domain, `a * b mod q`,
    /// using plain integer arithmetic independent of the Montgomery [`Mul`]
    /// impl. The standard-domain oracle for NTT products, shared by the unit
    /// tests and the `expose-internals` property tests.
    pub fn mul_reference(self, rhs: Self) -> Self {
        let product = u32::from(self.value()) * u32::from(rhs.value());

        Self::new((product % u32::from(parameters::Q)) as u16)
    }
}

impl PartialEq for FieldElement {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Eq for FieldElement {}

impl From<u8> for FieldElement {
    #[inline]
    fn from(value: u8) -> Self {
        Self::new(u16::from(value))
    }
}

impl From<u16> for FieldElement {
    #[inline]
    fn from(value: u16) -> Self {
        Self::new(value)
    }
}

impl From<FieldElement> for u16 {
    #[inline]
    fn from(value: FieldElement) -> Self {
        value.value()
    }
}

impl Add for FieldElement {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for FieldElement {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for FieldElement {
    type Output = Self;

    /// Montgomery multiplication: `a * b -> a*b*R^-1 mod q`. Correct standard-
    /// domain products arise when one operand is a Montgomery-form constant
    /// (the NTT zetas), which cancels the `R^-1`.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(Self::montgomery_reduce(self.0 as i32 * rhs.0 as i32))
    }
}

impl Neg for FieldElement {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
