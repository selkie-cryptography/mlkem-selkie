//! Polynomial ring arithmetic over finite fields for [FIPS 203].
//!
//! Arithmetic in the rings Rq and Tq, whose elements are polynomials defined
//! over ℤq, which are isomorphic to each other. The number-theoretic transform
//! (NTT) is a computationally efficient isomorphism between these rings that
//! allows for efficient arithmetic over matrixes and vectors of ring elements
//! in Rq.
//!
//! [FIPS-203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf

use std::{
    iter::IntoIterator,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub},
};

use crate::parameters::{self, ParameterSet};

/// ζ, a primitive 256-th root of unity modulo q.
///
/// Used in the representation of polynomials in the number theoretic transform
/// domain Tq and calculations over those.
///
/// Since q is the prime 3329 = 28 · 13 +1, and n = 256, there are 128 primitive
/// 256-th roots of unity and no primitive 512-th roots of unity in Zq: thus
/// ζ^128 ≡ −1.
///
/// Described in [section 4.3] of FIPS 203.
///
/// [section 4.3]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf#equation.4.8
const ZETA: FieldElement = FieldElement(17);

/// The steps taken in the loops of the NTT (and NTT⁻¹) algorithms.
///
/// The series is listed in [NTT] direction; NTT⁻¹ can use them in reverse.
///
/// [NTT]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf#algorithm.8
const NTT_SERIES: [usize; 7] = [128, 64, 32, 16, 8, 4, 2];

/// Returns the integer represented by bit-reversing the unsigned 7-bit value
/// that corresponds to the input integer i ∈ {0,...,127}.
///
/// Bit reversal of a seven-bit integer r. Specifcally, if r = r0 + 2r1 + 4r2
/// +···+ 64r6 with ri ∈ {0,1}, then BitRev₇(r) = r6 + 2r5 + 4r4 +···+ 64r0.
///
/// Described in [section 4.3] of FIPS 203.
///
/// [section 4.3]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf#equation.4.8
fn bit_rev_7(i: u8) -> u8 {
    let mut reversed: u8 = 0;

    for bit in 0..u8::BITS - 1 {
        reversed <<= 1;
        reversed |= (i & (1 << bit)) >> bit;
    }

    reversed
}

////////////////////////////////////////////////////////////////////////////////
// Field elements
////////////////////////////////////////////////////////////////////////////////

/// Elements of ℤ mod q, where q = 3329 for all ML-KEM parameter sets.
// TODO: OPTIMIZE.
//
// This is not production-grade code, neither by performance nor security. Replace.
//
// Simple things first, just get the structure in place for the field arithmetic before making it
// faster with things like signed integers, Montgomery representation, Barrett reduction, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldElement(u16);

impl FieldElement {
    /// Associated const pre-populated instance of zero.
    const ZERO: Self = Self(0);

    /// Instatiate a new `FieldElement` from a int value, modulo the modulus Q.
    fn new(value: u16) -> Self {
        Self(value % parameters::Q)
    }

    /// Exponentiation in the field.
    // TODO: replace with more efficient implementation.
    pub fn pow(&self, exponent: Self) -> Self {
        let mut result = Self::new(1);

        for _ in 0..exponent.into() {
            // Relies on the `Mul` impl to ensure the modular reduction.
            result = result * (*self);
        }

        return result;
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

impl From<u32> for FieldElement {
    fn from(value: u32) -> Self {
        let modulus_u32 = u32::from(parameters::Q);

        // TODO: get rid of this unwrap(), it can panic.
        Self::new((value % modulus_u32).try_into().unwrap())
    }
}

impl Add for FieldElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // u32s to handle overflow
        Self::from(u32::from(self.0) + u32::from(rhs.0))
    }
}

impl Mul for FieldElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // u32s to handle overflow
        Self::from(u32::from(self.0) * u32::from(rhs.0))
    }
}

impl Sub for FieldElement {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // i32s to handle overflow
        // TODO: get rid of unwrap()'s, they can panic.
        let value = (i32::try_from(self.0).unwrap() - i32::try_from(rhs.0).unwrap())
            .rem_euclid(parameters::Q.into());

        Self::from(u16::try_from(value).unwrap())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Polynomial ring elements (Rq and Tq)
////////////////////////////////////////////////////////////////////////////////

/// Elements of the polynomial rings Rq and Tq. Polynomial of degree N defined
/// over
///
/// We are blessed that all of these polynomials have exactly n = 256
/// coefficients, and are all defined over Zq, including the Tq ring, the NTT
/// representation target. We can therefore fix the number n as 256 and the
/// polynomial coefficients as [`FieldElement`]s. We can also share a lot of
/// operations between polynomials in Rq and in Tq, while preserving the fact
/// that a polynomial NTT representation is definitely a different type than the
/// original polynomial defined in Rq.
pub trait PolynomialRingElement: Copy + Index<usize> {
    /// Default value is a polynomial where all coefficients are zero.
    const ZERO: Self;

    /// Constructor for a new polynomial element instance.
    fn new(coefficients: [FieldElement; parameters::N]) -> Self;

    /// Get coefficients of this polynomial.
    fn coefficients(&self) -> [FieldElement; parameters::N];
}

impl<P> Add for P
where
    P: PolynomialRingElement,
{
    type Output = Self;

    fn add(self, rhs: P) -> Self {
        let mut result = [FieldElement::ZERO; parameters::N];

        for (i, (l, r)) in self.into_iter().zip(rhs.into_iter()).enumerate() {
            result[i] = l + r;
        }

        return Self::new(result);
    }
}

impl<P> IntoIterator for P
where
    P: PolynomialRingElement,
{
    type Item = FieldElement;

    type IntoIter = std::array::IntoIter<FieldElement, 256>;

    fn into_iter(self) -> Self::IntoIter {
        self.coefficients().into_iter()
    }
}

impl<P> Sub for P
where
    P: PolynomialRingElement,
{
    type Output = Self;

    fn sub(self, rhs: P) -> Self {
        let mut result = [FieldElement::ZERO; parameters::N];

        for (i, (l, r)) in self.into_iter().zip(rhs.into_iter()).enumerate() {
            result[i] = l - r;
        }

        return Self::new(result);
    }
}

/// Elements in the polynomial ring Rq over ℤq.
///
/// This is the default domain of values in ML-KEM, as opposed the NTT
/// representation of values which is in the Tq polynomial ring for some
/// computations.
// TODO: is it safe to derive `Eq` and `PartialEq` like so?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RqElement([FieldElement; parameters::N]);

impl RqElement {
    /// Transform a polynomial element _f_ in Rq into its NTT representation _f̂_
    /// in Tq.
    ///
    /// Implements [Algorithm 8, `NTT(f)`], from [FIPS 203].
    ///
    /// [Algorithm 8, `NTT(f)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf#algorithm.8
    pub fn ntt(self) -> TqElement {
        let mut f_hat = self.0.clone();
        let mut k = 1;

        // ln 3: `for (len <- 128; len >= 2; len <- len/2)`
        for len in NTT_SERIES {
            // ln 4: `for start <- 0; start < 256; start <- start + 2·len)`
            for start in (0..256).step_by(2 * len) {
                // ln 5: `zeta <- ζ ^BitRev₇(k) mod q`
                let zeta = ZETA.pow(bit_rev_7(k).into());
                // ln 6: `k = k + 1`
                k += 1;
                // ln 7: `for (j <- start; j < start + len; j++)`
                for j in 0..(start + len) {
                    // ln 8: `t <- zeta · f̂[j+len]`
                    let t = zeta * f_hat[j + len];
                    // ln 9: `f̂[j+len] <- f̂[j] - t`
                    f_hat[j + len] = f_hat[j] - t;
                    // ln 10: `f̂[j] <- f̂[j] + t`
                    f_hat[j] = f_hat[j] + t;
                }
            }
        }

        return TqElement::new(f_hat);
    }
}

impl Index<usize> for RqElement {
    type Output = FieldElement;

    fn index(&self, index: usize) -> &FieldElement {
        &self.0[index]
    }
}

impl PolynomialRingElement for RqElement {
    /// Default value is a polynomial where all coefficients are zero.
    const ZERO: Self = Self([FieldElement::ZERO; parameters::N]);

    /// Constructor for a new polynomial element instance from values.
    fn new(coefficients: [FieldElement; parameters::N]) -> Self {
        Self(coefficients)
    }

    /// Get the polynomial coefficients.
    fn coefficients(&self) -> [FieldElement; parameters::N] {
        self.0
    }
}

/// Elements in the polynomial ring Tq over ℤq.
///
/// This is the NTT representation of values in the Tq polynomial ring for some
/// computations.
///
/// NTT: number-theoretic transform, and its representation.
///
/// From section 4.3 of [FIPS 203]:
///
/// "The number-theoretic transform (or NTT) can be viewed as a specialized,
/// exact version of the discrete Fourier transform. In the case of ML-KEM, the
/// NTT is used to improve the effciency of multiplication in the ring Rq.
/// Recall that Rq is the ring Zq[X]/(Xn +1) consisting of polynomials of the
/// form f = f0 + f1X + ··· + f255X^255 where fj ∈ Zq for all j, equipped with
/// arithmetic modulo Xn +1.
///
/// The ring Rq is naturally isomorphic to another ring, denoted Tq, which is a
/// direct sum of 128 quadratic extensions of Zq. The NTT is a computationally
/// effcient isomorphism between these two rings. On input a polynomial f ∈ Rq,
/// the NTT outputs an element f̂ := NTT(f) of the ring Tq, where f̂ is called the
/// “NTT representation” of f. The isomorphism property implies that f ×Rq
/// g = NTT−1(f̂×Tq gˆ), (4.8) where ×Rq and ×Tq denote multiplication in Rq and
/// Tq, respectively. Moreover, since Tq is a product of 128 rings, each
/// consisting of degree-one polynomials, the operation ×Tq is much more
/// effcient than the operation ×Rq . For these reasons, the NTT is considered
/// to be an integral part of ML-KEM and not merely an optimization.
///
/// As the rings Rq and Tq have a vector space structure over Zq, the most
/// natural abstract data type to represent elements from either of these rings
/// is Zn q. For this reason, the choice of data structure for the inputs and
/// outputs of NTT and NTT−1 are length-n arrays of integers modulo q; these
/// arrays are understood to represent elements of Tq or Rq, respectively (see
/// Section 2.4). Both NTT and NTT−1 can be computed in-place. In fact,
/// Algorithms 8 and 9 demonstrate an effcient means of computing NTT and NTT−1
/// in-place. However, for clarity in understanding the distinction of the
/// algebraic objects before and after the conversion, the algorithms are
/// written with explicit inputs and outputs."
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
// TODO: is it safe to derive `Eq` and `PartialEq` like so?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TqElement([FieldElement; parameters::N]);

impl TqElement {
    /// Transform a polynomial element f̂ in Tq from its NTT representation into
    /// f in Rq.
    ///
    /// Implements [Algorithm 9, `NTT⁻¹(f̂)`], from [FIPS 203].
    ///
    /// [Algorithm 9, `NTT⁻¹(f̂)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf#algorithm.9
    pub fn ntt_inverse(self) -> RqElement {
        let mut f = self.0.clone();
        let mut k: u8 = 127;

        // ln 3: `for (len <- 2; len <= 128; len <- 2·len)`
        for len in NTT_SERIES.iter().rev() {
            // ln 4: `for start <- 0; start < 256; start <- start + 2·len)`
            for start in (0..256).step_by(2 * len) {
                // ln 5: `zeta <- ζ ^BitRev₇(k) mod q`
                let zeta = ZETA.pow(bit_rev_7(k).into());
                // ln 6: `k = k - 1`
                k -= 1;
                // ln 7: `for (j <- start; j < start + len; j++)`
                for j in 0..(start + len) {
                    // ln 8: `t <- f[j]`
                    let t = f[j];
                    // ln 9: `f[j] <- t + f[j+len]`
                    f[j] = t + f[j + len];
                    // ln 10: `f[j+len] <- zeta · (f[j+len] - t)`
                    f[j + len] = zeta * (f[j + len] - t);
                }
            }
        }

        const INVERSE_128: FieldElement = FieldElement(3303);

        // ln 14: `f <- f · 3033 mod q`
        for i in 0..parameters::N {
            f[i] = f[i] * INVERSE_128;
        }

        // ln 15: `return f`
        return RqElement::new(f);
    }
}

impl AddAssign for TqElement {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl Index<usize> for TqElement {
    type Output = FieldElement;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl PolynomialRingElement for TqElement {
    /// Default value is a polynomial where all coefficients are zero.
    const ZERO: Self = Self([FieldElement::ZERO; parameters::N]);

    /// Constructor for a new polynomial element instance from values.
    fn new(coefficients: [FieldElement; parameters::N]) -> Self {
        Self(coefficients)
    }

    /// Get the polynomial coefficients.
    fn coefficients(&self) -> [FieldElement; parameters::N] {
        self.0
    }
}

impl Mul for TqElement {
    type Output = Self;

    /// Compute the product of two NTT representations (polynomials in Tq).
    ///
    /// Implements [Algorithm 10], `MultiplyNTTs(f̂, ĝ)`, from FIPS 203.
    ///
    /// [Algorithm 10]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf#algorithm.10
    fn mul(self, rhs: Self) -> Self::Output {
        /// Compute the product of two degree-one polynomials with respect to a
        /// quadratic modulus.
        ///
        /// Implements [Algorithm 11], `BaseCaseMultiply(a₀, a₁, b₀, b₁, γ)`,
        /// from FIPS 203.
        ///
        /// [Algorithm 11]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf#algorithm.11
        fn base_case_multiply(
            a0: FieldElement,
            a1: FieldElement,
            b0: FieldElement,
            b1: FieldElement,
            γ: FieldElement,
        ) -> (FieldElement, FieldElement) {
            let c0 = (a0 * b0) + (a1 * b1 * γ);
            let c1 = (a0 * b1) + (a1 * b0);

            return (c0, c1);
        }

        let (f_hat, g_hat) = (self, rhs);

        let mut h_hat = TqElement::ZERO;

        for i in 0..128 {
            let two_i = 2 * i;
            let two_i_plus_1 = two_i + 1;

            (h_hat.0[two_i], h_hat.0[two_i_plus_1]) = base_case_multiply(
                f_hat.0[two_i],
                f_hat.0[two_i_plus_1],
                g_hat.0[two_i],
                g_hat.0[two_i_plus_1],
                // Casting `i` into `u8` is safe as it will never be larger than a 'u7'.
                ZETA.pow((2 * bit_rev_7(i as u8) + 1).into()),
            );
        }

        return h_hat;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vectors of polynomial ring elements
////////////////////////////////////////////////////////////////////////////////

// TODO: when associated const generic expressions are stable, pluck `K` from
// the `ParameterSet` instances.
//
// The rank / modular dimension(s) K of the vectors and matrices over Rq and Tq
// are implemented as const generic parameter statements that must be passed in
// for every instance.  This is not the worst ever, but it would be really great
// and cleaner to instaniate by plucking the value of K out of `ParameterSet` at
// compile time instead of having to define it multiple times.
//
// Since we don't need `ParameterSet` directly now, `Vector` and `Matrix` are
// not defined generically over it.

/// Vector of polynomial ring elements in Rq of length K.
pub struct RqVector<const K: usize>([RqElement; K]);

impl<const K: usize> RqVector<K> {
    const ZERO: Self = Self([RqElement::ZERO; K]);
}

impl<const K: usize> Add for RqVector<K> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut result = Self::ZERO;

        for i in 0..K {
            // Casting `i` as `u8` is safe as `i` will never be larger than `K`, which is at
            // most 8 for settings such as ML-DSA-87.
            result[i] = self.0[i] + rhs.0[i];
        }

        return result;
    }
}

impl<const K: usize> Index<usize> for RqVector<K> {
    type Output = RqElement;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const K: usize> IndexMut<usize> for RqVector<K> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const K: usize> IntoIterator for RqVector<K> {
    type Item = RqElement;

    type IntoIter = std::array::IntoIter<Self::Item, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Vector of polynomial ring elements in Tq of length K.
pub struct TqVector<const K: usize>([TqElement; K]);

impl<const K: usize> TqVector<K> {
    const ZERO: Self = Self([TqElement::ZERO; K]);
}

impl<const K: usize> Add for TqVector<K> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut result = Self::ZERO;

        for i in 0..K {
            result.0[i] = self.0[i] + rhs.0[i];
        }

        return result;
    }
}

impl<const K: usize> Index<usize> for TqVector<K> {
    type Output = TqElement;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const K: usize> IndexMut<usize> for TqVector<K> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const K: usize> IntoIterator for TqVector<K> {
    type Item = TqElement;

    type IntoIter = std::array::IntoIter<Self::Item, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// Implements vᵀ ∘ vᵀ, the dot product of two vectors over Tq, resulting in a
// `TqElement`.
impl<const K: usize> Mul for TqVector<K> {
    type Output = TqElement;

    /// Computes a dot product; the result is in the base ring Tq and is
    /// represented by a polynomial ring element.
    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = TqElement::ZERO;

        for j in 0..K {
            // Each of these is an invocation of `MultiplyNTTs()`.
            result += self.0[j] * rhs.0[j];
        }

        return result;
    }
}

// Implements f̂ ∘ vᵀ, the scalar multiplication of a polynomial ring element by
// a vector of polynomial ring elements over Tq, resulting in a new vector of
// `TqElement`s.
impl<const K: usize> Mul<TqVector<K>> for TqElement {
    type Output = TqVector<K>;

    /// Scalar multiplication of a vector by a polynomial ring element in the
    /// NTT domain Tq, resulting in a vector.
    fn mul(self, rhs: TqVector<K>) -> Self::Output {
        let mut result = TqVector::ZERO;

        for i in 0..K {
            // Each of these is an invocation of `MultiplyNTTs()`.
            result[i] = self * rhs[i];
        }

        return result;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Matrices of polynomial ring elements
////////////////////////////////////////////////////////////////////////////////

/// A k x k matrix of polynomial ring elements in Tq.
///
/// Despite the description in [FIPS 203] that one can view 'vectors as the
/// special case of matrices with only one column', we are representing matrices
/// as a set of row vectors.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
pub struct TqMatrix<const K: usize>([TqVector<K>; K]);

impl<const K: usize> TqMatrix<K> {
    const ZERO: Self = Self([TqVector::<K>::ZERO; K]);

    /// Return the transpose of this matrix.
    pub fn transpose(self) -> Self {
        let mut transpose = Self::ZERO;

        for (i, row) in self.0.iter().enumerate() {
            for (j, element) in row.0.iter().enumerate() {
                transpose.0[j].0[i] = *element;
            }
        }

        return transpose;
    }
}

/// Implements matrix multiplication by a column vector, with all matrix and
/// vector elements in Tq.
///
/// Corresponds to equation 2.12 in section 2.4.7 of [FIPS 203].
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
impl<const K: usize> Mul<TqVector<K>> for TqMatrix<K> {
    type Output = TqVector<K>;

    fn mul(self, rhs: TqVector<K>) -> Self::Output {
        let mut result = TqVector::<K>::ZERO;

        for i in 0..K {
            for j in 0..K {
                result[i] += self.0[i][j] * rhs[j];
            }
        }

        return result;
    }
}
