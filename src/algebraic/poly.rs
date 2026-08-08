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

use core::{
    array,
    ops::{Add, AddAssign, Deref, DerefMut, Index, Mul, Sub, SubAssign},
};

use zeroize::{DefaultIsZeroes, Zeroize};

use self::arch::ProductAccumulator;
use crate::{algebraic::field::FieldElement, parameters};

mod arch;

#[cfg(test)]
mod tests;

/// A polynomial of the ring Rq or Tq: 256 coefficients in Zq.
///
/// Both rings have n = 256 coefficients over Zq, so the standard-domain
/// [`RqElement`] and the NTT-domain [`TqElement`] share this trait for their
/// shared structure while remaining distinct types.
pub trait PolynomialRingElement: Clone + Index<usize, Output = FieldElement> {
    /// The polynomial all of whose coefficients are zero.
    const ZERO: Self;

    /// Constructs a polynomial from its coefficients.
    fn new(coefficients: [FieldElement; parameters::N]) -> Self;

    /// Returns this polynomial's coefficients.
    // reason: the public accessor for `expose-internals` consumers and tests;
    // in-crate serialization reads coefficients through the arch kernels
    // instead, so the default build has no caller.
    #[allow(dead_code)]
    fn coefficients(&self) -> [FieldElement; parameters::N];
}

/// The private guts of a ring element or its multiplication cache: a
/// `FieldElement` array split off so `Copy` and zeroize speed stop
/// conflicting.
///
/// One-shot volatile zeroizing needs `DefaultIsZeroes`, which needs `Copy`
/// (the per-coefficient fallback is much slower especially in decaps) — but
/// `Copy` on the wrappers would let these large values duplicate silently.
/// `Copy` here and not there gives both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Inner<const LEN: usize = { parameters::N }>([FieldElement; LEN]);

impl<const LEN: usize> Default for Inner<LEN> {
    fn default() -> Self {
        Self([FieldElement::ZERO; LEN])
    }
}

impl<const LEN: usize> DefaultIsZeroes for Inner<LEN> {}

impl<const LEN: usize> Deref for Inner<LEN> {
    type Target = [FieldElement; LEN];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const LEN: usize> DerefMut for Inner<LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Elements of the polynomial ring Rq over Zq.
///
/// This is the standard domain of ML-KEM values, as opposed to the NTT
/// representation [`TqElement`]. Deliberately not `Copy`: an implicit
/// 512-byte copy is a compile error, so element traffic is explicit. No
/// `ZeroizeOnDrop`; secret bare-element locals live inside a `ZeroizeOnDrop`
/// `RqVector`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RqElement(Inner);

impl Default for RqElement {
    fn default() -> Self {
        Self(Inner([FieldElement::ZERO; parameters::N]))
    }
}

// All-zeros is a valid `RqElement`, so it can be scrubbed as one bulk write.
impl Zeroize for RqElement {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl RqElement {
    /// Transforms a polynomial `f` in Rq into its NTT representation `f_hat` in
    /// Tq, then Barrett-reduces the result so the coefficients stay small for
    /// base multiplication.
    ///
    /// Implements [Algorithm 9, `NTT(f)`] from FIPS 203.
    ///
    /// [Algorithm 9, `NTT(f)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.9
    pub fn ntt(&self) -> TqElement {
        let mut coefficients = self.0;
        arch::ntt(&mut coefficients);

        TqElement::new(*coefficients)
    }

    /// `Compress_d` of every coefficient, via the architecture backend.
    ///
    /// Constant-time on secret-derived inputs (the `K-PKE` ciphertext
    /// components and recovered message), as `FieldElement::compress` is.
    #[must_use]
    pub(crate) fn compressed(&self, d: usize) -> [u16; parameters::N] {
        arch::compress(&self.0, d)
    }

    /// Constructs a polynomial by `Decompress_d` of every `d`-bit value, via
    /// the architecture backend.
    pub(crate) fn decompress(values: &[u16; parameters::N], d: usize) -> Self {
        Self(Inner(arch::decompress(values, d)))
    }
}

impl PolynomialRingElement for RqElement {
    const ZERO: Self = Self(Inner([FieldElement::ZERO; parameters::N]));

    fn new(coefficients: [FieldElement; parameters::N]) -> Self {
        Self(Inner(coefficients))
    }

    fn coefficients(&self) -> [FieldElement; parameters::N] {
        *self.0
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

    fn add(mut self, rhs: Self) -> Self {
        self += &rhs;

        self
    }
}

impl AddAssign<&RqElement> for RqElement {
    fn add_assign(&mut self, rhs: &Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a = *a + *b;
        }
    }
}

impl Sub for RqElement {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self -= &rhs;

        self
    }
}

impl SubAssign<&RqElement> for RqElement {
    fn sub_assign(&mut self, rhs: &Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a = *a - *b;
        }
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
/// For performance, coefficients are stored evens-then-odds — the 128
/// even-position (degree-0) coefficients, then the 128 odd-position
/// (degree-1) — so base multiplication loads each pair half as a contiguous
/// run instead of shuffling every pair apart per multiplication: the
/// `nttpack` technique of the [Kyber AVX2 reference implementation]. The
/// layout is internal; every public surface speaks natural coefficient
/// order, converting at the boundaries.
///
/// [section 4.3]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.3
/// [Kyber AVX2 reference implementation]: https://github.com/pq-crystals/kyber/tree/main/avx2
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TqElement(Inner);

impl Default for TqElement {
    fn default() -> Self {
        Self(Inner([FieldElement::ZERO; parameters::N]))
    }
}

// As [`RqElement`].
impl Zeroize for TqElement {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

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
    /// Reduces lazily (len-2 and len-16 stages only); inputs must satisfy
    /// `|x| ≤ 16383` so first-stage sums fit `i16` — comfortably above what
    /// accumulated basemul products reach.
    ///
    /// [Algorithm 10, `NTT⁻¹(f_hat)`]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.10
    pub fn ntt_inverse(&self) -> RqElement {
        let mut coefficients = arch::unpack(&self.0);
        arch::ntt_inverse(&mut coefficients);

        RqElement::new(coefficients)
    }

    /// Scales every coefficient by `R`, undoing the `R^-1` left by base
    /// multiplication so an NTT-domain product can be added to true NTT values
    /// (`K-PKE.KeyGen`'s `t_hat = A . s_hat + e_hat`).
    pub fn to_montgomery(&self) -> Self {
        Self(Inner(self.0.map(FieldElement::to_montgomery)))
    }

    /// Precomputes this polynomial's asymmetric base-multiplication cache.
    #[must_use]
    pub fn mul_cache(&self) -> TqMulCache {
        // reason: indices 128 + i and i are provably in bounds for i in
        // 0..128; the second (odd) half holds the degree-1 coefficients.
        #[allow(clippy::indexing_slicing)]
        TqMulCache(Inner(array::from_fn(|i| {
            self.0[parameters::N / 2 + i] * arch::GAMMA_MONT[i]
        })))
    }

    /// Computes the accumulated dot product `sum_j f_j * g_j` of two
    /// component slices, where `caches` holds each `g_j`'s [`TqMulCache`].
    ///
    /// The raw `i32` coefficient products of all components are summed first
    /// and Montgomery-reduced once per coefficient, matching the sum of
    /// per-component [`Mul`] products mod q with `4 * len - 1` fewer
    /// reductions per coefficient pair. The result is scaled by `R^-1`
    /// (Montgomery convention), as [`Mul`]'s is.
    ///
    /// The accumulator bound in `FieldElement::from_product_sum` admits at
    /// most four components with coefficients bounded by `3q/2`; every caller
    /// dots vectors of `K <= 4` NTT-domain polynomials within that bound.
    pub(crate) fn accumulated_dot(f: &[Self], g: &[Self], caches: &[TqMulCache]) -> Self {
        let mut acc = ProductAccumulator::default();

        for ((f_j, g_j), cache) in f.iter().zip(g).zip(caches) {
            acc.accumulate(&f_j.0, &g_j.0, &cache.0);
        }

        Self(Inner(acc.reduce()))
    }

    /// The canonical `[0, q)` representative of every coefficient, via the
    /// architecture backend (the `ByteEncode_12` serialization values).
    #[must_use]
    pub(crate) fn canonical(&self) -> [u16; parameters::N] {
        arch::canonical(&self.0)
    }
}

impl PolynomialRingElement for TqElement {
    const ZERO: Self = Self(Inner([FieldElement::ZERO; parameters::N]));

    // Splits the natural-order input into the evens-then-odds storage.
    fn new(coefficients: [FieldElement; parameters::N]) -> Self {
        Self(Inner(arch::pack(&coefficients)))
    }

    // Re-interleaves the storage back to natural coefficient order.
    fn coefficients(&self) -> [FieldElement; parameters::N] {
        arch::unpack(&self.0)
    }
}

impl Index<usize> for TqElement {
    type Output = FieldElement;

    /// Indexes by natural coefficient position, mapped onto the split
    /// storage.
    // reason: see `RqElement`'s `Index` impl — indexing is the trait's
    // contract; the mapped index stays below `N`.
    #[allow(clippy::indexing_slicing)]
    fn index(&self, index: usize) -> &FieldElement {
        &self.0[(index % 2) * (parameters::N / 2) + index / 2]
    }
}

impl Add for TqElement {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += &rhs;

        self
    }
}

impl AddAssign<&TqElement> for TqElement {
    fn add_assign(&mut self, rhs: &Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a = *a + *b;
        }
    }
}

impl AddAssign for TqElement {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
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
        &self * &rhs
    }
}

impl Mul for &TqElement {
    type Output = TqElement;

    /// The borrowing form of `TqElement`'s `Mul`, for operands that stay in
    /// place.
    fn mul(self, rhs: &TqElement) -> TqElement {
        TqElement(Inner(arch::multiply(&self.0, &rhs.0)))
    }
}

/// The precomputed asymmetric base-multiplication cache of a [`TqElement`]
/// `g`: the 128 products `gamma_i * g_(2i+1)`.
///
/// The degree-0 half of a base-case product is `f_e * g_e + f_o * (gamma *
/// g_o)`; the parenthesized factor depends only on `g`, so a polynomial reused
/// across the components of a matrix-vector or dot product pays its 128
/// Montgomery multiplications once instead of once per product.
/// `crate::algebraic::vector::CachedTqVector` bundles caches with their
/// polynomials so the two cannot be cross-wired.
// Deliberately not `Copy`, as the ring elements: an implicit 256-byte copy
// is a compile error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TqMulCache(Inner<{ parameters::N / 2 }>);

impl Default for TqMulCache {
    fn default() -> Self {
        Self(Inner([FieldElement::ZERO; parameters::N / 2]))
    }
}

// Secret-derived caches (of `s_hat`, of encryption randomness `y_hat`) scrub
// as one bulk write, as the polynomial types do.
impl Zeroize for TqMulCache {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}
