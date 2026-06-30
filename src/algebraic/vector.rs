//! Vectors of polynomial ring elements, of length `P::K`.
//!
//! Backed by the parameter set's [`ParameterSet::KArray`] (`[_; K]`), so a
//! vector lives entirely on the stack with no heap allocation.

use core::ops::{Add, Index, Mul};

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{
    algebraic::poly::{PolynomialRingElement, RqElement, TqElement},
    parameters::ParameterSet,
};

/// A vector of `K` polynomial ring elements in Rq.
///
/// `ZeroizeOnDrop` covers the CBD-sampled secrets `s`/`e`/`y`/`e_1` and any
/// transient vector that goes out of scope.
#[derive(Clone, Debug, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct RqVector<P: ParameterSet> {
    /// The `P::K` component polynomials.
    polys: P::KArray<RqElement>,
}

impl<P: ParameterSet> RqVector<P> {
    /// Constructs a vector by applying `f` to each component index in `0..K`.
    pub fn from_fn(f: impl FnMut(usize) -> RqElement) -> Self {
        Self {
            polys: P::k_array_from_fn(f),
        }
    }

    /// Returns the component polynomials.
    pub fn as_slice(&self) -> &[RqElement] {
        self.polys.as_ref()
    }

    /// Applies the NTT to every component, mapping this vector into Tq.
    pub fn ntt(&self) -> TqVector<P> {
        TqVector::from_fn(|i| self[i].ntt())
    }
}

impl<P: ParameterSet> Index<usize> for RqVector<P> {
    type Output = RqElement;

    // reason: the `Index` contract is to index and panic on out-of-bounds.
    #[allow(clippy::indexing_slicing)]
    fn index(&self, index: usize) -> &RqElement {
        &self.polys.as_ref()[index]
    }
}

impl<P: ParameterSet> Add for RqVector<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_fn(|i| self[i] + rhs[i])
    }
}

/// A vector of `K` polynomial ring elements in Tq (NTT representation).
///
/// `ZeroizeOnDrop` for the same reason as `RqVector`.
#[derive(Clone, Debug, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct TqVector<P: ParameterSet> {
    /// The `P::K` component polynomials in NTT form.
    polys: P::KArray<TqElement>,
}

impl<P: ParameterSet> TqVector<P> {
    /// Constructs a vector by applying `f` to each component index in `0..K`.
    pub fn from_fn(f: impl FnMut(usize) -> TqElement) -> Self {
        Self {
            polys: P::k_array_from_fn(f),
        }
    }

    /// Returns the component polynomials.
    pub fn as_slice(&self) -> &[TqElement] {
        self.polys.as_ref()
    }

    /// Applies the inverse NTT to every component, mapping this vector into Rq.
    pub fn ntt_inverse(&self) -> RqVector<P> {
        RqVector::from_fn(|i| self[i].ntt_inverse())
    }

    /// Scales every component by `R` (Montgomery), undoing the `R^-1` left by a
    /// matrix-vector base multiplication so the result can be added to true NTT
    /// values.
    pub fn to_montgomery(&self) -> Self {
        Self::from_fn(|i| self[i].to_montgomery())
    }
}

impl<P: ParameterSet> Add for TqVector<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_fn(|i| self[i] + rhs[i])
    }
}

impl<P: ParameterSet> Mul for &TqVector<P> {
    type Output = TqElement;

    /// Computes the dot product `self^T . rhs`, accumulating `P::K` pointwise
    /// NTT products into a single `TqElement`.
    fn mul(self, rhs: &TqVector<P>) -> TqElement {
        let mut result = TqElement::ZERO;

        for (a, b) in self.polys.as_ref().iter().zip(rhs.polys.as_ref()) {
            result += *a * *b;
        }

        result
    }
}

impl<P: ParameterSet> Index<usize> for TqVector<P> {
    type Output = TqElement;

    // reason: the `Index` contract is to index and panic on out-of-bounds.
    #[allow(clippy::indexing_slicing)]
    fn index(&self, index: usize) -> &TqElement {
        &self.polys.as_ref()[index]
    }
}
