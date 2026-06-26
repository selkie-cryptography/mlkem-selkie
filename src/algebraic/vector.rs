//! Vectors of polynomial ring elements, of length `P::K`.
//!
//! Stored on the heap pending stabilization of `generic_const_exprs`, which
//! would let the length be a const-generic array pulled from the
//! [`ParameterSet`].

use core::{
    marker::PhantomData,
    ops::{Add, Index, Mul},
};

use crate::{
    algebraic::poly::{PolynomialRingElement, RqElement, TqElement},
    parameters::ParameterSet,
};

/// A vector of `K` polynomial ring elements in Rq.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RqVector<P: ParameterSet> {
    /// The `P::K` component polynomials.
    polys: Vec<RqElement>,
    /// Binds the vector length to a parameter set without storing it.
    _marker: PhantomData<P>,
}

impl<P: ParameterSet> RqVector<P> {
    /// Constructs a vector from its component polynomials.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `polys.len() == P::K`.
    pub fn from_vec(polys: Vec<RqElement>) -> Self {
        debug_assert_eq!(polys.len(), P::K);

        Self {
            polys,
            _marker: PhantomData,
        }
    }

    /// Returns the component polynomials.
    pub fn as_slice(&self) -> &[RqElement] {
        &self.polys
    }

    /// Applies the NTT to every component, mapping this vector into Tq.
    pub fn ntt(&self) -> TqVector<P> {
        TqVector::from_vec(self.polys.iter().map(|f| f.ntt()).collect())
    }
}

impl<P: ParameterSet> Add for RqVector<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let polys = self
            .polys
            .into_iter()
            .zip(rhs.polys)
            .map(|(a, b)| a + b)
            .collect();

        Self::from_vec(polys)
    }
}

/// A vector of `K` polynomial ring elements in Tq (NTT representation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TqVector<P: ParameterSet> {
    /// The `P::K` component polynomials in NTT form.
    polys: Vec<TqElement>,
    /// Binds the vector length to a parameter set without storing it.
    _marker: PhantomData<P>,
}

impl<P: ParameterSet> TqVector<P> {
    /// Constructs a vector from its component polynomials.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `polys.len() == P::K`.
    pub fn from_vec(polys: Vec<TqElement>) -> Self {
        debug_assert_eq!(polys.len(), P::K);

        Self {
            polys,
            _marker: PhantomData,
        }
    }

    /// Returns the component polynomials.
    pub fn as_slice(&self) -> &[TqElement] {
        &self.polys
    }

    /// Applies the inverse NTT to every component, mapping this vector into Rq.
    pub fn ntt_inverse(&self) -> RqVector<P> {
        RqVector::from_vec(self.polys.iter().map(|f| f.ntt_inverse()).collect())
    }

    /// Scales every component by `R` (Montgomery), undoing the `R^-1` left by a
    /// matrix-vector base multiplication so the result can be added to true NTT
    /// values.
    pub fn to_montgomery(&self) -> Self {
        Self::from_vec(self.polys.iter().map(|f| f.to_montgomery()).collect())
    }
}

impl<P: ParameterSet> Add for TqVector<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let polys = self
            .polys
            .into_iter()
            .zip(rhs.polys)
            .map(|(a, b)| a + b)
            .collect();

        Self::from_vec(polys)
    }
}

impl<P: ParameterSet> Mul for &TqVector<P> {
    type Output = TqElement;

    /// Computes the dot product `self^T . rhs`, accumulating `P::K` pointwise
    /// NTT products into a single `TqElement`.
    fn mul(self, rhs: &TqVector<P>) -> TqElement {
        let mut result = TqElement::ZERO;

        for (a, b) in self.polys.iter().zip(&rhs.polys) {
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
        &self.polys[index]
    }
}
