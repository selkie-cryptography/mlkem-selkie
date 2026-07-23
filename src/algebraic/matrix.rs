//! `K`-by-`K` matrices of polynomial ring elements in Tq.

use core::ops::{Index, Mul};

use crate::{
    algebraic::vector::{CachedTqVector, TqVector},
    parameters::ParameterSet,
};

#[cfg(test)]
mod tests;

/// A `K`-by-`K` matrix of polynomial ring elements in Tq.
///
/// Despite the description in FIPS 203 that one can view vectors as the special
/// case of matrices with a single column, we store the matrix as a sequence of
/// `P::K` row vectors, backed by the parameter set's stack-allocated
/// [`ParameterSet::KArray`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TqMatrix<P: ParameterSet> {
    /// The `P::K` rows, each a `TqVector<P>` of length `P::K`.
    rows: P::KArray<TqVector<P>>,
}

impl<P: ParameterSet> TqMatrix<P> {
    /// Constructs a matrix by applying `f` to each row index in `0..K`.
    pub fn from_fn(f: impl FnMut(usize) -> TqVector<P>) -> Self {
        Self {
            rows: P::k_array_from_fn(f),
        }
    }

    /// Returns the transpose of this matrix.
    ///
    /// `K-PKE.Encrypt` multiplies by `A^T` while reusing the same `A` that
    /// `K-PKE.KeyGen` built.
    pub fn transpose(&self) -> Self {
        Self::from_fn(|j| TqVector::from_fn(|i| self[i][j]))
    }
}

impl<P: ParameterSet> Index<usize> for TqMatrix<P> {
    type Output = TqVector<P>;

    // reason: the `Index` contract is to index and panic on out-of-bounds.
    #[allow(clippy::indexing_slicing)]
    fn index(&self, index: usize) -> &TqVector<P> {
        &self.rows.as_ref()[index]
    }
}

impl<P: ParameterSet> Mul<&TqVector<P>> for &TqMatrix<P> {
    type Output = TqVector<P>;

    /// Multiplies this matrix by a column vector, all entries in Tq.
    ///
    /// Corresponds to equation 2.12 in [section 2.4.7] of FIPS 203.
    ///
    /// [section 2.4.7]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsubsection.2.4.7
    fn mul(self, rhs: &TqVector<P>) -> TqVector<P> {
        TqVector::from_fn(|i| &self[i] * rhs)
    }
}

impl<P: ParameterSet> Mul<&CachedTqVector<P>> for &TqMatrix<P> {
    type Output = TqVector<P>;

    /// Multiplies this matrix by a cached column vector: each row's dot
    /// product reuses the column's per-component caches by accumulated base
    /// multiplication.
    fn mul(self, rhs: &CachedTqVector<P>) -> TqVector<P> {
        TqVector::from_fn(|i| &self[i] * rhs)
    }
}
