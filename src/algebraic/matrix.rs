//! `K`-by-`K` matrices of polynomial ring elements in Tq.

use core::{
    marker::PhantomData,
    ops::{Index, Mul},
};

use crate::{algebraic::vector::TqVector, parameters::ParameterSet};

#[cfg(test)]
mod tests;

/// A `K`-by-`K` matrix of polynomial ring elements in Tq.
///
/// Despite the description in FIPS 203 that one can view vectors as the special
/// case of matrices with a single column, we store the matrix as a sequence of
/// `P::K` row vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TqMatrix<P: ParameterSet> {
    /// The `P::K` rows, each a `TqVector<P>` of length `P::K`.
    rows: Vec<TqVector<P>>,
    /// Binds the matrix dimension to a parameter set without storing it.
    _marker: PhantomData<P>,
}

impl<P: ParameterSet> TqMatrix<P> {
    /// Constructs a matrix from its row vectors.
    ///
    /// # Panics
    ///
    /// Debug-asserts that there are exactly `P::K` rows.
    pub fn from_rows(rows: Vec<TqVector<P>>) -> Self {
        debug_assert_eq!(rows.len(), P::K);

        Self {
            rows,
            _marker: PhantomData,
        }
    }

    /// Returns the transpose of this matrix.
    ///
    /// `K-PKE.Encrypt` multiplies by `A^T` while reusing the same `A` that
    /// `K-PKE.KeyGen` built.
    pub fn transpose(&self) -> Self {
        let rows = (0..P::K)
            .map(|j| {
                // `row[j]` uses `TqVector`'s `Index`; iterating `self.rows`
                // avoids indexing the backing `Vec`.
                let column = self.rows.iter().map(|row| row[j]).collect();
                TqVector::from_vec(column)
            })
            .collect();

        Self::from_rows(rows)
    }
}

impl<P: ParameterSet> Index<usize> for TqMatrix<P> {
    type Output = TqVector<P>;

    // reason: the `Index` contract is to index and panic on out-of-bounds.
    #[allow(clippy::indexing_slicing)]
    fn index(&self, index: usize) -> &TqVector<P> {
        &self.rows[index]
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
        let polys = self.rows.iter().map(|row| row * rhs).collect();

        TqVector::from_vec(polys)
    }
}
