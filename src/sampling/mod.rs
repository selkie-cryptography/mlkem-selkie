//! Sampling of ring elements from pseudorandom byte streams for [FIPS 203].
//!
//! Extends the ring types of [`crate::algebraic`] with two samplers:
//!
//! - [`TqElement::sample_ntt`] (Algorithm 7) rejection-samples a uniform
//!   NTT-domain polynomial from a SHAKE128 [`XofReader`], used to build the
//!   public matrix `A` from the seed `rho`.
//! - [`RqElement::sample_cbd`] (Algorithm 8) samples a polynomial from the
//!   centered binomial distribution `D_eta`, used for the secret and noise
//!   vectors.
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf

use sha3::digest::XofReader;

use crate::{
    algebraic::{FieldElement, PolynomialRingElement, RqElement, TqElement},
    parameters::{self, Q},
};

#[cfg(test)]
mod tests;

impl TqElement {
    /// `SampleNTT`: rejection-samples a uniformly random element of Tq from a
    /// SHAKE128 output stream.
    ///
    /// Reads three bytes at a time, interpreting them as two 12-bit candidates,
    /// and keeps each candidate that is less than q until 256 coefficients have
    /// been collected.
    ///
    /// Implements [Algorithm 7] of FIPS 203.
    ///
    /// [Algorithm 7]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.7
    // reason: `count < N` guards both writes, so every index is in bounds.
    #[allow(clippy::indexing_slicing)]
    pub fn sample_ntt(reader: &mut impl XofReader) -> Self {
        let mut coefficients = [FieldElement::ZERO; parameters::N];
        let mut count = 0;
        let mut triple = [0u8; 3];

        while count < parameters::N {
            reader.read(&mut triple);
            let [b0, b1, b2] = triple;

            let d1 = u16::from(b0) | (u16::from(b1 & 0x0F) << 8);
            let d2 = (u16::from(b1) >> 4) | (u16::from(b2) << 4);

            if d1 < Q {
                coefficients[count] = FieldElement::new(d1);
                count += 1;
            }
            if d2 < Q && count < parameters::N {
                coefficients[count] = FieldElement::new(d2);
                count += 1;
            }
        }

        Self::new(coefficients)
    }
}

impl RqElement {
    /// `SamplePolyCBD_eta`: samples a polynomial in Rq from the centered
    /// binomial distribution `D_eta`.
    ///
    /// Consumes `64 * eta` bytes as a bit stream; coefficient `i` is the
    /// difference of two sums of `eta` consecutive bits, giving a value in
    /// `{-eta, ..., eta}` reduced modulo q.
    ///
    /// Implements [Algorithm 8] of FIPS 203.
    ///
    /// [Algorithm 8]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.8
    pub fn sample_cbd(eta: usize, bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 64 * eta);

        let mut bits = bytes
            .iter()
            .flat_map(|byte| (0..8).map(move |k| (byte >> k) & 1));

        let coefficients = core::array::from_fn(|_| {
            let x: u16 = bits.by_ref().take(eta).map(u16::from).sum();
            let y: u16 = bits.by_ref().take(eta).map(u16::from).sum();

            FieldElement::from(x) - FieldElement::from(y)
        });

        Self::new(coefficients)
    }
}
