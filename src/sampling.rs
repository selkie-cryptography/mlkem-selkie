//! Sampling of ring elements from pseudorandom byte streams for [FIPS 203].
//!
//! Extends the ring types of [`crate::algebraic`] with two samplers:
//!
//! - [`TqElement::sample_ntt`] (Algorithm 7) rejection-samples a uniform
//!   NTT-domain polynomial from a SHAKE128 XOF state;
//!   [`TqElement::sample_ntt_x4`] is the batched form that builds the public
//!   matrix `A` from `rho` four streams at a time.
//! - [`RqElement::sample_cbd`] (Algorithm 8) samples a polynomial from the
//!   centered binomial distribution `D_eta`, used for the secret and noise
//!   vectors.
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf

use crate::{
    algebraic::{FieldElement, PolynomialRingElement, RqElement, TqElement},
    functions::{Shake128X4, XOF},
    parameters::{self, Eta, Q},
};

#[cfg(test)]
mod tests;

impl TqElement {
    /// `SampleNTT`: rejection-samples a uniformly random element of Tq from
    /// `XOF(rho, j, i)` per Algorithm 13/14 of FIPS 203.
    ///
    /// Reads three bytes at a time from the [`XOF`] stream, interpreting them
    /// as two 12-bit candidates, and keeps each candidate that is less than `q`
    /// until 256 coefficients have been collected. [`Self::sample_ntt_x4`] is
    /// the batched form that drives four streams at once for matrix expansion.
    ///
    /// Implements [Algorithm 7] of FIPS 203.
    ///
    /// [Algorithm 7]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.7
    // reason: `count < N` guards both writes, so every index is in bounds.
    #[allow(clippy::indexing_slicing)]
    pub fn sample_ntt(rho: &[u8; 32], i: u8, j: u8) -> Self {
        let mut reader = XOF(rho, i, j);
        let mut coefficients = [FieldElement::ZERO; parameters::N];
        let mut count = 0;
        let mut triple = [0u8; 3];

        while count < parameters::N {
            reader.squeeze(&mut triple);
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

    /// Batched `SampleNTT`: rejection-samples four uniform Tq elements in
    /// parallel, one per 34-byte seed (`rho ‖ j ‖ i`), driving the platform's
    /// batched Keccak through [`Shake128X4`].
    ///
    /// Each output equals [`Self::sample_ntt`] on that seed's SHAKE128 stream;
    /// this is the matrix-expansion hot path (`TqMatrix::expand`).
    #[must_use]
    pub fn sample_ntt_x4(seeds: &[[u8; 34]; 4]) -> [Self; 4] {
        let mut state = Shake128X4::absorb(seeds);
        let mut coefficients = [[FieldElement::ZERO; parameters::N]; 4];
        let mut counts = [0usize; 4];

        let blocks = state.squeeze_first_three_blocks();
        let mut done = Self::reject_into(&blocks, &mut coefficients, &mut counts);

        while !done {
            let blocks = state.squeeze_next_block();
            done = Self::reject_into(&blocks, &mut coefficients, &mut counts);
        }

        coefficients.map(Self::new)
    }

    /// Rejection-samples one squeezed block per lane into `coefficients`,
    /// advancing each lane's `counts` entry. Returns whether every lane has
    /// reached `N` coefficients.
    // reason: `count < N` guards every write, so each index is in bounds.
    #[allow(clippy::indexing_slicing)]
    fn reject_into<const B: usize>(
        blocks: &[[u8; B]; 4],
        coefficients: &mut [[FieldElement; parameters::N]; 4],
        counts: &mut [usize; 4],
    ) -> bool {
        let lanes = blocks
            .iter()
            .zip(coefficients.iter_mut())
            .zip(counts.iter_mut());

        for ((block, lane), count) in lanes {
            for chunk in block.chunks_exact(3) {
                if *count >= parameters::N {
                    break;
                }

                if let [b0, b1, b2] = chunk {
                    let d1 = u16::from(*b0) | (u16::from(*b1 & 0x0F) << 8);
                    let d2 = (u16::from(*b1) >> 4) | (u16::from(*b2) << 4);

                    if d1 < Q {
                        lane[*count] = FieldElement::new(d1);
                        *count += 1;
                    }
                    if d2 < Q && *count < parameters::N {
                        lane[*count] = FieldElement::new(d2);
                        *count += 1;
                    }
                }
            }
        }

        counts.iter().all(|&count| count >= parameters::N)
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
    pub fn sample_cbd(eta: Eta, bytes: &[u8]) -> Self {
        let eta = usize::from(eta);
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
