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
    /// The two `eta`-bit sums are computed word-parallel rather than bit by
    /// bit: masking and adding shifted copies of an input word sums each group
    /// of `eta` adjacent bits in place (a bitfield popcount), so one word load
    /// yields several coefficients. This is branch-free and data-independent —
    /// required, since `bytes` is secret PRF output for the noise/secret polys.
    ///
    /// [Algorithm 8]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.8
    pub fn sample_cbd(eta: Eta, bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 64 * usize::from(eta));

        match eta {
            Eta::Two => Self::sample_cbd_eta2(bytes),
            Eta::Three => Self::sample_cbd_eta3(bytes),
        }
    }

    /// `SamplePolyCBD_2`: each 4-byte word yields 8 coefficients. `(t & M) +
    /// ((t >> 1) & M)` with `M = 0x5555_5555` sums every adjacent bit-pair into
    /// its own 2-bit field; coefficient `j` differences the fields at `4j`
    /// (its `a`) and `4j + 2` (its `b`).
    fn sample_cbd_eta2(bytes: &[u8]) -> Self {
        const MASK: u32 = 0x5555_5555;

        let mut coefficients = [FieldElement::ZERO; parameters::N];

        for (word, out) in bytes.chunks_exact(4).zip(coefficients.chunks_exact_mut(8)) {
            let t = u32::from_le_bytes(word.try_into().expect("chunks_exact(4)"));
            let sums = (t & MASK) + ((t >> 1) & MASK);

            for (j, coeff) in out.iter_mut().enumerate() {
                let a = (sums >> (4 * j)) & 0x3;
                let b = (sums >> (4 * j + 2)) & 0x3;

                *coeff = FieldElement::from(a as u16) - FieldElement::from(b as u16);
            }
        }

        Self::new(coefficients)
    }

    /// `SamplePolyCBD_3`: each 3-byte (24-bit) word yields 4 coefficients.
    /// Three masked shifts with `M = 0x0024_9249` sum every group of 3 adjacent
    /// bits into its own 3-bit field; coefficient `j` differences the fields at
    /// `6j` and `6j + 3`.
    fn sample_cbd_eta3(bytes: &[u8]) -> Self {
        const MASK: u32 = 0x0024_9249;

        let mut coefficients = [FieldElement::ZERO; parameters::N];

        for (word, out) in bytes.chunks_exact(3).zip(coefficients.chunks_exact_mut(4)) {
            let &[b0, b1, b2] = word else {
                unreachable!("chunks_exact(3)")
            };

            let t = u32::from_le_bytes([b0, b1, b2, 0]);
            let sums = (t & MASK) + ((t >> 1) & MASK) + ((t >> 2) & MASK);

            for (j, coeff) in out.iter_mut().enumerate() {
                let a = (sums >> (6 * j)) & 0x7;
                let b = (sums >> (6 * j + 3)) & 0x7;

                *coeff = FieldElement::from(a as u16) - FieldElement::from(b as u16);
            }
        }

        Self::new(coefficients)
    }
}
