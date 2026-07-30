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
//!   vectors; `CbdSampler` runs whole same-eta stream groups through the widest
//!   batched `PRF` each tail allows.
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf

use core::array;

use zeroize::{Zeroize, ZeroizeOnDrop};

use self::arch::{EMPTY_REJECT_BUFFER, RejectBuffer};
use crate::{
    algebraic::{FieldElement, PolynomialRingElement, RqElement, RqVector, TqElement},
    functions::{PRF, PRF_x2, PRF_x4, PrfBatch, PrfOutput, Shake128, Shake128X4, XOF, XOF_x4},
    parameters::{self, Eta, ParameterSet},
};

mod arch;

/// The first `SampleNTT` squeeze: three SHAKE128 blocks, enough to fill a ring
/// element with overwhelming probability before the top-up loop runs at all.
const THREE_BLOCKS: usize = 3 * Shake128X4::RATE;

#[cfg(test)]
mod tests;

impl TqElement {
    /// `SampleNTT`: rejection-samples a uniformly random element of Tq from
    /// `XOF(rho, j, i)` per Algorithm 13/14 of FIPS 203.
    ///
    /// Squeezes the [`XOF`] stream one SHAKE128 block at a time, interpreting
    /// every three bytes as two 12-bit candidates, and keeps each candidate
    /// that is less than q until 256 coefficients have been collected (the
    /// architecture backend's rejection kernel). [`Self::sample_ntt_x4`] is
    /// the batched form that drives four streams at once for matrix
    /// expansion.
    ///
    /// Implements [Algorithm 7] of FIPS 203.
    ///
    /// [Algorithm 7]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.7
    pub fn sample_ntt(rho: &[u8; 32], i: u8, j: u8) -> Self {
        let mut reader = XOF(rho, i, j);
        let mut buffer = EMPTY_REJECT_BUFFER;
        let mut count = 0;
        let mut block = [0u8; Shake128::RATE];

        while count < parameters::N {
            reader.read(&mut block);
            arch::reject(&block, &mut buffer, &mut count);
        }

        Self::from_reject_buffer(&buffer)
    }

    /// Copies the first `N` coefficients of a filled rejection buffer.
    // reason: `RejectBuffer` is `N + REJECT_SLACK` long, so `i < N` is in
    // bounds.
    #[allow(clippy::indexing_slicing)]
    fn from_reject_buffer(buffer: &RejectBuffer) -> Self {
        Self::new(array::from_fn(|i| buffer[i]))
    }

    /// Batched `SampleNTT`: rejection-samples four uniform Tq elements in
    /// parallel, one per `(i, j)` pair, driving the platform's batched Keccak
    /// through [`XOF_x4`].
    ///
    /// Each output equals [`Self::sample_ntt`] on the same `(rho, i, j)`; this
    /// is the matrix-expansion hot path (`TqMatrix::expand`).
    #[must_use]
    pub fn sample_ntt_x4(rho: &[u8; 32], indices: [(u8, u8); 4]) -> [Self; 4] {
        let mut state = XOF_x4(rho, indices);
        let mut buffers = [EMPTY_REJECT_BUFFER; 4];
        let mut counts = [0usize; 4];

        let mut blocks = [[0u8; THREE_BLOCKS]; 4];
        let [b0, b1, b2, b3] = &mut blocks;
        state.squeeze([b0, b1, b2, b3]);
        let mut done = Self::reject_into(&blocks, &mut buffers, &mut counts);

        while !done {
            let mut blocks = [[0u8; Shake128X4::RATE]; 4];
            let [b0, b1, b2, b3] = &mut blocks;
            state.squeeze([b0, b1, b2, b3]);
            done = Self::reject_into(&blocks, &mut buffers, &mut counts);
        }

        [
            Self::from_reject_buffer(&buffers[0]),
            Self::from_reject_buffer(&buffers[1]),
            Self::from_reject_buffer(&buffers[2]),
            Self::from_reject_buffer(&buffers[3]),
        ]
    }

    /// Rejection-samples one squeezed block per lane into `buffers` via the
    /// architecture backend's kernel, advancing each lane's `counts` entry.
    /// Returns whether every lane has reached `N` coefficients.
    fn reject_into<const B: usize>(
        blocks: &[[u8; B]; 4],
        buffers: &mut [RejectBuffer; 4],
        counts: &mut [usize; 4],
    ) -> bool {
        let lanes = blocks.iter().zip(buffers.iter_mut()).zip(counts.iter_mut());

        for ((block, buffer), count) in lanes {
            arch::reject(block, buffer, count);
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

/// One refill's worth of `PRF` lanes, at whichever width the tail allowed.
// reason: the four-way variant is the common case and the crate is heap-free;
// boxing would trade a stack copy for an allocation.
#[allow(clippy::large_enum_variant)]
enum PrfLanes {
    /// A four-way batch.
    Four(PrfBatch<4>),
    /// A two-way batch.
    Two(PrfBatch<2>),
    /// A single scalar squeeze.
    One(PrfOutput),
}

impl PrfLanes {
    /// Returns the number of lanes this refill squeezed.
    fn len(&self) -> usize {
        match self {
            Self::Four(_) => 4,
            Self::Two(_) => 2,
            Self::One(_) => 1,
        }
    }

    /// Returns lane `index`'s `64 * eta` bytes.
    // reason: callers index below `len`, and every width has lane 0.
    #[allow(clippy::indexing_slicing)]
    fn lane(&self, index: usize) -> &[u8] {
        match self {
            Self::Four(batch) => batch.lanes()[index],
            Self::Two(batch) => batch.lanes()[index],
            Self::One(output) => {
                debug_assert_eq!(index, 0);
                output
            }
        }
    }
}

/// Batched CBD noise sampler: yields [`RqElement`]s from the `count`
/// consecutive `PRF` streams `PRF(seed, first)` through
/// `PRF(seed, first + count - 1)`.
///
/// Each refill squeezes the widest batched `PRF` the remaining streams can
/// fill — four lanes, two, then the scalar path — so declaring `count` up
/// front is what spares the tail from squeezing lanes nobody reads.
///
/// Holds a copy of the seed, secret for the noise/secret polynomials;
/// dropping the sampler zeroizes it.
#[derive(Zeroize, ZeroizeOnDrop)]
pub(crate) struct CbdSampler {
    /// The distribution parameter, passed through to `PRF` and CBD.
    #[zeroize(skip)]
    eta: Eta,
    /// The 32-byte `PRF` seed shared by every stream.
    seed: [u8; 32],
    /// The domain separator the next refill starts at (`N` in Algorithms
    /// 13/14).
    #[zeroize(skip)]
    next: u8,
    /// Streams promised at construction and not yet yielded.
    #[zeroize(skip)]
    remaining: u8,
    /// The current refill's `PRF` lanes. Not zeroized: transient noise bytes,
    /// matching the stack buffers of the scalar path; volatile-wiping them
    /// per drop costs ~5% of encapsulation.
    #[zeroize(skip)]
    batch: PrfLanes,
    /// Lanes of `batch` already yielded.
    #[zeroize(skip)]
    taken: usize,
}

impl CbdSampler {
    /// Starts the run of `count` streams `PRF(seed, first)`,
    /// `PRF(seed, first + 1)`, ... by squeezing its first batch.
    pub(crate) fn new(eta: Eta, seed: &[u8; 32], first: u8, count: u8) -> Self {
        let batch = Self::squeeze(eta, seed, first, count);

        Self {
            eta,
            seed: *seed,
            next: first.wrapping_add(batch.len() as u8),
            remaining: count,
            batch,
            taken: 0,
        }
    }

    /// Squeezes the widest `PRF` batch `remaining` streams can fill. Three
    /// remaining streams still take the four-way batch: one four-way
    /// permutation beats a two-way plus a scalar one.
    fn squeeze(eta: Eta, seed: &[u8; 32], b: u8, remaining: u8) -> PrfLanes {
        match remaining {
            0..=1 => PrfLanes::One(PRF(eta, seed, b)),
            2 => PrfLanes::Two(PRF_x2(eta, seed, b)),
            _ => PrfLanes::Four(PRF_x4(eta, seed, b)),
        }
    }

    /// Samples the next stream's element.
    pub(crate) fn sample_element(&mut self) -> RqElement {
        debug_assert!(self.remaining > 0, "sampled past the declared count");

        if self.taken == self.batch.len() {
            self.batch = Self::squeeze(self.eta, &self.seed, self.next, self.remaining);
            self.next = self.next.wrapping_add(self.batch.len() as u8);
            self.taken = 0;
        }

        let lane = self.batch.lane(self.taken);
        self.taken += 1;
        self.remaining = self.remaining.saturating_sub(1);

        RqElement::sample_cbd(self.eta, lane)
    }

    /// Samples the next `K` streams as a vector.
    pub(crate) fn sample_vector<P: ParameterSet>(&mut self) -> RqVector<P> {
        RqVector::from_fn(|_| self.sample_element())
    }
}
