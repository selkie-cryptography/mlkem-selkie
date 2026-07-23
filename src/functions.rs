//! Cryptographic function instantiations for [ML-KEM] from [section 4.1], plus
//! the batched-Keccak primitives that vectorize the parallel sampling streams.
//!
//! ML-KEM instantiates its symmetric primitives with the SHA-3 family: `H` and
//! `G` are SHA3-256 and SHA3-512, `J` is SHAKE256, `PRF` is SHAKE256 with a
//! length-scaled output, and `XOF` is SHAKE128 exposed as a streaming reader.
//! All are backed by the constant-time `sha3_selkie` crate, which selects its
//! Keccak backend (Arm SHA-3 extension, AVX2, or portable scalar) at compile
//! time.
//!
//! Matrix expansion ([`SampleNTT`]) and CBD sampling each run many independent
//! SHAKE streams, so [`Shake128X4`] and [`shake256_x4`] squeeze four lanes at
//! once on `sha3_selkie`'s batched sponges. Their output is bit-identical to
//! the scalar `XOF`/`PRF`; the scalar paths remain the per-stream reference.
//!
//! [ML-KEM]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
//! [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
//! [`SampleNTT`]: crate::sampling

use core::ops::Deref;

// Re-exported so `XOF` callers can name its streaming return type.
pub use sha3_selkie::Shake128Reader;
use sha3_selkie::{Sha3_256, Sha3_512, Shake128, Shake256, Shake256X4};

use crate::parameters::Eta;

#[cfg(test)]
mod tests;

/// [eXtendable-output function][FIPS 203] (`XOF`): a lightly constrained
/// invocation of SHAKE128.
///
/// Takes one 32-byte input and two 1-byte inputs and produces a streaming,
/// variable-length output for the rejection sampler `SampleNTT` (Algorithm 7 in
/// [FIPS 203]). Because `SampleNTT` cannot know in advance how many bytes it
/// will need, the return type is a streaming SHAKE128 reader rather than a
/// fixed-size buffer. This is the scalar per-stream path; matrix expansion
/// drives four streams at once through the batched [`Shake128X4`].
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
// XXX: normally we newtype 32 raw bytes to distinguish them from any other
// array of 32 bytes, but following the convention for hashes and PRFs we let
// this seed `rho` be raw bytes.
#[must_use]
pub fn XOF(rho: &[u8; 32], i: u8, j: u8) -> Shake128Reader {
    let mut hasher = Shake128::new();
    hasher.update(rho);
    hasher.update(&[i, j]);

    hasher.finalize_xof()
}

/// The output of [`PRF`]: an exactly-sized SHAKE256 squeeze, one variant per
/// value of [`Eta`].
///
/// Each variant holds precisely `64 * eta` bytes with no slack, so [`Deref`]
/// can only ever yield a length matching the requested `eta`; there is no
/// unused tail to read by accident.
pub enum PrfOutput {
    /// `eta = 2`: a `64 * 2`-byte squeeze.
    Eta2([u8; 64 * 2]),
    /// `eta = 3`: a `64 * 3`-byte squeeze.
    Eta3([u8; 64 * 3]),
}

impl Deref for PrfOutput {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            PrfOutput::Eta2(bytes) => bytes,
            PrfOutput::Eta3(bytes) => bytes,
        }
    }
}

/// `PRF` from [FIPS 203 section 4.1][FIPS 203] takes the distribution parameter
/// [`Eta`], a 32-byte input `s`, and a 1-byte domain separator `b`, and returns
/// its `64 * eta`-byte output as a [`PrfOutput`].
///
/// This is SHAKE256 with `b` post-fixed to `s`. The output is an exactly-sized
/// stack array (no allocation), one [`PrfOutput`] variant per [`Eta`].
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn PRF(eta: Eta, s: &[u8; 32], b: u8) -> PrfOutput {
    let mut hasher = Shake256::new();
    hasher.update(s);
    hasher.update(&[b]);

    let mut reader = hasher.finalize_xof();

    match eta {
        Eta::Two => {
            let mut bytes = [0u8; 64 * 2];
            reader.read(&mut bytes);
            PrfOutput::Eta2(bytes)
        }
        Eta::Three => {
            let mut bytes = [0u8; 64 * 3];
            reader.read(&mut bytes);
            PrfOutput::Eta3(bytes)
        }
    }
}

/// Runs four independent SHAKE256 squeezes in parallel: `outputs[i]` receives
/// `SHAKE256(inputs[i])` truncated to its length.
///
/// The four lanes run in lockstep on [`Shake256X4`]'s batched sponge, and the
/// output is bit-identical to four scalar SHAKE256 squeezes; this backs the
/// parallel `PRF` streams of CBD sampling (`RqVector::sample_cbd`).
pub fn shake256_x4(inputs: [&[u8]; 4], outputs: [&mut [u8]; 4]) {
    Shake256X4::absorb(inputs).squeeze(outputs);
}

/// SHAKE128 rate (squeeze-block size) in bytes.
pub const SHAKE128_BLOCK: usize = 168;

/// Three SHAKE128 blocks — the first squeeze for `SampleNTT`, enough to sample
/// a full ring element with overwhelming probability.
pub const SHAKE128_THREE_BLOCKS: usize = 3 * SHAKE128_BLOCK;

/// Four parallel SHAKE128 squeeze streams over distinct 34-byte seeds, for
/// batched `SampleNTT` matrix expansion.
///
/// [`Self::absorb`] takes the four seeds; the squeeze methods then produce
/// fixed blocks for all four lanes at once on `sha3_selkie`'s batched sponge.
/// The squeezed bytes are bit-identical to four independent scalar SHAKE128
/// streams, so each lane matches the scalar [`XOF`] on its seed.
pub struct Shake128X4(sha3_selkie::Shake128X4Reader);

impl Shake128X4 {
    /// Absorbs four 34-byte seeds (`rho ‖ j ‖ i`), one per lane.
    #[must_use]
    pub fn absorb(seeds: &[[u8; 34]; 4]) -> Self {
        Self(sha3_selkie::Shake128X4::absorb(
            seeds.each_ref().map(<[u8; 34]>::as_slice),
        ))
    }

    /// Squeezes the first three blocks (504 bytes) from each lane.
    pub fn squeeze_first_three_blocks(&mut self) -> [[u8; SHAKE128_THREE_BLOCKS]; 4] {
        let mut out = [[0u8; SHAKE128_THREE_BLOCKS]; 4];
        self.0.squeeze(
            out.each_mut()
                .map(<[u8; SHAKE128_THREE_BLOCKS]>::as_mut_slice),
        );

        out
    }

    /// Squeezes one further block (168 bytes) from each lane.
    pub fn squeeze_next_block(&mut self) -> [[u8; SHAKE128_BLOCK]; 4] {
        let mut out = [[0u8; SHAKE128_BLOCK]; 4];
        self.0
            .squeeze(out.each_mut().map(<[u8; SHAKE128_BLOCK]>::as_mut_slice));

        out
    }
}

/// `H` from [section 4.1]: SHA3-256 into 32 bytes.
///
/// [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn H(preimage: &[u8]) -> [u8; 32] {
    Sha3_256::digest(preimage)
}

/// `J` from [section 4.1]: SHAKE256 of `z ‖ c` truncated to 32 bytes. Absorbs
/// `z` and `c` incrementally to skip a joined-buffer allocation.
///
/// [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn J(z: &[u8; 32], c: &[u8]) -> [u8; 32] {
    let mut hasher = Shake256::new();
    hasher.update(z);
    hasher.update(c);

    let mut reader = hasher.finalize_xof();
    let mut out = [0u8; 32];
    reader.read(&mut out);

    out
}

/// `G` from [section 4.1]: SHA3-512, split into two 32-byte halves.
///
/// [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn G(preimage: &[u8]) -> ([u8; 32], [u8; 32]) {
    let digest = Sha3_512::digest(preimage);

    let (left, right) = digest.split_at(32);
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a.copy_from_slice(left);
    b.copy_from_slice(right);

    (a, b)
}
