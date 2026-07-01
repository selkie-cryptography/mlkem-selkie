//! Cryptographic function instantiations for [ML-KEM] from [section 4.1], plus
//! the batched-Keccak primitives that vectorize the parallel sampling streams.
//!
//! ML-KEM instantiates its symmetric primitives with the SHA-3 family: `H` and
//! `G` are SHA3-256 and SHA3-512, `J` is SHAKE256, `PRF` is SHAKE256 with a
//! length-scaled output, and `XOF` is SHAKE128 exposed as a streaming reader.
//!
//! Matrix expansion ([`SampleNTT`]) and CBD sampling each run many independent
//! SHAKE streams, so [`Shake128X4`] and [`shake256_x4`] squeeze four lanes at
//! once, dispatched at compile time (the `mlkem_selkie_arch` cfg from
//! `build.rs`) to AVX2 4-way, NEON 2-way, or scalar Keccak. Their output is
//! bit-identical to the scalar `XOF`/`PRF`; the scalar paths remain the
//! per-stream reference and the portable fallback.
//!
//! [ML-KEM]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
//! [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
//! [`SampleNTT`]: crate::sampling

use core::ops::Deref;

#[cfg(mlkem_selkie_arch = "avx2")]
use libcrux_sha3::avx2::x4::incremental as keccak4;
#[cfg(mlkem_selkie_arch = "neon")]
use libcrux_sha3::neon::x2::incremental as keccak2;
use sha3::{
    Digest, Sha3_256, Sha3_512, Shake128, Shake128Reader, Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};

use crate::parameters::Eta;

#[cfg(test)]
mod tests;

/// [eXtendable-output function][FIPS 203] (`XOF`): a lightly constrained
/// invocation of SHAKE128.
///
/// Takes one 32-byte input and two 1-byte inputs and produces a streaming,
/// variable-length output for the rejection sampler `SampleNTT` (Algorithm 7 in
/// [FIPS 203]). Because `SampleNTT` cannot know in advance how many bytes it
/// will need, the return type is a streaming [`XofReader`] rather than a
/// fixed-size buffer. This is the scalar per-stream path; matrix expansion
/// drives four streams at once through the batched [`Shake128X4`].
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
// XXX: normally we newtype 32 raw bytes to distinguish them from any other
// array of 32 bytes, but following the convention for hashes and PRFs we let
// this seed `rho` be raw bytes.
#[must_use]
pub fn XOF(rho: &[u8; 32], i: u8, j: u8) -> Shake128Reader {
    let mut h = Shake128::default();
    h.update(rho);
    h.update(&[i, j]);

    h.finalize_xof()
}

/// The output of [`PRF`]: an exactly-sized SHAKE256 squeeze, one variant per
/// value of [`Eta`].
///
/// Each variant holds precisely `64 * eta` bytes with no slack, so [`Deref`]
/// and [`AsRef`] can only ever yield a length matching the requested `eta`;
/// there is no unused tail to read by accident.
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
    let mut h = Shake256::default();
    h.update(s);
    h.update(&[b]);
    let mut reader = h.finalize_xof();

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
/// Dispatches at compile time (via the `mlkem_selkie_arch` cfg from `build.rs`)
/// to batched Keccak — AVX2 4-way, or NEON 2-way run twice — and otherwise to
/// four scalar squeezes. The batched output is bit-identical to calling the
/// scalar SHAKE256 four times; this backs the parallel `PRF` streams of CBD
/// sampling (`RqVector::sample_cbd`).
pub fn shake256_x4(inputs: [&[u8]; 4], outputs: [&mut [u8]; 4]) {
    #[cfg(mlkem_selkie_arch = "avx2")]
    {
        let [i0, i1, i2, i3] = inputs;
        let [o0, o1, o2, o3] = outputs;
        libcrux_sha3::avx2::x4::shake256(i0, i1, i2, i3, o0, o1, o2, o3);
    }

    #[cfg(mlkem_selkie_arch = "neon")]
    {
        let [i0, i1, i2, i3] = inputs;
        let [o0, o1, o2, o3] = outputs;
        libcrux_sha3::neon::x2::shake256(i0, i1, o0, o1);
        libcrux_sha3::neon::x2::shake256(i2, i3, o2, o3);
    }

    #[cfg(not(any(mlkem_selkie_arch = "avx2", mlkem_selkie_arch = "neon")))]
    for (input, output) in inputs.into_iter().zip(outputs) {
        let mut h = Shake256::default();
        h.update(input);
        h.finalize_xof().read(output);
    }
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
/// fixed blocks for all four lanes at once. Compile-time dispatch (the
/// `mlkem_selkie_arch` cfg from `build.rs`) maps the four lanes onto AVX2
/// 4-way, NEON 2-way run twice, or four scalar SHAKE128 readers — the squeezed
/// bytes are bit-identical to four independent scalar SHAKE128 streams.
#[cfg(mlkem_selkie_arch = "avx2")]
pub struct Shake128X4(keccak4::KeccakState);

/// See the AVX2 variant; this holds two NEON 2-way states (lanes 0–1 and 2–3).
#[cfg(mlkem_selkie_arch = "neon")]
pub struct Shake128X4([keccak2::KeccakState; 2]);

/// See the AVX2 variant; this holds four scalar SHAKE128 readers.
#[cfg(not(any(mlkem_selkie_arch = "avx2", mlkem_selkie_arch = "neon")))]
pub struct Shake128X4([Shake128Reader; 4]);

impl Shake128X4 {
    /// Absorbs four 34-byte seeds (`rho ‖ j ‖ i`), one per lane.
    #[cfg(mlkem_selkie_arch = "avx2")]
    #[must_use]
    pub fn absorb(seeds: &[[u8; 34]; 4]) -> Self {
        let [d0, d1, d2, d3] = seeds;
        let mut state = keccak4::init();
        keccak4::shake128_absorb_final(&mut state, d0, d1, d2, d3);

        Self(state)
    }

    /// Absorbs four 34-byte seeds (`rho ‖ j ‖ i`), one per lane.
    #[cfg(mlkem_selkie_arch = "neon")]
    #[must_use]
    pub fn absorb(seeds: &[[u8; 34]; 4]) -> Self {
        let [d0, d1, d2, d3] = seeds;
        let mut lo = keccak2::init();
        let mut hi = keccak2::init();
        keccak2::shake128_absorb_final(&mut lo, d0, d1);
        keccak2::shake128_absorb_final(&mut hi, d2, d3);

        Self([lo, hi])
    }

    /// Absorbs four 34-byte seeds (`rho ‖ j ‖ i`), one per lane.
    #[cfg(not(any(mlkem_selkie_arch = "avx2", mlkem_selkie_arch = "neon")))]
    #[must_use]
    pub fn absorb(seeds: &[[u8; 34]; 4]) -> Self {
        // Decompose each seed back into `(rho, j, i)` and feed it through `XOF`
        // — keeping the scalar fallback aligned with the per-stream entry point.
        // `XOF`'s `update(rho); update(&[a, b])` matches `SHAKE128(rho ‖ a ‖ b)`,
        // and the seed is `rho ‖ j ‖ i`, so call as `XOF(&rho, j, i)`.
        Self(seeds.each_ref().map(|seed| {
            let [rho @ .., j, i] = *seed;
            XOF(&rho, j, i)
        }))
    }

    /// Squeezes the first three blocks (504 bytes) from each lane.
    #[cfg(mlkem_selkie_arch = "avx2")]
    pub fn squeeze_first_three_blocks(&mut self) -> [[u8; SHAKE128_THREE_BLOCKS]; 4] {
        let mut out = [[0u8; SHAKE128_THREE_BLOCKS]; 4];
        let [o0, o1, o2, o3] = &mut out;
        keccak4::shake128_squeeze_first_three_blocks(&mut self.0, o0, o1, o2, o3);

        out
    }

    /// Squeezes the first three blocks (504 bytes) from each lane.
    #[cfg(mlkem_selkie_arch = "neon")]
    pub fn squeeze_first_three_blocks(&mut self) -> [[u8; SHAKE128_THREE_BLOCKS]; 4] {
        let mut out = [[0u8; SHAKE128_THREE_BLOCKS]; 4];
        let [o0, o1, o2, o3] = &mut out;
        let [lo, hi] = &mut self.0;
        keccak2::shake128_squeeze_first_three_blocks(lo, o0, o1);
        keccak2::shake128_squeeze_first_three_blocks(hi, o2, o3);

        out
    }

    /// Squeezes the first three blocks (504 bytes) from each lane.
    #[cfg(not(any(mlkem_selkie_arch = "avx2", mlkem_selkie_arch = "neon")))]
    pub fn squeeze_first_three_blocks(&mut self) -> [[u8; SHAKE128_THREE_BLOCKS]; 4] {
        let mut out = [[0u8; SHAKE128_THREE_BLOCKS]; 4];
        for (reader, lane) in self.0.iter_mut().zip(&mut out) {
            reader.read(lane);
        }

        out
    }

    /// Squeezes one further block (168 bytes) from each lane.
    #[cfg(mlkem_selkie_arch = "avx2")]
    pub fn squeeze_next_block(&mut self) -> [[u8; SHAKE128_BLOCK]; 4] {
        let mut out = [[0u8; SHAKE128_BLOCK]; 4];
        let [o0, o1, o2, o3] = &mut out;
        keccak4::shake128_squeeze_next_block(&mut self.0, o0, o1, o2, o3);

        out
    }

    /// Squeezes one further block (168 bytes) from each lane.
    #[cfg(mlkem_selkie_arch = "neon")]
    pub fn squeeze_next_block(&mut self) -> [[u8; SHAKE128_BLOCK]; 4] {
        let mut out = [[0u8; SHAKE128_BLOCK]; 4];
        let [o0, o1, o2, o3] = &mut out;
        let [lo, hi] = &mut self.0;
        keccak2::shake128_squeeze_next_block(lo, o0, o1);
        keccak2::shake128_squeeze_next_block(hi, o2, o3);

        out
    }

    /// Squeezes one further block (168 bytes) from each lane.
    #[cfg(not(any(mlkem_selkie_arch = "avx2", mlkem_selkie_arch = "neon")))]
    pub fn squeeze_next_block(&mut self) -> [[u8; SHAKE128_BLOCK]; 4] {
        let mut out = [[0u8; SHAKE128_BLOCK]; 4];
        for (reader, lane) in self.0.iter_mut().zip(&mut out) {
            reader.read(lane);
        }

        out
    }
}

/// `H` from [section 4.1] takes a variable-length byte input and returns a
/// 32-byte output.
///
/// This is SHA3-256 by another name.
///
/// [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn H(preimage: &[u8]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    Digest::update(&mut h, preimage);

    h.finalize().into()
}

/// `J` from [section 4.1] hashes the 32-byte rejection seed `z` followed by the
/// ciphertext `c`, returning a 32-byte output.
///
/// This is SHAKE256 truncated to 32 bytes. `ML-KEM.Decaps` is its only caller
/// and always evaluates `J(z ‖ c)` (Algorithm 18), so rather than take a single
/// `B*` preimage like the abstract `J`, this absorbs `z` and `c` in two
/// `update`s — identical to hashing the concatenation, but without allocating a
/// joined buffer for the ciphertext-sized input.
///
/// [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn J(z: &[u8; 32], c: &[u8]) -> [u8; 32] {
    let mut h = Shake256::default();
    h.update(z);
    h.update(c);
    let mut reader = h.finalize_xof();

    let mut output = [0u8; 32];
    reader.read(&mut output);

    output
}

/// `G` from [section 4.1] takes a variable-length byte input and returns two
/// 32-byte outputs.
///
/// This is SHA3-512, split into two 32-byte halves.
///
/// [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn G(preimage: &[u8]) -> ([u8; 32], [u8; 32]) {
    let mut h = Sha3_512::new();
    Digest::update(&mut h, preimage);
    let digest = h.finalize();

    let (left, right) = digest.split_at(32);

    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a.copy_from_slice(left);
    b.copy_from_slice(right);

    (a, b)
}
