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
use libcrux_sha3::portable::{
    self, KeccakState,
    incremental::{
        Shake256Xof, Xof, shake128_absorb_final, shake128_init,
        shake128_squeeze_first_three_blocks, shake128_squeeze_next_block,
    },
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
/// will need, the return type is a streaming SHAKE128 state rather than a
/// fixed-size buffer. This is the scalar per-stream path; matrix expansion
/// drives four streams at once through the batched [`Shake128X4`].
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
// XXX: normally we newtype 32 raw bytes to distinguish them from any other
// array of 32 bytes, but following the convention for hashes and PRFs we let
// this seed `rho` be raw bytes.
#[must_use]
pub fn XOF(rho: &[u8; 32], i: u8, j: u8) -> Shake128Reader {
    let mut buf = [0u8; 34];
    buf[..32].copy_from_slice(rho);
    buf[32] = i;
    buf[33] = j;

    let mut state = shake128_init();
    shake128_absorb_final(&mut state, &buf);

    Shake128Reader::new(state)
}

/// Streaming SHAKE128 reader on top of libcrux's fixed-block
/// `shake128_squeeze_*` primitives. Matches the squeeze layout used by
/// [`Shake128X4`] and by `libcrux_sha3::{avx2,neon}::x{4,2}::shake128_*`, so
/// scalar and batched streams are bit-identical for identical seeds.
pub struct Shake128Reader {
    state: KeccakState,
    // Buffered squeezed bytes; sized for the first three blocks squeeze.
    // Subsequent refills only use the first `SHAKE128_BLOCK` slot.
    buf: [u8; SHAKE128_THREE_BLOCKS],
    filled: usize,
    pos: usize,
    squeezed_first_three: bool,
}

impl Shake128Reader {
    fn new(state: KeccakState) -> Self {
        Self {
            state,
            buf: [0u8; SHAKE128_THREE_BLOCKS],
            filled: 0,
            pos: 0,
            squeezed_first_three: false,
        }
    }

    /// Copies bytes into `dst`, block-buffering so partial-block reads work.
    // reason: `take` is clamped to `dst.len() - i` and `filled - pos`, so both
    // slice ranges stay within `dst` and `buf` respectively; the invariants
    // `pos <= filled <= SHAKE128_THREE_BLOCKS` and `i <= dst.len()` follow
    // from the loop guard and `pos = 0; filled = block_size` after each refill.
    #[allow(clippy::indexing_slicing)]
    pub fn squeeze(&mut self, dst: &mut [u8]) {
        let mut i = 0;
        while i < dst.len() {
            if self.pos >= self.filled {
                if !self.squeezed_first_three {
                    shake128_squeeze_first_three_blocks(&mut self.state, &mut self.buf);
                    self.filled = SHAKE128_THREE_BLOCKS;
                    self.squeezed_first_three = true;
                } else {
                    shake128_squeeze_next_block(&mut self.state, &mut self.buf[..SHAKE128_BLOCK]);
                    self.filled = SHAKE128_BLOCK;
                }
                self.pos = 0;
            }
            let take = (self.filled - self.pos).min(dst.len() - i);
            dst[i..i + take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
            i += take;
        }
    }
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
    let mut xof = Shake256Xof::new();
    xof.absorb(s);
    xof.absorb_final(&[b]);

    match eta {
        Eta::Two => {
            let mut bytes = [0u8; 64 * 2];
            xof.squeeze(&mut bytes);
            PrfOutput::Eta2(bytes)
        }
        Eta::Three => {
            let mut bytes = [0u8; 64 * 3];
            xof.squeeze(&mut bytes);
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
        portable::shake256(output, input);
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
/// 4-way, NEON 2-way run twice, or four scalar SHAKE128 states — the squeezed
/// bytes are bit-identical to four independent scalar SHAKE128 streams.
#[cfg(mlkem_selkie_arch = "avx2")]
pub struct Shake128X4(keccak4::KeccakState);

/// See the AVX2 variant; this holds two NEON 2-way states (lanes 0–1 and 2–3).
#[cfg(mlkem_selkie_arch = "neon")]
pub struct Shake128X4([keccak2::KeccakState; 2]);

/// See the AVX2 variant; this holds four scalar SHAKE128 states.
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
        for (xof, lane) in self.0.iter_mut().zip(&mut out) {
            xof.squeeze(lane);
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
        for (xof, lane) in self.0.iter_mut().zip(&mut out) {
            xof.squeeze(lane);
        }

        out
    }
}

/// `H` from [section 4.1]: SHA3-256 into 32 bytes.
///
/// [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn H(preimage: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    portable::sha256(&mut out, preimage);

    out
}

/// `J` from [section 4.1]: SHAKE256 of `z ‖ c` truncated to 32 bytes. Absorbs
/// `z` and `c` incrementally to skip a joined-buffer allocation.
///
/// [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn J(z: &[u8; 32], c: &[u8]) -> [u8; 32] {
    let mut xof = Shake256Xof::new();
    xof.absorb(z);
    xof.absorb_final(c);

    let mut out = [0u8; 32];
    xof.squeeze(&mut out);

    out
}

/// `G` from [section 4.1]: SHA3-512, split into two 32-byte halves.
///
/// [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[must_use]
pub fn G(preimage: &[u8]) -> ([u8; 32], [u8; 32]) {
    let mut digest = [0u8; 64];
    portable::sha512(&mut digest, preimage);

    let (left, right) = digest.split_at(32);

    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a.copy_from_slice(left);
    b.copy_from_slice(right);

    (a, b)
}
