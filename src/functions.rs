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
//! SHAKE streams, so [`XOF_x4`] and [`PRF_x4`] squeeze four lanes at
//! once on `sha3_selkie`'s batched sponges. Their output is bit-identical to
//! the scalar `XOF`/`PRF`; the scalar paths remain the per-stream reference.
//!
//! [ML-KEM]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
//! [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
//! [`SampleNTT`]: crate::sampling

use core::{array, ops::Deref};

use sha3_selkie::{Sha3_256, Sha3_512, Shake256, Shake256X2, Shake256X4};
// Re-exported so `XOF` callers can name its return type: the same hasher, in
// its squeezing phase.
pub use sha3_selkie::{Shake128, Shake128X4, Squeezing};

use crate::parameters::Eta;

#[cfg(test)]
mod tests;

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

/// The output of a batched `PRF`: `N` lanes of exactly `64 * eta` bytes,
/// carrying the same no-slack guarantee as [`PrfOutput`] across a whole batch.
///
/// One discriminant for the batch rather than one per lane. Lanes always share
/// an `eta` — they come from a single lockstep sponge — so a `[PrfOutput; N]`
/// would encode that shared fact `N` times, pad every `eta = 2` lane out to the
/// `eta = 3` size, and force a re-wrapping copy of each lane after the squeeze.
/// Here the squeeze writes straight into the variant's payload.
pub enum PrfBatch<const N: usize> {
    /// `eta = 2`: `N` lanes of `64 * 2` bytes.
    Eta2([[u8; 64 * 2]; N]),
    /// `eta = 3`: `N` lanes of `64 * 3` bytes.
    Eta3([[u8; 64 * 3]; N]),
}

impl<const N: usize> PrfBatch<N> {
    /// Returns the `N` SHAKE256 inputs this batch squeezes from: `s ‖ (b + i)`
    /// for lane `i`, the consecutive domain separators of [`PRF`].
    fn inputs(s: &[u8; 32], b: u8) -> [[u8; 33]; N] {
        array::from_fn(|lane| {
            let mut input = [0u8; 33];
            let (prefix, suffix) = input.split_at_mut(32);
            prefix.copy_from_slice(s);
            suffix.copy_from_slice(&[b.wrapping_add(lane as u8)]);

            input
        })
    }

    /// Returns each lane's output bytes in lane order, every one exactly
    /// `64 * eta` long.
    #[must_use]
    pub fn lanes(&self) -> [&[u8]; N] {
        match self {
            Self::Eta2(lanes) => lanes.each_ref().map(<[u8; 64 * 2]>::as_slice),
            Self::Eta3(lanes) => lanes.each_ref().map(<[u8; 64 * 3]>::as_slice),
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

/// Two independent [`PRF`] streams, the two-lane counterpart of [`PRF_x4`].
#[must_use]
#[inline]
pub fn PRF_x2(eta: Eta, s: &[u8; 32], b: u8) -> PrfBatch<2> {
    let inputs = PrfBatch::<2>::inputs(s, b);
    let inputs = inputs.each_ref().map(<[u8; 33]>::as_slice);

    match eta {
        Eta::Two => {
            let mut lanes = [[0u8; 64 * 2]; 2];
            let [l0, l1] = &mut lanes;
            Shake256X2::absorb(inputs).squeeze([l0, l1]);

            PrfBatch::Eta2(lanes)
        }
        Eta::Three => {
            let mut lanes = [[0u8; 64 * 3]; 2];
            let [l0, l1] = &mut lanes;
            Shake256X2::absorb(inputs).squeeze([l0, l1]);

            PrfBatch::Eta3(lanes)
        }
    }
}

/// Four independent [`PRF`] streams sharing an [`Eta`] and a seed `s`, over the
/// consecutive domain separators `b`, `b + 1`, `b + 2`, `b + 3`.
///
/// Each lane squeezes exactly `64 * eta` bytes, as [`PRF`] does. Exactness
/// costs more here than on the scalar path: the lanes share one lockstep sponge
/// cursor, so a lane reading past the current rate block costs *every* lane
/// another `Keccak-f[1600]`. At `eta = 2` the 128-byte output fits inside one
/// 136-byte SHAKE256 block, where squeezing a uniform `64 * 3` would spill into
/// a second.
#[must_use]
#[inline]
pub fn PRF_x4(eta: Eta, s: &[u8; 32], b: u8) -> PrfBatch<4> {
    let inputs = PrfBatch::<4>::inputs(s, b);
    let inputs = inputs.each_ref().map(<[u8; 33]>::as_slice);

    match eta {
        Eta::Two => {
            let mut lanes = [[0u8; 64 * 2]; 4];
            let [l0, l1, l2, l3] = &mut lanes;
            Shake256X4::absorb(inputs).squeeze([l0, l1, l2, l3]);

            PrfBatch::Eta2(lanes)
        }
        Eta::Three => {
            let mut lanes = [[0u8; 64 * 3]; 4];
            let [l0, l1, l2, l3] = &mut lanes;
            Shake256X4::absorb(inputs).squeeze([l0, l1, l2, l3]);

            PrfBatch::Eta3(lanes)
        }
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

/// [eXtendable-output function][FIPS 203] (`XOF`): a lightly constrained
/// invocation of SHAKE128.
///
/// Takes one 32-byte input and two 1-byte inputs and produces a streaming,
/// variable-length output for the rejection sampler `SampleNTT` (Algorithm 7 in
/// [FIPS 203]). Because `SampleNTT` cannot know in advance how many bytes it
/// will need, it returns the SHAKE128 sponge in its squeezing phase rather than
/// a fixed-size buffer. This is the scalar per-stream path; matrix expansion
/// drives four streams at once through [`XOF_x4`].
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
// XXX: normally we newtype 32 raw bytes to distinguish them from any other
// array of 32 bytes, but following the convention for hashes and PRFs we let
// this seed `rho` be raw bytes.
#[must_use]
pub fn XOF(rho: &[u8; 32], i: u8, j: u8) -> Shake128<Squeezing> {
    let mut hasher = Shake128::new();
    hasher.update(rho);
    hasher.update(&[i, j]);

    hasher.finalize_xof()
}

/// Four independent [`XOF`] streams over one `rho`, one per `(i, j)` pair.
///
/// Each lane absorbs the same `rho ‖ i ‖ j` that [`XOF`] does, so lane `n`
/// squeezes bit-identically to `XOF(rho, i, j)` on its own pair. Encoding the
/// seed here rather than at the call sites keeps FIPS 203's input layout in one
/// place, shared with the scalar path.
#[must_use]
pub fn XOF_x4(rho: &[u8; 32], indices: [(u8, u8); 4]) -> Shake128X4<Squeezing> {
    let seeds: [[u8; 34]; 4] = indices.map(|(i, j)| {
        let mut seed = [0u8; 34];
        let (prefix, suffix) = seed.split_at_mut(32);
        prefix.copy_from_slice(rho);
        suffix.copy_from_slice(&[i, j]);

        seed
    });

    Shake128X4::absorb(seeds.each_ref().map(<[u8; 34]>::as_slice))
}
