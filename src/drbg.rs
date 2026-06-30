//! AES256-CTR-DRBG per [NIST SP 800-90A][sp80090a], §10.2.1 (no derivation
//! function, no reseed counter, no additional input).
//!
//! Vendored from the sibling `sqisign-selkie` crate. ML-KEM itself only needs
//! an [`RngCore`] for `KeyGen` and `Encaps`; this deterministic generator
//! exists so that NIST PQC known-answer tests, which seed `randombytes_init`
//! with a 48-byte entropy input and then draw the per-test `d`, `z`, and `m`,
//! can be replayed byte-for-byte. The self-contained Wycheproof vectors supply
//! their seeds directly and use the derandomized API instead.
//!
//! # Why 48 bytes of seed
//!
//! AES-256 has `keylen = 32` and `blocklen = 16`, so SP 800-90A §10.2.1 Table 3
//! gives `seedlen = keylen + blocklen = 48` bytes (384 bits). The NIST PQC test
//! harness passes a 48-byte `entropy_input` to `randombytes_init` for every
//! test vector — the `seed = ...` line in each KAT `.rsp` file.
//!
//! [sp80090a]: https://doi.org/10.6028/NIST.SP.800-90Ar1

// reason: vendored SP 800-90A reference code; every index is into a fixed-size
// array or an explicitly bounded slice, so the bounds are provable by
// inspection and the indexed form mirrors the standard's pseudocode.
#![allow(clippy::indexing_slicing)]

use aes::{
    Aes256Enc,
    cipher::{Block, BlockCipherEncrypt, KeyInit},
};
use rand_core::{CryptoRng, Error, RngCore};
use zeroize::{Zeroize, ZeroizeOnDrop};

#[cfg(test)]
mod tests;

/// AES-256 key length in bytes.
const KEYLEN: usize = 32;

/// AES block length in bytes.
const BLOCKLEN: usize = 16;

/// Size in bytes of the 48-byte seed consumed by [`Aes256CtrDrbg::new`].
/// Matches the NIST PQC test harness `randombytes_init` entropy input.
pub const SEEDLEN: usize = KEYLEN + BLOCKLEN;

/// AES256-CTR-DRBG state: 32-byte Key + 16-byte V counter.
///
/// `ZeroizeOnDrop`: both fields are entropy-derived material whose disclosure
/// would let an attacker replay or predict the DRBG stream.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Aes256CtrDrbg {
    /// 32-byte AES-256 key (`Key` in SP 800-90A §10.2.1).
    key: [u8; KEYLEN],
    /// 16-byte counter (`V` in SP 800-90A §10.2.1).
    v: [u8; BLOCKLEN],
    /// Total bytes delivered to callers via `fill` since instantiation.
    /// Test-only probe for diffing byte consumption against a reference.
    #[cfg(test)]
    consumed: u64,
}

impl Aes256CtrDrbg {
    /// Instantiates the DRBG from 48 bytes of entropy input.
    ///
    /// Matches `randombytes_init(entropy_input, NULL, 256)` in the NIST
    /// reference code: start from an all-zero Key/V, then run CTR_DRBG_Update
    /// with `entropy_input` as the provided data.
    #[must_use]
    pub fn new(seed: &[u8; SEEDLEN]) -> Self {
        let mut d = Self {
            key: [0u8; KEYLEN],
            v: [0u8; BLOCKLEN],
            #[cfg(test)]
            consumed: 0,
        };
        d.update(Some(seed));

        d
    }

    /// Returns the total number of output bytes delivered to callers since
    /// instantiation.
    #[cfg(test)]
    pub(crate) fn bytes_consumed(&self) -> u64 {
        self.consumed
    }

    /// CTR_DRBG_Update (SP 800-90A §10.2.1.2).
    ///
    /// Expands the key schedule from the current `Key`. When the update follows
    /// a generate step that already expanded the same `Key` (see
    /// [`Self::randombytes`]), use [`Self::update_with_cipher`] to reuse that
    /// schedule instead of expanding twice.
    fn update(&mut self, provided_data: Option<&[u8; SEEDLEN]>) {
        let cipher = Aes256Enc::new((&self.key).into());
        self.update_with_cipher(&cipher, provided_data);
    }

    /// CTR_DRBG_Update using an already-expanded key schedule for the current
    /// `Key`. `cipher` must be `Aes256Enc::new(self.key)`.
    fn update_with_cipher(&mut self, cipher: &Aes256Enc, provided_data: Option<&[u8; SEEDLEN]>) {
        let mut blocks = [Block::<Aes256Enc>::default(); SEEDLEN / BLOCKLEN];
        for block in &mut blocks {
            Self::increment_v(&mut self.v);
            *block = self.v.into();
        }
        cipher.encrypt_blocks(&mut blocks);

        let mut temp = [0u8; SEEDLEN];
        for (i, block) in blocks.iter().enumerate() {
            temp[i * BLOCKLEN..(i + 1) * BLOCKLEN].copy_from_slice(block);
        }
        if let Some(pd) = provided_data {
            for i in 0..SEEDLEN {
                temp[i] ^= pd[i];
            }
        }
        self.key.copy_from_slice(&temp[..KEYLEN]);
        self.v.copy_from_slice(&temp[KEYLEN..]);
    }

    /// CTR_DRBG_Generate (SP 800-90A §10.2.1.5), no additional input.
    fn randombytes(&mut self, out: &mut [u8]) {
        // One key schedule for the whole call: the generate loop and the
        // trailing update both encrypt under the current `Key`, which is
        // unchanged until the update writes the new one. Generate up to `PAR`
        // counter blocks per AES call so the backend's parallel block pipeline
        // is fed instead of one block at a time.
        const PAR: usize = 8;
        let cipher = Aes256Enc::new((&self.key).into());
        let mut blocks = [Block::<Aes256Enc>::default(); PAR];

        let mut i = 0;
        while i < out.len() {
            let nblocks = (out.len() - i).div_ceil(BLOCKLEN).min(PAR);
            for block in &mut blocks[..nblocks] {
                Self::increment_v(&mut self.v);
                *block = self.v.into();
            }
            cipher.encrypt_blocks(&mut blocks[..nblocks]);
            for block in &blocks[..nblocks] {
                let take = BLOCKLEN.min(out.len() - i);
                out[i..i + take].copy_from_slice(&block[..take]);
                i += take;
            }
        }

        self.update_with_cipher(&cipher, None);
        #[cfg(test)]
        {
            self.consumed += out.len() as u64;
        }
    }

    /// Big-endian increment of the 16-byte V counter.
    fn increment_v(v: &mut [u8; BLOCKLEN]) {
        for j in (0..BLOCKLEN).rev() {
            if v[j] == 0xFF {
                v[j] = 0;
            } else {
                v[j] = v[j].wrapping_add(1);
                return;
            }
        }
    }
}

impl RngCore for Aes256CtrDrbg {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0u8; 4];
        self.randombytes(&mut bytes);

        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        self.randombytes(&mut bytes);

        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.randombytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.randombytes(dest);

        Ok(())
    }
}

impl CryptoRng for Aes256CtrDrbg {}
