//! Hash_DRBG-SHA3 per [SP 800-90A Rev 1][sp80090a] §10.1.1. One-shot use only:
//! reseed / additional_input / personalization paths are unimplemented — the
//! library instantiates, calls `fill_bytes` once, and drops.
//!
//! [sp80090a]: https://doi.org/10.6028/NIST.SP.800-90Ar1

// reason: SP 800-90A pseudocode indexes into V, C, and the hash output by
// spec-defined offsets; the array sizes are const-generic and all offsets are
// bounded by their loop guards, so the bounds are provable by inspection.
#![allow(clippy::indexing_slicing)]

use core::marker::PhantomData;

use libcrux_sha3::portable;
use rand_core::{CryptoRng, Error, RngCore};
use zeroize::{Zeroize, ZeroizeOnDrop};

#[cfg(test)]
mod tests;

/// Underlying hash for [`HashDrbg`]. Implemented for the three SHA3 fixed-
/// output flavors below.
pub trait DrbgHash {
    /// Digest length in bytes.
    const OUTLEN: usize;
    /// One-shot fixed-length hash: `out.len() == Self::OUTLEN`.
    fn hash(out: &mut [u8], data: &[u8]);
}

/// SP 800-90A §10.1.1 Hash_DRBG. Callers use one of the strength-labeled
/// aliases below rather than naming `H` and `SEEDLEN` directly.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct HashDrbg<H, const SEEDLEN: usize> {
    v: [u8; SEEDLEN],
    c: [u8; SEEDLEN],
    reseed_counter: u64,
    #[zeroize(skip)]
    _hash: PhantomData<H>,
}

/// Marker type for the SHA3-256 flavor (128-bit strength).
#[derive(Zeroize)]
pub struct Sha3_256Hash;

impl DrbgHash for Sha3_256Hash {
    const OUTLEN: usize = 32;
    fn hash(out: &mut [u8], data: &[u8]) {
        portable::sha256(out, data);
    }
}

/// Marker type for the SHA3-384 flavor (192-bit strength).
#[derive(Zeroize)]
pub struct Sha3_384Hash;

impl DrbgHash for Sha3_384Hash {
    const OUTLEN: usize = 48;
    fn hash(out: &mut [u8], data: &[u8]) {
        portable::sha384(out, data);
    }
}

/// Marker type for the SHA3-512 flavor (256-bit strength).
#[derive(Zeroize)]
pub struct Sha3_512Hash;

impl DrbgHash for Sha3_512Hash {
    const OUTLEN: usize = 64;
    fn hash(out: &mut [u8], data: &[u8]) {
        portable::sha512(out, data);
    }
}

/// 128-bit strength Hash_DRBG (SP 800-90A Rev 1 Table 2: `seedlen = 440`).
pub type HashDrbgSha3_256 = HashDrbg<Sha3_256Hash, 55>;

/// 192-bit strength Hash_DRBG (SP 800-90A Rev 1 Table 2: `seedlen = 888`).
pub type HashDrbgSha3_384 = HashDrbg<Sha3_384Hash, 111>;

/// 256-bit strength Hash_DRBG (SP 800-90A Rev 1 Table 2: `seedlen = 888`).
pub type HashDrbgSha3_512 = HashDrbg<Sha3_512Hash, 111>;

/// Stack scratch buffer sized for the largest hash input this DRBG assembles:
/// `1 (counter) + 4 (length) + 1 (prefix) + SEEDLEN (up to 111) = 117` bytes.
const HASH_INPUT_BUF: usize = 128;

/// Max digest length across the supported hashes (SHA3-512 = 64).
const MAX_DIGEST: usize = 64;

impl<H, const SEEDLEN: usize> HashDrbg<H, SEEDLEN>
where
    H: DrbgHash,
{
    /// Instantiate per SP 800-90A §10.1.1.2. `entropy_input` must carry
    /// `security_strength` bits of entropy; supplying `1.5 * strength` bits
    /// lets the same buffer cover the nonce requirement (§8.6.7).
    #[must_use]
    pub fn new(entropy_input: &[u8]) -> Self {
        let mut v = [0u8; SEEDLEN];
        Self::hash_df(&[entropy_input], &mut v);

        let mut c = [0u8; SEEDLEN];
        Self::hash_df(&[&[0x00u8], &v], &mut c);

        Self {
            v,
            c,
            reseed_counter: 1,
            _hash: PhantomData,
        }
    }

    /// `Hash_df`, SP 800-90A §10.4.1.
    fn hash_df(input_pieces: &[&[u8]], out: &mut [u8]) {
        let outlen = H::OUTLEN;
        let no_of_bits_to_return: u32 = (out.len() as u32) * 8;
        let iterations = out.len().div_ceil(outlen);

        let mut buf = [0u8; HASH_INPUT_BUF];
        let mut digest = [0u8; MAX_DIGEST];

        let mut counter: u8 = 1;
        for i in 0..iterations {
            let mut n = 0;
            buf[n] = counter;
            n += 1;
            buf[n..n + 4].copy_from_slice(&no_of_bits_to_return.to_be_bytes());
            n += 4;
            for piece in input_pieces {
                buf[n..n + piece.len()].copy_from_slice(piece);
                n += piece.len();
            }
            H::hash(&mut digest[..outlen], &buf[..n]);

            let start = i * outlen;
            let end = (start + outlen).min(out.len());
            let take = end - start;
            out[start..end].copy_from_slice(&digest[..take]);

            counter = counter.wrapping_add(1);
        }
    }

    /// `Hashgen`, SP 800-90A §10.1.1.4 step 3.
    fn hashgen(&self, out: &mut [u8]) {
        let outlen = H::OUTLEN;
        let iterations = out.len().div_ceil(outlen);
        let mut data = self.v;
        let mut digest = [0u8; MAX_DIGEST];

        for i in 0..iterations {
            H::hash(&mut digest[..outlen], &data);

            let start = i * outlen;
            let end = (start + outlen).min(out.len());
            let take = end - start;
            out[start..end].copy_from_slice(&digest[..take]);

            add_be_u8(&mut data, 1);
        }
    }

    /// Generate per SP 800-90A §10.1.1.4 (no `additional_input`). The
    /// reseed-required check is elided: we never approach the `2^48`
    /// `reseed_interval` in a one-shot use.
    fn generate(&mut self, out: &mut [u8]) {
        self.hashgen(out);

        // H = Hash(0x03 || V)
        let mut buf = [0u8; HASH_INPUT_BUF];
        buf[0] = 0x03;
        buf[1..1 + SEEDLEN].copy_from_slice(&self.v);
        let mut digest = [0u8; MAX_DIGEST];
        H::hash(&mut digest[..H::OUTLEN], &buf[..1 + SEEDLEN]);

        add_be_slice(&mut self.v, &digest[..H::OUTLEN]);
        {
            let c = self.c;
            add_be_slice(&mut self.v, &c);
        }
        add_be_slice(&mut self.v, &self.reseed_counter.to_be_bytes());

        self.reseed_counter = self
            .reseed_counter
            .checked_add(1)
            .expect("Hash_DRBG reseed_counter overflow");
    }
}

/// `v += addend (mod 2^(8*N))`, big-endian.
fn add_be_u8<const N: usize>(v: &mut [u8; N], addend: u8) {
    let mut carry: u16 = u16::from(addend);
    for j in (0..N).rev() {
        let sum = u16::from(v[j]) + carry;
        v[j] = sum as u8;
        carry = sum >> 8;
        if carry == 0 {
            break;
        }
    }
}

/// `v += addend (mod 2^(8*N))`, big-endian. `addend` may be any length; bytes
/// above `2^(8*N)` are the mod-reduction.
fn add_be_slice<const N: usize>(v: &mut [u8; N], addend: &[u8]) {
    let a_len = addend.len();
    let mut carry: u16 = 0;
    for i in 0..N {
        let v_byte = u16::from(v[N - 1 - i]);
        let a_byte = if i < a_len {
            u16::from(addend[a_len - 1 - i])
        } else {
            0
        };
        let sum = v_byte + a_byte + carry;
        v[N - 1 - i] = sum as u8;
        carry = sum >> 8;
    }
}

impl<H, const SEEDLEN: usize> RngCore for HashDrbg<H, SEEDLEN>
where
    H: DrbgHash,
{
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0u8; 4];
        self.generate(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        self.generate(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.generate(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.generate(dest);
        Ok(())
    }
}

impl<H, const SEEDLEN: usize> CryptoRng for HashDrbg<H, SEEDLEN> where H: DrbgHash {}

/// One-shot: source entropy from the OS, instantiate the concrete
/// [`HashDrbg`], and fill `dst`.
///
/// # Panics
///
/// Panics if the OS entropy source is unavailable.
pub trait DrbgFor {
    /// See [`DrbgFor`].
    fn fill_from_os(dst: &mut [u8]);
}

// `security_strength * 1.5 / 8` bytes per SP 800-90A §8.6.7 (one draw covers
// entropy + nonce when they share a source).
const ENTROPY_LEN_128: usize = 24;
const ENTROPY_LEN_192: usize = 36;
const ENTROPY_LEN_256: usize = 48;

const GETRANDOM_ERR: &str = "OS entropy source (`getrandom`) unavailable";

impl DrbgFor for HashDrbgSha3_256 {
    fn fill_from_os(dst: &mut [u8]) {
        let mut entropy = [0u8; ENTROPY_LEN_128];
        getrandom::getrandom(&mut entropy).expect(GETRANDOM_ERR);
        let mut drbg = Self::new(&entropy);
        entropy.zeroize();
        drbg.fill_bytes(dst);
    }
}

impl DrbgFor for HashDrbgSha3_384 {
    fn fill_from_os(dst: &mut [u8]) {
        let mut entropy = [0u8; ENTROPY_LEN_192];
        getrandom::getrandom(&mut entropy).expect(GETRANDOM_ERR);
        let mut drbg = Self::new(&entropy);
        entropy.zeroize();
        drbg.fill_bytes(dst);
    }
}

impl DrbgFor for HashDrbgSha3_512 {
    fn fill_from_os(dst: &mut [u8]) {
        let mut entropy = [0u8; ENTROPY_LEN_256];
        getrandom::getrandom(&mut entropy).expect(GETRANDOM_ERR);
        let mut drbg = Self::new(&entropy);
        entropy.zeroize();
        drbg.fill_bytes(dst);
    }
}
