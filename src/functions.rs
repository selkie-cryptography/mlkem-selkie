//! Cryptographic function instantiations (hashes, PRFs, XOF) for [ML-KEM] from
//! [section 4.1].
//!
//! ML-KEM instantiates its symmetric primitives with the SHA-3 family: `H` and
//! `G` are SHA3-256 and SHA3-512, `J` is SHAKE256, `PRF` is SHAKE256 with a
//! length-scaled output, and `XOF` is SHAKE128 exposed as a streaming reader so
//! that the rejection sampler [`SampleNTT`] can pull as many bytes as it needs.
//!
//! [ML-KEM]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
//! [section 4.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
//! [`SampleNTT`]: crate::sampling

use core::ops::Deref;

use sha3::{
    Digest, Sha3_256, Sha3_512, Shake128, Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};

use crate::parameters::Eta;

/// [eXtendable-output function][FIPS 203] (`XOF`): a lightly constrained
/// invocation of SHAKE128.
///
/// Takes one 32-byte input and two 1-byte inputs and produces a streaming,
/// variable-length output. Invoked to provide a stream of pseudorandom bytes
/// for the sampling algorithm `SampleNTT` (Algorithm 7 in [FIPS 203]). As
/// `SampleNTT` performs rejection sampling, the total number of bytes needed is
/// not known when `XOF` is invoked, hence the streaming [`XofReader`] return
/// type rather than a fixed-size buffer.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
// XXX: normally we newtype 32 raw bytes to distinguish them from any other
// array of 32 bytes, but following the convention for hashes and PRFs we let
// this seed `rho` be raw bytes.
#[must_use]
pub fn XOF(rho: &[u8; 32], i: u8, j: u8) -> impl XofReader {
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

impl AsRef<[u8]> for PrfOutput {
    fn as_ref(&self) -> &[u8] {
        self
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
