//! Cryptographic function instantiations (hashes, PRFs, XOF) for [ML-KEM] from section 4.1.
//!
//! [ML-KEM]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Digest, Sha3_256, Sha3_512, Shake128, Shake256,
};

use crate::parameters::ParameterSet;

/// [eXtendable-output function][FIPS 203] (`XOF`): a lightly constrained invocation of SHAKE128.
///
/// The function XOF takes one 32-byte input and two 1-byte inputs, and produces a variable-length
/// output. Invoked to provide a stream of pseudorandom bytes for the sampling algorithm `SampleNTT`
/// (Algorithm 6 in [FIPS 203]). As `SampleNTT` performs rejection sampling, the total number of
/// needed bytes will not be known at the time that `XOF` is invoked, hence, variable-length output.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
// XXX: normally we newtype 32 raw bytes to distinguish them from any other array or slice of 32
// bytes, but as seems to be the convention for functions like hashes and PRFs, we allow this input
// rho to be some raw 32 bytes
pub(crate) fn XOF(rho: [u8; 32], i: u8, j: u8) -> &[u8] {
    let mut preimage = [0u8; 34];

    preimage[..32].copy_from_slice(&rho);
    preimage[32..].copy_from_slice(&[i]);
    preimage[33..].copy_from_slice(&[j]);

    let mut h = Shake128::default();
    h.update(&preimage);
    let mut reader = h.finalize_xof();

    let mut output = [0u8; ETA_X_64];
    reader.read(&mut output);
    return output;
}

/// `PRF` from [FIPS 203 section 4.1][FIPS 203] takes a parameter η of value 2 or 3, a 32-byte input `s`, and a 1-byte
/// input `b`, and returns a 64 * η -byte output.
///
/// This is SHAKE256 with the 32 byte input post-fixed and the output length scaled in 64-byte
/// chunks.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
pub(crate) fn PRF<const ETA_X_64: usize>(s: [u8; 32], b: u8) -> [u8; ETA_X_64] {
    let mut preimage = [0u8; 33];

    preimage[..32].copy_from_slice(&s);
    preimage[32..].copy_from_slice(&[b]);

    let mut h = Shake256::default();
    h.update(&preimage);
    let mut reader = h.finalize_xof();

    let mut output = [0u8; ETA_X_64];
    reader.read(&mut output);
    return output;
}

/// `H` from section 4.1 takes a variable-length input of bytes and returns a 32 byte output.
///
/// This is SHA3-256 by another name.
pub fn H(preimage: &[u8]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    Digest::update(&mut h, preimage);

    let mut output = [0u8; 32];
    output.copy_from_slice(h.finalize().as_slice());
    return output;
}

/// `J` from section 4.1 takes a variable-length input of bytes and returns a 32 byte output.
///
/// This is SHAKE256 by another name.
pub fn J(preimage: &[u8]) -> [u8; 32] {
    let mut h = Shake256::default();
    h.update(preimage);
    let mut reader = h.finalize_xof();

    let mut output = [0u8; 32];
    reader.read(&mut output);
    return output;
}

/// `G` from section 4.1 takes a variable-length input of bytes and returns two 32 byte outputs.
///
/// This is SHA3-512, split in two and returned as an array of arrays of bytes (u8's).
pub fn G(preimage: &[u8]) -> [[u8; 32]; 2] {
    let mut h = Sha3_512::new();
    Digest::update(&mut h, preimage);

    let mut output = [[0u8; 32], [0u8; 32]];
    let (left, right) = h.finalize().as_slice().split_at(32);
    output[0].copy_from_slice(left);
    output[1].copy_from_slice(right);
    return output;
}
