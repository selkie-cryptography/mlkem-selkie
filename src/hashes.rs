//! Hash function instatiations for [ML-KEM] from section 4.1.
//!
//! [ML-KEM]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf

use sha3::{Digest, Sha3_256, Sha3_512, Shake128, Shake256};

/// `H` from section 4.1 takes a variable-length input of bytes and returns a 32 byte output.
///
/// This is SHA3-256 by another name.
pub fn H(preimage: &[u8]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    h.update(preimage);

    let mut output = [0u8; 32];
    output.copy_from_slice(h.finalize().as_slice());
    return output;
}

/// `J` from section 4.1 takes a variable-length input of bytes and returns a 32 byte output.
///
/// This is SHAKE256 by another name.
pub fn J(preimage: &[u8]) -> [u8; 32] {
    let mut h = Shake256::new();
    h.update(preimage);

    let mut output = [0u8; 32];
    output.copy_from_slice(h.finalize(output).as_slice());
    return output;
}

/// `G` from section 4.1 takes a variable-length input of bytes and returns two 32 byte outputs.
///
/// This is SHA3-512, split in two and returned as an array of arrays of bytes (u8's).
pub fn G(preimage: &[u8]) -> [[u8; 32]; 2] {
    let mut h = Sha3_512::new();
    h.update(preimage);

    let mut output = [[0u8; 32], [0u8; 32]];
    let (left, right) = h.finalize().as_slice().split_at(32);
    output[0].copy_from_slice(left);
    output[1].copy_from_slice(right);
    return output;
}
