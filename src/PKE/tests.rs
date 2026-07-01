//! Unit tests for K-PKE key generation, encryption, and decryption.

use super::*;
use crate::parameters::{MLKEM512, MLKEM768};

/// A freshly generated key pair encrypts and decrypts an arbitrary message back
/// to itself.
fn encrypt_decrypt_roundtrip<P: ParameterSet>() {
    let seed = KeyGenRandomnessSeed::<P>::new([0x42; 32]);
    let KeyPair { dk_pke, ek_pke } = KeyPair::<P>::new_derand(seed);

    let message = [0x9Au8; 32];
    let randomness = [0x17u8; 32];

    let ciphertext = ek_pke.encrypt(&message, &randomness);
    let recovered = dk_pke.decrypt(&ciphertext);

    assert_eq!(recovered, message);
}

/// K-PKE round-trips for ML-KEM-512.
#[test]
fn roundtrip_mlkem512() {
    encrypt_decrypt_roundtrip::<MLKEM512>();
}

/// K-PKE round-trips for ML-KEM-768 (different `K` and `eta_1`).
#[test]
fn roundtrip_mlkem768() {
    encrypt_decrypt_roundtrip::<MLKEM768>();
}
