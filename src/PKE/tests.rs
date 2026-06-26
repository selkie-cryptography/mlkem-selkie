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

/// Encryption and decryption key serialization round-trips.
#[test]
fn key_serialization_roundtrip() {
    let seed = KeyGenRandomnessSeed::<MLKEM512>::new([7u8; 32]);
    let KeyPair { dk_pke, ek_pke } = KeyPair::<MLKEM512>::new_derand(seed);

    let ek_bytes = ek_pke.to_bytes();
    let dk_bytes = dk_pke.to_bytes();

    assert_eq!(ek_bytes.len(), MLKEM512::PKE_ENCRYPTION_KEY_SIZE);
    assert_eq!(dk_bytes.len(), MLKEM512::PKE_DECRYPTION_KEY_SIZE);

    let reparsed_ek = EncryptionKey::<MLKEM512>::from_bytes(&ek_bytes);
    let reparsed_dk = DecryptionKey::<MLKEM512>::from_bytes(&dk_bytes);

    assert_eq!(reparsed_ek.to_bytes(), ek_bytes);
    assert_eq!(reparsed_dk.to_bytes(), dk_bytes);
}
