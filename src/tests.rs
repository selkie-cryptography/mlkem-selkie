//! End-to-end tests of the ML-KEM public API.

use rand::rngs::OsRng;

use super::*;

/// A generated key pair encapsulates and decapsulates to the same shared
/// secret.
fn encaps_decaps_agree<P: ParameterSet>() {
    let keypair = KeyPair::<P>::generate(&mut OsRng);

    let (sender_secret, ciphertext) = keypair.encapsulation_key.encapsulate(&mut OsRng);
    let receiver_secret = keypair.decapsulation_key.decapsulate(&ciphertext);

    assert_eq!(sender_secret.as_bytes(), receiver_secret.as_bytes());
}

/// Encaps/decaps agree across all three parameter sets.
#[test]
fn roundtrip_all_parameter_sets() {
    encaps_decaps_agree::<MLKEM512>();
    encaps_decaps_agree::<MLKEM768>();
    encaps_decaps_agree::<MLKEM1024>();
}

/// The per-parameter-set aliased API resolves and round-trips.
#[test]
fn aliased_module_roundtrip() {
    let keypair: crate::mlkem768::KeyPair = crate::mlkem768::KeyPair::generate(&mut OsRng);

    let (sender, ciphertext) = keypair.encapsulation_key.encapsulate(&mut OsRng);
    let parsed = crate::mlkem768::Ciphertext::from_bytes(ciphertext.as_bytes()).expect("valid ct");
    let receiver = keypair.decapsulation_key.decapsulate(&parsed);

    assert_eq!(sender.as_bytes(), receiver.as_bytes());
}

/// A derandomized key pair is reproducible from its seed, and its keys
/// serialize to the expected sizes.
#[test]
fn derand_keygen_is_deterministic() {
    let seed = [0x33u8; 64];

    let first = KeyPair::<MLKEM768>::generate_derand(&seed);
    let second = KeyPair::<MLKEM768>::generate_derand(&seed);

    assert_eq!(
        first.encapsulation_key.to_bytes(),
        second.encapsulation_key.to_bytes()
    );
    assert_eq!(
        first.decapsulation_key.to_bytes(),
        second.decapsulation_key.to_bytes()
    );
    assert_eq!(
        first.encapsulation_key.to_bytes().len(),
        MLKEM768::ENCAPS_KEY_SIZE
    );
    assert_eq!(
        first.decapsulation_key.to_bytes().len(),
        MLKEM768::DECAPS_KEY_SIZE
    );
}

/// Key and ciphertext serialization round-trips through the public parsers.
#[test]
fn public_serialization_roundtrip() {
    let keypair = KeyPair::<MLKEM512>::generate_derand(&[1u8; 64]);

    let ek_bytes = keypair.encapsulation_key.to_bytes();
    let dk_bytes = keypair.decapsulation_key.to_bytes();

    let ek = EncapsulationKey::<MLKEM512>::from_bytes(&ek_bytes).expect("valid ek");
    let dk = DecapsulationKey::<MLKEM512>::from_bytes(&dk_bytes).expect("valid dk");

    assert_eq!(ek.to_bytes(), ek_bytes);
    assert_eq!(dk.to_bytes(), dk_bytes);

    let (_secret, ciphertext) = ek.encapsulate_derand(&[0x55u8; 32]);
    let parsed = Ciphertext::<MLKEM512>::from_bytes(ciphertext.as_bytes()).expect("valid ct");

    assert_eq!(parsed.as_bytes(), ciphertext.as_bytes());
}

/// A wrong-length ciphertext is rejected by the parser.
#[test]
fn ciphertext_length_validation() {
    let too_short = vec![0u8; MLKEM512::CIPHERTEXT_SIZE - 1];

    assert!(matches!(
        Ciphertext::<MLKEM512>::from_bytes(&too_short),
        Err(Error::InvalidCiphertextLength)
    ));
}
