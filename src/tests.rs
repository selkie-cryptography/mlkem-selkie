//! End-to-end tests of the ML-KEM public API.

use super::*;

/// A generated key pair encapsulates and decapsulates to the same shared
/// secret.
fn encaps_decaps_agree<P: ParameterSet>() {
    let keypair = DecapsulationKey::<P>::generate();

    let (sender_secret, ciphertext) = keypair.encapsulation_key().encapsulate();
    let receiver_secret = keypair.decapsulate(&ciphertext);

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
    let keypair: mlkem768::DecapsulationKey = mlkem768::DecapsulationKey::generate();

    let (sender, ciphertext) = keypair.encapsulation_key().encapsulate();
    let parsed = mlkem768::Ciphertext::from_bytes(ciphertext.as_bytes()).expect("valid ct");
    let receiver = keypair.decapsulate(&parsed);

    assert_eq!(sender.as_bytes(), receiver.as_bytes());
}

/// Consecutive `generate()` calls produce distinct key pairs — the public
/// randomized entry point actually consumes randomness. Kills mutants that
/// no-op the internal entropy path (default: `getrandom` direct; `--features
/// fips`: `P::fill_from_fips_drbg`). Covers all three parameter sets so the
/// per-P impls and their strength-matched DRBG free functions are all hit.
#[test]
fn generate_is_randomized() {
    check::<MLKEM512>();
    check::<MLKEM768>();
    check::<MLKEM1024>();

    fn check<P: ParameterSet>() {
        let a = DecapsulationKey::<P>::generate();
        let b = DecapsulationKey::<P>::generate();

        assert_ne!(
            a.encapsulation_key().to_bytes().as_ref(),
            b.encapsulation_key().to_bytes().as_ref(),
        );
    }
}

/// A derandomized key pair is reproducible from its seed, and its keys
/// serialize to the expected sizes.
#[test]
fn derand_keygen_is_deterministic() {
    let seed = [0x33u8; 64];

    let first = DecapsulationKey::<MLKEM768>::generate_derand(&seed);
    let second = DecapsulationKey::<MLKEM768>::generate_derand(&seed);

    assert_eq!(
        first.encapsulation_key().to_bytes(),
        second.encapsulation_key().to_bytes()
    );
    assert_eq!(first.to_bytes(), second.to_bytes());
    assert_eq!(
        first.encapsulation_key().to_bytes().len(),
        MLKEM768::ENCAPS_KEY_SIZE
    );
    assert_eq!(first.to_bytes().len(), MLKEM768::DECAPS_KEY_SIZE);
}

/// Key and ciphertext serialization round-trips through the public parsers.
#[test]
fn public_serialization_roundtrip() {
    let keypair = DecapsulationKey::<MLKEM512>::generate_derand(&[1u8; 64]);

    let ek_bytes = keypair.encapsulation_key().to_bytes();
    let dk_bytes = keypair.to_bytes();

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
