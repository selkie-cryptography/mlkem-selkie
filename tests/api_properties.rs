//! Property-based tests of the ML-KEM public API.
//!
//! Complements the fixed KAT / Wycheproof vectors with randomized round-trips
//! across all three parameter sets.

use mlkem_selkie::{
    Ciphertext, DecapsulationKey, EncapsulationKey, KeyPair, MLKEM512, MLKEM768, MLKEM1024,
    ParameterSet,
};
use proptest::{array::uniform32, prelude::*};

/// Reassembles a 64-byte `d ‖ z` keygen seed from two 32-byte halves (proptest
/// derives `Arbitrary` for `[u8; 32]`, not `[u8; 64]`).
fn seed(d: [u8; 32], z: [u8; 32]) -> [u8; 64] {
    let mut seed = [0u8; 64];
    seed[..32].copy_from_slice(&d);
    seed[32..].copy_from_slice(&z);

    seed
}

/// Encaps then Decaps agree, and every serialization round-trips, for `P`.
fn check_roundtrip<P: ParameterSet>(seed: &[u8; 64], message: &[u8; 32]) {
    let keypair = KeyPair::<P>::generate_derand(seed);

    let (sender, ciphertext) = keypair.encapsulation_key.encapsulate_derand(message);
    let receiver = keypair.decapsulation_key.decapsulate(&ciphertext);
    assert_eq!(sender.as_bytes(), receiver.as_bytes());

    let ek_bytes = keypair.encapsulation_key.to_bytes();
    let ek = EncapsulationKey::<P>::from_bytes(&ek_bytes).expect("valid ek");
    assert_eq!(ek.to_bytes(), ek_bytes);

    let dk_bytes = keypair.decapsulation_key.to_bytes();
    let dk = DecapsulationKey::<P>::from_bytes(&dk_bytes).expect("valid dk");
    assert_eq!(dk.to_bytes(), dk_bytes);

    let parsed = Ciphertext::<P>::from_bytes(ciphertext.as_bytes()).expect("valid ct");
    assert_eq!(parsed.as_bytes(), ciphertext.as_bytes());
}

proptest! {
    /// Encaps/decaps correctness and serialization round-trips hold for every
    /// parameter set and every seed/message.
    #[test]
    fn roundtrip(d in uniform32(any::<u8>()), z in uniform32(any::<u8>()), m in uniform32(any::<u8>())) {
        let seed = seed(d, z);

        check_roundtrip::<MLKEM512>(&seed, &m);
        check_roundtrip::<MLKEM768>(&seed, &m);
        check_roundtrip::<MLKEM1024>(&seed, &m);
    }

    /// Decapsulation is a deterministic function of `(dk, c)` — re-running it on
    /// a malleated ciphertext yields the same implicit-rejection secret.
    #[test]
    fn decaps_is_deterministic(
        d in uniform32(any::<u8>()),
        z in uniform32(any::<u8>()),
        m in uniform32(any::<u8>()),
        index in 0usize..(32 * (10 * 2 + 4)),
        bit in 0u8..8,
    ) {
        let keypair = KeyPair::<MLKEM512>::generate_derand(&seed(d, z));
        let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&m);

        let mut bytes = ciphertext.as_bytes().to_vec();
        bytes[index] ^= 1 << bit;
        let malleated = Ciphertext::<MLKEM512>::from_bytes(&bytes).expect("same-length ct");

        let first = keypair.decapsulation_key.decapsulate(&malleated);
        let second = keypair.decapsulation_key.decapsulate(&malleated);

        prop_assert_eq!(first.as_bytes(), second.as_bytes());
    }

    /// A ciphertext of any wrong length is rejected by the parser.
    #[test]
    fn wrong_length_ciphertext_rejected(len in 0usize..2048) {
        prop_assume!(len != MLKEM512::CIPHERTEXT_SIZE);

        prop_assert!(Ciphertext::<MLKEM512>::from_bytes(&vec![0u8; len]).is_err());
    }
}
