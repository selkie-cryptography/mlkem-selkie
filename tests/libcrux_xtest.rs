//! Randomized cross-implementation interop tests against [libcrux] ML-KEM.
//!
//! libcrux is an independent, formally-verified ML-KEM implementation; agreeing
//! with it over many random seeds is strong evidence of FIPS 203 conformance
//! beyond the fixed KAT files. Unlike the BoringSSL interop (a C subprocess
//! oracle), libcrux is a Rust crate, so this is a direct in-process cross-test
//! that runs by default.
//!
//! Each iteration, for a random `d ‖ z` keygen seed and a random encapsulation
//! message `m`, checks three things:
//!
//! 1. **Keygen byte-identity**: both implementations derive the same `ek` and
//!    `dk` from the seed.
//! 2. **Their encaps, our decaps**: libcrux encapsulates under the shared `ek`;
//!    our `Decaps` recovers libcrux's shared secret.
//! 3. **Our encaps, their decaps**: we encapsulate `m`; libcrux's `Decaps`
//!    recovers our shared secret.
//!
//! Seeds come from our own exported AES-256-CTR-DRBG so the run is
//! deterministic and reproducible.
//!
//! [libcrux]: https://github.com/cryspen/libcrux

use mlkem_selkie::{Ciphertext, DecapsulationKey, MLKEM512, MLKEM768, MLKEM1024};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

/// Number of random interop iterations per parameter set.
///
/// 1000 random seeds per set (3000 iterations, 9000 directional cross-checks)
/// gives good coverage of input-dependent edge cases — compression boundaries
/// and near-modulus coefficients — while keeping the default run quick.
const ITERATIONS: usize = 1000;

/// Draws a fresh `(seed, message)` pair from the deterministic DRBG.
fn next_inputs(rng: &mut ChaCha8Rng) -> ([u8; 64], [u8; 32]) {
    let mut seed = [0u8; 64];
    let mut message = [0u8; 32];
    rng.fill_bytes(&mut seed);
    rng.fill_bytes(&mut message);

    (seed, message)
}

#[test]
fn interop_mlkem512() {
    let mut rng = ChaCha8Rng::from_seed([0x51; 32]);

    for i in 0..ITERATIONS {
        let (seed, message) = next_inputs(&mut rng);

        let ours = DecapsulationKey::<MLKEM512>::generate_derand(&seed);
        let theirs = libcrux_ml_kem::mlkem512::generate_key_pair(seed);

        assert_eq!(
            ours.encapsulation_key().to_bytes().as_ref(),
            theirs.public_key().as_slice(),
            "iter {i}: ek mismatch",
        );
        assert_eq!(
            ours.to_bytes().as_ref(),
            theirs.private_key().as_slice(),
            "iter {i}: dk mismatch",
        );

        let (their_ct, their_ss) =
            libcrux_ml_kem::mlkem512::encapsulate(theirs.public_key(), message);
        let our_ss =
            ours.decapsulate(&Ciphertext::<MLKEM512>::from_bytes(their_ct.as_slice()).unwrap());
        assert_eq!(
            our_ss.as_bytes(),
            &their_ss,
            "iter {i}: their-encaps/our-decaps"
        );

        let (our_ss2, our_ct) = ours.encapsulation_key().encapsulate_derand(&message);
        let their_ct2 = libcrux_ml_kem::mlkem512::MlKem512Ciphertext::from(
            <[u8; 768]>::try_from(our_ct.as_bytes()).unwrap(),
        );
        let their_ss2 = libcrux_ml_kem::mlkem512::decapsulate(theirs.private_key(), &their_ct2);
        assert_eq!(
            our_ss2.as_bytes(),
            &their_ss2,
            "iter {i}: our-encaps/their-decaps"
        );
    }
}

#[test]
fn interop_mlkem768() {
    let mut rng = ChaCha8Rng::from_seed([0x76; 32]);

    for i in 0..ITERATIONS {
        let (seed, message) = next_inputs(&mut rng);

        let ours = DecapsulationKey::<MLKEM768>::generate_derand(&seed);
        let theirs = libcrux_ml_kem::mlkem768::generate_key_pair(seed);

        assert_eq!(
            ours.encapsulation_key().to_bytes().as_ref(),
            theirs.public_key().as_slice(),
            "iter {i}: ek mismatch",
        );
        assert_eq!(
            ours.to_bytes().as_ref(),
            theirs.private_key().as_slice(),
            "iter {i}: dk mismatch",
        );

        let (their_ct, their_ss) =
            libcrux_ml_kem::mlkem768::encapsulate(theirs.public_key(), message);
        let our_ss =
            ours.decapsulate(&Ciphertext::<MLKEM768>::from_bytes(their_ct.as_slice()).unwrap());
        assert_eq!(
            our_ss.as_bytes(),
            &their_ss,
            "iter {i}: their-encaps/our-decaps"
        );

        let (our_ss2, our_ct) = ours.encapsulation_key().encapsulate_derand(&message);
        let their_ct2 = libcrux_ml_kem::mlkem768::MlKem768Ciphertext::from(
            <[u8; 1088]>::try_from(our_ct.as_bytes()).unwrap(),
        );
        let their_ss2 = libcrux_ml_kem::mlkem768::decapsulate(theirs.private_key(), &their_ct2);
        assert_eq!(
            our_ss2.as_bytes(),
            &their_ss2,
            "iter {i}: our-encaps/their-decaps"
        );
    }
}

#[test]
fn interop_mlkem1024() {
    let mut rng = ChaCha8Rng::from_seed([0x10; 32]);

    for i in 0..ITERATIONS {
        let (seed, message) = next_inputs(&mut rng);

        let ours = DecapsulationKey::<MLKEM1024>::generate_derand(&seed);
        let theirs = libcrux_ml_kem::mlkem1024::generate_key_pair(seed);

        assert_eq!(
            ours.encapsulation_key().to_bytes().as_ref(),
            theirs.public_key().as_slice(),
            "iter {i}: ek mismatch",
        );
        assert_eq!(
            ours.to_bytes().as_ref(),
            theirs.private_key().as_slice(),
            "iter {i}: dk mismatch",
        );

        let (their_ct, their_ss) =
            libcrux_ml_kem::mlkem1024::encapsulate(theirs.public_key(), message);
        let our_ss =
            ours.decapsulate(&Ciphertext::<MLKEM1024>::from_bytes(their_ct.as_slice()).unwrap());
        assert_eq!(
            our_ss.as_bytes(),
            &their_ss,
            "iter {i}: their-encaps/our-decaps"
        );

        let (our_ss2, our_ct) = ours.encapsulation_key().encapsulate_derand(&message);
        let their_ct2 = libcrux_ml_kem::mlkem1024::MlKem1024Ciphertext::from(
            <[u8; 1568]>::try_from(our_ct.as_bytes()).unwrap(),
        );
        let their_ss2 = libcrux_ml_kem::mlkem1024::decapsulate(theirs.private_key(), &their_ct2);
        assert_eq!(
            our_ss2.as_bytes(),
            &their_ss2,
            "iter {i}: our-encaps/their-decaps"
        );
    }
}
