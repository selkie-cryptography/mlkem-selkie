//! dudect constant-time t-test for the field arithmetic and decapsulation.
//!
//! `cargo bench --bench dudect --features expose-internals`. The pass threshold
//! is `|t| < 4.5`.
//!
//! NOTE: the field arithmetic is currently variable-time and decapsulation's
//! implicit-rejection check is a plain `==`, so these are *expected to fail*
//! today (the `TODO(ct)` markers). They are in place to validate the eventual
//! constant-time rework.
//!
//! Input randomness comes from our own `Aes256CtrDrbg` rather than dudect's
//! `BenchRng`, sidestepping a `rand` version mismatch between the two crates.

use std::hint::black_box;

use dudect_bencher::{BenchRng, Class, CtRunner, ctbench_main};
use mlkem_selkie::{Ciphertext, KeyPair, MLKEM512, algebraic::FieldElement, drbg::Aes256CtrDrbg};
use rand_core::RngCore;

/// A random field element.
fn random_fe(rng: &mut Aes256CtrDrbg) -> FieldElement {
    FieldElement::new(rng.next_u32() as u16)
}

/// A random class selector.
fn random_class(rng: &mut Aes256CtrDrbg) -> Class {
    if rng.next_u32() & 1 == 1 {
        Class::Left
    } else {
        Class::Right
    }
}

/// Field multiplication: zero (Left) vs random (Right) operands.
fn fe_mul(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = Aes256CtrDrbg::new(&[0x11; 48]);

    for _ in 0..100_000 {
        let class = random_class(&mut rng);
        let a = match class {
            Class::Left => FieldElement::ZERO,
            Class::Right => random_fe(&mut rng),
        };
        let b = random_fe(&mut rng);

        runner.run_one(class, || black_box(black_box(a) * black_box(b)));
    }
}

/// Field addition: zero (Left) vs random (Right) operands.
fn fe_add(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = Aes256CtrDrbg::new(&[0x22; 48]);

    for _ in 0..100_000 {
        let class = random_class(&mut rng);
        let a = match class {
            Class::Left => FieldElement::ZERO,
            Class::Right => random_fe(&mut rng),
        };
        let b = random_fe(&mut rng);

        runner.run_one(class, || black_box(black_box(a) + black_box(b)));
    }
}

/// Decapsulation: valid (Left) vs malleated (Right) ciphertext under a fixed
/// key — the implicit-rejection path must be constant-time.
fn decaps(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = Aes256CtrDrbg::new(&[0x33; 48]);

    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();
    let mut malleated = valid.clone();
    malleated[0] ^= 1;
    let decapsulation_key = keypair.decapsulation_key;

    for _ in 0..2_000 {
        let class = random_class(&mut rng);
        let bytes = match class {
            Class::Left => &valid,
            Class::Right => &malleated,
        };
        let ciphertext = Ciphertext::<MLKEM512>::from_bytes(bytes).expect("valid length");

        runner.run_one(class, || {
            black_box(decapsulation_key.decapsulate(black_box(&ciphertext)))
        });
    }
}

ctbench_main!(fe_mul, fe_add, decaps);
