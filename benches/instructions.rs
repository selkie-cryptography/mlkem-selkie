//! Deterministic callgrind instruction counts via gungraun.
//!
//! `cargo bench --bench instructions --features expose-internals` (needs
//! Valgrind + the matching `gungraun-runner`). Operands are built outside the
//! measured region by the `#[bench::case]` setup expressions, so only the
//! operation itself is counted.

use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};
use mlkem_selkie::{EncapsulationKey, KeyPair, MLKEM768, algebraic::FieldElement};

/// A fixed `d ‖ z` key-generation seed.
const SEED: [u8; 64] = [0x42; 64];

/// A fixed encapsulation message.
const MESSAGE: [u8; 32] = [0x17; 32];

#[library_benchmark]
#[bench::case(FieldElement::new(1234), FieldElement::new(5678))]
fn fe_mul(a: FieldElement, b: FieldElement) -> FieldElement {
    black_box(a) * black_box(b)
}

#[library_benchmark]
#[bench::case(FieldElement::new(1234), FieldElement::new(5678))]
fn fe_add(a: FieldElement, b: FieldElement) -> FieldElement {
    black_box(a) + black_box(b)
}

/// Builds the ML-KEM-768 encapsulation key used by the `encaps` bench.
fn encapsulation_key() -> EncapsulationKey<MLKEM768> {
    KeyPair::<MLKEM768>::generate_derand(&SEED).encapsulation_key
}

/// Builds an ML-KEM-768 key pair and a valid ciphertext for the `decaps` bench.
fn keypair_and_ciphertext() -> (KeyPair<MLKEM768>, mlkem_selkie::Ciphertext<MLKEM768>) {
    let keypair = KeyPair::<MLKEM768>::generate_derand(&SEED);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&MESSAGE);

    (keypair, ciphertext)
}

#[library_benchmark]
fn keygen() -> KeyPair<MLKEM768> {
    KeyPair::<MLKEM768>::generate_derand(black_box(&SEED))
}

#[library_benchmark]
#[bench::case(encapsulation_key())]
fn encaps(ek: EncapsulationKey<MLKEM768>) {
    black_box(ek.encapsulate_derand(black_box(&MESSAGE)));
}

#[library_benchmark]
#[bench::case(keypair_and_ciphertext())]
fn decaps(input: (KeyPair<MLKEM768>, mlkem_selkie::Ciphertext<MLKEM768>)) {
    let (keypair, ciphertext) = input;
    black_box(
        keypair
            .decapsulation_key
            .decapsulate(black_box(&ciphertext)),
    );
}

library_benchmark_group!(name = field; benchmarks = fe_mul, fe_add);
library_benchmark_group!(name = mlkem; benchmarks = keygen, encaps, decaps);

main!(library_benchmark_groups = field, mlkem);
