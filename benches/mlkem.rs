//! Wall-clock benchmarks for the ML-KEM public API.
//!
//! `cargo bench --bench mlkem` (add `-- keygen` etc. to filter). Each benchmark
//! is instantiated for all three parameter sets via divan's `types`.

use divan::{Bencher, black_box};
use mlkem_selkie::{
    Ciphertext, EncapsulationKey, KeyPair, MLKEM512, MLKEM768, MLKEM1024, ParameterSet,
};

fn main() {
    divan::main();
}

/// A fixed `d ‖ z` key-generation seed.
const SEED: [u8; 64] = [0x42; 64];

/// A fixed encapsulation message.
const MESSAGE: [u8; 32] = [0x17; 32];

/// `ML-KEM.KeyGen_internal` from a fixed seed.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024], sample_count = 100)]
fn keygen<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    bencher.bench(|| KeyPair::<P>::generate_derand(black_box(&SEED)));
}

/// `ML-KEM.Encaps_internal` under a freshly generated key.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn encaps<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let keypair = KeyPair::<P>::generate_derand(&SEED);

    bencher.bench(|| {
        keypair
            .encapsulation_key
            .encapsulate_derand(black_box(&MESSAGE))
    });
}

/// `ML-KEM.Decaps` of a valid ciphertext (the common, no-rejection path).
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn decaps<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let keypair = KeyPair::<P>::generate_derand(&SEED);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&MESSAGE);

    bencher.bench(|| {
        keypair
            .decapsulation_key
            .decapsulate(black_box(&ciphertext))
    });
}

/// `EncapsulationKey::from_bytes`, including the section 7.2 modulus check.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn parse_encapsulation_key<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let bytes = KeyPair::<P>::generate_derand(&SEED)
        .encapsulation_key
        .to_bytes();

    bencher.bench(|| EncapsulationKey::<P>::from_bytes(black_box(&bytes)));
}

/// `Ciphertext::from_bytes` length validation.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn parse_ciphertext<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let keypair = KeyPair::<P>::generate_derand(&SEED);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&MESSAGE);
    let bytes = ciphertext.as_bytes().to_vec();

    bencher.bench(|| Ciphertext::<P>::from_bytes(black_box(&bytes)));
}
