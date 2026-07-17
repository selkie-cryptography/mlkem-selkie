//! Wall-clock benchmarks for the ML-KEM public API.
//!
//! `cargo bench --bench mlkem` (add `-- keygen` etc. to filter). Each benchmark
//! is instantiated for all three parameter sets via divan's `types`.

use divan::{Bencher, black_box};
use mlkem_selkie::{
    Ciphertext, DecapsulationKey, EncapsulationKey, KeyPair, MLKEM512, MLKEM768, MLKEM1024,
    ParameterSet,
};

fn main() {
    divan::main();
}

/// A fixed `d ‖ z` key-generation seed.
const SEED: [u8; 64] = [0x42; 64];

/// A fixed encapsulation message.
const MESSAGE: [u8; 32] = [0x17; 32];

/// `ML-KEM.KeyGen_internal` from a fixed seed — no OS randomness. Matches
/// the primary-comparison bench in libcrux / graviola.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024], sample_count = 100)]
fn keygen_derand<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    bencher
        .counter(1u32)
        .bench(|| KeyPair::<P>::generate_derand(black_box(&SEED)));
}

/// `ML-KEM.KeyGen` with OS entropy — [`keygen_derand`] plus `getrandom`.
/// The delta against [`keygen_derand`] is the getrandom overhead.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024], sample_count = 100)]
fn keygen_rand<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    bencher.counter(1u32).bench(|| KeyPair::<P>::generate());
}

/// `ML-KEM.Encaps_internal` under a fixed key from a fixed message.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn encaps_derand<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let keypair = KeyPair::<P>::generate_derand(&SEED);

    bencher.counter(1u32).bench(|| {
        keypair
            .encapsulation_key
            .encapsulate_derand(black_box(&MESSAGE))
    });
}

/// `ML-KEM.Encaps` with OS entropy — [`encaps_derand`] plus `getrandom` for
/// the fresh message.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn encaps_rand<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let keypair = KeyPair::<P>::generate_derand(&SEED);

    bencher
        .counter(1u32)
        .bench(|| keypair.encapsulation_key.encapsulate());
}

/// `ML-KEM.Decaps` of a valid ciphertext (the common, no-rejection path).
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn decaps<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let keypair = KeyPair::<P>::generate_derand(&SEED);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&MESSAGE);

    bencher.counter(1u32).bench(|| {
        keypair
            .decapsulation_key
            .decapsulate(black_box(&ciphertext))
    });
}

/// `EncapsulationKey::to_bytes` — pack the `t_hat` NTT-domain key and `rho`
/// seed into `384*K + 32` bytes.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn serialize_encapsulation_key<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let ek = KeyPair::<P>::generate_derand(&SEED).encapsulation_key;

    bencher.counter(1u32).bench(|| black_box(&ek).to_bytes());
}

/// `DecapsulationKey::to_bytes` — pack `dk_PKE ‖ ek ‖ H(ek) ‖ z` into
/// `768*K + 96` bytes.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn serialize_decapsulation_key<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let dk = KeyPair::<P>::generate_derand(&SEED).decapsulation_key;

    bencher.counter(1u32).bench(|| black_box(&dk).to_bytes());
}

/// `EncapsulationKey::from_bytes`, including the section 7.2 modulus check.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn parse_encapsulation_key<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let bytes = KeyPair::<P>::generate_derand(&SEED)
        .encapsulation_key
        .to_bytes();

    bencher
        .counter(1u32)
        .bench(|| EncapsulationKey::<P>::from_bytes(black_box(bytes.as_ref())));
}

/// `DecapsulationKey::from_bytes`, including the section 7.3 `H(ek)`
/// consistency check.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn parse_decapsulation_key<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    // to_vec: divan's closure captures need `Sync`, which the
    // `P::DecapsKeySerialization` associated type doesn't itself guarantee.
    let bytes: Vec<u8> = KeyPair::<P>::generate_derand(&SEED)
        .decapsulation_key
        .to_bytes()
        .as_ref()
        .to_vec();

    bencher
        .counter(1u32)
        .bench(|| DecapsulationKey::<P>::from_bytes(black_box(&bytes)));
}

/// `Ciphertext::from_bytes` length validation.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn parse_ciphertext<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let keypair = KeyPair::<P>::generate_derand(&SEED);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&MESSAGE);
    let bytes = ciphertext.as_bytes().to_vec();

    bencher
        .counter(1u32)
        .bench(|| Ciphertext::<P>::from_bytes(black_box(&bytes)));
}

/// `SharedSecret::as_bytes` — the accessor an HKDF/keyschedule caller uses
/// to consume the KEM output. Effectively a struct-field ref-return, but
/// worth pinning as a baseline so its cost is visible in the roundtrip
/// decomposition.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024])]
fn shared_secret_as_bytes<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    let keypair = KeyPair::<P>::generate_derand(&SEED);
    let (shared_secret, _) = keypair.encapsulation_key.encapsulate_derand(&MESSAGE);

    bencher
        .counter(1u32)
        .bench(|| black_box(&shared_secret).as_bytes());
}

/// Combined one-shot KEM: `generate` + `ek.as_bytes` + `ek.from_bytes` +
/// `encapsulate` + `decapsulate` (with the `Ciphertext` passed directly, no
/// wire serialize/parse). Matches the scope of graviola's `mlkem768-combined`
/// bench for a direct KEM/sec head-to-head. Pair with [`roundtrip`] for the
/// wire-including view.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024], sample_count = 100)]
fn combined<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    bencher.counter(1u32).bench(|| {
        let keypair = KeyPair::<P>::generate();

        let ek_wire = keypair.encapsulation_key.to_bytes();
        let ek = EncapsulationKey::<P>::from_bytes(black_box(ek_wire.as_ref()))
            .expect("re-parsed key must be valid");

        let (_ss_bob, ciphertext) = ek.encapsulate();

        black_box(
            keypair
                .decapsulation_key
                .decapsulate(black_box(&ciphertext)),
        )
    });
}

/// Full TLS 1.3-style KEM roundtrip per iteration:
///
/// 1. Alice: `KeyPair::generate()` (OS entropy).
/// 2. Alice → wire: `EncapsulationKey::to_bytes`.
/// 3. Bob: `EncapsulationKey::from_bytes`.
/// 4. Bob: `encapsulate` (OS entropy).
/// 5. Bob → wire: `Ciphertext::as_bytes` copy.
/// 6. Alice: `Ciphertext::from_bytes`.
/// 7. Alice: `decapsulate`.
///
/// The reported time is what a single hybrid-KEM handshake session
/// costs end-to-end. Comparable to graviola / libcrux full-handshake
/// benches.
#[divan::bench(types = [MLKEM512, MLKEM768, MLKEM1024], sample_count = 100)]
fn roundtrip<P: ParameterSet>(bencher: Bencher<'_, '_>) {
    bencher.counter(1u32).bench(|| {
        let keypair = KeyPair::<P>::generate();

        let ek_wire = keypair.encapsulation_key.to_bytes();
        let ek = EncapsulationKey::<P>::from_bytes(black_box(ek_wire.as_ref()))
            .expect("re-parsed key must be valid");

        let (ss_bob, ciphertext) = ek.encapsulate();

        let ct_wire: Vec<u8> = ciphertext.as_bytes().to_vec();
        let ct = Ciphertext::<P>::from_bytes(black_box(&ct_wire))
            .expect("re-parsed ciphertext must be valid");

        let ss_alice = keypair.decapsulation_key.decapsulate(&ct);

        // Consume the SS bytes as an HKDF-like caller would, so the
        // compiler can't DCE the chain. Explicit `drop` folds the
        // `ZeroizeOnDrop` cost into the timing region (returning the
        // tuple would let divan drop it after the timer stops).
        black_box(ss_alice.as_bytes());
        black_box(ss_bob.as_bytes());
        drop(ss_alice);
        drop(ss_bob);
    });
}
