//! dudect constant-time t-test for encapsulation and decapsulation.
//!
//! `cargo bench --bench dudect --features expose-internals`. The pass threshold
//! is `|t| < 5.0`. Each bench takes 100k samples: at small counts (~2k) the
//! max-t statistic is dominated by scheduler noise on shared CI runners and
//! produces spurious excursions past the threshold. CI runs the binary three
//! times and gates on the per-bench median |t|, since even at 100k samples a
//! single reading on a shared runner can spike on a true null.
//!
//! Key generation is intentionally not covered here: `SampleNTT`'s rejection-
//! sampling iteration count varies with the (public) `rho`, which dominates
//! the t-statistic without representing a real CT bug. Keygen secret-
//! independence is checked by `tests/ctgrind.rs::keygen_secret_independent`
//! under Valgrind taint tracking instead, which ignores rejection-loop
//! timing and flags only secret-dependent branches.
//!
//! Input randomness comes from our own `Aes256CtrDrbg` rather than dudect's
//! `BenchRng`, sidestepping a `rand` version mismatch between the two crates.

use std::hint::black_box;

use dudect_bencher::{BenchRng, Class, CtRunner, ctbench_main};
use mlkem_selkie::{Ciphertext, KeyPair, MLKEM512, drbg::Aes256CtrDrbg};
use rand_core::RngCore;

/// A random class selector.
fn random_class(rng: &mut Aes256CtrDrbg) -> Class {
    if rng.next_u32() & 1 == 1 {
        Class::Left
    } else {
        Class::Right
    }
}

/// Encapsulation: fixed encaps randomness `m` (Left) vs random `m` (Right)
/// under a fixed encapsulation key. `m` is the secret input from which the
/// shared secret and ciphertext are derived; a CT implementation must show
/// no `m`-dependent timing variance.
fn encaps(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = Aes256CtrDrbg::new(&[0x55; 48]);
    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let encapsulation_key = keypair.encapsulation_key;
    let fixed_m = [0x55u8; 32];

    for _ in 0..100_000 {
        let class = random_class(&mut rng);
        let mut random_m = [0u8; 32];
        rng.fill_bytes(&mut random_m);
        let m = match class {
            Class::Left => &fixed_m,
            Class::Right => &random_m,
        };

        runner.run_one(class, || {
            black_box(encapsulation_key.encapsulate_derand(black_box(m)))
        });
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

    for _ in 0..100_000 {
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

ctbench_main!(encaps, decaps);
