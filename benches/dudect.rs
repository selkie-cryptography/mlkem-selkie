//! dudect constant-time t-test for encapsulation and decapsulation.
//!
//! `cargo bench --bench dudect --features expose-internals`. The pass threshold
//! is `|t| < 5.0`. Each bench takes 100k samples: at small counts (~2k) the
//! max-t statistic is dominated by scheduler noise on shared CI runners and
//! produces spurious excursions past the threshold. CI runs the binary five
//! times and gates on the per-bench median |t|, since even at 100k samples a
//! single reading on a shared runner can spike on a true null.
//!
//! [`decaps_null`] pairs with [`decaps`] as a runner-noise diagnostic: both
//! classes of input are byte-identical, so any non-trivial `|t|` measures
//! the runner's noise floor, not a real leak.
//!
//! Key generation is intentionally not covered here: `SampleNTT`'s rejection-
//! sampling iteration count varies with the (public) `rho`, which dominates
//! the t-statistic without representing a real CT bug. Keygen secret-
//! independence is checked by `tests/ctgrind.rs::keygen_secret_independent`
//! under Valgrind taint tracking instead, which ignores rejection-loop
//! timing and flags only secret-dependent branches.
//!
//! Input randomness comes from `ChaCha8Rng` rather than dudect's `BenchRng`,
//! sidestepping a `rand` version mismatch between the two crates.

use std::hint::black_box;

use dudect_bencher::{BenchRng, Class, CtRunner, ctbench_main};
use mlkem_selkie::{Ciphertext, DecapsulationKey, MLKEM512};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

/// A random class selector.
fn random_class(rng: &mut ChaCha8Rng) -> Class {
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
    let mut rng = ChaCha8Rng::from_seed([0x55; 32]);
    let keypair = DecapsulationKey::<MLKEM512>::generate_derand(&[0x42; 64]);
    let encapsulation_key = keypair.encapsulation_key();
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

/// Null for [`encaps`]: byte-identical `m` for both classes. Any signal is
/// runner noise; the paired sign test against `encaps` cancels it, so a
/// leak has to raise `encaps` above the noise floor to fire the gate.
fn encaps_null(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = ChaCha8Rng::from_seed([0x55; 32]);
    let keypair = DecapsulationKey::<MLKEM512>::generate_derand(&[0x42; 64]);
    let encapsulation_key = keypair.encapsulation_key().clone();
    let fixed_m = [0x55u8; 32];

    for _ in 0..100_000 {
        let class = random_class(&mut rng);

        runner.run_one(class, || {
            black_box(encapsulation_key.encapsulate_derand(black_box(&fixed_m)))
        });
    }
}

/// Decapsulation: valid (Left) vs malleated (Right) ciphertext under a fixed
/// key — the implicit-rejection path must be constant-time.
fn decaps(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = ChaCha8Rng::from_seed([0x33; 32]);

    let keypair = DecapsulationKey::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key().encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();
    let mut malleated = valid.clone();
    malleated[0] ^= 1;
    let decapsulation_key = keypair;

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

/// Null test: same ciphertext for both classes. Any non-trivial `|t|` here
/// is pure host noise — there's no possible information leak because the
/// two "classes" of input are byte-identical. Feeds a paired diagnostic
/// against the real [`decaps`] bench: if `|t|` on `decaps_null` matches
/// or exceeds `|t|` on `decaps` on the same runner, the shared-vCPU noise
/// floor is above the gate threshold and the `decaps` reading is not
/// distinguishable from noise. If `decaps_null` sits near zero while
/// `decaps` reads high, the `decaps` signal is real.
fn decaps_null(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = ChaCha8Rng::from_seed([0x33; 32]);

    let keypair = DecapsulationKey::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key().encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();
    let decapsulation_key = keypair;

    for _ in 0..100_000 {
        let class = random_class(&mut rng);
        let ciphertext = Ciphertext::<MLKEM512>::from_bytes(&valid).expect("valid length");

        runner.run_one(class, || {
            black_box(decapsulation_key.decapsulate(black_box(&ciphertext)))
        });
    }
}

ctbench_main!(encaps, encaps_null, decaps, decaps_null);
