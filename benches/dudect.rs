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
use mlkem_selkie::{Ciphertext, KeyPair, MLKEM512, PKE, ParameterSet};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use subtle::{ConditionallySelectable, ConstantTimeEq};

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
    let mut rng = ChaCha8Rng::from_seed([0x33; 32]);

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

    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();
    let decapsulation_key = keypair.decapsulation_key;

    for _ in 0..100_000 {
        let class = random_class(&mut rng);
        let ciphertext = Ciphertext::<MLKEM512>::from_bytes(&valid).expect("valid length");

        runner.run_one(class, || {
            black_box(decapsulation_key.decapsulate(black_box(&ciphertext)))
        });
    }
}

/// Isolated `K-PKE.Decrypt`: valid (Left) vs malleated (Right) ciphertext under
/// a fixed key, with no re-encryption, no FO comparison, and no
/// `conditional_select` on top. Pairs with [`decaps`]:
///
/// - If `|t|` here matches `decaps` (~9), the leak sits inside K-PKE.Decrypt
///   proper (`ByteDecode` / `Decompress` / `s_hat . NTT(u)` /
///   `compress_message`).
/// - If `|t|` drops toward the null (~3), the leak is in the re-encrypt path,
///   the `ct_eq` comparison, or emerges from state carried across those steps.
///
/// The isolated encrypt case is already exercised by [`encaps`] (fixed vs
/// random `m` produces the same class-dependent internal state that a decaps
/// re-encryption would), so no separate `pke_encrypt` bench is added.
fn pke_decrypt(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = ChaCha8Rng::from_seed([0x77; 32]);

    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();
    let mut malleated = valid.clone();
    malleated[0] ^= 1;

    let dk_bytes = keypair.decapsulation_key.to_bytes();
    let dk_pke = PKE::DecryptionKey::<MLKEM512>::from_bytes(
        &dk_bytes.as_ref()[..MLKEM512::PKE_DECRYPTION_KEY_SIZE],
    );

    for _ in 0..100_000 {
        let class = random_class(&mut rng);
        let bytes = match class {
            Class::Left => &valid,
            Class::Right => &malleated,
        };
        let ciphertext = PKE::Ciphertext::<MLKEM512>::from_bytes(bytes);

        runner.run_one(class, || black_box(dk_pke.decrypt(black_box(&ciphertext))));
    }
}

/// `K-PKE.Decrypt` immediately followed by `K-PKE.Encrypt` inside one timed
/// closure, using the recovered `m'` (class-dependent) and a fixed `r`. Runs
/// the same code both PKE halves' isolated benches ran, but *in sequence*, so
/// any cache / branch-predictor state seeded by `decrypt` colours the
/// subsequent `encrypt`. Catches a leak that only appears when the two
/// halves are executed back-to-back on the same call — the diagnostic gap
/// [`pke_decrypt`] and [`encaps`] cannot fill on their own.
///
/// Diverges from real decaps by omitting `G(m' || h_ek)` for the randomness
/// derivation: `r` is fixed across classes so the timed closure exercises
/// only the two PKE calls and the cache handoff between them, not SHAKE.
fn decrypt_then_encrypt(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = ChaCha8Rng::from_seed([0xAA; 32]);

    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();
    let mut malleated = valid.clone();
    malleated[0] ^= 1;

    let dk_bytes = keypair.decapsulation_key.to_bytes();
    let dk_pke_end = MLKEM512::PKE_DECRYPTION_KEY_SIZE;
    let ek_pke_end = dk_pke_end + MLKEM512::PKE_ENCRYPTION_KEY_SIZE;
    let dk_pke =
        PKE::DecryptionKey::<MLKEM512>::from_bytes(&dk_bytes.as_ref()[..dk_pke_end]);
    let ek_pke =
        PKE::EncryptionKey::<MLKEM512>::from_bytes(&dk_bytes.as_ref()[dk_pke_end..ek_pke_end]);
    let fixed_r = [0u8; 32];

    for _ in 0..100_000 {
        let class = random_class(&mut rng);
        let bytes = match class {
            Class::Left => &valid,
            Class::Right => &malleated,
        };
        let ct = PKE::Ciphertext::<MLKEM512>::from_bytes(bytes);

        runner.run_one(class, || {
            let m_prime = dk_pke.decrypt(black_box(&ct));
            black_box(ek_pke.encrypt(black_box(&m_prime), black_box(&fixed_r)))
        });
    }
}

/// The Fujisaki–Okamoto tail alone: `ct_eq(c, c') + conditional_select(K_bar,
/// K', matches)`. `c` and `c'` are pre-computed per class so the timed closure
/// exercises only the `subtle` primitives — no NTT, no compress, no SHAKE.
/// The Left class has `c == c'` (valid ciphertext round-trips exactly); the
/// Right class has `c != c'` (malleated bytes cause re-encryption divergence).
/// If this bench alone reproduces the decaps signal, the leak is in `subtle`'s
/// slice comparison or byte-select loop on our compiler / µarch. If it sits
/// at the null floor, the leak is elsewhere in decaps.
fn fo_tail_only(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = ChaCha8Rng::from_seed([0xBB; 32]);

    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();
    let mut malleated = valid.clone();
    malleated[0] ^= 1;

    let dk_bytes = keypair.decapsulation_key.to_bytes();
    let dk_pke_end = MLKEM512::PKE_DECRYPTION_KEY_SIZE;
    let ek_pke_end = dk_pke_end + MLKEM512::PKE_ENCRYPTION_KEY_SIZE;
    let dk_pke =
        PKE::DecryptionKey::<MLKEM512>::from_bytes(&dk_bytes.as_ref()[..dk_pke_end]);
    let ek_pke =
        PKE::EncryptionKey::<MLKEM512>::from_bytes(&dk_bytes.as_ref()[dk_pke_end..ek_pke_end]);
    let fixed_r = [0u8; 32];

    // Pre-compute the (c, c') pair for each class outside the timed region.
    let ct_valid = PKE::Ciphertext::<MLKEM512>::from_bytes(&valid);
    let ct_malleated = PKE::Ciphertext::<MLKEM512>::from_bytes(&malleated);
    let m_valid = dk_pke.decrypt(&ct_valid);
    let m_malleated = dk_pke.decrypt(&ct_malleated);
    let cprime_valid = ek_pke.encrypt(&m_valid, &fixed_r).as_bytes().to_vec();
    let cprime_malleated = ek_pke.encrypt(&m_malleated, &fixed_r).as_bytes().to_vec();

    // Fake `K'` / `K_bar` — the select never branches, so their contents don't
    // affect timing; distinct values guard against the optimizer eliding the loop.
    let k_prime = [0xEEu8; 32];
    let k_bar = [0xFFu8; 32];

    for _ in 0..100_000 {
        let class = random_class(&mut rng);
        let (ct_bytes, cprime_bytes) = match class {
            Class::Left => (&valid, &cprime_valid),
            Class::Right => (&malleated, &cprime_malleated),
        };

        runner.run_one(class, || {
            let matches = black_box(ct_bytes.as_slice())
                .ct_eq(black_box(cprime_bytes.as_slice()));
            let mut secret = [0u8; 32];
            for (out, (kp, kb)) in secret.iter_mut().zip(k_prime.iter().zip(k_bar.iter())) {
                *out = u8::conditional_select(kb, kp, matches);
            }
            black_box(secret)
        });
    }
}

/// Null for [`decrypt_then_encrypt`]: byte-identical ciphertext for both
/// classes. Same `m'`, same `c'`, same decrypt→encrypt work. Any signal
/// here is runner noise; the paired diagnostic against
/// `decrypt_then_encrypt` catches a leak whose median `|t|` would slip
/// under the standalone 5.0 threshold (as decaps' 3.35 did).
fn decrypt_then_encrypt_null(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = ChaCha8Rng::from_seed([0xAA; 32]);

    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();

    let dk_bytes = keypair.decapsulation_key.to_bytes();
    let dk_pke_end = MLKEM512::PKE_DECRYPTION_KEY_SIZE;
    let ek_pke_end = dk_pke_end + MLKEM512::PKE_ENCRYPTION_KEY_SIZE;
    let dk_pke =
        PKE::DecryptionKey::<MLKEM512>::from_bytes(&dk_bytes.as_ref()[..dk_pke_end]);
    let ek_pke =
        PKE::EncryptionKey::<MLKEM512>::from_bytes(&dk_bytes.as_ref()[dk_pke_end..ek_pke_end]);
    let fixed_r = [0u8; 32];

    for _ in 0..100_000 {
        let class = random_class(&mut rng);
        let ct = PKE::Ciphertext::<MLKEM512>::from_bytes(&valid);

        runner.run_one(class, || {
            let m_prime = dk_pke.decrypt(black_box(&ct));
            black_box(ek_pke.encrypt(black_box(&m_prime), black_box(&fixed_r)))
        });
    }
}

/// Null for [`fo_tail_only`]: byte-identical `(c, c')` pair for both classes.
/// Same `ct_eq` input, same select input. Any signal is runner noise.
fn fo_tail_only_null(runner: &mut CtRunner, _rng: &mut BenchRng) {
    let mut rng = ChaCha8Rng::from_seed([0xBB; 32]);

    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();

    let dk_bytes = keypair.decapsulation_key.to_bytes();
    let dk_pke_end = MLKEM512::PKE_DECRYPTION_KEY_SIZE;
    let ek_pke_end = dk_pke_end + MLKEM512::PKE_ENCRYPTION_KEY_SIZE;
    let dk_pke =
        PKE::DecryptionKey::<MLKEM512>::from_bytes(&dk_bytes.as_ref()[..dk_pke_end]);
    let ek_pke =
        PKE::EncryptionKey::<MLKEM512>::from_bytes(&dk_bytes.as_ref()[dk_pke_end..ek_pke_end]);
    let fixed_r = [0u8; 32];

    let ct = PKE::Ciphertext::<MLKEM512>::from_bytes(&valid);
    let m_valid = dk_pke.decrypt(&ct);
    let cprime = ek_pke.encrypt(&m_valid, &fixed_r).as_bytes().to_vec();

    let k_prime = [0xEEu8; 32];
    let k_bar = [0xFFu8; 32];

    for _ in 0..100_000 {
        let class = random_class(&mut rng);

        runner.run_one(class, || {
            let matches =
                black_box(valid.as_slice()).ct_eq(black_box(cprime.as_slice()));
            let mut secret = [0u8; 32];
            for (out, (kp, kb)) in secret.iter_mut().zip(k_prime.iter().zip(k_bar.iter())) {
                *out = u8::conditional_select(kb, kp, matches);
            }
            black_box(secret)
        });
    }
}

ctbench_main!(
    encaps,
    decaps,
    decaps_null,
    pke_decrypt,
    decrypt_then_encrypt,
    decrypt_then_encrypt_null,
    fo_tail_only,
    fo_tail_only_null
);
