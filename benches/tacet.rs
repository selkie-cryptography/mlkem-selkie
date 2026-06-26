//! Tacet constant-time analysis with attacker-model-aware exploitability.
//!
//! `cargo bench --bench tacet --features expose-internals`.
//!
//! NOTE: the field arithmetic is currently variable-time (the `% q` reductions;
//! see the `TODO(ct)` markers), and decapsulation's implicit-rejection
//! comparison is a plain `==`, so these checks are *expected to report leakage*
//! today. They are in place so the constant-time rework can be validated
//! against them. Inputs are byte arrays (which impl `Hash`, as `InputPair`
//! requires) and converted inside the closure so both classes pay the same
//! conversion cost.

use mlkem_selkie::{Ciphertext, KeyPair, MLKEM512, algebraic::FieldElement};
use rand_core::RngCore;
use tacet::{AttackerModel, Outcome, TimingOracle, helpers::InputPair};

/// Draws `N` random bytes from the OS RNG.
fn random_bytes<const N: usize>() -> [u8; N] {
    let mut buf = [0u8; N];
    rand_core::OsRng.fill_bytes(&mut buf);

    buf
}

/// A field element from the first two bytes of a slice.
fn fe(bytes: &[u8]) -> FieldElement {
    FieldElement::new(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// The attacker models to evaluate each operation under.
const MODELS: &[(&str, AttackerModel)] = &[
    ("shared_hw", AttackerModel::SharedHardware),
    ("pq_sentinel", AttackerModel::PostQuantumSentinel),
    ("adjacent", AttackerModel::AdjacentNetwork),
];

/// Prints a one-line verdict for an operation under a model.
fn report(name: &str, model_name: &str, outcome: &Outcome) {
    match outcome {
        Outcome::Pass {
            leak_probability, ..
        } => println!("PASS  {name:<14} [{model_name:<12}] leak_prob={leak_probability:.4}"),
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => println!(
            "FAIL  {name:<14} [{model_name:<12}] leak_prob={leak_probability:.4} exploit={exploitability:?}"
        ),
        Outcome::Inconclusive { reason, .. } => {
            println!("SKIP  {name:<14} [{model_name:<12}] inconclusive: {reason:?}")
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            println!("SKIP  {name:<14} [{model_name:<12}] unmeasurable: {recommendation}")
        }
        _ => println!("????  {name:<14} [{model_name:<12}]"),
    }
}

fn main() {
    println!("tacet constant-time analysis");
    println!("============================\n");

    // Field multiplication: zero vs random operands.
    for &(model_name, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(|| [0u8; 4], random_bytes::<4>),
            |bytes| {
                let _ = std::hint::black_box(fe(&bytes[..2]) * fe(&bytes[2..]));
            },
        );
        report("fe_mul", model_name, &outcome);
    }

    // Field addition: zero vs random operands.
    for &(model_name, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(|| [0u8; 4], random_bytes::<4>),
            |bytes| {
                let _ = std::hint::black_box(fe(&bytes[..2]) + fe(&bytes[2..]));
            },
        );
        report("fe_add", model_name, &outcome);
    }

    // Field subtraction: equal operands (result 0) vs random.
    for &(model_name, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(
                || {
                    let a: [u8; 2] = random_bytes();
                    [a[0], a[1], a[0], a[1]]
                },
                random_bytes::<4>,
            ),
            |bytes| {
                let _ = std::hint::black_box(fe(&bytes[..2]) - fe(&bytes[2..]));
            },
        );
        report("fe_sub", model_name, &outcome);
    }

    // Decapsulation: a valid ciphertext (no rejection) vs a malleated one
    // (implicit rejection). Under a fixed key the timing must not distinguish
    // the two paths.
    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&[0x55; 32]);
    let valid = ciphertext.as_bytes().to_vec();
    let mut malleated = valid.clone();
    malleated[0] ^= 1;
    let decapsulation_key = keypair.decapsulation_key;

    for &(model_name, model) in MODELS {
        let outcome = TimingOracle::for_attacker(model).test(
            InputPair::new(|| valid.clone(), || malleated.clone()),
            |bytes| {
                let ciphertext = Ciphertext::<MLKEM512>::from_bytes(bytes).expect("valid length");
                let _ = std::hint::black_box(decapsulation_key.decapsulate(&ciphertext));
            },
        );
        report("decaps", model_name, &outcome);
    }

    println!("\ntacet analysis complete");
}
