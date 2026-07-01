// Gated to Linux because crabgrind's build step looks up
// <valgrind/valgrind.h> on the system include paths, which is absent on a
// stock macOS install. CI runs this on the Linux image with Valgrind
// installed; on every other target the file compiles to an empty test binary.
#![cfg(target_os = "linux")]
//! Secret-dependent branch / memory-access tests using Valgrind memcheck.
//!
//! Marks secret inputs as "undefined" via Valgrind client requests, then runs
//! ML-KEM operations. Valgrind reports an error if any branch or memory access
//! depends on the tainted (secret) data.
//!
//! Run with:
//! ```text
//! cargo test --test ctgrind --features expose-internals --no-run
//! valgrind --tool=memcheck --error-exitcode=1 \
//!   target/debug/deps/ctgrind-* --test-threads=1
//! ```
//!
//! NOTE: the field reductions are branch-free (signed Montgomery / Barrett),
//! and `ML-KEM.Decaps`'s implicit-rejection check compares and selects over the
//! secret-derived ciphertext bytes in constant time (`subtle`'s `ct_eq` /
//! `conditional_select`; see `DecapsulationKey::decapsulate`), so the
//! `decapsulate` test should report no secret-dependent branch under Valgrind.

use core::ffi::c_void;

use crabgrind::memcheck::{self, MemState};
use mlkem_selkie::{
    Ciphertext, DecapsulationKey, KeyPair, MLKEM512, ParameterSet, algebraic::FieldElement,
};

/// Marks a byte slice as "secret" (undefined) for Valgrind. A no-op when not
/// running under Valgrind.
fn mark_secret(data: &[u8]) {
    let _ = memcheck::mark_memory(
        data.as_ptr() as *const c_void,
        data.len(),
        MemState::Undefined,
    );
}

/// Marks a byte slice as "public" (defined) for Valgrind.
fn mark_public(data: &[u8]) {
    let _ = memcheck::mark_memory(
        data.as_ptr() as *const c_void,
        data.len(),
        MemState::Defined,
    );
}

/// A field element from the first two bytes of a slice.
fn fe(bytes: &[u8]) -> FieldElement {
    FieldElement::new(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Returns `true` if the caller should skip — slow top-level tests run minutes
/// under Valgrind, so they are opt-in via `CTGRIND_SLOW`.
fn skip_unless_slow(name: &str) -> bool {
    // Treat unset *and* empty as off: the CI workflow's `env:` mapping has no
    // way to skip setting a variable on conditional `false`, so it passes
    // `CTGRIND_SLOW=''` instead. `var_os().is_none()` only catches the unset
    // case and would let the empty string run the slow tests.
    let enabled = std::env::var_os("CTGRIND_SLOW").is_some_and(|v| !v.is_empty());
    if !enabled {
        eprintln!("[ctgrind] skipping {name}; set CTGRIND_SLOW=1 to enable");
        return true;
    }

    false
}

#[test]
fn field_mul_secret_independent() {
    let bytes = [0x42u8; 4];
    mark_secret(&bytes);

    let result = fe(&bytes[..2]) * fe(&bytes[2..]);

    let out = result.value().to_le_bytes();
    mark_public(&out);
}

#[test]
fn field_add_secret_independent() {
    let bytes = [0x42u8; 4];
    mark_secret(&bytes);

    let result = fe(&bytes[..2]) + fe(&bytes[2..]);

    let out = result.value().to_le_bytes();
    mark_public(&out);
}

#[test]
fn field_sub_secret_independent() {
    let bytes = [0x42u8; 4];
    mark_secret(&bytes);

    let result = fe(&bytes[..2]) - fe(&bytes[2..]);

    let out = result.value().to_le_bytes();
    mark_public(&out);
}

#[test]
fn keygen_secret_independent() {
    if skip_unless_slow("keygen_secret_independent") {
        return;
    }

    // The keygen seed `d ‖ z` is secret; the resulting `s`, `e`, and `s_hat`
    // are all derived from it, so a branch on any of them lights up.
    let mut seed = [0x42u8; 64];
    mark_secret(&seed);

    let keypair = KeyPair::<MLKEM512>::generate_derand(&seed);

    // The encapsulation key is public output; declassify it. The secret
    // decapsulation key is dropped silently.
    let ek_bytes = keypair.encapsulation_key.to_bytes();
    mark_public(&ek_bytes);

    seed.fill(0);
    mark_public(&seed);
}

#[test]
fn encapsulate_secret_independent() {
    if skip_unless_slow("encapsulate_secret_independent") {
        return;
    }

    // Public encapsulation key.
    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let encapsulation_key = keypair.encapsulation_key;

    // The 32-byte encapsulation randomness `m` is secret: the shared secret
    // and the ciphertext are both derived from it, so a branch anywhere in
    // `K-PKE.Encrypt` or the `G(m || H(ek))` split lights up.
    let mut m = [0x55u8; 32];
    mark_secret(&m);

    let (shared, ciphertext) = encapsulation_key.encapsulate_derand(&m);

    // The ciphertext is public output; the shared secret would flow into a
    // caller's KDF. Declassify both here so their tainted bytes don't leak
    // into unrelated test cleanup code.
    let ciphertext_bytes = ciphertext.as_bytes().to_vec();
    mark_public(&ciphertext_bytes);

    let shared_bytes = *shared.as_bytes();
    mark_public(&shared_bytes);

    m.fill(0);
    mark_public(&m);
}

#[test]
fn decapsulate_secret_independent() {
    if skip_unless_slow("decapsulate_secret_independent") {
        return;
    }

    // Public key pair and ciphertext.
    let keypair = KeyPair::<MLKEM512>::generate_derand(&[0x42; 64]);
    let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&[0x55; 32]);
    let ciphertext_bytes = ciphertext.as_bytes().to_vec();

    // Serialize the decapsulation key and taint only its secret portions: the
    // K-PKE decryption key `s_hat` (the prefix) and the rejection seed `z` (the
    // suffix). The embedded encapsulation key and its hash are public.
    let mut dk_bytes = keypair.decapsulation_key.to_bytes();
    mark_secret(&dk_bytes[..MLKEM512::PKE_DECRYPTION_KEY_SIZE]);
    mark_secret(&dk_bytes[MLKEM512::DECAPS_KEY_SIZE - 32..]);

    let decapsulation_key =
        DecapsulationKey::<MLKEM512>::from_bytes(&dk_bytes).expect("valid decapsulation key");
    let ciphertext =
        Ciphertext::<MLKEM512>::from_bytes(&ciphertext_bytes).expect("valid ciphertext");

    let shared = decapsulation_key.decapsulate(&ciphertext);

    let shared_bytes = *shared.as_bytes();
    mark_public(&shared_bytes);

    dk_bytes.fill(0);
    mark_public(&dk_bytes);
}
