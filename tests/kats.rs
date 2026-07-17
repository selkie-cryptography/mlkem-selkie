//! NIST ACVP (FIPS 203) ML-KEM validation KAT cross-checks.
//!
//! Replays the official NIST ACVP demo vectors — the same functional KATs used
//! for FIPS 140-3 / CAVP module validation — vendored under `tests/vectors`:
//!
//! - `acvp_ml_kem_keygen.json` (`ML-KEM-keyGen-FIPS203`): derive `(ek, dk)`
//!   from the `d`/`z` seed halves.
//! - `acvp_ml_kem_encap_decap.json` (`ML-KEM-encapDecap-FIPS203`): four group
//!   functions — `encapsulation` (AFT: `m` -> `(k, c)`), `decapsulation` (VAL:
//!   `(dk, c)` -> `k`, including the implicit-rejection path), and the
//!   `encapsulationKeyCheck` / `decapsulationKeyCheck` input-validation groups
//!   whose `testPassed` flag our parsers must reproduce.
//!
//! These are a *separate* source from the Wycheproof vectors, not a subset: a
//! different (ACVP) format and set of cases. Wycheproof carries more
//! adversarial negative tests; these are the canonical NIST validation KATs.
//! Same hand-rolled serde + `include_str!` approach as `wycheproof.rs`.

use mlkem_selkie::{
    Ciphertext, DecapsulationKey, EncapsulationKey, MLKEM512, MLKEM768, MLKEM1024, ParameterSet,
};
use serde::Deserialize;

/// An ACVP test file (`internalProjection.json`): header plus test groups.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestFile {
    algorithm: String,
    test_groups: Vec<TestGroup>,
}

/// A group of ACVP test cases sharing a parameter set and operation.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestGroup {
    parameter_set: String,
    /// Absent in the keyGen file; one of `encapsulation`, `decapsulation`,
    /// `encapsulationKeyCheck`, `decapsulationKeyCheck` in encapDecap.
    #[serde(default)]
    function: Option<String>,
    tests: Vec<TestCase>,
}

/// An ACVP test case. Fields are populated per group function.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestCase {
    tc_id: u32,
    #[serde(default)]
    d: String,
    #[serde(default)]
    z: String,
    #[serde(default)]
    ek: String,
    #[serde(default)]
    dk: String,
    #[serde(default)]
    c: String,
    #[serde(default, rename = "k")]
    k: String,
    #[serde(default)]
    m: String,
    #[serde(default)]
    test_passed: Option<bool>,
}

/// Decodes a hex string into a fixed-size array.
fn hex_array<const N: usize>(hex: &str) -> [u8; N] {
    hex::decode(hex)
        .expect("hex")
        .try_into()
        .unwrap_or_else(|v: Vec<u8>| panic!("expected {N} bytes, got {}", v.len()))
}

/// Runs every test in a group against the parameter set `P`.
fn run_group<P: ParameterSet>(group: &TestGroup) {
    for test in &group.tests {
        match group.function.as_deref() {
            None => keygen_case::<P>(test),
            Some("encapsulation") => encapsulation_case::<P>(test),
            Some("decapsulation") => decapsulation_case::<P>(test),
            Some("encapsulationKeyCheck") => encapsulation_key_check::<P>(test),
            Some("decapsulationKeyCheck") => decapsulation_key_check::<P>(test),
            Some(other) => panic!("tc {}: unknown function {other:?}", test.tc_id),
        }
    }
}

/// `ML-KEM.KeyGen_internal`: `d ‖ z` must reproduce `(ek, dk)`.
fn keygen_case<P: ParameterSet>(test: &TestCase) {
    let mut seed = [0u8; 64];
    seed[..32].copy_from_slice(&hex_array::<32>(&test.d));
    seed[32..].copy_from_slice(&hex_array::<32>(&test.z));

    let keypair = DecapsulationKey::<P>::generate_derand(&seed);

    assert_eq!(
        keypair.encapsulation_key().to_bytes().as_ref(),
        hex::decode(&test.ek).expect("ek hex"),
        "tc {}: ek mismatch",
        test.tc_id
    );
    assert_eq!(
        keypair.to_bytes().as_ref(),
        hex::decode(&test.dk).expect("dk hex"),
        "tc {}: dk mismatch",
        test.tc_id
    );
}

/// `ML-KEM.Encaps_internal`: `m` under `ek` must reproduce `(k, c)`.
fn encapsulation_case<P: ParameterSet>(test: &TestCase) {
    let ek = EncapsulationKey::<P>::from_bytes(&hex::decode(&test.ek).expect("ek hex"))
        .expect("valid encapsulation key");
    let (shared, ciphertext) = ek.encapsulate_derand(&hex_array::<32>(&test.m));

    assert_eq!(
        ciphertext.as_bytes(),
        hex::decode(&test.c).expect("c hex"),
        "tc {}: ciphertext mismatch",
        test.tc_id
    );
    assert_eq!(
        shared.as_bytes().as_slice(),
        hex::decode(&test.k).expect("k hex").as_slice(),
        "tc {}: shared secret mismatch",
        test.tc_id
    );
}

/// `ML-KEM.Decaps`: `(dk, c)` must reproduce `k` (including implicit
/// rejection).
fn decapsulation_case<P: ParameterSet>(test: &TestCase) {
    let dk = DecapsulationKey::<P>::from_bytes(&hex::decode(&test.dk).expect("dk hex"))
        .expect("valid decapsulation key");
    let ciphertext = Ciphertext::<P>::from_bytes(&hex::decode(&test.c).expect("c hex"))
        .expect("valid ciphertext");
    let shared = dk.decapsulate(&ciphertext);

    assert_eq!(
        shared.as_bytes().as_slice(),
        hex::decode(&test.k).expect("k hex").as_slice(),
        "tc {}: shared secret mismatch",
        test.tc_id
    );
}

/// Encapsulation-key validation: `from_bytes` success must match `testPassed`.
fn encapsulation_key_check<P: ParameterSet>(test: &TestCase) {
    let accepted =
        EncapsulationKey::<P>::from_bytes(&hex::decode(&test.ek).expect("ek hex")).is_ok();

    assert_eq!(
        accepted,
        test.test_passed.expect("testPassed"),
        "tc {}: encapsulation-key validity mismatch",
        test.tc_id
    );
}

/// Decapsulation-key validation: `from_bytes` success must match `testPassed`.
fn decapsulation_key_check<P: ParameterSet>(test: &TestCase) {
    let accepted =
        DecapsulationKey::<P>::from_bytes(&hex::decode(&test.dk).expect("dk hex")).is_ok();

    assert_eq!(
        accepted,
        test.test_passed.expect("testPassed"),
        "tc {}: decapsulation-key validity mismatch",
        test.tc_id
    );
}

/// Parses a file and dispatches every group to its parameter set.
fn run(json: &str) {
    let file: TestFile = serde_json::from_str(json).expect("parse ACVP JSON");
    assert_eq!(file.algorithm, "ML-KEM");

    for group in &file.test_groups {
        match group.parameter_set.as_str() {
            "ML-KEM-512" => run_group::<MLKEM512>(group),
            "ML-KEM-768" => run_group::<MLKEM768>(group),
            "ML-KEM-1024" => run_group::<MLKEM1024>(group),
            other => panic!("unknown parameter set {other:?}"),
        }
    }
}

#[test]
fn acvp_keygen() {
    run(include_str!("vectors/acvp_ml_kem_keygen.json"));
}

#[test]
fn acvp_encap_decap() {
    run(include_str!("vectors/acvp_ml_kem_encap_decap.json"));
}
