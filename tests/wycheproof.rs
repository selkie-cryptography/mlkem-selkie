//! Wycheproof / CCTV ML-KEM test-vector cross-checks.
//!
//! Replays the [C2SP/wycheproof] ML-KEM vectors (vendored under
//! `tests/vectors`) against the public `mlkem-selkie` API. There are four
//! vector shapes per parameter set:
//!
//! - `*_keygen_seed_test`: derive `(ek, dk)` from a 64-byte `d ‖ z` seed.
//! - `*_encaps_test`: `Encaps_derand(ek, m)` must reproduce `(K, c)`, and
//!   malformed encapsulation keys (modulus overflow) must be rejected.
//! - `*_test`: derive a key pair from a seed, then `Decaps(c)` must reproduce
//!   `K` (including the implicit-rejection path).
//! - `*_semi_expanded_decaps_test`: `Decaps` over an explicit `dk`, exercising
//!   length and decapsulation-key validation.
//!
//! Following `sqisign-selkie`, the Wycheproof JSON schema is modeled with local
//! serde structs (no `wycheproof` crate) and loaded at compile time with
//! `include_str!`.
//!
//! [C2SP/wycheproof]: https://github.com/C2SP/wycheproof

use mlkem_selkie::{
    Ciphertext, DecapsulationKey, EncapsulationKey, MLKEM512, MLKEM768, MLKEM1024, ParameterSet,
};
use serde::Deserialize;

/// A Wycheproof test file: a header plus a list of test groups.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestFile<T> {
    algorithm: String,
    number_of_tests: usize,
    test_groups: Vec<TestGroup<T>>,
}

/// A group of test vectors sharing a parameter set.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestGroup<T> {
    parameter_set: String,
    tests: Vec<T>,
}

/// A `MLKEMKeyGen` vector: derive `(ek, dk)` from a seed.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct KeyGenTest {
    tc_id: u32,
    seed: String,
    ek: String,
    dk: String,
    result: String,
}

/// A `MLKEMEncapsTest` vector: encapsulate `m` under `ek`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EncapsTest {
    tc_id: u32,
    m: String,
    ek: String,
    #[serde(default)]
    c: String,
    #[serde(default, rename = "K")]
    k: String,
    result: String,
}

/// A `MLKEMTest` vector: keygen from a seed, then decapsulate `c`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DecapsFromSeedTest {
    tc_id: u32,
    seed: String,
    #[serde(default)]
    ek: String,
    #[serde(default)]
    c: String,
    #[serde(default, rename = "K")]
    k: String,
    result: String,
}

/// A `MLKEMDecapsValidationTest` vector: decapsulate over an explicit `dk`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DecapsValidationTest {
    tc_id: u32,
    dk: String,
    c: String,
    #[serde(default, rename = "K")]
    k: String,
    result: String,
}

/// Parses a test file, asserting the algorithm and parameter-set labels match.
fn parse<T: serde::de::DeserializeOwned>(json: &str, expected_set: &str) -> TestFile<T> {
    let file: TestFile<T> = serde_json::from_str(json).expect("parse Wycheproof JSON");

    assert_eq!(file.algorithm, "ML-KEM");
    for group in &file.test_groups {
        assert_eq!(group.parameter_set, expected_set);
    }

    file
}

/// Decodes a hex string into a fixed-size array.
fn hex_array<const N: usize>(hex: &str) -> [u8; N] {
    hex::decode(hex)
        .expect("hex")
        .try_into()
        .unwrap_or_else(|v: Vec<u8>| panic!("expected {N} bytes, got {}", v.len()))
}

/// Runs a `keygen_seed` file: each seed must reproduce the expected `(ek, dk)`.
fn run_keygen<P: ParameterSet>(json: &str, expected_set: &str) {
    let file = parse::<KeyGenTest>(json, expected_set);

    let mut tested = 0;
    for group in &file.test_groups {
        for test in &group.tests {
            assert_eq!(
                test.result, "valid",
                "tc {}: keygen only has valid",
                test.tc_id
            );

            let seed = hex_array::<64>(&test.seed);
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

            tested += 1;
        }
    }

    assert_eq!(tested, file.number_of_tests);
}

/// Runs an `encaps` file: valid keys must reproduce `(K, c)`; modulus-overflow
/// keys must be rejected by the parser.
fn run_encaps<P: ParameterSet>(json: &str, expected_set: &str) {
    let file = parse::<EncapsTest>(json, expected_set);

    let mut tested = 0;
    for group in &file.test_groups {
        for test in &group.tests {
            let ek_bytes = hex::decode(&test.ek).expect("ek hex");
            let m = hex_array::<32>(&test.m);

            match test.result.as_str() {
                "valid" => {
                    let ek = EncapsulationKey::<P>::from_bytes(&ek_bytes)
                        .expect("valid encapsulation key");
                    let (shared, ciphertext) = ek.encapsulate_derand(&m);

                    assert_eq!(
                        ciphertext.as_bytes(),
                        hex::decode(&test.c).expect("c hex"),
                        "tc {}: ciphertext mismatch",
                        test.tc_id
                    );
                    assert_eq!(
                        shared.as_bytes().as_slice(),
                        hex::decode(&test.k).expect("K hex").as_slice(),
                        "tc {}: shared secret mismatch",
                        test.tc_id
                    );
                }
                "invalid" => {
                    assert!(
                        EncapsulationKey::<P>::from_bytes(&ek_bytes).is_err(),
                        "tc {}: malformed encapsulation key was accepted",
                        test.tc_id
                    );
                }
                other => panic!("tc {}: unknown result {other:?}", test.tc_id),
            }

            tested += 1;
        }
    }

    assert_eq!(tested, file.number_of_tests);
}

/// Runs a `*_test` file: keygen from the seed, then decapsulate `c` to `K`.
fn run_decaps_from_seed<P: ParameterSet>(json: &str, expected_set: &str) {
    let file = parse::<DecapsFromSeedTest>(json, expected_set);

    let mut tested = 0;
    for group in &file.test_groups {
        for test in &group.tests {
            match test.result.as_str() {
                "valid" => {
                    let seed = hex_array::<64>(&test.seed);
                    let keypair = DecapsulationKey::<P>::generate_derand(&seed);

                    if !test.ek.is_empty() {
                        assert_eq!(
                            keypair.encapsulation_key().to_bytes().as_ref(),
                            hex::decode(&test.ek).expect("ek hex"),
                            "tc {}: ek mismatch",
                            test.tc_id
                        );
                    }

                    let ciphertext =
                        Ciphertext::<P>::from_bytes(&hex::decode(&test.c).expect("c hex"))
                            .expect("valid ciphertext");
                    let shared = keypair.decapsulate(&ciphertext);

                    assert_eq!(
                        shared.as_bytes().as_slice(),
                        hex::decode(&test.k).expect("K hex").as_slice(),
                        "tc {}: shared secret mismatch",
                        test.tc_id
                    );
                }
                "invalid" => {
                    // An invalid vector is rejected either because its seed is
                    // truncated (cannot form a key) or because its ciphertext is
                    // the wrong length (cannot be parsed).
                    let seed = hex::decode(&test.seed).expect("seed hex");
                    let c_bytes = hex::decode(&test.c).expect("c hex");

                    let rejected = match <[u8; 64]>::try_from(seed) {
                        Err(_) => true,
                        Ok(_) => Ciphertext::<P>::from_bytes(&c_bytes).is_err(),
                    };

                    assert!(rejected, "tc {}: invalid vector was accepted", test.tc_id);
                }
                other => panic!("tc {}: unknown result {other:?}", test.tc_id),
            }

            tested += 1;
        }
    }

    assert_eq!(tested, file.number_of_tests);
}

/// Runs a `semi_expanded_decaps` file: decapsulate over an explicit `dk`,
/// rejecting length and key-consistency failures.
fn run_decaps_validation<P: ParameterSet>(json: &str, expected_set: &str) {
    let file = parse::<DecapsValidationTest>(json, expected_set);

    let mut tested = 0;
    for group in &file.test_groups {
        for test in &group.tests {
            let dk_bytes = hex::decode(&test.dk).expect("dk hex");
            let c_bytes = hex::decode(&test.c).expect("c hex");

            match test.result.as_str() {
                "valid" => {
                    let dk = DecapsulationKey::<P>::from_bytes(&dk_bytes)
                        .expect("valid decapsulation key");
                    let ciphertext =
                        Ciphertext::<P>::from_bytes(&c_bytes).expect("valid ciphertext");
                    let shared = dk.decapsulate(&ciphertext);

                    assert_eq!(
                        shared.as_bytes().as_slice(),
                        hex::decode(&test.k).expect("K hex").as_slice(),
                        "tc {}: shared secret mismatch",
                        test.tc_id
                    );
                }
                "invalid" => {
                    let accepted = DecapsulationKey::<P>::from_bytes(&dk_bytes).is_ok()
                        && Ciphertext::<P>::from_bytes(&c_bytes).is_ok();
                    assert!(
                        !accepted,
                        "tc {}: invalid decapsulation vector was accepted",
                        test.tc_id
                    );
                }
                other => panic!("tc {}: unknown result {other:?}", test.tc_id),
            }

            tested += 1;
        }
    }

    assert_eq!(tested, file.number_of_tests);
}

#[test]
fn mlkem_512_keygen() {
    run_keygen::<MLKEM512>(
        include_str!("vectors/mlkem_512_keygen_seed_test.json"),
        "ML-KEM-512",
    );
}

#[test]
fn mlkem_768_keygen() {
    run_keygen::<MLKEM768>(
        include_str!("vectors/mlkem_768_keygen_seed_test.json"),
        "ML-KEM-768",
    );
}

#[test]
fn mlkem_1024_keygen() {
    run_keygen::<MLKEM1024>(
        include_str!("vectors/mlkem_1024_keygen_seed_test.json"),
        "ML-KEM-1024",
    );
}

#[test]
fn mlkem_512_encaps() {
    run_encaps::<MLKEM512>(
        include_str!("vectors/mlkem_512_encaps_test.json"),
        "ML-KEM-512",
    );
}

#[test]
fn mlkem_768_encaps() {
    run_encaps::<MLKEM768>(
        include_str!("vectors/mlkem_768_encaps_test.json"),
        "ML-KEM-768",
    );
}

#[test]
fn mlkem_1024_encaps() {
    run_encaps::<MLKEM1024>(
        include_str!("vectors/mlkem_1024_encaps_test.json"),
        "ML-KEM-1024",
    );
}

#[test]
fn mlkem_512_decaps_from_seed() {
    run_decaps_from_seed::<MLKEM512>(include_str!("vectors/mlkem_512_test.json"), "ML-KEM-512");
}

#[test]
fn mlkem_768_decaps_from_seed() {
    run_decaps_from_seed::<MLKEM768>(include_str!("vectors/mlkem_768_test.json"), "ML-KEM-768");
}

#[test]
fn mlkem_1024_decaps_from_seed() {
    run_decaps_from_seed::<MLKEM1024>(include_str!("vectors/mlkem_1024_test.json"), "ML-KEM-1024");
}

#[test]
fn mlkem_512_semi_expanded_decaps() {
    run_decaps_validation::<MLKEM512>(
        include_str!("vectors/mlkem_512_semi_expanded_decaps_test.json"),
        "ML-KEM-512",
    );
}

#[test]
fn mlkem_768_semi_expanded_decaps() {
    run_decaps_validation::<MLKEM768>(
        include_str!("vectors/mlkem_768_semi_expanded_decaps_test.json"),
        "ML-KEM-768",
    );
}

#[test]
fn mlkem_1024_semi_expanded_decaps() {
    run_decaps_validation::<MLKEM1024>(
        include_str!("vectors/mlkem_1024_semi_expanded_decaps_test.json"),
        "ML-KEM-1024",
    );
}
