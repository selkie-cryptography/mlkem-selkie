//! Unit tests for the SP 800-90A Hash_DRBG-SHA3 implementation.

use core::marker::PhantomData;

use rand_core::RngCore;

use super::*;

/// Two DRBGs seeded with the same entropy input produce the same stream — the
/// core determinism property SP 800-90A relies on.
#[test]
fn same_entropy_produces_same_stream() {
    let entropy = [0xA5u8; 24];
    let mut a = HashDrbgSha3_256::new(&entropy);
    let mut b = HashDrbgSha3_256::new(&entropy);

    let mut a_buf = [0u8; 128];
    let mut b_buf = [0u8; 128];
    a.fill_bytes(&mut a_buf);
    b.fill_bytes(&mut b_buf);

    assert_eq!(a_buf, b_buf);
}

/// Different entropy inputs produce different streams — a weak but load-bearing
/// sanity check that Hash_df's counter/length prefix actually reaches the hash.
#[test]
fn different_entropy_diverges() {
    let mut a = HashDrbgSha3_256::new(&[0x00u8; 24]);
    let mut b = HashDrbgSha3_256::new(&[0xFFu8; 24]);

    let mut a_buf = [0u8; 128];
    let mut b_buf = [0u8; 128];
    a.fill_bytes(&mut a_buf);
    b.fill_bytes(&mut b_buf);

    assert_ne!(a_buf, b_buf);
}

/// Requesting output shorter than one hash block, exactly one block, and
/// multiple blocks all deliver bytes — the Hashgen loop's iteration count
/// covers the three cases (`0 < N < outlen`, `N == outlen`, `N > outlen`).
#[test]
fn covers_partial_full_and_multiblock_output_sizes() {
    let entropy = [0x11u8; 24];
    for &len in &[1usize, 16, 32, 33, 128, 512] {
        let mut drbg = HashDrbgSha3_256::new(&entropy);
        let mut buf = vec![0u8; len];
        drbg.fill_bytes(&mut buf);
        assert!(
            buf.iter().any(|&b| b != 0),
            "output is all-zero at len={len}"
        );
    }
}

/// All three strength-matched aliases actually build and produce distinct
/// streams for the same-shape entropy prefix. Guards against a
/// copy-paste bug in the type aliases or the `fill_from_fips_drbg_sha3_*` fns.
#[test]
fn all_three_strengths_produce_distinct_streams() {
    // Zero-pad to the widest entropy length; each impl only reads the prefix
    // it expects (24 / 36 / 48 bytes).
    let entropy = [0x77u8; 48];

    let mut buf_256 = [0u8; 64];
    HashDrbgSha3_256::new(&entropy[..24]).fill_bytes(&mut buf_256);

    let mut buf_384 = [0u8; 64];
    HashDrbgSha3_384::new(&entropy[..36]).fill_bytes(&mut buf_384);

    let mut buf_512 = [0u8; 64];
    HashDrbgSha3_512::new(&entropy).fill_bytes(&mut buf_512);

    assert_ne!(buf_256, buf_384);
    assert_ne!(buf_384, buf_512);
    assert_ne!(buf_256, buf_512);
}

/// The `RngCore` convenience methods (`next_u32`, `next_u64`,
/// `try_fill_bytes`) are deterministic from the seed: two DRBGs seeded
/// identically produce the same sequence of outputs, and `try_fill_bytes` is
/// infallible for `HashDrbg`.
#[test]
fn rng_core_helpers_are_deterministic_from_seed() {
    let entropy = [0xC3u8; 48];

    let mut a = HashDrbgSha3_512::new(&entropy);
    let mut b = HashDrbgSha3_512::new(&entropy);

    assert_eq!(a.next_u32(), b.next_u32());
    assert_eq!(a.next_u64(), b.next_u64());

    let mut a_buf = [0u8; 20];
    let mut b_buf = [0u8; 20];
    a.try_fill_bytes(&mut a_buf)
        .expect("try_fill_bytes is infallible for HashDrbg");
    b.try_fill_bytes(&mut b_buf)
        .expect("try_fill_bytes is infallible for HashDrbg");
    assert_eq!(a_buf, b_buf);
}

/// `add_be_u8` propagates carry across a byte boundary. Adding 1 to
/// `[..., 0xFF]` sets the LSB to zero and increments the next byte, mirroring
/// hashgen's `data + 1` step. Guards against carry mutations that stay
/// invisible in Hashgen's typical inputs.
#[test]
fn add_be_u8_propagates_carry_across_boundary() {
    let mut v = [0u8; 4];
    v[3] = 0xFF;
    add_be_u8(&mut v, 1);
    assert_eq!(v, [0, 0, 1, 0]);

    let mut v = [0u8; 3];
    v[2] = 0xFF;
    v[1] = 0xFF;
    add_be_u8(&mut v, 1);
    assert_eq!(v, [1, 0, 0]);
}

/// SP 800-90A optional-shape entry points used only by the ACVP KAT harness
/// below: instantiate-with-nonce-and-personalization (§10.1.1.2),
/// [`HashDrbg::reseed`] (§10.1.1.3), and generate-with-additional-input
/// (§10.1.1.4). The library's production path never touches these — the KEM
/// instantiates once and calls `fill_bytes` once — but the NIST ACVP
/// `hashDRBG-1.0` vectors do, and running against them is the only way to
/// catch mutations against the DRBG's spec-transliterated arithmetic.
impl<H, const SEEDLEN: usize> HashDrbg<H, SEEDLEN>
where
    H: DrbgHash,
{
    /// §10.1.1.2 with an explicit `nonce` and `personalization` — the three
    /// inputs are concatenated into `seed_material` before Hash_df.
    #[must_use]
    fn new_with_perso(entropy_input: &[u8], nonce: &[u8], personalization: &[u8]) -> Self {
        let mut v = [0u8; SEEDLEN];
        Self::hash_df(&[entropy_input, nonce, personalization], &mut v);

        let mut c = [0u8; SEEDLEN];
        Self::hash_df(&[&[0x00u8], &v], &mut c);

        Self {
            v,
            c,
            reseed_counter: 1,
            _hash: PhantomData,
        }
    }

    /// §10.1.1.3 `Hash_DRBG_Reseed`.
    fn reseed(&mut self, entropy_input: &[u8], additional_input: &[u8]) {
        let mut new_v = [0u8; SEEDLEN];
        Self::hash_df(
            &[&[0x01u8], &self.v, entropy_input, additional_input],
            &mut new_v,
        );

        let mut new_c = [0u8; SEEDLEN];
        Self::hash_df(&[&[0x00u8], &new_v], &mut new_c);

        self.v = new_v;
        self.c = new_c;
        self.reseed_counter = 1;
    }

    /// §10.1.1.4 `Hash_DRBG_Generate` with `additional_input`. Runs the
    /// optional additional_input step (§10.1.1.4 step 2), then delegates to
    /// the mainline `generate` — sharing the Hashgen + V-update code with
    /// production ensures mutants against those paths are exercised here.
    fn generate_with_additional_input(&mut self, additional_input: &[u8], out: &mut [u8]) {
        if !additional_input.is_empty() {
            let mut input: Vec<u8> = Vec::with_capacity(1 + SEEDLEN + additional_input.len());
            input.push(0x02);
            input.extend_from_slice(&self.v);
            input.extend_from_slice(additional_input);
            let mut w = [0u8; MAX_DIGEST];
            H::hash(&mut w[..H::OUTLEN], &input);
            add_be_slice(&mut self.v, &w[..H::OUTLEN]);
        }

        self.generate(out);
    }
}

/// NIST ACVP `hashDRBG-1.0` SHA3-256 KAT harness. The vendored JSON is the
/// SHA3-256 subset (PR-enabled and PR-disabled groups) of
/// `usnistgov/ACVP-Server/gen-val/json-files/hashDRBG-1.0`, with the
/// `returnedBits` from `expectedResults.json` inlined. The same generic
/// `HashDrbg<H, SEEDLEN>` code path underlies SHA3-384 and SHA3-512, so
/// pinning one hash flavor covers the arithmetic for all three.
#[test]
fn acvp_hashdrbg_sha3_256() {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct TestVectors {
        #[serde(rename = "testGroups")]
        test_groups: Vec<TestGroup>,
    }

    #[derive(Deserialize)]
    struct TestGroup {
        #[serde(rename = "predResistance")]
        pred_resistance: bool,
        #[serde(rename = "returnedBitsLen")]
        returned_bits_len: usize,
        tests: Vec<TestCase>,
    }

    #[derive(Deserialize)]
    struct TestCase {
        #[serde(rename = "tcId")]
        tc_id: u64,
        #[serde(rename = "entropyInput")]
        entropy_input: String,
        nonce: String,
        #[serde(rename = "persoString")]
        perso_string: String,
        #[serde(rename = "otherInput")]
        other_input: Vec<Step>,
        #[serde(rename = "returnedBits")]
        returned_bits: String,
    }

    #[derive(Deserialize)]
    struct Step {
        #[serde(rename = "intendedUse")]
        intended_use: String,
        #[serde(rename = "additionalInput", default)]
        additional_input: String,
        #[serde(rename = "entropyInput", default)]
        entropy_input: String,
    }

    fn h(s: &str) -> Vec<u8> {
        hex::decode(s).expect("hex")
    }

    let vectors: TestVectors =
        serde_json::from_str(include_str!("vectors/hashdrbg_sha3_256.json"))
            .expect("parse ACVP JSON");

    let mut ran = 0usize;
    for group in &vectors.test_groups {
        for test in &group.tests {
            let mut drbg = HashDrbgSha3_256::new_with_perso(
                &h(&test.entropy_input),
                &h(&test.nonce),
                &h(&test.perso_string),
            );

            let out_len = group.returned_bits_len / 8;
            let mut out = vec![0u8; out_len];

            for step in &test.other_input {
                let additional = h(&step.additional_input);
                match step.intended_use.as_str() {
                    "reSeed" => drbg.reseed(&h(&step.entropy_input), &additional),
                    "generate" => {
                        if group.pred_resistance {
                            // §9.3.1: reseed with (entropy, additional), then
                            // generate with additional = null.
                            drbg.reseed(&h(&step.entropy_input), &additional);
                            drbg.generate_with_additional_input(&[], &mut out);
                        } else {
                            drbg.generate_with_additional_input(&additional, &mut out);
                        }
                    }
                    other => panic!("unknown intendedUse: {other}"),
                }
            }

            let expected = h(&test.returned_bits);
            assert_eq!(
                out, expected,
                "tcId {} (PR={}) returnedBits mismatch",
                test.tc_id, group.pred_resistance,
            );
            ran += 1;
        }
    }

    assert!(ran > 0, "no ACVP test cases ran");
}
