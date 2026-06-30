//! Randomized cross-implementation interop test against BoringSSL ML-KEM-768.
//!
//! Mirrors sqisign-selkie's C-reference interop pattern: a tiny C oracle
//! (`tools/boringssl_xtest/oracle.c`) is built against a prebuilt BoringSSL and
//! driven as a subprocess over a hex line protocol. BoringSSL is not vendored —
//! building it needs cmake + go — so this test is `#[ignore]`d and only runs
//! when pointed at a built oracle:
//!
//! ```text
//! ORACLE=$(tools/boringssl_xtest/build.sh)
//! MLKEM_BSSL_ORACLE="$ORACLE" cargo test --test boringssl_xtest -- --ignored --nocapture
//! ```
//!
//! Each iteration sends `<seed> <our_ciphertext>` and the oracle returns
//! `<ek> <bssl_ct> <bssl_ss> <our_ss>`, which lets us check, from one round
//! trip:
//!
//! 1. **Keygen byte-identity**: our derandomized keygen `ek` equals BoringSSL's
//!    seed-derived `ek`.
//! 2. **Their encaps, our decaps**: we decapsulate BoringSSL's ciphertext to
//!    BoringSSL's shared secret.
//! 3. **Our encaps, their decaps**: BoringSSL decapsulated our ciphertext to
//!    the same shared secret we computed.

use std::{
    io::Write,
    process::{Command, Stdio},
};

use mlkem_selkie::{KeyPair, MLKEM768, drbg::Aes256CtrDrbg};
use rand_core::RngCore;

/// Number of random interop iterations (ML-KEM-768).
const ITERATIONS: usize = 1000;

#[test]
#[ignore = "requires a prebuilt BoringSSL oracle; set MLKEM_BSSL_ORACLE (see module docs)"]
fn interop_boringssl_mlkem768() {
    let Ok(oracle) = std::env::var("MLKEM_BSSL_ORACLE") else {
        eprintln!(
            "skipping: MLKEM_BSSL_ORACLE not set (build with tools/boringssl_xtest/build.sh)"
        );
        return;
    };

    // Deterministic inputs: derive a key pair from each seed, encapsulate `m`,
    // and remember the resulting (shared secret, ciphertext) per iteration.
    let mut drbg = Aes256CtrDrbg::new(&[0xB5; 48]);
    let mut keypairs = Vec::with_capacity(ITERATIONS);
    let mut our_secrets = Vec::with_capacity(ITERATIONS);
    let mut input = String::new();

    for _ in 0..ITERATIONS {
        let mut seed = [0u8; 64];
        let mut message = [0u8; 32];
        drbg.fill_bytes(&mut seed);
        drbg.fill_bytes(&mut message);

        let keypair = KeyPair::<MLKEM768>::generate_derand(&seed);
        let (shared, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&message);

        input.push_str(&hex::encode(seed));
        input.push(' ');
        input.push_str(&hex::encode(ciphertext.as_bytes()));
        input.push('\n');

        keypairs.push(keypair);
        our_secrets.push(*shared.as_bytes());
    }

    // Spawn the oracle, feed all input from a writer thread (so a full stdout
    // pipe cannot deadlock us), and collect its output.
    let mut child = Command::new(&oracle)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn BoringSSL oracle");

    let mut stdin = child.stdin.take().expect("oracle stdin");
    let writer = std::thread::spawn(move || {
        stdin.write_all(input.as_bytes()).expect("write to oracle");
    });

    let output = child.wait_with_output().expect("await oracle");
    writer.join().expect("writer thread");
    assert!(
        output.status.success(),
        "oracle exited with {:?}",
        output.status
    );

    let stdout = String::from_utf8(output.stdout).expect("oracle output is utf-8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines.len(),
        ITERATIONS,
        "oracle returned {} lines",
        lines.len()
    );

    for (i, line) in lines.iter().enumerate() {
        assert_ne!(*line, "FAIL", "iter {i}: oracle reported FAIL");

        let fields: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 4, "iter {i}: expected 4 fields");
        let ek = hex::decode(fields[0]).expect("ek hex");
        let bssl_ct = hex::decode(fields[1]).expect("bssl_ct hex");
        let bssl_ss = hex::decode(fields[2]).expect("bssl_ss hex");
        let our_ss_via_bssl = hex::decode(fields[3]).expect("our_ss hex");

        let keypair = &keypairs[i];

        // 1. Keygen byte-identity.
        assert_eq!(
            keypair.encapsulation_key.to_bytes().as_ref(),
            ek,
            "iter {i}: ek mismatch"
        );

        // 2. Their encaps, our decaps.
        let ciphertext = mlkem_selkie::Ciphertext::<MLKEM768>::from_bytes(&bssl_ct)
            .expect("valid BoringSSL ciphertext");
        let recovered = keypair.decapsulation_key.decapsulate(&ciphertext);
        assert_eq!(
            recovered.as_bytes().as_slice(),
            bssl_ss.as_slice(),
            "iter {i}: their-encaps/our-decaps shared secret mismatch",
        );

        // 3. Our encaps, their decaps.
        assert_eq!(
            our_secrets[i].as_slice(),
            our_ss_via_bssl.as_slice(),
            "iter {i}: our-encaps/their-decaps shared secret mismatch",
        );
    }
}
