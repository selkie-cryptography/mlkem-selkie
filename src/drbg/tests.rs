//! Unit tests for the SP 800-90A Hash_DRBG-SHA3 implementation.

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
