//! Known-answer self-test for the AES256-CTR-DRBG.

use rand_core::RngCore;

use super::*;

/// Seeding the DRBG with a fixed 48-byte entropy input and drawing 128 bytes
/// reproduces the SP 800-90A reference output stream.
///
/// This pins the generator independently of any ML-KEM logic: a divergence here
/// would mean the AES-CTR-DRBG, or the byte ordering in `randombytes`, is out
/// of step with the NIST reference, and every downstream KAT replay would drift
/// for reasons unrelated to the scheme.
#[test]
fn matches_reference_seed_first_128_bytes() {
    const SEED: [u8; SEEDLEN] = [
        0x06, 0x15, 0x50, 0x23, 0x4D, 0x15, 0x8C, 0x5E, 0xC9, 0x55, 0x95, 0xFE, 0x04, 0xEF, 0x7A,
        0x25, 0x76, 0x7F, 0x2E, 0x24, 0xCC, 0x2B, 0xC4, 0x79, 0xD0, 0x9D, 0x86, 0xDC, 0x9A, 0xBC,
        0xFD, 0xE7, 0x05, 0x6A, 0x8C, 0x26, 0x6F, 0x9E, 0xF9, 0x7E, 0xD0, 0x85, 0x41, 0xDB, 0xD2,
        0xE1, 0xFF, 0xA1,
    ];
    const EXPECTED_HEX: &str = "\
        7c9935a0b07694aa0c6d10e4db6b1add\
        2fd81a25ccb148032dcd739936737f2d\
        b505d7cfad1b497499323c8686325e47\
        92f267aafa3f87ca60d01cb54f29202a\
        3e784ccb7ebcdcfd45542b7f6af77874\
        2e0f4479175084aa488b3b74340678aa\
        38e22e9628b0a161fdeb0bd252173b9c\
        4e4cd0dbbd9cd3f10ef5fe5e4b034745";

    let mut drbg = Aes256CtrDrbg::new(&SEED);
    let mut buf = [0u8; 128];
    drbg.fill_bytes(&mut buf);

    let got: String = buf.iter().map(|b| format!("{b:02x}")).collect();

    assert_eq!(got, EXPECTED_HEX);
    assert_eq!(drbg.bytes_consumed(), 128);
}

/// The `RngCore` convenience methods (`next_u32`, `next_u64`,
/// `try_fill_bytes`) are deterministic from the seed: two DRBGs seeded
/// identically produce the same sequence of outputs, and `try_fill_bytes`
/// is infallible.
#[test]
fn rng_core_helpers_are_deterministic_from_seed() {
    let seed = [0xA5u8; SEEDLEN];

    let mut a = Aes256CtrDrbg::new(&seed);
    let mut b = Aes256CtrDrbg::new(&seed);

    assert_eq!(a.next_u32(), b.next_u32());
    assert_eq!(a.next_u64(), b.next_u64());

    let mut a_buf = [0u8; 20];
    let mut b_buf = [0u8; 20];
    a.try_fill_bytes(&mut a_buf)
        .expect("try_fill_bytes is infallible for Aes256CtrDrbg");
    b.try_fill_bytes(&mut b_buf)
        .expect("try_fill_bytes is infallible for Aes256CtrDrbg");
    assert_eq!(a_buf, b_buf);
}
