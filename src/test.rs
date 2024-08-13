/// Test-related functionality.

/// Known Answer Tests (KATs)
///
/// Follows the [encoding conventions] in the known answer tests (KATs) for [ML-KEM].
///
/// [ML-KEM]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
/// [encoding conventions]: https://github.com/cryspen/libcrux/tree/5fc2cbad58f3f3e515490502e82b1c4600d5e6e3/tests/kyber_kats
pub struct KnownAnswerTest {
    keygen_seed: KeyGenRandomness,
    encaps_key_hash: [u8; 32],
    decaps_key_hash: [u8; 32],
    encaps_seed: [u8; 32],
    ciphertext_hash: [u8; 32],
    shared_secret: [u8; 32],
}
