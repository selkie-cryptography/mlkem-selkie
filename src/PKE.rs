//! K-PKE: internal-only, IND-CPA-secure public key encryption.
//!
//! Implements section 5 of the NIST [FIPS 203] standard.
//!
//! [FIPS-203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf

use crate::{
    algebraic,
    functions::{G, PRF},
    parameters::ParameterSet,
};

/// A K-PKE ciphertext resulting from `K-PKE.Encrypt()`, defined in section 5 of [FIPS-203].
///
/// Should _not_ be used outside the bounds of ML-KEM, including [being treated as 'public' data and
/// computed over as such][cryspen-verified-mlkem]: `K-PKE` ciphertexts are created inside `ML-KEM.Decaps()` that are _not
/// public_ and must be computed over in a side-channel-free and value-independent manner.
///
/// [FIPS-203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
/// [cryspen-verified-mlkem]: https://cryspen.com/post/ml-kem-implementation/
pub(crate) struct Ciphertext<P: ParameterSet>;

/// A message to be encrypted via `K-PKE.Encrypt()` defined in [FIPS 203], section 5.
///
/// Distinct from a message that has been decrypted by `K-PKE.Decrypt()`, and distinct from the
/// messages (shared secret) returned from `ML-KEM.Encaps()` and `ML-KEM.Decaps()`.
///
/// [FIPS-203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
pub(crate) struct MessageForEncryption<P: ParameterSet>([u8; 32]);

/// A message decrypted via `K-PKE.Decrypt()` defined in [FIPS 203], section 5.
///
/// Distinct from a message that is to be encrypted by `K-PKE.Encrypt()`, and distinct from the
/// messages (shared secret) returned from `ML-KEM.Encaps()` and `ML-KEM.Decaps()`.
///
/// [FIPS-203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
pub(crate) struct DecryptedMessage<P: ParameterSet>([u8; 32]);

/// A K-PKE encryption key from section 5 of [FIPS-203].
///
/// Should _not_ be used outside the bounds of ML-KEM.
///
/// [FIPS-203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
// TODO: revisit making interior bytes pub(crate)
pub(crate) struct EncryptionKey<P: ParameterSet>();

/// A K-PKE decryption key from section 5 of [FIPS-203].
///
/// Should _not_ be used outside the bounds of ML-KEM.
///
/// [FIPS-203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
pub(crate) struct DecryptionKey<P: ParameterSet>();

impl<P> DecryptionKey<P>
where
    P: ParameterSet,
{
    /// Byte serialization of the K-PKE encapsulation key.
    ///
    /// Should be _internal only_.
    fn serialize(self) -> P::PKEDecryptionKeySerialization {
        todo!()
    }
}

/// Seed randomness used to generate the matrix Â during `K-PKE` (and thus `ML-KEM`) [key
/// generation][FIPS 203].
///
/// See section 5.1 of [FIPS 203].
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
struct MatrixSeedRandomness(pub(self) [u8; 32]);

/// Seed randomness used to sample the secret vector `s` and the noise `e` during `K-PKE` (and thus
/// `ML-KEM`) [key generation][FIPS 203].
///
/// See section 5.1 of [FIPS 203].
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
struct SecretAndNoiseSamplingRandomness(pub(self) [u8; 32]);

/// K-PKE key generation randomness seed from section 5 of [FIPS-203].
///
/// Should _not_ be used outside the bounds of ML-KEM.
///
/// [FIPS-203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
pub(crate) struct KeyGenRandomnessSeed<P: ParameterSet>([u8; 32]);

impl<P> KeyGenRandomnessSeed<P>
where
    P: ParameterSet,
{
    /// K-PKE key generation randomness seed constructor.
    pub(crate) fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl<P> From<KeyGenRandomnessSeed<P>> for (MatrixSeedRandomness, SecretAndNoiseSamplingRandomness)
where
    P: ParameterSet,
{
    fn from(
        d: KeyGenRandomnessSeed<P>,
    ) -> (MatrixSeedRandomness, SecretAndNoiseSamplingRandomness) {
        let [rho, sigma] = G(&d.0);

        return (
            MatrixSeedRandomness(rho),
            SecretAndNoiseSamplingRandomness(sigma),
        );
    }
}

/// A K-PKE key pair derived via Algorithm 12 in [FIPS 203], section 5.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
// TODO: revisit making struct members pub(crate)
pub(crate) struct KeyPair<P: ParameterSet> {
    /// K-PKE decryption key key.
    pub(crate) dk_pke: DecryptionKey<P>,
    /// ML-KEM encryption key.
    pub(crate) ek_pke: EncryptionKey<P>,
}

impl<P> KeyPair<P>
where
    P: ParameterSet,
{
    /// Generate a `K-PKE` key pair based on the provided seed randomness.
    ///
    /// This diverges from Algorithm 12 in section 5.1 of [FIPS 203] but is otherwise aligned.  We
    /// do not provided a similar randomized (internally sourcing fresh randomness) implementation
    /// as for `ML-KEM.KeyGen()` as this associated function should never be exposed in the public
    /// API, it is only used internally, and the `ML-KEM.KeyGen()` implementation that calls it will
    /// source fresh randomness for it.
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    pub(crate) fn new_derand(seed: KeyGenRandomnessSeed<P>) -> Result<KeyPair<P>, Error> {
        // (ρ,σ) ← G(d)                  ▷ expand to two pseudorandom 32-byte seeds
        let (rho, sigma) = seed.into();

        // Generate matrix Â
    }

    /// Get the K-PKE encryption key.
    pub(crate) fn ek(self) -> EncryptionKey<P> {
        self.ek_pke
    }
}
