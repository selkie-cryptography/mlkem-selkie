//! ML-KEM parameter sets from section 7 of the NIST [FIPS 203] standard.
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf

/// "n is set to 256 because the goal is to encapsulate keys with 256 bits of
/// entropy (i.e., use a plaintext size of 256 bits in Kyber.CPAPKE.Enc).
/// Smaller values of n would require to encode multiple key bits into one
/// polynomial coefficient, which requires lower noise levels and
/// therefore lowers security. Larger values of n would reduce the capability to
/// easily scale security via parameter k." - [CRYSTALS-Kyber version 3.02]
///
/// [CRYSTALS-Kyber version 3.02]: https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
pub(crate) const N: usize = 256;

/// "We choose q as a small prime satisfying n | (q − 1); this is required to
/// enable the fast NTT-based multiplication. There are two smaller primes for
/// which this property holds, namely 257 and 769.  However, for those primes we
/// would not be able to achieve negligible failure probability required for CCA
/// security, so we chose the next largest, i.e., q = 3329." - [CRYSTALS-Kyber
/// version 3.02]
///
/// [CRYSTALS-Kyber version 3.02]: https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
// TODO: should this be a `FieldElement`?
pub(crate) const Q: u16 = 3329;

/// Parameter sets for ML-KEM as defined in the NIST [FIPS 203] standard,
/// section 7.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
// TODO: revisit the *Serialization associated types when default values are
// stable / when generic_const_exprs is stable
//
// Every parameter set is a zero-sized marker, so these supertraits cost
// nothing: `Copy` lets the `PhantomData<P>` newtypes (`RqVector<P>`,
// `EncapsulationKey<P>`, ...) derive `Clone` without a separate `P: Clone`
// bound at every use site, and `Send + Sync` make the key/ciphertext types
// thread-safe (and lets the divan benchmarks run their closures across
// threads).
pub trait ParameterSet: Copy + Send + Sync {
    /// Represents the dimensions of the vectors *s* and *e* in `K-PKE.KeyGen()`
    /// and the dimensions of the matrix *Â* and the vectors *r*, *e_1*, and
    /// *e_2* in `K-PKE.Encrypt()`, as defined in section 5 of the NIST
    /// [FIPS 203] standard.
    ///
    /// "k is selected to fix the lattice dimension as a multiple of n; changing
    /// k is the main mechanism in Kyber to scale security (and as a
    /// consequence, efficiency) to different levels." - [CRYSTALS-Kyber
    /// version 3.02]
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    /// [CRYSTALS-Kyber version 3.02]: https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
    const K: usize;

    /// Represents the distribution `η₁` for generating the vectors *s* and *e*
    /// in `K-PKE.KeyGen()` and the vector *r* in `K-PKE.Encrypt()`, as
    /// defined in section 5 of the NIST [FIPS 203] standard.
    ///
    /// "The parameter η₁ defines the noise of *s* and *e* in Algorithm 4 and of
    /// *r* in Algorithm 5." - [CRYSTALS-Kyber version 3.02]
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    /// [CRYSTALS-Kyber version 3.02]: https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
    const ETA_1: usize;

    /// Represents the distribution η₂ for generating the vectors *e_1* and
    /// *e_2* in `K-PKE.Encrypt()`, as defined in section 5 of the NIST
    /// [FIPS 203] standard.
    ///
    /// "The parameter η₂ defines the noise of e1 and e2 in Algorithm 5." -
    /// [CRYSTALS-Kyber version 3.02]
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    /// [CRYSTALS-Kyber version 3.02]: https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
    const ETA_2: usize;

    /// Represents one of the byte range parameters for `Compress`,
    /// `Decompress`, `ByteEncode`, and `ByteDecode` as used in
    /// `K-PKE.Encrypt()` and `K-PKE.Decrypt()`, as defined in section 5 of
    /// the NIST [FIPS 203] standard.
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    const D_U: usize;

    /// Represents the other byte range parameter for `Compress`,
    /// `Decompress`, `ByteEncode`, and `ByteDecode` as used in
    /// `K-PKE.Encrypt()` and `K-PKE.Decrypt()`, as defined in section 5 of
    /// the NIST [FIPS 203] standard.
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    const D_V: usize;

    /// The derived `PRF_η₁` output length in bytes.
    const ETA1_X_64: usize = 64 * Self::ETA_1;

    /// The derived `PRF_η₂` output length in bytes.
    const ETA2_X_64: usize = 64 * Self::ETA_2;

    /// The derived K-PKE decryption key size in bytes.
    const PKE_DECRYPTION_KEY_SIZE: usize = 384 * Self::K;

    /// Decryption key byte serialization, a byte array of fixed length
    /// [`Self::PKE_DECRYPTION_KEY_SIZE`].
    type PKEDecryptionKeySerialization: AsRef<[u8]> + TryFrom<Vec<u8>>;

    /// The derived K-PKE encryption key size in bytes.
    const PKE_ENCRYPTION_KEY_SIZE: usize = (384 * Self::K) + 32;

    /// Encryption key byte serialization, a byte array of fixed length
    /// [`Self::PKE_ENCRYPTION_KEY_SIZE`].
    type PKEEncryptionKeySerialization: AsRef<[u8]> + TryFrom<Vec<u8>>;

    /// The derived secret decapsulation key size in bytes.
    const DECAPS_KEY_SIZE: usize = (768 * Self::K) + 96;

    /// Decapsulation key byte serialization, a byte array of fixed length
    /// [`Self::DECAPS_KEY_SIZE`].
    type DecapsKeySerialization: AsRef<[u8]> + TryFrom<Vec<u8>>;

    /// The derived public encapsulation key size in bytes.
    const ENCAPS_KEY_SIZE: usize = (384 * Self::K) + 32;

    /// Encapsulation key byte serialization, a byte array of fixed length
    /// [`Self::ENCAPS_KEY_SIZE`].
    type EncapsKeySerialization: AsRef<[u8]> + TryFrom<Vec<u8>>;

    /// The derived public ciphertext size in bytes.
    const CIPHERTEXT_SIZE: usize = 32 * ((Self::D_U * Self::K) + Self::D_V);

    /// Ciphertext byte serialization, a byte array of fixed length
    /// [`Self::CIPHERTEXT_SIZE`].
    type CiphertextSerialization: AsRef<[u8]> + TryFrom<Vec<u8>>;
}

/// The ML-KEM-512 parameter set defined in [FIPS 203], section 7.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
#[cfg(feature = "mlkem512")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MLKEM512;

#[cfg(feature = "mlkem512")]
impl ParameterSet for MLKEM512 {
    const K: usize = 2;
    const ETA_1: usize = 3;
    const ETA_2: usize = 2;
    const D_U: usize = 10;
    const D_V: usize = 4;

    type PKEDecryptionKeySerialization = [u8; Self::PKE_DECRYPTION_KEY_SIZE];
    type PKEEncryptionKeySerialization = [u8; Self::PKE_ENCRYPTION_KEY_SIZE];

    type EncapsKeySerialization = [u8; Self::ENCAPS_KEY_SIZE];
    type DecapsKeySerialization = [u8; Self::DECAPS_KEY_SIZE];
    type CiphertextSerialization = [u8; Self::CIPHERTEXT_SIZE];
}

/// The ML-KEM-768 parameter set defined in [FIPS 203], section 7.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
#[cfg(feature = "mlkem768")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MLKEM768;

#[cfg(feature = "mlkem768")]
impl ParameterSet for MLKEM768 {
    const K: usize = 3;
    const ETA_1: usize = 2;
    const ETA_2: usize = 2;
    const D_U: usize = 10;
    const D_V: usize = 4;

    type PKEDecryptionKeySerialization = [u8; Self::PKE_DECRYPTION_KEY_SIZE];
    type PKEEncryptionKeySerialization = [u8; Self::PKE_ENCRYPTION_KEY_SIZE];

    type EncapsKeySerialization = [u8; Self::ENCAPS_KEY_SIZE];
    type DecapsKeySerialization = [u8; Self::DECAPS_KEY_SIZE];
    type CiphertextSerialization = [u8; Self::CIPHERTEXT_SIZE];
}

/// The ML-KEM-1024 parameter set defined in [FIPS 203], section 7.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
#[cfg(feature = "mlkem1024")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MLKEM1024;

#[cfg(feature = "mlkem1024")]
impl ParameterSet for MLKEM1024 {
    const K: usize = 4;
    const ETA_1: usize = 2;
    const ETA_2: usize = 2;
    const D_U: usize = 11;
    const D_V: usize = 5;

    type PKEDecryptionKeySerialization = [u8; Self::PKE_DECRYPTION_KEY_SIZE];
    type PKEEncryptionKeySerialization = [u8; Self::PKE_ENCRYPTION_KEY_SIZE];

    type EncapsKeySerialization = [u8; Self::ENCAPS_KEY_SIZE];
    type DecapsKeySerialization = [u8; Self::DECAPS_KEY_SIZE];
    type CiphertextSerialization = [u8; Self::CIPHERTEXT_SIZE];
}
