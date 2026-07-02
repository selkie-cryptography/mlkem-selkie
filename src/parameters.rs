//! ML-KEM parameter sets from section 7 of the NIST [FIPS 203] standard.
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf

use core::fmt::Debug;

use zeroize::Zeroize;

/// The bounds an element type must satisfy to live in a
/// [`ParameterSet::KArray`]: enough for the vector and matrix newtypes to derive
/// `Clone`/`Debug`/`PartialEq`/`Eq`, remain `Send + Sync`, and `Zeroize`.
pub trait KElement: Clone + Debug + PartialEq + Eq + Send + Sync + Zeroize {}

impl<T: Clone + Debug + PartialEq + Eq + Send + Sync + Zeroize> KElement for T {}

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

/// The centered-binomial-distribution parameter `eta`, which FIPS 203 fixes to
/// either 2 or 3 across every parameter set (the `ETA_1` and `ETA_2` feeding
/// `PRF_eta` in [section 4.1][FIPS 203]). Modeling it as a closed enum keeps
/// `PRF` and CBD sampling from ever handling a width outside `{2, 3}`.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.1
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eta {
    /// `eta = 2`.
    Two,
    /// `eta = 3`.
    Three,
}

impl From<Eta> for usize {
    fn from(eta: Eta) -> usize {
        match eta {
            Eta::Two => 2,
            Eta::Three => 3,
        }
    }
}

/// Parameter sets for ML-KEM as defined in the NIST [FIPS 203] standard,
/// section 7.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
// TODO: collapse the *Serialization associated types and their `*_from_fn`
// builders into plain `[u8; SIZE]` return types once `generic_const_exprs` is
// stable — they exist only because a generic `[u8; Self::SIZE]` is not yet
// expressible on stable Rust.
//
// Every parameter set is a zero-sized marker, so these supertraits cost
// nothing. `Copy`/`Clone` let the `P`-parameterized newtypes (`RqVector<P>`,
// `TqMatrix<P>`, `EncapsulationKey<P>`, ...) derive `Clone` without a separate
// `P: Clone` bound at every use site; `Debug`/`PartialEq`/`Eq` likewise enable
// those derives, and are required so a `TqVector<P>` (itself only `Eq` when `P`
// is) can sit inside another parameter set's `KArray`; and `Send + Sync` make
// the key/ciphertext types thread-safe (and let the divan benchmarks run their
// closures across threads).
pub trait ParameterSet: Copy + Send + Sync + Debug + PartialEq + Eq {
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
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
    /// [CRYSTALS-Kyber version 3.02]: https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
    const K: usize;

    /// A length-[`Self::K`] array of `T`: the heap-free backing store for the
    /// `K`-dimensional vectors and matrices over the polynomial rings.
    ///
    /// Concrete in each parameter set (`[T; 2]`, `[T; 3]`, `[T; 4]`), which
    /// sidesteps the unstable `generic_const_exprs` that a generic `[T;
    /// Self::K]` would otherwise require.
    type KArray<T>: AsRef<[T]> + Clone + Debug + PartialEq + Eq + Send + Sync + Zeroize
    where
        T: KElement;

    /// Builds a [`Self::KArray`] by applying `f` to each index in `0..K`.
    fn k_array_from_fn<T>(f: impl FnMut(usize) -> T) -> Self::KArray<T>
    where
        T: KElement;

    /// Represents the distribution `η₁` for generating the vectors *s* and *e*
    /// in `K-PKE.KeyGen()` and the vector *r* in `K-PKE.Encrypt()`, as
    /// defined in section 5 of the NIST [FIPS 203] standard.
    ///
    /// "The parameter η₁ defines the noise of *s* and *e* in Algorithm 4 and of
    /// *r* in Algorithm 5." - [CRYSTALS-Kyber version 3.02]
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
    /// [CRYSTALS-Kyber version 3.02]: https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
    const ETA_1: Eta;

    /// Represents the distribution η₂ for generating the vectors *e_1* and
    /// *e_2* in `K-PKE.Encrypt()`, as defined in section 5 of the NIST
    /// [FIPS 203] standard.
    ///
    /// "The parameter η₂ defines the noise of e1 and e2 in Algorithm 5." -
    /// [CRYSTALS-Kyber version 3.02]
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
    /// [CRYSTALS-Kyber version 3.02]: https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
    const ETA_2: Eta;

    /// Represents one of the byte range parameters for `Compress`,
    /// `Decompress`, `ByteEncode`, and `ByteDecode` as used in
    /// `K-PKE.Encrypt()` and `K-PKE.Decrypt()`, as defined in section 5 of
    /// the NIST [FIPS 203] standard.
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
    const D_U: usize;

    /// Represents the other byte range parameter for `Compress`,
    /// `Decompress`, `ByteEncode`, and `ByteDecode` as used in
    /// `K-PKE.Encrypt()` and `K-PKE.Decrypt()`, as defined in section 5 of
    /// the NIST [FIPS 203] standard.
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
    const D_V: usize;

    /// The derived K-PKE decryption key size in bytes.
    ///
    /// Load-bearing at the KEM boundary: `DecapsulationKey::from_bytes`
    /// splits its input at this offset. There is no standalone
    /// `PKEDecryptionKeySerialization` because the K-PKE decryption key
    /// reaches the wire only through `DecapsulationKey::to_bytes`, which
    /// chains its byte-encoding iterator into [`Self::decaps_key_from_fn`].
    const PKE_DECRYPTION_KEY_SIZE: usize = 384 * Self::K;

    /// The derived K-PKE encryption key size in bytes.
    ///
    /// Load-bearing at the K-PKE boundary: `EncryptionKey::from_bytes`
    /// debug-asserts against it. No matching `PKEEncryptionKeySerialization`
    /// type — the K-PKE encryption key reaches the wire only through its
    /// streaming byte-encoding iterator, chained into
    /// [`Self::encaps_key_from_fn`] at the KEM boundary.
    const PKE_ENCRYPTION_KEY_SIZE: usize = (384 * Self::K) + 32;

    /// The derived secret decapsulation key size in bytes.
    const DECAPS_KEY_SIZE: usize = (768 * Self::K) + 96;

    /// Decapsulation key byte serialization, a byte array of fixed length
    /// [`Self::DECAPS_KEY_SIZE`].
    type DecapsKeySerialization: AsRef<[u8]>;

    /// Builds a [`Self::DecapsKeySerialization`] on the stack by applying `f`
    /// to each byte index. Concrete per parameter set, like
    /// [`Self::ciphertext_from_fn`].
    fn decaps_key_from_fn(f: impl FnMut(usize) -> u8) -> Self::DecapsKeySerialization;

    /// The derived public encapsulation key size in bytes.
    const ENCAPS_KEY_SIZE: usize = (384 * Self::K) + 32;

    /// Encapsulation key byte serialization, a byte array of fixed length
    /// [`Self::ENCAPS_KEY_SIZE`].
    ///
    /// `Send + Sync` (like [`Self::CiphertextSerialization`]) so the value
    /// `EncapsulationKey::to_bytes` returns can cross threads — the divan
    /// benchmarks capture it.
    type EncapsKeySerialization: AsRef<[u8]> + Send + Sync;

    /// Builds a [`Self::EncapsKeySerialization`] by applying `f` to each byte
    /// index, assembling the encapsulation-key buffer on the stack so
    /// `to_bytes` and `H(ek)` need no heap. Concrete per parameter set,
    /// like [`Self::ciphertext_from_fn`].
    fn encaps_key_from_fn(f: impl FnMut(usize) -> u8) -> Self::EncapsKeySerialization;

    /// The derived public ciphertext size in bytes.
    const CIPHERTEXT_SIZE: usize = 32 * ((Self::D_U * Self::K) + Self::D_V);

    /// Ciphertext byte serialization, a byte array of fixed length
    /// [`Self::CIPHERTEXT_SIZE`].
    ///
    /// Carries `Send + Sync` (like [`Self::KArray`]) so the heap-free
    /// ciphertext type that wraps it stays thread-safe — the divan benchmarks
    /// move ciphertexts across threads.
    type CiphertextSerialization: AsRef<[u8]> + Send + Sync;

    /// Builds a [`Self::CiphertextSerialization`] by applying `f` to each byte
    /// index in `0..CIPHERTEXT_SIZE`.
    ///
    /// Assembles the fixed-size ciphertext buffer on the stack, so
    /// `K-PKE.Encrypt` can serialize the compressed `(u, v)` pair straight into
    /// it without a heap allocation. Concrete in each parameter set for the
    /// same reason as [`Self::k_array_from_fn`]: a generic `[u8;
    /// Self::CIPHERTEXT_SIZE]` would otherwise require the unstable
    /// `generic_const_exprs`.
    fn ciphertext_from_fn(f: impl FnMut(usize) -> u8) -> Self::CiphertextSerialization;
}

/// The ML-KEM-512 parameter set defined in [FIPS 203], section 7.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
#[cfg(feature = "mlkem512")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MLKEM512;

#[cfg(feature = "mlkem512")]
impl ParameterSet for MLKEM512 {
    const K: usize = 2;
    const ETA_1: Eta = Eta::Three;
    const ETA_2: Eta = Eta::Two;
    const D_U: usize = 10;
    const D_V: usize = 4;

    type KArray<T>
        = [T; 2]
    where
        T: KElement;

    fn k_array_from_fn<T>(f: impl FnMut(usize) -> T) -> [T; 2]
    where
        T: KElement,
    {
        core::array::from_fn(f)
    }

    fn ciphertext_from_fn(f: impl FnMut(usize) -> u8) -> [u8; Self::CIPHERTEXT_SIZE] {
        core::array::from_fn(f)
    }

    fn encaps_key_from_fn(f: impl FnMut(usize) -> u8) -> [u8; Self::ENCAPS_KEY_SIZE] {
        core::array::from_fn(f)
    }

    fn decaps_key_from_fn(f: impl FnMut(usize) -> u8) -> [u8; Self::DECAPS_KEY_SIZE] {
        core::array::from_fn(f)
    }

    type EncapsKeySerialization = [u8; Self::ENCAPS_KEY_SIZE];
    type DecapsKeySerialization = [u8; Self::DECAPS_KEY_SIZE];
    type CiphertextSerialization = [u8; Self::CIPHERTEXT_SIZE];
}

/// The ML-KEM-768 parameter set defined in [FIPS 203], section 7.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
#[cfg(feature = "mlkem768")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MLKEM768;

#[cfg(feature = "mlkem768")]
impl ParameterSet for MLKEM768 {
    const K: usize = 3;
    const ETA_1: Eta = Eta::Two;
    const ETA_2: Eta = Eta::Two;
    const D_U: usize = 10;
    const D_V: usize = 4;

    type KArray<T>
        = [T; 3]
    where
        T: KElement;

    fn k_array_from_fn<T>(f: impl FnMut(usize) -> T) -> [T; 3]
    where
        T: KElement,
    {
        core::array::from_fn(f)
    }

    fn ciphertext_from_fn(f: impl FnMut(usize) -> u8) -> [u8; Self::CIPHERTEXT_SIZE] {
        core::array::from_fn(f)
    }

    fn encaps_key_from_fn(f: impl FnMut(usize) -> u8) -> [u8; Self::ENCAPS_KEY_SIZE] {
        core::array::from_fn(f)
    }

    fn decaps_key_from_fn(f: impl FnMut(usize) -> u8) -> [u8; Self::DECAPS_KEY_SIZE] {
        core::array::from_fn(f)
    }

    type EncapsKeySerialization = [u8; Self::ENCAPS_KEY_SIZE];
    type DecapsKeySerialization = [u8; Self::DECAPS_KEY_SIZE];
    type CiphertextSerialization = [u8; Self::CIPHERTEXT_SIZE];
}

/// The ML-KEM-1024 parameter set defined in [FIPS 203], section 7.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
#[cfg(feature = "mlkem1024")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MLKEM1024;

#[cfg(feature = "mlkem1024")]
impl ParameterSet for MLKEM1024 {
    const K: usize = 4;
    const ETA_1: Eta = Eta::Two;
    const ETA_2: Eta = Eta::Two;
    const D_U: usize = 11;
    const D_V: usize = 5;

    type KArray<T>
        = [T; 4]
    where
        T: KElement;

    fn k_array_from_fn<T>(f: impl FnMut(usize) -> T) -> [T; 4]
    where
        T: KElement,
    {
        core::array::from_fn(f)
    }

    fn ciphertext_from_fn(f: impl FnMut(usize) -> u8) -> [u8; Self::CIPHERTEXT_SIZE] {
        core::array::from_fn(f)
    }

    fn encaps_key_from_fn(f: impl FnMut(usize) -> u8) -> [u8; Self::ENCAPS_KEY_SIZE] {
        core::array::from_fn(f)
    }

    fn decaps_key_from_fn(f: impl FnMut(usize) -> u8) -> [u8; Self::DECAPS_KEY_SIZE] {
        core::array::from_fn(f)
    }

    type EncapsKeySerialization = [u8; Self::ENCAPS_KEY_SIZE];
    type DecapsKeySerialization = [u8; Self::DECAPS_KEY_SIZE];
    type CiphertextSerialization = [u8; Self::CIPHERTEXT_SIZE];
}
