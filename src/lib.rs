// [ML-KEM] Selkie
//
// [ML-KEM]: https://doi.org/10.6028/NIST.FIPS.203

#![doc(
    html_logo_url = "https://user-images.githubusercontent.com/552961/197638905-f5144be3-a2f2-48c2-9ecb-26e4e34d8d8a.svg#gh-light-mode-only"
)]
#![doc = include_str!("../README.md")]
#![allow(mixed_script_confusables)]
#![allow(non_snake_case)]
#![deny(missing_docs, clippy::indexing_slicing, clippy::unwrap_used)]
// `deny`, not `forbid`: the only `unsafe` in the crate is the SIMD intrinsics in
// the `poly::arch::{neon,avx2}` backends, each call carrying a `// SAFETY:` note.
// Everything outside those modules stays unsafe-free.
#![deny(unsafe_code)]
#![warn(rust_2018_idioms)]

use rand_core::{CryptoRng, RngCore};

// The internal building blocks (field/ring arithmetic, K-PKE, the samplers, the
// serialization layer, and the symmetric primitives) are private by default and
// re-exposed publicly under the `expose-internals` feature for white-box tests,
// fuzzing, and benchmarks.
#[cfg(not(feature = "expose-internals"))]
mod PKE;
#[cfg(feature = "expose-internals")]
pub mod PKE;

#[cfg(not(feature = "expose-internals"))]
mod algebraic;
#[cfg(feature = "expose-internals")]
pub mod algebraic;

#[cfg(not(feature = "expose-internals"))]
mod encoding;
#[cfg(feature = "expose-internals")]
pub mod encoding;

#[cfg(not(feature = "expose-internals"))]
mod functions;
#[cfg(feature = "expose-internals")]
pub mod functions;

#[cfg(not(feature = "expose-internals"))]
mod sampling;
#[cfg(feature = "expose-internals")]
pub mod sampling;

mod drbg;
mod parameters;

#[cfg(test)]
mod tests;

pub use drbg::{Aes256CtrDrbg, SEEDLEN};
#[cfg(feature = "mlkem512")]
pub use parameters::MLKEM512;
#[cfg(feature = "mlkem768")]
pub use parameters::MLKEM768;
#[cfg(feature = "mlkem1024")]
pub use parameters::MLKEM1024;
pub use parameters::{Eta, ParameterSet};

use crate::{
    PKE::Ciphertext as PkeCiphertext,
    functions::{G, H, J},
};

/// Errors returned by the public ML-KEM API.
///
/// These arise only from the input-validation checks of [section 7] of
/// FIPS 203: malformed key or ciphertext encodings. A well-formed but malleated
/// ciphertext is *not* an error — `Decaps` returns an implicit-rejection shared
/// secret for it.
///
/// [section 7]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#section.7
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// An encapsulation key was not `384 * K + 32` bytes long.
    InvalidEncapsulationKeyLength,
    /// An encapsulation key encoded a coefficient not in `0..q` (the "modulus
    /// check" of section 7.2).
    EncapsulationKeyModulusCheckFailed,
    /// A decapsulation key was not `768 * K + 96` bytes long.
    InvalidDecapsulationKeyLength,
    /// A decapsulation key's embedded encapsulation-key hash did not match.
    DecapsulationKeyHashMismatch,
    /// A ciphertext was not `32 * (D_U * K + D_V)` bytes long.
    InvalidCiphertextLength,
}

/// A 32-byte ML-KEM shared secret `K`.
///
/// The output of both [`EncapsulationKey::encapsulate`] and
/// [`DecapsulationKey::decapsulate`]. Carries no `PartialEq`: callers that must
/// compare shared secrets should do so in constant time.
pub struct SharedSecret([u8; 32]);

impl SharedSecret {
    /// Returns the shared secret bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// An ML-KEM ciphertext.
///
/// A thin typed wrapper over the K-PKE ciphertext bytes, sized
/// `32 * (D_U * K + D_V)` for the parameter set `P`.
pub struct Ciphertext<P: ParameterSet>(PkeCiphertext<P>);

impl<P: ParameterSet> Ciphertext<P> {
    /// Parses a ciphertext from bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCiphertextLength`] unless the input is
    /// `P::CIPHERTEXT_SIZE` bytes long.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != P::CIPHERTEXT_SIZE {
            return Err(Error::InvalidCiphertextLength);
        }

        Ok(Self(PkeCiphertext::from_bytes(bytes.to_vec())))
    }

    /// Returns the serialized ciphertext bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

/// The SHA3-256 hash `H(ek)` of an encapsulation key.
///
/// Bound into the shared-secret derivation of `ML-KEM.Encaps` and `Decaps`, and
/// stored inside the decapsulation key so the binding can be re-checked when
/// the key is parsed.
#[derive(Clone, Copy, PartialEq, Eq)]
struct EncapsulationKeyHash([u8; 32]);

impl EncapsulationKeyHash {
    /// Returns the 32 hash bytes.
    fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// The Fujisaki–Okamoto implicit-rejection seed `z`: 32 random bytes that
/// derive the rejection shared secret in `ML-KEM.Decaps`.
// No `PartialEq`/`Eq`: this is secret key material.
struct RejectionSeed([u8; 32]);

impl RejectionSeed {
    /// Returns the 32 seed bytes.
    fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl From<[u8; 32]> for RejectionSeed {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// An ML-KEM encapsulation (public) key.
///
/// Identical to the underlying K-PKE encryption key; see [FIPS 203 section
/// 6.1].
///
/// [FIPS 203 section 6.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.6.1
#[derive(Clone)]
pub struct EncapsulationKey<P: ParameterSet> {
    /// The K-PKE encryption key.
    ek_pke: PKE::EncryptionKey<P>,
}

impl<P: ParameterSet> From<&EncapsulationKey<P>> for EncapsulationKeyHash {
    fn from(ek: &EncapsulationKey<P>) -> Self {
        Self(H(&ek.to_bytes()))
    }
}

impl<P: ParameterSet> EncapsulationKey<P> {
    /// Serializes the encapsulation key to `384 * K + 32` bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.ek_pke.to_bytes()
    }

    /// Parses an encapsulation key from bytes, applying the [section 7.2] input
    /// validation.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidEncapsulationKeyLength`] if the length is wrong.
    /// - [`Error::EncapsulationKeyModulusCheckFailed`] if any coefficient is
    ///   not in `0..q`, detected by a `ByteDecode_12` / `ByteEncode_12`
    ///   round-trip.
    ///
    /// [section 7.2]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.7.2
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != P::ENCAPS_KEY_SIZE {
            return Err(Error::InvalidEncapsulationKeyLength);
        }

        let ek_pke = PKE::EncryptionKey::<P>::from_bytes(bytes);

        // Modulus check: the parsed key must re-encode to the original bytes.
        if ek_pke.to_bytes() != bytes {
            return Err(Error::EncapsulationKeyModulusCheckFailed);
        }

        Ok(Self { ek_pke })
    }

    /// `ML-KEM.Encaps`: generates a shared secret and a ciphertext
    /// encapsulating it under this key, sourcing fresh randomness.
    ///
    /// Implements [Algorithm 20] of FIPS 203 (its [Algorithm 17] internal
    /// core).
    ///
    /// [Algorithm 20]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.20
    /// [Algorithm 17]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.17
    pub fn encapsulate<R: CryptoRng + RngCore>(
        &self,
        rng: &mut R,
    ) -> (SharedSecret, Ciphertext<P>) {
        let mut m = [0u8; 32];
        rng.fill_bytes(&mut m);

        self.encapsulate_derand(&m)
    }

    /// `ML-KEM.Encaps_internal`: the derandomized core of
    /// [`Self::encapsulate`], taking the message `m` explicitly.
    ///
    /// Exposed for known-answer and Wycheproof testing, where the encapsulation
    /// message is fixed by the vector.
    ///
    /// Implements [Algorithm 17] of FIPS 203.
    ///
    /// [Algorithm 17]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.17
    #[must_use]
    pub fn encapsulate_derand(&self, m: &[u8; 32]) -> (SharedSecret, Ciphertext<P>) {
        // (K, r) <- G(m || H(ek))
        let mut g_input = m.to_vec();
        g_input.extend_from_slice(&H(&self.to_bytes()));
        let (k, r) = G(&g_input);

        let ciphertext = self.ek_pke.encrypt(m, &r);

        (SharedSecret(k), Ciphertext(ciphertext))
    }
}

/// An ML-KEM decapsulation (secret) key.
///
/// Bundles the K-PKE decryption key with the material needed to run the
/// Fujisaki–Okamoto re-encryption check: the encapsulation key, its hash, and
/// the implicit-rejection seed `z`. See [FIPS 203 section 6.1].
///
/// [FIPS 203 section 6.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.6.1
// No `PartialEq`/`Eq`/`Hash`: this is secret key material.
pub struct DecapsulationKey<P: ParameterSet> {
    /// The K-PKE decryption key.
    dk_pke: PKE::DecryptionKey<P>,
    /// The encapsulation key, needed to re-encrypt during decapsulation.
    ek: EncapsulationKey<P>,
    /// `H(ek)`, mixed into the shared-secret derivation.
    h_ek: EncapsulationKeyHash,
    /// The implicit-rejection seed `z`.
    z: RejectionSeed,
}

impl<P: ParameterSet> DecapsulationKey<P> {
    /// Returns the corresponding encapsulation key.
    #[must_use]
    pub fn encapsulation_key(&self) -> &EncapsulationKey<P> {
        &self.ek
    }

    /// Serializes the decapsulation key to `768 * K + 96` bytes, as
    /// `dk_PKE ‖ ek ‖ H(ek) ‖ z`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.dk_pke.to_bytes();
        bytes.extend_from_slice(&self.ek.to_bytes());
        bytes.extend_from_slice(self.h_ek.as_bytes());
        bytes.extend_from_slice(self.z.as_bytes());

        bytes
    }

    /// Parses a decapsulation key from bytes.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidDecapsulationKeyLength`] if the length is wrong.
    /// - [`Error::DecapsulationKeyHashMismatch`] if the embedded `H(ek)` does
    ///   not match a freshly computed hash of the embedded encapsulation key.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != P::DECAPS_KEY_SIZE {
            return Err(Error::InvalidDecapsulationKeyLength);
        }

        let (dk_pke_bytes, rest) = bytes.split_at(P::PKE_DECRYPTION_KEY_SIZE);
        let (ek_bytes, rest) = rest.split_at(P::ENCAPS_KEY_SIZE);
        let (h_ek_bytes, z_bytes) = rest.split_at(32);

        let dk_pke = PKE::DecryptionKey::<P>::from_bytes(dk_pke_bytes);
        let ek = EncapsulationKey {
            ek_pke: PKE::EncryptionKey::<P>::from_bytes(ek_bytes),
        };

        let mut h_ek_array = [0u8; 32];
        h_ek_array.copy_from_slice(h_ek_bytes);
        let h_ek = EncapsulationKeyHash(h_ek_array);
        if EncapsulationKeyHash::from(&ek) != h_ek {
            return Err(Error::DecapsulationKeyHashMismatch);
        }

        let mut z = [0u8; 32];
        z.copy_from_slice(z_bytes);

        Ok(Self {
            dk_pke,
            ek,
            h_ek,
            z: RejectionSeed::from(z),
        })
    }

    /// `ML-KEM.Decaps`: recovers the shared secret from a ciphertext, with
    /// implicit rejection.
    ///
    /// Implements [Algorithm 18] of FIPS 203 (the internal core of
    /// [Algorithm 21]). On a ciphertext that does not re-encrypt to itself, the
    /// returned secret is derived from the rejection seal `z` rather than from
    /// the decrypted message.
    ///
    /// [Algorithm 18]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.18
    /// [Algorithm 21]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.21
    // TODO(ct): the re-encryption comparison and the field arithmetic it feeds
    // are variable-time. The decrypted message m' and ciphertext c' are
    // secret-derived (Algorithm 18 lines 5, 8); make the comparison and
    // selection constant-time before production use.
    #[must_use]
    pub fn decapsulate(&self, ciphertext: &Ciphertext<P>) -> SharedSecret {
        // m' <- K-PKE.Decrypt(dk_PKE, c)
        let m_prime = self.dk_pke.decrypt(&ciphertext.0);

        // (K', r') <- G(m' || h)
        let mut g_input = m_prime.to_vec();
        g_input.extend_from_slice(self.h_ek.as_bytes());
        let (k_prime, r_prime) = G(&g_input);

        // K_bar <- J(z || c)
        let mut j_input = self.z.as_bytes().to_vec();
        j_input.extend_from_slice(ciphertext.as_bytes());
        let k_bar = J(&j_input);

        // c' <- K-PKE.Encrypt(ek_PKE, m', r'); implicit reject if c != c'.
        let c_prime = self.ek.ek_pke.encrypt(&m_prime, &r_prime);

        if ciphertext.as_bytes() == c_prime.as_bytes() {
            SharedSecret(k_prime)
        } else {
            SharedSecret(k_bar)
        }
    }
}

/// An ML-KEM key pair: an encapsulation key and its decapsulation key.
///
/// See [FIPS 203 section 6.1].
///
/// [FIPS 203 section 6.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.6.1
pub struct KeyPair<P: ParameterSet> {
    /// The secret decapsulation key.
    pub decapsulation_key: DecapsulationKey<P>,
    /// The public encapsulation key.
    pub encapsulation_key: EncapsulationKey<P>,
}

impl<P: ParameterSet> KeyPair<P> {
    /// `ML-KEM.KeyGen`: generates a fresh key pair, sourcing all randomness
    /// from `rng`.
    ///
    /// Implements [Algorithm 19] of FIPS 203. The 32-byte seeds `d` and `z` are
    /// drawn from `rng`, which must be a NIST SP 800-90A/B/C-approved RBG of
    /// the security strength required by the parameter set (see section
    /// 3.3).
    ///
    /// [Algorithm 19]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.19
    pub fn generate<R: CryptoRng + RngCore>(rng: &mut R) -> Self {
        let mut seed = [0u8; 64];
        rng.fill_bytes(&mut seed);

        Self::generate_derand(&seed)
    }

    /// `ML-KEM.KeyGen_internal`: the derandomized core of [`Self::generate`],
    /// taking the 64-byte seed `d ‖ z` explicitly.
    ///
    /// The seed concatenates the K-PKE key-generation seed `d` (first 32 bytes)
    /// and the implicit-rejection seed `z` (last 32 bytes), following the
    /// known-answer-test encoding convention. Exposed for KAT and Wycheproof
    /// replay.
    ///
    /// Implements [Algorithm 16] of FIPS 203.
    ///
    /// [Algorithm 16]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.16
    #[must_use]
    pub fn generate_derand(seed: &[u8; 64]) -> Self {
        let (d, z) = seed.split_at(32);

        let mut d_seed = [0u8; 32];
        d_seed.copy_from_slice(d);
        let mut z_seed = [0u8; 32];
        z_seed.copy_from_slice(z);

        let PKE::KeyPair { dk_pke, ek_pke } =
            PKE::KeyPair::new_derand(PKE::KeyGenRandomnessSeed::<P>::new(d_seed));

        let encapsulation_key = EncapsulationKey { ek_pke };
        let h_ek = EncapsulationKeyHash::from(&encapsulation_key);

        let decapsulation_key = DecapsulationKey {
            dk_pke,
            ek: encapsulation_key.clone(),
            h_ek,
            z: RejectionSeed::from(z_seed),
        };

        Self {
            decapsulation_key,
            encapsulation_key,
        }
    }
}

/// ML-KEM-512 type aliases.
///
/// Convenience specializations of the generic API for the [`MLKEM512`]
/// parameter set, so callers can write `mlkem512::KeyPair` rather than
/// `KeyPair<MLKEM512>`. Gated behind the `mlkem512` feature.
#[cfg(feature = "mlkem512")]
pub mod mlkem512 {
    pub use crate::{Error, SharedSecret};

    /// An ML-KEM-512 key pair.
    pub type KeyPair = crate::KeyPair<crate::MLKEM512>;
    /// An ML-KEM-512 encapsulation (public) key.
    pub type EncapsulationKey = crate::EncapsulationKey<crate::MLKEM512>;
    /// An ML-KEM-512 decapsulation (secret) key.
    pub type DecapsulationKey = crate::DecapsulationKey<crate::MLKEM512>;
    /// An ML-KEM-512 ciphertext.
    pub type Ciphertext = crate::Ciphertext<crate::MLKEM512>;
}

/// ML-KEM-768 type aliases.
///
/// Convenience specializations of the generic API for the [`MLKEM768`]
/// parameter set, so callers can write `mlkem768::KeyPair` rather than
/// `KeyPair<MLKEM768>`. Gated behind the `mlkem768` feature.
#[cfg(feature = "mlkem768")]
pub mod mlkem768 {
    pub use crate::{Error, SharedSecret};

    /// An ML-KEM-768 key pair.
    pub type KeyPair = crate::KeyPair<crate::MLKEM768>;
    /// An ML-KEM-768 encapsulation (public) key.
    pub type EncapsulationKey = crate::EncapsulationKey<crate::MLKEM768>;
    /// An ML-KEM-768 decapsulation (secret) key.
    pub type DecapsulationKey = crate::DecapsulationKey<crate::MLKEM768>;
    /// An ML-KEM-768 ciphertext.
    pub type Ciphertext = crate::Ciphertext<crate::MLKEM768>;
}

/// ML-KEM-1024 type aliases.
///
/// Convenience specializations of the generic API for the [`MLKEM1024`]
/// parameter set, so callers can write `mlkem1024::KeyPair` rather than
/// `KeyPair<MLKEM1024>`. Gated behind the `mlkem1024` feature.
#[cfg(feature = "mlkem1024")]
pub mod mlkem1024 {
    pub use crate::{Error, SharedSecret};

    /// An ML-KEM-1024 key pair.
    pub type KeyPair = crate::KeyPair<crate::MLKEM1024>;
    /// An ML-KEM-1024 encapsulation (public) key.
    pub type EncapsulationKey = crate::EncapsulationKey<crate::MLKEM1024>;
    /// An ML-KEM-1024 decapsulation (secret) key.
    pub type DecapsulationKey = crate::DecapsulationKey<crate::MLKEM1024>;
    /// An ML-KEM-1024 ciphertext.
    pub type Ciphertext = crate::Ciphertext<crate::MLKEM1024>;
}
