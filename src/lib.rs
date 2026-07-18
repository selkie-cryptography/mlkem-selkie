#![doc = include_str!("../README.md")]
#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![deny(missing_docs, clippy::indexing_slicing, clippy::unwrap_used)]
// `deny`, not `forbid`: the only `unsafe` in the crate is the SIMD intrinsics in
// the `poly::arch::{neon,avx2}` backends, each call carrying a `// SAFETY:` note.
// Everything outside those modules stays unsafe-free.
#![deny(unsafe_code)]
#![warn(rust_2018_idioms, unused_lifetimes, unused_qualifications)]

use subtle::{ConditionallySelectable, ConstantTimeEq};
use zeroize::{Zeroize, ZeroizeOnDrop};

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

#[cfg(not(feature = "expose-internals"))]
mod parameters;
#[cfg(feature = "expose-internals")]
pub mod parameters;

#[cfg(test)]
mod tests;

// Valgrind declassification hooks for the constant-time test harness. Only
// compiled under `--features ctgrind`; call sites cfg-gate too.
#[cfg(feature = "ctgrind")]
mod ctgrind;

#[cfg(feature = "mlkem512")]
pub use parameters::MLKEM512;
#[cfg(feature = "mlkem768")]
pub use parameters::MLKEM768;
#[cfg(feature = "mlkem1024")]
pub use parameters::MLKEM1024;
pub use parameters::ParameterSet;

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
/// compare shared secrets should do so in constant time. A copy of
/// `*as_bytes()` into a plain `[u8; 32]` does not inherit the zeroization.
#[derive(Zeroize, ZeroizeOnDrop)]
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
    /// Parses a ciphertext from bytes; see the [`TryFrom`] impl for the error
    /// conditions.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        Self::try_from(bytes)
    }

    /// Returns the serialized ciphertext bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl<P: ParameterSet> TryFrom<&[u8]> for Ciphertext<P> {
    type Error = Error;

    /// Parses a ciphertext from bytes, applying the [section 7.3] ciphertext
    /// type check.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCiphertextLength`] unless the input is
    /// `P::CIPHERTEXT_SIZE` bytes long ([section 7.3] check 1).
    ///
    /// [section 7.3]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.7.3
    fn try_from(bytes: &[u8]) -> Result<Self, Error> {
        // Ciphertext type check (FIPS 203 section 7.3, decapsulation input
        // check 1): c must be 32 * (D_U * K + D_V) bytes, else
        // `InvalidCiphertextLength`. Run on every decapsulation, per section 7.3.
        if bytes.len() != P::CIPHERTEXT_SIZE {
            return Err(Error::InvalidCiphertextLength);
        }

        Ok(Self(PkeCiphertext::from_bytes(bytes)))
    }
}

/// The SHA3-256 hash `H(ek)` of an encapsulation key.
///
/// Bound into the shared-secret derivation of `ML-KEM.Encaps` and `Decaps`, and
/// stored inside the decapsulation key so the binding can be re-checked when
/// the key is parsed.
#[derive(Clone, Copy, PartialEq, Eq, Zeroize)]
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
#[derive(Zeroize, ZeroizeOnDrop)]
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
/// 6.1]. Callers usually reach one via
/// [`DecapsulationKey::encapsulation_key`] or by parsing the on-wire bytes
/// through [`Self::from_bytes`].
///
/// [FIPS 203 section 6.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.6.1
#[derive(Clone)]
pub struct EncapsulationKey<P: ParameterSet> {
    /// The K-PKE encryption key.
    ek_pke: PKE::EncryptionKey<P>,
}

impl<P: ParameterSet> From<&EncapsulationKey<P>> for EncapsulationKeyHash {
    fn from(ek: &EncapsulationKey<P>) -> Self {
        Self(H(ek.to_bytes().as_ref()))
    }
}

impl<P: ParameterSet> EncapsulationKey<P> {
    /// Serializes the encapsulation key to its `384 * K + 32` bytes
    /// (`P::EncapsKeySerialization`), assembled on the stack with no heap
    /// allocation.
    #[must_use]
    pub fn to_bytes(&self) -> P::EncapsKeySerialization {
        let mut bytes = self.ek_pke.bytes();
        P::encaps_key_from_fn(|_| bytes.next().unwrap_or(0))
    }

    /// Parses an encapsulation key from bytes; see the [`TryFrom`] impl for the
    /// section 7.2 validation and error conditions.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        Self::try_from(bytes)
    }

    /// `ML-KEM.Encaps`: generates a shared secret and a ciphertext
    /// encapsulating it under this key.
    ///
    /// Implements [Algorithm 20] of FIPS 203 (its [Algorithm 17] internal
    /// core). The encapsulation randomness `m` is drawn from the OS via
    /// [`getrandom`].
    ///
    /// # Panics
    ///
    /// Panics if the OS entropy source is unavailable.
    ///
    /// [Algorithm 20]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.20
    /// [Algorithm 17]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.17
    /// [`getrandom`]: getrandom::getrandom
    #[must_use]
    pub fn encapsulate(&self) -> (SharedSecret, Ciphertext<P>) {
        let mut m = [0u8; 32];

        getrandom::getrandom(&mut m)
            .expect("ML-KEM.Encaps: OS entropy source (`getrandom`) unavailable");

        let result = self.encapsulate_derand(&m);
        m.zeroize();

        result
    }

    /// `ML-KEM.Encaps_internal`: the derandomized core of
    /// [`Self::encapsulate`], taking the 32 bytes of encapsulation
    /// randomness `m` explicitly.
    ///
    /// Specified by [Algorithm 17] of FIPS 203 §6. `pub` only under the
    /// `expose-internals` feature (for KAT replay); `pub(crate)` otherwise,
    /// since [`Self::encapsulate`] calls it.
    ///
    /// [Algorithm 17]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.17
    #[cfg(feature = "expose-internals")]
    #[must_use]
    pub fn encapsulate_derand(&self, m: &[u8; 32]) -> (SharedSecret, Ciphertext<P>) {
        // (K, r) <- G(m || H(ek)); preimage assembled in a 64-byte stack buffer.
        let mut g_input = [0u8; 64];
        let (m_part, h_part) = g_input.split_at_mut(32);
        m_part.copy_from_slice(m);
        h_part.copy_from_slice(&H(self.to_bytes().as_ref()));
        let (k, mut r) = G(&g_input);

        let ciphertext = self.ek_pke.encrypt(m, &r);

        // `m` is borrowed (caller's responsibility); `g_input` and `r` are
        // ours to zeroize.
        g_input.zeroize();
        r.zeroize();

        (SharedSecret(k), Ciphertext(ciphertext))
    }

    #[cfg(not(feature = "expose-internals"))]
    #[must_use]
    pub(crate) fn encapsulate_derand(&self, m: &[u8; 32]) -> (SharedSecret, Ciphertext<P>) {
        let mut g_input = [0u8; 64];
        let (m_part, h_part) = g_input.split_at_mut(32);
        m_part.copy_from_slice(m);
        h_part.copy_from_slice(&H(self.to_bytes().as_ref()));
        let (k, mut r) = G(&g_input);

        let ciphertext = self.ek_pke.encrypt(m, &r);

        g_input.zeroize();
        r.zeroize();

        (SharedSecret(k), Ciphertext(ciphertext))
    }
}

impl<P: ParameterSet> TryFrom<&[u8]> for EncapsulationKey<P> {
    type Error = Error;

    /// Parses an encapsulation key from bytes, applying the [section 7.2]
    /// encapsulation key check.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidEncapsulationKeyLength`] — the type check: the input
    ///   is not `384 * K + 32` bytes ([section 7.2] check 1).
    /// - [`Error::EncapsulationKeyModulusCheckFailed`] — the modulus check
    ///   (equation 7.1): some coefficient is not in `0..q`, so the
    ///   `ByteDecode_12` / `ByteEncode_12` round-trip does not reproduce the
    ///   input ([section 7.2] check 2).
    ///
    /// [section 7.2]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.7.2
    fn try_from(bytes: &[u8]) -> Result<Self, Error> {
        // Type check (FIPS 203 section 7.2, encapsulation key check 1): ek must
        // be 384 * K + 32 bytes, else `InvalidEncapsulationKeyLength`.
        if bytes.len() != P::ENCAPS_KEY_SIZE {
            return Err(Error::InvalidEncapsulationKeyLength);
        }

        let ek_pke = PKE::EncryptionKey::<P>::from_bytes(bytes);

        // Modulus check (FIPS 203 section 7.2, encapsulation key check 2,
        // equation 7.1): ByteEncode_12(ByteDecode_12(ek)) must equal ek. The
        // round-trip differs iff a coefficient decoded from a value >= q, so a
        // mismatch yields `EncapsulationKeyModulusCheckFailed`. Streamed rather
        // than materialized: the input `bytes` is public and this comparison
        // short-circuits on the first mismatch either way.
        if !ek_pke.bytes().eq(bytes.iter().copied()) {
            return Err(Error::EncapsulationKeyModulusCheckFailed);
        }

        Ok(Self { ek_pke })
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
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct DecapsulationKey<P: ParameterSet> {
    /// The K-PKE decryption key.
    dk_pke: PKE::DecryptionKey<P>,
    /// The encapsulation key, needed to re-encrypt during decapsulation.
    /// Public, `Zeroize` not needed.
    #[zeroize(skip)]
    ek: EncapsulationKey<P>,
    /// `H(ek)`, mixed into the shared-secret derivation. Public, `Zeroize` not
    /// needed.
    #[zeroize(skip)]
    h_ek: EncapsulationKeyHash,
    /// The implicit-rejection seed `z`.
    z: RejectionSeed,
}

impl<P: ParameterSet> DecapsulationKey<P> {
    /// `ML-KEM.KeyGen`: generates a fresh decapsulation key.
    ///
    /// Implements [Algorithm 19] of FIPS 203. The 64-byte `d ‖ z` keygen seed
    /// is drawn from the OS via [`getrandom`]. Recover the corresponding
    /// public key with [`Self::encapsulation_key`].
    ///
    /// # Panics
    ///
    /// Panics if the OS entropy source is unavailable.
    ///
    /// [Algorithm 19]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.19
    /// [`getrandom`]: getrandom::getrandom
    #[must_use]
    pub fn generate() -> Self {
        let mut seed = [0u8; 64];

        getrandom::getrandom(&mut seed)
            .expect("ML-KEM.KeyGen: OS entropy source (`getrandom`) unavailable");

        let dk = Self::generate_derand(&seed);
        seed.zeroize();

        dk
    }

    /// `ML-KEM.KeyGen_internal`: the derandomized core of [`Self::generate`],
    /// taking the 64-byte seed `d ‖ z` explicitly.
    ///
    /// Specified by [Algorithm 16] of FIPS 203 §6. The seed concatenates the
    /// K-PKE key-generation seed `d` (first 32 bytes) and the
    /// implicit-rejection seed `z` (last 32 bytes). Exposed for KAT replay
    /// **and** for hybrid-KEM constructions like [X-Wing] that derive
    /// `d ‖ z` deterministically from a combined seed and call this
    /// directly; both halves must come from an SP 800-90A/B/C RBG of the
    /// parameter set's security strength (FIPS 203 §3.3).
    ///
    /// [Algorithm 16]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.16
    /// [X-Wing]: https://datatracker.ietf.org/doc/draft-connolly-cfrg-xwing-kem/
    #[must_use]
    pub fn generate_derand(seed: &[u8; 64]) -> Self {
        let (d, z) = seed.split_at(32);

        let mut d_seed = [0u8; 32];
        d_seed.copy_from_slice(d);
        let mut z_seed = [0u8; 32];
        z_seed.copy_from_slice(z);

        let PKE::KeyPair { dk_pke, ek_pke } =
            PKE::KeyPair::new_derand(PKE::KeyGenRandomnessSeed::<P>::new(d_seed));

        let ek = EncapsulationKey { ek_pke };
        let h_ek = EncapsulationKeyHash::from(&ek);

        Self {
            dk_pke,
            ek,
            h_ek,
            z: RejectionSeed::from(z_seed),
        }
    }

    /// Returns a reference to the encapsulation key derived from this
    /// decapsulation key. Clone if you need ownership.
    #[must_use]
    pub fn encapsulation_key(&self) -> &EncapsulationKey<P> {
        &self.ek
    }

    /// Serializes the decapsulation key to its `768 * K + 96` bytes
    /// (`P::DecapsKeySerialization`), as `dk_PKE ‖ ek ‖ H(ek) ‖ z`, assembled
    /// on the stack with no heap allocation.
    #[must_use]
    pub fn to_bytes(&self) -> P::DecapsKeySerialization {
        let mut bytes = self
            .dk_pke
            .bytes()
            .chain(self.ek.ek_pke.bytes())
            .chain(*self.h_ek.as_bytes())
            .chain(*self.z.as_bytes());

        P::decaps_key_from_fn(|_| bytes.next().unwrap_or(0))
    }

    /// Parses a decapsulation key from bytes; see the [`TryFrom`] impl for the
    /// error conditions.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        Self::try_from(bytes)
    }

    /// `ML-KEM.Decaps`: recovers the shared secret from a ciphertext, with
    /// implicit rejection.
    ///
    /// Implements [Algorithm 18] of FIPS 203 (the internal core of
    /// [Algorithm 21]). On a ciphertext that does not re-encrypt to itself, the
    /// returned secret is derived from the rejection seal `z` rather than from
    /// the recovered encapsulation randomness `m'`.
    ///
    /// # Constant-time
    ///
    /// The recovered randomness `m'` and the re-encryption `c'` are
    /// secret-derived
    /// ([Algorithm 18] lines 5, 8). The ciphertext comparison and the
    /// `K'`/`K_bar` selection use `subtle` (`ct_eq` /
    /// `conditional_select`), so neither the equality result nor which
    /// secret is returned leaks through a branch or an early exit. The
    /// field arithmetic feeding `m'` is itself constant-time (Montgomery/
    /// Barrett; see the `algebraic` module).
    ///
    /// [Algorithm 18]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.18
    /// [Algorithm 21]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.21
    #[must_use]
    pub fn decapsulate(&self, ciphertext: &Ciphertext<P>) -> SharedSecret {
        // m' <- K-PKE.Decrypt(dk_PKE, c)
        let mut m_prime = self.dk_pke.decrypt(&ciphertext.0);

        // (K', r') <- G(m' || h); preimage assembled in a 64-byte stack buffer.
        let mut g_input = [0u8; 64];
        let (m_part, h_part) = g_input.split_at_mut(32);
        m_part.copy_from_slice(&m_prime);
        h_part.copy_from_slice(self.h_ek.as_bytes());
        let (mut k_prime, mut r_prime) = G(&g_input);
        g_input.zeroize();

        // K_bar <- J(z || c); absorbed in two parts, no joined buffer.
        let mut k_bar = J(self.z.as_bytes(), ciphertext.as_bytes());

        // c' <- K-PKE.Encrypt(ek_PKE, m', r')
        let c_prime = self.ek.ek_pke.encrypt(&m_prime, &r_prime);

        // Implicit rejection (Algorithm 18 lines 8-10): return K' if c == c',
        // else the rejection secret K_bar — compared and selected in constant
        // time so the outcome over secret-derived bytes never branches or
        // short-circuits.
        let matches = ciphertext.as_bytes().ct_eq(c_prime.as_bytes());

        let mut secret = [0u8; 32];
        for (out, (kp, kb)) in secret.iter_mut().zip(k_prime.iter().zip(k_bar.iter())) {
            *out = u8::conditional_select(kb, kp, matches);
        }

        // Zeroize the secret-derived FO transients (`c_prime` is not on this
        // list — it equals the public ciphertext on the success path).
        m_prime.zeroize();
        r_prime.zeroize();
        k_prime.zeroize();
        k_bar.zeroize();

        SharedSecret(secret)
    }
}

impl<P: ParameterSet> TryFrom<&[u8]> for DecapsulationKey<P> {
    type Error = Error;

    /// Parses a decapsulation key from bytes, applying the [section 7.3]
    /// decapsulation key checks.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidDecapsulationKeyLength`] — the decapsulation key type
    ///   check: the input is not `768 * K + 96` bytes ([section 7.3] check 2).
    /// - [`Error::DecapsulationKeyHashMismatch`] — the hash check (equation
    ///   7.2): the embedded `H(ek)` does not match a fresh hash of the embedded
    ///   encapsulation key ([section 7.3] check 3).
    ///
    /// [section 7.3]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.7.3
    fn try_from(bytes: &[u8]) -> Result<Self, Error> {
        // Decapsulation key type check (FIPS 203 section 7.3, decapsulation
        // input check 2): dk must be 768 * K + 96 bytes, else
        // `InvalidDecapsulationKeyLength`.
        if bytes.len() != P::DECAPS_KEY_SIZE {
            return Err(Error::InvalidDecapsulationKeyLength);
        }

        // dk = dk_PKE ‖ ek ‖ H(ek) ‖ z
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

        // Hash check (FIPS 203 section 7.3, decapsulation input check 3,
        // equation 7.2): the stored H(ek) must equal a fresh hash of the
        // embedded ek, else `DecapsulationKeyHashMismatch`.
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
}

/// ML-KEM-512 type aliases.
///
/// Convenience specializations of the generic API for the [`MLKEM512`]
/// parameter set, so callers can write `mlkem512::DecapsulationKey` rather than
/// `DecapsulationKey<MLKEM512>`. Gated behind the `mlkem512` feature.
#[cfg(feature = "mlkem512")]
pub mod mlkem512 {
    pub use crate::{Error, SharedSecret};

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
/// parameter set, so callers can write `mlkem768::DecapsulationKey` rather than
/// `DecapsulationKey<MLKEM768>`. Gated behind the `mlkem768` feature.
#[cfg(feature = "mlkem768")]
pub mod mlkem768 {
    pub use crate::{Error, SharedSecret};

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
/// parameter set, so callers can write `mlkem1024::DecapsulationKey` rather
/// than `DecapsulationKey<MLKEM1024>`. Gated behind the `mlkem1024` feature.
#[cfg(feature = "mlkem1024")]
pub mod mlkem1024 {
    pub use crate::{Error, SharedSecret};

    /// An ML-KEM-1024 encapsulation (public) key.
    pub type EncapsulationKey = crate::EncapsulationKey<crate::MLKEM1024>;
    /// An ML-KEM-1024 decapsulation (secret) key.
    pub type DecapsulationKey = crate::DecapsulationKey<crate::MLKEM1024>;
    /// An ML-KEM-1024 ciphertext.
    pub type Ciphertext = crate::Ciphertext<crate::MLKEM1024>;
}
