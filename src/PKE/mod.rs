//! K-PKE: the internal, IND-CPA-secure public-key encryption scheme.
//!
//! Implements [section 5] of FIPS 203: `K-PKE.KeyGen` (Algorithm 13),
//! `K-PKE.Encrypt` (Algorithm 14), and `K-PKE.Decrypt` (Algorithm 15).
//!
//! K-PKE is never exposed directly. ML-KEM wraps it in the Fujisaki–Okamoto
//! transform; in particular, ciphertexts produced inside `ML-KEM.Decaps` are
//! secret-dependent and must be compared in constant time by the caller.
//!
//! [section 5]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#section.5

use crate::{
    algebraic::{RqElement, RqVector, TqElement, TqMatrix, TqVector},
    functions::{G, PRF, XOF},
    parameters::{Eta, ParameterSet},
};

#[cfg(test)]
mod tests;

/// K-PKE key generation randomness seed `d` (Algorithm 13 input).
///
/// Should never be exposed outside ML-KEM.
pub struct KeyGenRandomnessSeed<P: ParameterSet>([u8; 32], core::marker::PhantomData<P>);

impl<P: ParameterSet> KeyGenRandomnessSeed<P> {
    /// Constructs the seed from 32 random bytes.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes, core::marker::PhantomData)
    }
}

/// A K-PKE encryption key: the NTT-domain vector `t_hat` and the matrix seed
/// `rho` (section 5 of FIPS 203).
#[derive(Clone)]
pub struct EncryptionKey<P: ParameterSet> {
    /// `t_hat = A . s_hat + e_hat`, the public vector in Tq.
    t_hat: TqVector<P>,
    /// The 32-byte seed from which the public matrix `A` is regenerated.
    rho: [u8; 32],
}

impl<P: ParameterSet> EncryptionKey<P> {
    /// Returns an iterator over the encryption key's serialized bytes,
    /// `ByteEncode_12(t_hat) ‖ rho`, yielded lazily so a fixed buffer can be
    /// packed without an intermediate allocation.
    pub(crate) fn bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.t_hat.byte_encode().chain(self.rho)
    }

    /// Serializes to `ByteEncode_12(t_hat) ‖ rho`, the fixed-size
    /// `P::PKEEncryptionKeySerialization`, assembled on the stack.
    pub fn to_bytes(&self) -> P::PKEEncryptionKeySerialization {
        let mut bytes = self.bytes();
        P::pke_encryption_key_from_fn(|_| bytes.next().unwrap_or(0))
    }

    /// Parses an encryption key from `384 * K + 32` bytes.
    ///
    /// # Panics
    ///
    /// Debug-asserts the input length; callers in `ML-KEM` validate the length
    /// at the public boundary before parsing (FIPS 203 section 7.2).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), P::PKE_ENCRYPTION_KEY_SIZE);

        let (encoded_t, rho_bytes) = bytes.split_at(384 * P::K);

        let t_hat = TqVector::<P>::byte_decode(encoded_t);
        let mut rho = [0u8; 32];
        rho.copy_from_slice(rho_bytes);

        Self { t_hat, rho }
    }

    /// `K-PKE.Encrypt`: encrypts a 32-byte message under explicit encryption
    /// randomness `r`.
    ///
    /// Implements [Algorithm 14] of FIPS 203.
    ///
    /// [Algorithm 14]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.14
    pub fn encrypt(&self, message: &[u8; 32], randomness: &[u8; 32]) -> Ciphertext<P> {
        let a_hat = TqMatrix::<P>::expand(&self.rho);

        let mut n = 0u8;
        let y = RqVector::<P>::sample_cbd(P::ETA_1, randomness, &mut n);
        let e1 = RqVector::<P>::sample_cbd(P::ETA_2, randomness, &mut n);
        let e2 = RqElement::sample_cbd(P::ETA_2, &PRF(P::ETA_2, randomness, n));

        let y_hat = y.ntt();

        // u = NTT⁻¹(A^T . y_hat) + e1
        let u = (&a_hat.transpose() * &y_hat).ntt_inverse() + e1;

        // mu = Decompress_1(ByteDecode_1(m))
        let mu = RqElement::from_message(message);

        // v = NTT⁻¹(t_hat^T . y_hat) + e2 + mu
        let v = (&self.t_hat * &y_hat).ntt_inverse() + e2 + mu;

        // Serialize Compress(u) ‖ Compress(v) straight into the fixed-size
        // ciphertext buffer; both halves yield their bytes lazily, so no
        // intermediate heap allocation backs the encoding.
        let mut bytes = u.compress_encode(P::D_U).chain(v.compress_encode(P::D_V));

        Ciphertext(P::ciphertext_from_fn(|_| bytes.next().unwrap_or(0)))
    }
}

/// A K-PKE decryption key: the NTT-domain secret vector `s_hat` (section 5 of
/// FIPS 203).
// No `PartialEq`/`Eq`: this is secret key material.
pub struct DecryptionKey<P: ParameterSet> {
    /// `s_hat = NTT(s)`, the secret vector in Tq.
    s_hat: TqVector<P>,
}

impl<P: ParameterSet> DecryptionKey<P> {
    /// Returns an iterator over the decryption key's serialized bytes,
    /// `ByteEncode_12(s_hat)`, yielded lazily.
    pub(crate) fn bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.s_hat.byte_encode()
    }

    /// Serializes to `ByteEncode_12(s_hat)`, the fixed-size
    /// `P::PKEDecryptionKeySerialization`, assembled on the stack.
    pub fn to_bytes(&self) -> P::PKEDecryptionKeySerialization {
        let mut bytes = self.bytes();
        P::pke_decryption_key_from_fn(|_| bytes.next().unwrap_or(0))
    }

    /// Parses a decryption key from `384 * K` bytes.
    ///
    /// # Panics
    ///
    /// Debug-asserts the input length; callers validate at the public boundary.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), P::PKE_DECRYPTION_KEY_SIZE);

        Self {
            s_hat: TqVector::<P>::byte_decode(bytes),
        }
    }

    /// `K-PKE.Decrypt`: recovers the 32-byte message from a ciphertext.
    ///
    /// Implements [Algorithm 15] of FIPS 203.
    ///
    /// [Algorithm 15]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.15
    // The recovered message is secret-derived in `ML-KEM.Decaps` (Algorithm 18
    // line 5); the field arithmetic here is constant-time (signed Montgomery /
    // Barrett, branch-free — see `crate::algebraic::field`).
    pub fn decrypt(&self, ciphertext: &Ciphertext<P>) -> [u8; 32] {
        let (c1, c2) = ciphertext.as_bytes().split_at(32 * P::D_U * P::K);

        let u = RqVector::<P>::decode_decompress(c1, P::D_U);
        let v = RqElement::decode_decompress(c2, P::D_V);

        // w = v - NTT⁻¹(s_hat^T . NTT(u))
        let w = v - (&self.s_hat * &u.ntt()).ntt_inverse();

        w.compress_message()
    }
}

/// A K-PKE ciphertext: the serialized, compressed `(u, v)` pair.
///
/// Stored as a fixed-size `[u8; CIPHERTEXT_SIZE]` (via
/// [`ParameterSet::CiphertextSerialization`]) so the value is heap-free and
/// `ML-KEM.Decaps` can compare it against a re-encryption for the
/// implicit-rejection check.
pub struct Ciphertext<P: ParameterSet>(P::CiphertextSerialization);

impl<P: ParameterSet> Ciphertext<P> {
    /// Wraps `32 * (D_U * K + D_V)` ciphertext bytes, copying them into the
    /// fixed-size buffer.
    ///
    /// # Panics
    ///
    /// Debug-asserts the input length; callers in `ML-KEM` validate the length
    /// at the public boundary before parsing (FIPS 203 section 7.2).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), P::CIPHERTEXT_SIZE);

        Self(P::ciphertext_from_fn(|i| {
            bytes.get(i).copied().unwrap_or(0)
        }))
    }

    /// Returns the serialized ciphertext bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_ref()
    }
}

/// A K-PKE key pair (Algorithm 13 output).
pub struct KeyPair<P: ParameterSet> {
    /// The secret decryption key.
    pub dk_pke: DecryptionKey<P>,
    /// The public encryption key.
    pub ek_pke: EncryptionKey<P>,
}

impl<P: ParameterSet> KeyPair<P> {
    /// `K-PKE.KeyGen`: derives a key pair deterministically from the seed `d`.
    ///
    /// Diverges from [Algorithm 13] only in that the randomness `d` is supplied
    /// rather than sampled internally; ML-KEM sources fresh randomness for it.
    /// Follows FIPS 203 final in expanding `(rho, sigma) ← G(d ‖ k)`, where `k`
    /// is the parameter byte `P::K` — the domain separator the initial public
    /// draft omitted.
    ///
    /// [Algorithm 13]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.13
    pub fn new_derand(seed: KeyGenRandomnessSeed<P>) -> Self {
        let mut g_input = [0u8; 33];
        let (d_part, k_part) = g_input.split_at_mut(32);
        d_part.copy_from_slice(&seed.0);
        k_part.copy_from_slice(&[P::K as u8]);
        let (rho, sigma) = G(&g_input);

        let a_hat = TqMatrix::<P>::expand(&rho);

        let mut n = 0u8;
        let s = RqVector::<P>::sample_cbd(P::ETA_1, &sigma, &mut n);
        let e = RqVector::<P>::sample_cbd(P::ETA_1, &sigma, &mut n);

        let s_hat = s.ntt();
        let e_hat = e.ntt();

        // t_hat = A . s_hat + e_hat. The matrix-vector base multiplication leaves
        // the product scaled by R^-1; `to_montgomery` restores the standard
        // domain before adding the true NTT noise e_hat.
        let t_hat = (&a_hat * &s_hat).to_montgomery() + e_hat;

        Self {
            dk_pke: DecryptionKey { s_hat },
            ek_pke: EncryptionKey { t_hat, rho },
        }
    }
}

impl<P: ParameterSet> TqMatrix<P> {
    /// Regenerates the public matrix `A_hat` from the seed `rho`.
    ///
    /// Each entry `A_hat[i][j] = SampleNTT(XOF(rho, j, i))`, per Algorithm 13
    /// lines 3-7 (and identically in Algorithm 14): the column index `j` is the
    /// first domain byte and the row index `i` the second, matching the FIPS
    /// 203 test vectors. `K-PKE.KeyGen` uses `A_hat` directly while
    /// `K-PKE.Encrypt` multiplies by its transpose.
    fn expand(rho: &[u8; 32]) -> Self {
        Self::from_fn(|i| {
            TqVector::<P>::from_fn(|j| TqElement::sample_ntt(&mut XOF(rho, j as u8, i as u8)))
        })
    }
}

impl<P: ParameterSet> RqVector<P> {
    /// Samples a length-`K` vector from the centered binomial distribution
    /// `D_eta`, advancing the PRF counter `n` once per component.
    fn sample_cbd(eta: Eta, seed: &[u8; 32], n: &mut u8) -> Self {
        Self::from_fn(|_| {
            let output = PRF(eta, seed, *n);
            *n += 1;

            RqElement::sample_cbd(eta, &output)
        })
    }
}
