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

use core::array;

use subtle::{Choice, ConstantTimeEq};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{
    algebraic::{
        CachedTqVector, PolynomialRingElement, RqElement, RqVector, TqElement, TqMatrix, TqVector,
    },
    functions::G,
    parameters::{PKE as _, ParameterSet},
    sampling::CbdSampler,
};

#[cfg(test)]
mod tests;

/// K-PKE key generation randomness seed `d` (Algorithm 13 input).
///
/// Should never be exposed outside ML-KEM.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct KeyGenRandomnessSeed<P: ParameterSet>(
    [u8; 32],
    #[zeroize(skip)] core::marker::PhantomData<P>,
);

impl<P: ParameterSet> KeyGenRandomnessSeed<P> {
    /// Constructs the seed from 32 random bytes.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes, core::marker::PhantomData)
    }
}

/// The largest [`PKE::ENCRYPTION_KEY_SIZE`] across parameter sets (`K = 4`):
/// `384 * 4 + 32`.
///
/// [`PKE::ENCRYPTION_KEY_SIZE`]: crate::parameters::PKE::ENCRYPTION_KEY_SIZE
const MAX_PKE_ENCRYPTION_KEY_SIZE: usize = 1568;

/// The K-PKE encryption key's serialization type: the encapsulation key's,
/// byte for byte (Algorithm 16 line 3).
pub(crate) type EncryptionKeySerialization<P> = <P as ParameterSet>::EncapsKeySerialization;

/// The K-PKE decryption key's serialization type, `[u8; 384 * K]`.
pub(crate) type DecryptionKeySerialization<P> =
    <<P as ParameterSet>::PKE as crate::parameters::PKE>::DecryptionKeySerialization;

/// A K-PKE encryption key: the NTT-domain vector `t_hat`, the matrix seed
/// `rho`, and the expanded transpose `A^T` cached for encryption (section 5 of
/// FIPS 203).
///
/// Caching `A^T` moves the SampleNTT matrix expansion off the hot
/// `K-PKE.Encrypt` path (which runs once per encapsulation) and onto key
/// construction (which runs once per key). `rho` is retained because it, not
/// the expanded matrix, is what serializes.
///
/// Public key material; contains no secrets, so it is not zeroized.
#[derive(Clone)]
pub struct EncryptionKey<P: ParameterSet> {
    /// `t_hat = A . s_hat + e_hat`, the public vector in Tq.
    t_hat: TqVector<P>,
    /// The 32-byte seed from which the public matrix `A` is regenerated.
    rho: [u8; 32],
    /// `A^T`, expanded once from `rho` so `encrypt` need not re-run SampleNTT.
    a_hat_transpose: TqMatrix<P>,
}

impl<P: ParameterSet> EncryptionKey<P> {
    /// Serializes the encryption key block-wise on the stack.
    pub(crate) fn to_bytes(&self) -> EncryptionKeySerialization<P> {
        let mut assembled = [0u8; MAX_PKE_ENCRYPTION_KEY_SIZE];
        debug_assert!(P::PKE::ENCRYPTION_KEY_SIZE <= assembled.len());

        let (encoded, _) = assembled.split_at_mut(P::PKE::ENCRYPTION_KEY_SIZE);
        let (t_part, rho_part) = encoded.split_at_mut(384 * P::K);
        for (chunk, block) in t_part
            .chunks_exact_mut(384)
            .zip(self.t_hat.byte_encoded_elements())
        {
            chunk.copy_from_slice(&block);
        }
        rho_part.copy_from_slice(&self.rho);

        P::encaps_key_from_fn(|i| assembled.get(i).copied().unwrap_or(0))
    }

    /// Parses an encryption key from `384 * K + 32` bytes.
    ///
    /// # Panics
    ///
    /// Debug-asserts the input length; callers in `ML-KEM` validate the length
    /// at the public boundary before parsing (FIPS 203 section 7.2).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), P::PKE::ENCRYPTION_KEY_SIZE);

        let (encoded_t, rho_bytes) = bytes.split_at(384 * P::K);

        let t_hat = TqVector::<P>::byte_decode(encoded_t);
        let mut rho = [0u8; 32];
        rho.copy_from_slice(rho_bytes);

        // Expand `A^T` once here so repeated `encrypt` calls under this parsed
        // key skip SampleNTT entirely.
        let a_hat_transpose = TqMatrix::<P>::expand(&rho).transpose();

        Self {
            t_hat,
            rho,
            a_hat_transpose,
        }
    }

    /// `K-PKE.Encrypt`: encrypts a 32-byte message under explicit encryption
    /// randomness `r`.
    ///
    /// Implements [Algorithm 14] of FIPS 203.
    ///
    /// [Algorithm 14]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.14
    pub fn encrypt(&self, message: &[u8; 32], randomness: &[u8; 32]) -> Ciphertext<P> {
        // y takes PRF streams 0..K at eta_1; e1 and e2 share eta_2, so their
        // K + 1 streams (K..2K, then 2K) run as one batched group.
        let y = CbdSampler::new(P::ETA_1, randomness, 0, P::K as u8).sample_vector();

        let mut noise = CbdSampler::new(P::ETA_2, randomness, P::K as u8, P::K as u8 + 1);
        let e1 = noise.sample_vector();
        let e2 = noise.sample_element();

        // `y_hat` is the reused operand of both products below; cache its
        // base-multiplication terms once.
        let y_hat = CachedTqVector::from(y.ntt());

        // u = NTT⁻¹(A^T . y_hat) + e1, using the `A^T` cached at key construction.
        let u = (&self.a_hat_transpose * &y_hat).ntt_inverse() + e1;

        // mu = Decompress_1(ByteDecode_1(m))
        let mu = RqElement::from_message(message);

        // v = NTT⁻¹(t_hat^T . y_hat) + e2 + mu
        let v = (&self.t_hat * &y_hat).ntt_inverse() + e2 + mu;

        Ciphertext::from_uv(&u, &v)
    }
}

/// A K-PKE decryption key: the NTT-domain secret vector `s_hat` (section 5 of
/// FIPS 203), with its base-multiplication caches.
///
/// `s_hat` is the reused operand of every `decrypt`'s dot product, so its
/// caches are computed once per key rather than per decryption.
// No `PartialEq`/`Eq`: this is secret key material.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct DecryptionKey<P: ParameterSet> {
    /// `s_hat = NTT(s)`, the secret vector in Tq, with its caches.
    s_hat: CachedTqVector<P>,
}

impl<P: ParameterSet> DecryptionKey<P> {
    /// Serializes the decryption key, `ByteEncode_12(s_hat)` (Algorithm 13
    /// line 19), block-wise on the stack.
    pub(crate) fn to_bytes(&self) -> DecryptionKeySerialization<P> {
        let blocks = self.s_hat.vector().byte_encoded();
        let bytes = blocks.as_ref().as_flattened();

        P::PKE::decryption_key_from_fn(|i| bytes.get(i).copied().unwrap_or(0))
    }

    /// Parses a decryption key from `384 * K` bytes.
    ///
    /// # Panics
    ///
    /// Debug-asserts the input length; callers validate at the public boundary.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), P::PKE::DECRYPTION_KEY_SIZE);

        Self {
            s_hat: CachedTqVector::from(TqVector::<P>::byte_decode(bytes)),
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

        // w = v - NTT⁻¹(s_hat^T . NTT(u)), reusing the caches of s_hat.
        let w = v - (&u.ntt() * &self.s_hat).ntt_inverse();

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
    /// Serializes `c = ByteEncode_{d_u}(Compress_{d_u}(u)) ‖
    /// ByteEncode_{d_v}(Compress_{d_v}(v))` ([Algorithm 14] lines 21-23).
    ///
    /// [Algorithm 14]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.14
    fn from_uv(u: &RqVector<P>, v: &RqElement) -> Self {
        // The largest ciphertext across parameter sets: 32 * (11 * 4 + 5).
        // The exact `[u8; P::CIPHERTEXT_SIZE]` local is not expressible on
        // stable Rust, and the collector reading one contiguous staging
        // buffer is what lets it compile to block copies.
        const MAX_CIPHERTEXT_SIZE: usize = 1568;

        let mut staged = [0u8; MAX_CIPHERTEXT_SIZE];
        debug_assert!(P::CIPHERTEXT_SIZE <= staged.len());

        let (u_part, v_part) = staged.split_at_mut(32 * P::D_U * P::K);
        for (chunk, poly) in u_part.chunks_exact_mut(32 * P::D_U).zip(u.as_slice()) {
            let packed = poly.compress_encode(P::D_U);
            chunk.copy_from_slice(packed.split_at(chunk.len()).0);
        }

        let (v_exact, _) = v_part.split_at_mut(32 * P::D_V);
        let packed = v.compress_encode(P::D_V);
        v_exact.copy_from_slice(packed.split_at(v_exact.len()).0);

        Self(P::ciphertext_from_fn(|i| {
            staged.get(i).copied().unwrap_or(0)
        }))
    }

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

impl<P: ParameterSet> ConstantTimeEq for Ciphertext<P> {
    /// Constant-time equality for the FO re-encryption check: XOR-folds the
    /// serializations word-wise into one accumulator — data-independent by
    /// construction — with a single `subtle` comparison at the end, faster
    /// than `subtle`'s per-byte slice comparison.
    fn ct_eq(&self, other: &Self) -> Choice {
        let (a, b) = (self.as_bytes(), other.as_bytes());
        debug_assert_eq!(a.len() % 8, 0);

        let mut acc = 0u64;
        let (a_words, _) = a.as_chunks::<8>();
        let (b_words, _) = b.as_chunks::<8>();
        for (x, y) in a_words.iter().zip(b_words) {
            acc |= u64::from_le_bytes(*x) ^ u64::from_le_bytes(*y);
        }

        acc.ct_eq(&0)
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
        let (rho, mut sigma) = G(&g_input);
        // g_input held the secret seed `d`; zeroize now that G has consumed it.
        g_input.zeroize();

        // `rho` is public per FIPS 203 (part of `ek_pke` below), but memcheck
        // taints it through `G(d ‖ k)` — see [`crate::ctgrind`].
        #[cfg(feature = "ctgrind")]
        crate::ctgrind::declassify(&rho);

        let a_hat = TqMatrix::<P>::expand(&rho);

        // s takes PRF streams 0..K and e takes K..2K; they share eta_1, so
        // all 2K streams run as one batched group.
        let mut noise = CbdSampler::new(P::ETA_1, &sigma, 0, 2 * P::K as u8);
        let s = noise.sample_vector();
        let e = noise.sample_vector();
        // Dropping the sampler zeroizes its copy of sigma.
        drop(noise);
        sigma.zeroize();

        // `s_hat` is reused by every row of `A . s_hat` and then by every
        // `decrypt` under this key; cache its base-multiplication terms once.
        let s_hat = CachedTqVector::from(s.ntt());
        let e_hat = e.ntt();

        // t_hat = A . s_hat + e_hat. The matrix-vector base multiplication leaves
        // the product scaled by R^-1; `to_montgomery` restores the standard
        // domain before adding the true NTT noise e_hat.
        let t_hat = (&a_hat * &s_hat).to_montgomery() + e_hat;

        // `A` is now spent by `t_hat`; transpose it once for the cached encrypt
        // matrix rather than re-expanding from `rho`.
        let a_hat_transpose = a_hat.transpose();

        Self {
            dk_pke: DecryptionKey { s_hat },
            ek_pke: EncryptionKey {
                t_hat,
                rho,
                a_hat_transpose,
            },
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
        let k = P::K;

        // `A_hat`'s `K*K` entries in row-major order — `A_hat[i][j] =
        // SampleNTT(rho ‖ j ‖ i)` — sampled four lanes at a time across the flat
        // entry list (not per row), so `K=2` fills one batch exactly with no
        // wasted lanes; the final partial batch (`K=3`) over-samples and discards.
        let mut next = 0usize;
        let mut batch = [TqElement::ZERO; 4];
        let mut taken = 4usize;

        let mut entries = core::iter::from_fn(move || {
            if taken == 4 {
                let base = next;
                let indices: [(u8, u8); 4] = array::from_fn(|lane| {
                    let n = base + lane;

                    ((n % k) as u8, (n / k) as u8)
                });
                batch = TqElement::sample_ntt_x4(rho, indices);
                taken = 0;
                next += 4;
            }

            let element = batch.get(taken).copied();
            taken += 1;

            element
        });

        Self::from_fn(|_| TqVector::<P>::from_fn(|_| entries.next().unwrap_or(TqElement::ZERO)))
    }
}
