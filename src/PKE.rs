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

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{
    algebraic::{PolynomialRingElement, RqElement, RqVector, TqElement, TqMatrix, TqVector},
    functions::{G, PRF, shake256_x4},
    parameters::{Eta, ParameterSet},
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
    /// Returns an iterator over the encryption key's serialized bytes,
    /// `ByteEncode_12(t_hat) ‖ rho`, yielded lazily so a fixed buffer can be
    /// packed without an intermediate allocation.
    pub(crate) fn bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.t_hat.byte_encode().chain(self.rho)
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
        let mut n = 0u8;
        let y = RqVector::<P>::sample_cbd(P::ETA_1, randomness, &mut n);
        let e1 = RqVector::<P>::sample_cbd(P::ETA_2, randomness, &mut n);
        let e2 = RqElement::sample_cbd(P::ETA_2, &PRF(P::ETA_2, randomness, n));

        let y_hat = y.ntt();

        // u = NTT⁻¹(A^T . y_hat) + e1, using the `A^T` cached at key construction.
        let u = (&self.a_hat_transpose * &y_hat).ntt_inverse() + e1;

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
#[derive(Zeroize, ZeroizeOnDrop)]
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
        let (rho, mut sigma) = G(&g_input);
        // g_input held the secret seed `d`; zeroize now that G has consumed it.
        g_input.zeroize();

        // `rho` is public per FIPS 203 (part of `ek_pke` below), but memcheck
        // taints it through `G(d ‖ k)` — see [`crate::ctgrind`].
        #[cfg(feature = "ctgrind")]
        crate::ctgrind::declassify(&rho);

        let a_hat = TqMatrix::<P>::expand(&rho);

        let mut n = 0u8;
        let s = RqVector::<P>::sample_cbd(P::ETA_1, &sigma, &mut n);
        let e = RqVector::<P>::sample_cbd(P::ETA_1, &sigma, &mut n);
        // sigma is no longer needed after CBD sampling.
        sigma.zeroize();

        let s_hat = s.ntt();
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
                let seeds: [[u8; 34]; 4] = core::array::from_fn(|lane| {
                    let n = base + lane;
                    let mut seed = [0u8; 34];
                    let (prefix, suffix) = seed.split_at_mut(32);
                    prefix.copy_from_slice(rho);
                    suffix.copy_from_slice(&[(n % k) as u8, (n / k) as u8]);

                    seed
                });
                batch = TqElement::sample_ntt_x4(&seeds);
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

impl<P: ParameterSet> RqVector<P> {
    /// Samples a length-`K` vector from the centered binomial distribution
    /// `D_eta`, advancing the PRF counter `n` once per component.
    ///
    /// The `K` `PRF` squeezes (`seed ‖ (n+0..n+K)`) run in one batched Keccak
    /// call; lanes past `K` are computed and discarded. Each lane squeezes the
    /// `64 * eta`-byte CBD input (the buffer is sized for the largest `eta`).
    fn sample_cbd(eta: Eta, seed: &[u8; 32], n: &mut u8) -> Self {
        let prf_len = 64 * usize::from(eta);

        let inputs: [[u8; 33]; 4] = core::array::from_fn(|lane| {
            let mut input = [0u8; 33];
            let (prefix, suffix) = input.split_at_mut(32);
            prefix.copy_from_slice(seed);
            suffix.copy_from_slice(&[n.wrapping_add(lane as u8)]);

            input
        });
        *n += P::K as u8;

        let mut outputs = [[0u8; 64 * 3]; 4];
        let [o0, o1, o2, o3] = &mut outputs;
        shake256_x4(
            inputs.each_ref().map(<[u8; 33]>::as_slice),
            [
                o0.as_mut_slice(),
                o1.as_mut_slice(),
                o2.as_mut_slice(),
                o3.as_mut_slice(),
            ],
        );

        let mut outputs = outputs.into_iter();
        Self::from_fn(|_| {
            let output = outputs.next().unwrap_or([0u8; 64 * 3]);
            let (bytes, _) = output.split_at(prf_len);

            RqElement::sample_cbd(eta, bytes)
        })
    }
}
