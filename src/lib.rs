// [ML-KEM] Selkie
//
// [ML-KEM]: https://doi.org/10.6028/NIST.FIPS.203.ipd

#![doc(
    html_logo_url = "https://user-images.githubusercontent.com/552961/197638905-f5144be3-a2f2-48c2-9ecb-26e4e34d8d8a.svg#gh-light-mode-only"
)]
#![doc = include_str!("../README.md")]
#![allow(mixed_script_confusables)]
#![allow(non_snake_case)]
#![deny(missing_docs, clippy::indexing_slicing, clippy::unwrap_used)]
#![forbid(unsafe_code)]
#![warn(rust_2018_idioms)]

use rand::{CryptoRng, RngCore};

mod PKE;
mod algebraic;
mod functions;
mod parameters;
#[cfg(feature = "test")]
mod test;

use crate::{
    functions::{G, H, J},
    parameters::ParameterSet,
};

/// Explicit errors generated throughout this specification.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Error {}

/// ML-KEM encapsulation key.
///
/// Just the K-PKE encryption key.
pub struct EncapsulationKey<P: ParameterSet>(PKE::EncryptionKey<P>);

impl<P> EncapsulationKey<P>
where
    P: ParameterSet,
{
    /// Byte serialization of the ML-KEM encapsulation key.
    fn serialize(self) -> P::EncapsKeySerialization {
        (self.0).0
    }
}

/// SHA3-256 hash of an ML-KEM encryption key.
///
/// "These 32 bytes are different than other 32 bytes."
struct EncapsulationKeyHash<P: ParameterSet>([u8; 32]);

impl<P> From<EncapsulationKey<P>> for EncapsulationKeyHash<P>
where
    P: ParameterSet,
{
    fn from(ek: EncapsulationKey<P>) -> EncapsulationKeyHash<P> {
        let h_ek = H(ek.serialize().as_ref());

        EncapsulationKeyHash::<P>(h_ek)
    }
}

/// Fujisaki–Okamoto transform implict-rejection randomness value _z_ of 32
/// random bytes.
struct RejectionRandomness<P: ParameterSet>([u8; 32]);

/// ML-KEM decapsulation key as-per [FIPS 203] section 6.
///
/// [FIPS 203]:
pub struct DecapsulationKey<P: ParameterSet> {
    /// K-PKE decryption key.
    dk_pke: PKE::DecryptionKey<P>,
    /// K-PKE encryption key (required in Decaps to re-compute the ciphertext).
    ek: EncapsulationKey<P>,
    /// Hash of the K-PKE encryption key, used in shared secret derivation.
    h_ek: EncapsulationKeyHash<P>,
    /// Random bytes that are used to derive the rejection value, so that
    /// Decaps() doesn't need to source randomness.
    z: RejectionRandomness<P>,
}

impl<P> DecapsulationKey<P>
where
    P: ParameterSet,
{
    /// Byte serialization of the ML-KEM decapsulation key.
    fn serialize(self) -> P::DecapsKeySerialization {
        // 🥲 ideally this would not need a Vec on the heap.
        let mut buf = vec![];

        buf.extend_from_slice(self.dk_pke.as_ref());
    }
}

/// The combined `K-PKE` key generation randomness _d_ and the `ML-KEM.Decaps()`
/// implicit-rejection randomness _z_ that together deterministicly generate an
/// `ML-KEM` key pair.
///
/// Follows the [encoding conventions] in the known answer tests (KATs) for
/// [ML-KEM].
///
/// [ML-KEM]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
/// [encoding conventions]: https://github.com/cryspen/libcrux/tree/5fc2cbad58f3f3e515490502e82b1c4600d5e6e3/tests/kyber_kats
pub(crate) struct KeyGenRandomness<P: ParameterSet>([u8; 64]);

impl<P> KeyGenRandomness<P>
where
    P: ParameterSet,
{
    /// Parses out the K-PKE key generation randomness _d_, as defined in
    /// Algorithm 12 of [FIPS 203].
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    pub(crate) fn d(self) -> PKE::KeyGenRandomnessSeed<P>
    where
        P: ParameterSet,
    {
        let mut d = [0u8; 32];
        d.copy_from_slice(&self.0[0..32]);
        PKE::KeyGenRandomnessSeed::<P>::new(d)
    }

    /// Parses out the implicit-rejection randomness _z_, as defined in
    /// Algorithm 15 in [FIPS 203].
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    pub(crate) fn z(self) -> RejectionRandomness<P> {
        let mut z = [0u8; 32];
        z.copy_from_slice(&self.0[32..]);
        RejectionRandomness::<P>(z)
    }
}

/// An ML-KEM keypair: an encapsulation key and a corresponding decapsulation
/// key.
///
/// See [FIPS 203] section 6.1.
///
/// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
pub struct KeyPair<P: ParameterSet> {
    /// ML-KEM secret decapsulation key.
    decaps: DecapsulationKey<P>,
    /// ML-KEM public encapsulation key.
    encaps: EncapsulationKey<P>,
}

impl<P> KeyPair<P>
where
    P: ParameterSet,
{
    /// Generates a new keypair, as specified by Algorithm 15 in section 6.1 of
    /// [FIPS 203].
    ///
    /// Sources all randomness for all subroutines upfront, including the
    /// implicit-rejection randomness _z_ and the `K-PKE` key generation
    /// seed randomness. This deviates from the exact Algorithm 15 for
    /// ML-KEM key generation and the subroutine Algorithm 12 for `K-PKE` key
    /// generation, which source their own randomness internally.
    ///
    /// "A fresh string of random bytes must be generated for every such
    /// invocation. These random bytes shall be generated using an approved
    /// RBG, as prescribed in NIST SP 800-90A, NIST SP 800-90B, and NIST SP
    /// 800-90C. Moreover, the RBG used shall have a security strength of at
    /// least 128 bits for ML-KEM-512, at least 192 bits for ML-KEM-768, and at
    /// least 256 bits for ML-KEM-1024." - [FIPS 203], section 3.3
    ///
    /// We're using an implementation of CTR_DRBG using AES-256 via the `drbg`
    /// crate.
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    pub fn new<R: CryptoRng + RngCore>(csprng: &mut R) -> Result<KeyPair<P>, Error> {
        let mut seed = [0u8; 64];

        csprng.try_fill_bytes(&mut seed)?;

        return Ok(Self::new_derand(KeyGenRandomness(seed)));
    }

    /// Generates a new keypair using provided seed randomness, as specified by
    /// Algorithm 15 in section 6.1 of [FIPS 203].
    ///
    /// The provided seed randomness provides all sources of randomness for all
    /// subroutines upfront, including the implicit-rejection randomness _z_
    /// and the `K-PKE` key generation seed randomness. This deviates from
    /// the exact Algorthm 15 for ML-KEM key generation and the subroutine
    /// Algorithm 12 for `K-PKE` key generation, which source their own
    /// randomness internally.
    ///
    /// [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.ipd.pdf
    // Does this need to be fallible, if randomness is handed in? Are there failure
    // cases when handed bad randomness?
    pub(crate) fn new_derand(seed: KeyGenRandomness<P>) -> KeyPair<P> {
        // z is 32 random bytes
        let z = seed.z();

        // d is 32 random bytes, used by K-PKE
        let d = seed.d();

        // Run key generation for K-PKE.
        let PKE::KeyPair { dk_pke, ek_pke } = PKE::KeyPair::new_derand(d)?;

        // The ML-KEM encapsulation key is literally the K-PKE encryption key.
        let ek = EncapsulationKey(ek_pke);

        // Hash the encapsulation key.
        let h_ek = EncapsulationKeyHash::from(ek);

        // Construct the ML-KEM decapsulation key.
        let dk = DecapsulationKey {
            dk_pke,
            ek,
            h_ek,
            z,
        };

        // Tada.
        return KeyPair {
            decaps: dk,
            encaps: ek,
        };
    }
}
