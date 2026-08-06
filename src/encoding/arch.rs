//! Architecture-dispatched bit-packing impls
//!
//! Every backend exposes [`pack`] and [`unpack`], the bit decomposition of
//! `ByteEncode_d` ([Algorithm 5]) and recomposition of `ByteDecode_d`
//! ([Algorithm 6]) shared by the key and ciphertext coding paths. The active
//! backend is chosen at compile time from the `mlkem_selkie_arch` cfg that
//! `build.rs` emits; absent it, the portable scalar [`generic`] backend is
//! used. The SIMD backends fall back to [`generic`] for widths without a
//! kernel and for the zero-padded short inputs the scalar forms tolerate.
//!
//! [Algorithm 5]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.5
//! [Algorithm 6]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.6

#[cfg(mlkem_selkie_arch = "avx2")]
mod avx2;
pub(super) mod generic;
#[cfg(mlkem_selkie_arch = "neon")]
mod neon;

#[cfg(mlkem_selkie_arch = "avx2")]
pub(super) use avx2::{pack, unpack};
#[cfg(not(any(mlkem_selkie_arch = "neon", mlkem_selkie_arch = "avx2")))]
pub(super) use generic::{pack, unpack};
#[cfg(mlkem_selkie_arch = "neon")]
pub(super) use neon::{pack, unpack};
