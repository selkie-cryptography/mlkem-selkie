//! Byte serialization and compression of ring elements for [FIPS 203].
//!
//! Extends the algebraic types of [`crate::algebraic`] with the
//! [section 4.2.1] serialization layer:
//!
//! - `ByteEncode_d` / `ByteDecode_d` pack and unpack arrays of `d`-bit integers
//!   (Algorithms 5 and 6). For `d = 12` this serializes NTT coefficients in
//!   `0..q`; for smaller `d` it serializes compressed coefficients in `0..2^d`.
//! - `Compress_d` / `Decompress_d` (equations 4.7 and 4.8) are fused with the
//!   bit packing, since FIPS 203 always applies them as a pair except for the
//!   `d = 12` key encoding, which performs no compression.
//!
//! These live in their own module rather than alongside the arithmetic impls:
//! serialization is a separable concern, and folding it into `algebraic.rs`
//! would nearly double that file.
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf
//! [section 4.2.1]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsubsection.4.2.1

use crate::{
    algebraic::{FieldElement, PolynomialRingElement, RqElement, RqVector, TqElement, TqVector},
    parameters::{N, ParameterSet},
};

#[cfg(test)]
mod tests;

/// Bits per coefficient when serializing NTT-domain polynomials (no
/// compression).
const D_12: usize = 12;

/// Packs `N` `d`-bit values as `BYTES` bytes, least-significant bit first.
///
/// Implements `BitsToBytes` composed with the bit decomposition of
/// `ByteEncode_d` (Algorithm 5): the bit at position `j` of `values[i]` is
/// written to global bit `i * d + j`. `BYTES` is an explicit const parameter
/// because `N * d / 8` is not expressible as a return type on stable Rust;
/// callers state the size their `d` implies.
///
/// # Panics
///
/// Debug-asserts `BYTES == N * d / 8`.
#[must_use]
fn pack_words<const BYTES: usize>(values: &[u16; N], d: usize) -> [u8; BYTES] {
    debug_assert_eq!(BYTES, N * d / 8);

    let mut out = [0u8; BYTES];
    let mut words = out.chunks_exact_mut(8);
    let mut accumulator: u128 = 0;
    let mut bits = 0usize;

    for &value in values {
        accumulator |= u128::from(value) << bits;
        bits += d;

        if bits >= 64 {
            if let Some(word) = words.next() {
                word.copy_from_slice(&(accumulator as u64).to_le_bytes());
            }
            accumulator >>= 64;
            bits -= 64;
        }
    }

    out
}

/// Packs `N` `d`-bit values into a byte stream, least-significant bit first,
/// for the parameter-dependent ciphertext widths whose byte counts cannot be
/// named as a return type on stable Rust; [`pack_words`] is the fixed-width
/// form.
fn pack(values: [u16; N], d: usize) -> impl Iterator<Item = u8> {
    let mut values = values.into_iter();
    let mut accumulator: u32 = 0;
    let mut bits = 0usize;

    core::iter::from_fn(move || {
        loop {
            if bits >= 8 {
                let byte = (accumulator & 0xFF) as u8;
                accumulator >>= 8;
                bits -= 8;

                return Some(byte);
            }

            let value = values.next()?;
            accumulator |= u32::from(value) << bits;
            bits += d;
        }
    })
}

/// Unpacks `N` `d`-bit values from bytes, least-significant bit first.
///
/// Inverse of [`pack_words`] and [`pack`]; implements the bit recomposition of
/// `ByteDecode_d` (Algorithm 6). Reads the leading `N * d / 8` bytes in
/// 64-bit words, each value in `0..2^d`; a short input is zero-padded.
fn unpack(bytes: &[u8], d: usize) -> [u16; N] {
    let mask = (1u32 << d) - 1;

    let mut words = bytes.chunks_exact(8);
    let mut tail = Some(words.remainder()).filter(|rest| !rest.is_empty());
    let mut next_word = move || {
        if let Some(chunk) = words.next() {
            u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)"))
        } else if let Some(rest) = tail.take() {
            let mut padded = [0u8; 8];
            padded.split_at_mut(rest.len()).0.copy_from_slice(rest);

            u64::from_le_bytes(padded)
        } else {
            0
        }
    };

    let mut accumulator: u128 = 0;
    let mut bits = 0usize;

    core::array::from_fn(|_| {
        if bits < d {
            accumulator |= u128::from(next_word()) << bits;
            bits += 64;
        }

        let value = (accumulator as u32 & mask) as u16;
        accumulator >>= d;
        bits -= d;

        value
    })
}

impl TqElement {
    /// `ByteEncode_12`: serializes the 256 NTT coefficients as 384 bytes.
    #[must_use]
    pub fn byte_encode(&self) -> [u8; 384] {
        pack_words(&self.canonical(), D_12)
    }

    /// `ByteDecode_12`: deserializes 384 bytes into 256 NTT coefficients.
    ///
    /// Each 12-bit value is reduced modulo q, so an input encoding a
    /// coefficient `>= q` does not round-trip — this is what the
    /// encapsulation-key modulus check of [section 7.2] relies on.
    ///
    /// [section 7.2]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.7.2
    pub fn byte_decode(bytes: &[u8]) -> Self {
        let coefficients = unpack(bytes, D_12).map(FieldElement::new);

        Self::new(coefficients)
    }
}

impl RqElement {
    /// `ByteEncode_d(Compress_d(self))`: compresses each coefficient to `d`
    /// bits and packs the result as `32 * d` bytes, yielded lazily; `d` varies
    /// by parameter set, so the width has no stable-Rust array type.
    pub fn compress_encode(&self, d: usize) -> impl Iterator<Item = u8> {
        pack(self.compressed(d), d)
    }

    /// `Decompress_d(ByteDecode_d(bytes))`: unpacks `d`-bit values and
    /// decompresses each back into Zq.
    pub fn decode_decompress(bytes: &[u8], d: usize) -> Self {
        Self::decompress(&unpack(bytes, d), d)
    }

    /// Serializes a 32-byte message into a polynomial via `Decompress_1`.
    ///
    /// The message bits are decompressed to either 0 or `(q + 1) / 2`, giving
    /// `mu` in `K-PKE.Encrypt`.
    pub fn from_message(message: &[u8; 32]) -> Self {
        Self::decode_decompress(message, 1)
    }

    /// Deserializes this polynomial back into a 32-byte message via
    /// `Compress_1`, recovering `m` in `K-PKE.Decrypt`.
    pub fn compress_message(&self) -> [u8; 32] {
        pack_words(&self.compressed(1), 1)
    }
}

impl<P: ParameterSet> TqVector<P> {
    /// `ByteEncode_12` applied componentwise, one 384-byte block per element.
    pub(crate) fn byte_encoded_elements(&self) -> impl Iterator<Item = [u8; 384]> + '_ {
        self.as_slice().iter().map(TqElement::byte_encode)
    }

    /// `ByteEncode_12` applied componentwise, the `K` 384-byte blocks as an
    /// array.
    pub(crate) fn byte_encoded(&self) -> P::KArray<[u8; 384]> {
        let mut blocks = self.byte_encoded_elements();

        // `k_array_from_fn` drives `K` iterations and the vector holds `K`
        // elements, so every `next()` yields a block; the zero fallback is
        // unreachable.
        P::k_array_from_fn(|_| blocks.next().unwrap_or([0u8; 384]))
    }

    /// `ByteDecode_12` applied componentwise to `384 * K` bytes.
    pub fn byte_decode(bytes: &[u8]) -> Self {
        let mut chunks = bytes.chunks_exact(384);

        // `from_fn` drives `K` iterations and `bytes` holds `384 * K`
        // bytes, so every `next()` yields a full chunk; the empty fallback is
        // unreachable.
        Self::from_fn(|_| TqElement::byte_decode(chunks.next().unwrap_or_default()))
    }
}

impl<P: ParameterSet> RqVector<P> {
    /// `ByteEncode_d(Compress_d(.))` applied componentwise: `32 * d * K` bytes,
    /// yielded lazily.
    pub fn compress_encode(&self, d: usize) -> impl Iterator<Item = u8> + '_ {
        self.as_slice()
            .iter()
            .flat_map(move |poly| poly.compress_encode(d))
    }

    /// `Decompress_d(ByteDecode_d(.))` applied componentwise.
    pub fn decode_decompress(bytes: &[u8], d: usize) -> Self {
        let mut chunks = bytes.chunks_exact(32 * d);

        // `from_fn` drives `K` iterations and `bytes` holds `32 * d * K`
        // bytes, so every `next()` yields a full chunk; the empty fallback is
        // unreachable.
        Self::from_fn(|_| RqElement::decode_decompress(chunks.next().unwrap_or_default(), d))
    }
}
