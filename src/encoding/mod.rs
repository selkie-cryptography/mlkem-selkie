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
    parameters::ParameterSet,
};

#[cfg(test)]
mod tests;

/// Bits per coefficient when serializing NTT-domain polynomials (no
/// compression).
const D_12: usize = 12;

/// Packs `d`-bit values into bytes, least-significant bit first.
///
/// Implements `BitsToBytes` composed with the bit decomposition of
/// `ByteEncode_d` (Algorithm 5): the bit at position `j` of `values[i]` is
/// written to global bit `i * d + j`, which lands in byte `(i * d + j) / 8`.
fn pack(values: &[u16], d: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * d / 8);
    let mut accumulator: u32 = 0;
    let mut bits = 0usize;

    for &value in values {
        accumulator |= u32::from(value) << bits;
        bits += d;

        while bits >= 8 {
            bytes.push((accumulator & 0xFF) as u8);
            accumulator >>= 8;
            bits -= 8;
        }
    }

    bytes
}

/// Unpacks `d`-bit values from bytes, least-significant bit first.
///
/// Inverse of [`pack`]; implements the bit recomposition of `ByteDecode_d`
/// (Algorithm 6). Returns `bytes.len() * 8 / d` values, each in `0..2^d`.
fn unpack(bytes: &[u8], d: usize) -> Vec<u16> {
    let count = bytes.len() * 8 / d;
    let mask = (1u32 << d) - 1;

    let mut values = Vec::with_capacity(count);
    let mut byte_stream = bytes.iter();
    let mut accumulator: u32 = 0;
    let mut bits = 0usize;

    for _ in 0..count {
        while bits < d {
            let byte = byte_stream.next().copied().unwrap_or(0);
            accumulator |= u32::from(byte) << bits;
            bits += 8;
        }

        values.push((accumulator & mask) as u16);
        accumulator >>= d;
        bits -= d;
    }

    values
}

impl TqElement {
    /// `ByteEncode_12`: serializes the 256 NTT coefficients into 384 bytes.
    pub fn byte_encode(&self) -> Vec<u8> {
        let values: Vec<u16> = self.coefficients().iter().map(|c| c.value()).collect();

        pack(&values, D_12)
    }

    /// `ByteDecode_12`: deserializes 384 bytes into 256 NTT coefficients.
    ///
    /// Each 12-bit value is reduced modulo q, so an input encoding a
    /// coefficient `>= q` does not round-trip — this is what the
    /// encapsulation-key modulus check of [section 7.2] relies on.
    ///
    /// [section 7.2]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.7.2
    pub fn byte_decode(bytes: &[u8]) -> Self {
        let values = unpack(bytes, D_12);
        let coefficients =
            core::array::from_fn(|i| FieldElement::new(*values.get(i).unwrap_or(&0)));

        Self::new(coefficients)
    }
}

impl RqElement {
    /// `ByteEncode_d(Compress_d(self))`: compresses each coefficient to `d`
    /// bits and packs the result into `32 * d` bytes.
    pub fn compress_encode(&self, d: usize) -> Vec<u8> {
        let values: Vec<u16> = self.coefficients().iter().map(|c| c.compress(d)).collect();

        pack(&values, d)
    }

    /// `Decompress_d(ByteDecode_d(bytes))`: unpacks `d`-bit values and
    /// decompresses each back into Zq.
    pub fn decode_decompress(bytes: &[u8], d: usize) -> Self {
        let values = unpack(bytes, d);
        let coefficients =
            core::array::from_fn(|i| FieldElement::decompress(*values.get(i).unwrap_or(&0), d));

        Self::new(coefficients)
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
        let encoded = self.compress_encode(1);

        core::array::from_fn(|i| *encoded.get(i).unwrap_or(&0))
    }
}

impl<P: ParameterSet> TqVector<P> {
    /// `ByteEncode_12` applied componentwise: `384 * K` bytes.
    pub fn byte_encode(&self) -> Vec<u8> {
        self.as_slice()
            .iter()
            .flat_map(TqElement::byte_encode)
            .collect()
    }

    /// `ByteDecode_12` applied componentwise to `384 * K` bytes.
    pub fn byte_decode(bytes: &[u8]) -> Self {
        let mut chunks = bytes.chunks_exact(384);

        // `from_fn` drives exactly `K` iterations and `bytes` holds `384 * K`
        // bytes, so every `next()` yields a full chunk; the empty fallback is
        // unreachable.
        Self::from_fn(|_| TqElement::byte_decode(chunks.next().unwrap_or_default()))
    }
}

impl<P: ParameterSet> RqVector<P> {
    /// `ByteEncode_d(Compress_d(.))` applied componentwise: `32 * d * K` bytes.
    pub fn compress_encode(&self, d: usize) -> Vec<u8> {
        self.as_slice()
            .iter()
            .flat_map(|poly| poly.compress_encode(d))
            .collect()
    }

    /// `Decompress_d(ByteDecode_d(.))` applied componentwise.
    pub fn decode_decompress(bytes: &[u8], d: usize) -> Self {
        let mut chunks = bytes.chunks_exact(32 * d);

        // `from_fn` drives exactly `K` iterations and `bytes` holds `32 * d * K`
        // bytes, so every `next()` yields a full chunk; the empty fallback is
        // unreachable.
        Self::from_fn(|_| RqElement::decode_decompress(chunks.next().unwrap_or_default(), d))
    }
}
