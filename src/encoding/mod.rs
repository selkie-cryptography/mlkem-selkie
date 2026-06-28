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

/// Packs `N` `d`-bit values into a byte stream, least-significant bit first.
///
/// Implements `BitsToBytes` composed with the bit decomposition of
/// `ByteEncode_d` (Algorithm 5): the bit at position `j` of `values[i]` is
/// written to global bit `i * d + j`, which lands in byte `(i * d + j) / 8`.
/// The `N * d / 8` packed bytes are yielded lazily, so a caller can drive them
/// into an owned buffer without an intermediate allocation.
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
/// Inverse of [`pack`]; implements the bit recomposition of `ByteDecode_d`
/// (Algorithm 6). Reads the leading `N * d / 8` bytes, each value in `0..2^d`;
/// a short input is zero-padded.
fn unpack(bytes: &[u8], d: usize) -> [u16; N] {
    let mask = (1u32 << d) - 1;

    let mut bytes = bytes.iter();
    let mut accumulator: u32 = 0;
    let mut bits = 0usize;

    core::array::from_fn(|_| {
        while bits < d {
            let byte = bytes.next().copied().unwrap_or(0);
            accumulator |= u32::from(byte) << bits;
            bits += 8;
        }

        let value = (accumulator & mask) as u16;
        accumulator >>= d;
        bits -= d;

        value
    })
}

impl TqElement {
    /// `ByteEncode_12`: serializes the 256 NTT coefficients as 384 bytes,
    /// yielded lazily.
    pub fn byte_encode(&self) -> impl Iterator<Item = u8> {
        pack(self.coefficients().map(|c| c.value()), D_12)
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
    /// bits and packs the result as `32 * d` bytes, yielded lazily.
    pub fn compress_encode(&self, d: usize) -> impl Iterator<Item = u8> {
        pack(self.coefficients().map(|c| c.compress(d)), d)
    }

    /// `Decompress_d(ByteDecode_d(bytes))`: unpacks `d`-bit values and
    /// decompresses each back into Zq.
    pub fn decode_decompress(bytes: &[u8], d: usize) -> Self {
        let coefficients = unpack(bytes, d).map(|v| FieldElement::decompress(v, d));

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
        let mut bytes = self.compress_encode(1);

        core::array::from_fn(|_| bytes.next().unwrap_or(0))
    }
}

impl<P: ParameterSet> TqVector<P> {
    /// `ByteEncode_12` applied componentwise: `384 * K` bytes.
    pub fn byte_encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(384 * P::K);
        out.extend(self.as_slice().iter().flat_map(TqElement::byte_encode));

        out
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
    /// `ByteEncode_d(Compress_d(.))` applied componentwise: `32 * d * K` bytes.
    pub fn compress_encode(&self, d: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 * d * P::K);
        out.extend(
            self.as_slice()
                .iter()
                .flat_map(|poly| poly.compress_encode(d)),
        );

        out
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
