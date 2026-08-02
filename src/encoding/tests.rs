//! Unit tests for byte serialization and compression.

use super::*;
use crate::{
    algebraic::FieldElement,
    parameters::{self, MLKEM512},
};

/// `pack` and `unpack` are inverse for every supported bit width, and
/// `pack_words` agrees with `pack` at the fixed widths.
#[test]
fn pack_unpack_roundtrip() {
    for &d in &[1usize, 4, 5, 10, 11, 12] {
        let values: [u16; N] = core::array::from_fn(|i| (i as u16) & ((1 << d) - 1));

        let packed: Vec<u8> = pack(values, d).collect();

        assert_eq!(packed.len(), N * d / 8);
        assert_eq!(unpack(&packed, d), values);
    }

    let values: [u16; N] = core::array::from_fn(|i| (i as u16) & 0xFFF);
    let words: [u8; 384] = pack_words(&values, 12);
    let streamed: Vec<u8> = pack(values, 12).collect();

    assert_eq!(words.as_slice(), streamed.as_slice());
}

/// `ByteEncode_12` round-trips NTT coefficients that are already in `0..q`.
#[test]
fn byte_encode_12_roundtrip() {
    let coeffs = core::array::from_fn(|i| FieldElement::new((31 * i as u16 + 5) % parameters::Q));
    let poly = TqElement::new(coeffs);

    let encoded = poly.byte_encode();

    assert_eq!(TqElement::byte_decode(&encoded), poly);
}

/// A coefficient `>= q` does not survive a `ByteDecode_12` / `ByteEncode_12`
/// round-trip, which is the basis of the modulus check.
#[test]
fn byte_decode_12_reduces_mod_q() {
    // Pack a raw 12-bit value of q (3329) into the first coefficient slot.
    let mut values = [0u16; N];
    if let Some(first) = values.first_mut() {
        *first = parameters::Q;
    }

    let bytes: [u8; 384] = pack_words(&values, 12);

    let decoded = TqElement::byte_decode(&bytes);

    // q mod q == 0, so the re-encoding differs from the malformed input.
    assert_ne!(decoded.byte_encode(), bytes);
}

/// Message bytes survive a `Decompress_1` / `Compress_1` round-trip.
#[test]
fn message_polynomial_roundtrip() {
    let message: [u8; 32] = core::array::from_fn(|i| (13 * i + 1) as u8);

    let polynomial = RqElement::from_message(&message);

    assert_eq!(polynomial.compress_message(), message);
}

/// A compressed vector round-trips through `decode_decompress` for the
/// ciphertext bit widths.
#[test]
fn vector_compress_roundtrip() {
    let d = MLKEM512::D_U;
    let vector = RqVector::<MLKEM512>::from_fn(|p| {
        let coeffs = core::array::from_fn(|i| {
            FieldElement::decompress(((i + p) as u16) & ((1 << d) - 1), d)
        });

        RqElement::new(coeffs)
    });

    let encoded: Vec<u8> = vector.compress_encode(d).collect();

    assert_eq!(RqVector::<MLKEM512>::decode_decompress(&encoded, d), vector);
}

/// A short input zero-pads: nine `0xFF` bytes at `d = 12` fill values 0..=5
/// (the 8-byte word plus the tail byte reach bit 72), and value 6 onward is
/// zero. Losing the partial-word tail would zero value 5's top eight bits.
#[test]
fn unpack_zero_pads_short_input() {
    let values = unpack(&[0xFF; 9], 12);

    assert_eq!(values[4], 0xFFF);
    assert_eq!(values[5], 0xFFF);
    assert_eq!(values[6], 0);
}
