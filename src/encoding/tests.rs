//! Unit tests for byte serialization and compression.

use super::*;
use crate::{
    algebraic::FieldElement,
    parameters::{self, MLKEM512},
};

/// `pack` and `unpack` are inverse for every supported bit width.
#[test]
fn pack_unpack_roundtrip() {
    for &d in &[1usize, 4, 5, 10, 11, 12] {
        let values: [u16; N] = core::array::from_fn(|i| (i as u16) & ((1 << d) - 1));

        let packed: Vec<u8> = pack(values, d).collect();

        assert_eq!(packed.len(), N * d / 8);
        assert_eq!(unpack(&packed, d), values);
    }
}

/// `ByteEncode_12` round-trips NTT coefficients that are already in `0..q`.
#[test]
fn byte_encode_12_roundtrip() {
    let coeffs = core::array::from_fn(|i| FieldElement::new((31 * i as u16 + 5) % parameters::Q));
    let poly = TqElement::new(coeffs);

    let encoded: Vec<u8> = poly.byte_encode().collect();

    assert_eq!(encoded.len(), 384);
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

    let bytes: Vec<u8> = pack(values, 12).collect();

    let decoded = TqElement::byte_decode(&bytes);

    // q mod q == 0, so the re-encoding differs from the malformed input.
    let reencoded: Vec<u8> = decoded.byte_encode().collect();
    assert_ne!(reencoded, bytes);
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

    assert_eq!(encoded.len(), 32 * d * MLKEM512::K);
    assert_eq!(RqVector::<MLKEM512>::decode_decompress(&encoded, d), vector);
}
