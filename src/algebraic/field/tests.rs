//! Unit tests for prime-field arithmetic.

use super::*;

/// Field subtraction is the additive inverse of addition.
#[test]
fn field_add_sub_roundtrip() {
    for a in 0..parameters::Q {
        for b in (0..parameters::Q).step_by(97) {
            let (a, b) = (FieldElement::new(a), FieldElement::new(b));

            assert_eq!((a + b) - b, a);
        }
    }
}

/// Negation followed by addition yields zero.
#[test]
fn field_neg_is_additive_inverse() {
    for a in (0..parameters::Q).step_by(13) {
        let a = FieldElement::new(a);

        assert_eq!(a + (-a), FieldElement::ZERO);
    }
}

/// `Decompress` then `Compress` is the identity on `d`-bit values.
#[test]
fn compress_decompress_roundtrip() {
    for &d in &[1usize, 4, 5, 10, 11] {
        for y in 0..(1u16 << d) {
            let decompressed = FieldElement::decompress(y, d);

            assert_eq!(decompressed.compress(d), y, "d = {d}, y = {y}");
        }
    }
}
