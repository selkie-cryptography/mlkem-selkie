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

/// `FieldElement::new` produces the same canonical representative as `value %
/// q` for every input in the full `u16` range — pinning the explicit Barrett
/// mul-shift reduction to the textbook semantics.
#[test]
fn new_matches_mod_q() {
    for value in 0u16..=u16::MAX {
        let oracle = value % parameters::Q;
        let actual = FieldElement::new(value).value();

        assert_eq!(actual, oracle, "value = {value}");
    }
}

/// `FieldElement::compress` produces the same result as the textbook
/// `numerator / q` for every (value, d) pair in the compress range — pinning
/// the explicit Barrett mul-shift `(num * COMPRESS_MULT) >> 33`.
#[test]
fn compress_matches_textbook_division() {
    for d in 1usize..=12 {
        let mask = (1u32 << d) - 1;
        for value in 0..parameters::Q {
            let element = FieldElement::new(value);
            let actual = element.compress(d);

            let numerator = (u32::from(value) << d) + u32::from(parameters::Q / 2);
            let textbook_quotient = numerator / u32::from(parameters::Q);
            let oracle = (textbook_quotient & mask) as u16;

            assert_eq!(actual, oracle, "d = {d}, value = {value}");
        }
    }
}

/// `From<u8>` and `From<FieldElement> for u16` agree with the canonical
/// `FieldElement::new` / `value` reductions.
#[test]
fn from_u8_and_to_u16_roundtrip() {
    for raw in 0u16..=u16::from(u8::MAX) {
        let from_u8 = FieldElement::from(raw as u8);
        let from_u16 = FieldElement::from(raw);

        assert_eq!(from_u8, from_u16);
        assert_eq!(u16::from(from_u16), raw % parameters::Q);
    }
}

/// `montgomery_table` agrees with applying `to_montgomery` element-wise to
/// the source array. The function is `const` and used at compile time to
/// build the NTT zeta table, but we call it at runtime here so coverage
/// instrumentation sees the body.
#[test]
fn montgomery_table_matches_element_wise() {
    let raw: [u16; 128] =
        core::array::from_fn(|i| ((i as u16).wrapping_mul(57) + 11) % parameters::Q);

    let table = FieldElement::montgomery_table(raw);

    for (i, (entry, source)) in table.iter().zip(raw.iter()).enumerate() {
        assert_eq!(
            *entry,
            FieldElement::new(*source).to_montgomery(),
            "i = {i}"
        );
    }
}
