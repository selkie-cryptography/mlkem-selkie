//! Unit tests for prime-field arithmetic.

use proptest::prelude::*;

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

/// `PartialEq` distinguishes non-congruent field elements. Guards against the
/// impl collapsing to `true` (or `false`) — the existing `assert_eq!`s only
/// cover the equal-case, so a constant-`true` `eq` would slip through.
#[test]
fn partial_eq_distinguishes_and_agrees() {
    let zero = FieldElement::ZERO;
    let one = FieldElement::from(1u8);

    assert_ne!(zero, one);
    assert_eq!(zero, FieldElement::ZERO);
    assert_eq!(one, FieldElement::from(1u8));

    // Two representatives of the same class (`0` and `q`) compare equal.
    assert_eq!(FieldElement::new(0), FieldElement::new(parameters::Q));

    // Every canonical residue in `[0, q)` is `Eq` to itself and distinct from
    // its successor mod q.
    for v in 0..parameters::Q {
        let a = FieldElement::new(v);
        let b = FieldElement::new((v + 1) % parameters::Q);

        assert_eq!(a, a);
        assert_ne!(a, b);
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

/// Exercises [`FieldElement`]'s reduction path across a fixed set of i16
/// boundary values that proptest can miss.
#[test]
fn field_element_boundary_values_no_overflow() {
    let q = parameters::Q as i16;
    let boundary = [
        i16::MIN,
        i16::MIN + 1,
        i16::MIN / 2,
        -q - 1,
        -q,
        -q + 1,
        -1,
        0,
        1,
        q - 1,
        q,
        q + 1,
        i16::MAX / 2,
        i16::MAX - 1,
        i16::MAX,
    ];
    for &raw in &boundary {
        let fe = FieldElement::from_montgomery_table(raw);
        let canonical = fe.value();

        assert!(
            u32::from(canonical) < u32::from(parameters::Q),
            "value = {canonical} not in [0, Q) for raw = {raw}",
        );
        assert_eq!(
            fe.reduce().reduce(),
            fe.reduce(),
            "reduce not idempotent at raw = {raw}",
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 4096,
        .. ProptestConfig::default()
    })]

    /// [`FieldElement::barrett_const_mul`] with `b_bar = round(b · 2^15 / q)`
    /// computes `a · b mod q` for every canonical `(a, b)` pair, matching the
    /// reference u32 modular multiply. Covers the whole NTT butterfly input
    /// space (`a` grows past `q` across stages, so the input range extends
    /// well beyond canonical).
    #[test]
    fn barrett_const_mul_matches_reference(
        a in -4 * (parameters::Q as i16)..=4 * (parameters::Q as i16),
        b in 0u16..parameters::Q,
    ) {
        let b_bar = ((u32::from(b) * 32_768 + u32::from(parameters::Q) / 2)
            / u32::from(parameters::Q)) as i16;

        let got = FieldElement::from_montgomery_table(a)
            .barrett_const_mul(b as i16, b_bar)
            .reduce()
            .value();

        // Reference: reduce `a` to canonical [0, q) first, then modular multiply.
        let a_canonical = a.rem_euclid(parameters::Q as i16) as u32;
        let want = ((a_canonical * u32::from(b)) % u32::from(parameters::Q)) as u16;

        prop_assert_eq!(got, want);
    }
}
