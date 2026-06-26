//! Property-based tests of the internal field and ring arithmetic.
//!
//! Requires `--features expose-internals`.

use mlkem_selkie::algebraic::{FieldElement, PolynomialRingElement, RqElement, TqElement};
use proptest::{collection::vec, prelude::*};

/// The ML-KEM modulus.
const Q: u16 = 3329;

/// The ring degree.
const N: usize = 256;

/// Shorthand for a canonical field element.
fn fe(value: u16) -> FieldElement {
    FieldElement::new(value)
}

/// Builds an Rq element from 256 coefficient values.
fn rq(coeffs: &[u16]) -> RqElement {
    RqElement::new(core::array::from_fn(|i| fe(coeffs[i])))
}

/// Reference multiplication in `Zq[X] / (X^256 + 1)`.
// reason: the convolution writes `result[i + j]`, so the indices i and j drive
// both the operand reads and a computed output index; an enumerate rewrite
// can't express the wrap-around accumulation cleanly.
#[allow(clippy::needless_range_loop)]
fn schoolbook(f: &[u16], g: &[u16]) -> RqElement {
    let mut result = [FieldElement::ZERO; N];

    for i in 0..N {
        for j in 0..N {
            let product = fe(f[i]) * fe(g[j]);
            let k = i + j;

            if k < N {
                result[k] = result[k] + product;
            } else {
                result[k - N] = result[k - N] - product;
            }
        }
    }

    RqElement::new(result)
}

proptest! {
    /// Field multiplication commutes.
    #[test]
    fn field_mul_commutes(a in 0u16..Q, b in 0u16..Q) {
        prop_assert_eq!(fe(a) * fe(b), fe(b) * fe(a));
    }

    /// Multiplication distributes over addition.
    #[test]
    fn field_distributive(a in 0u16..Q, b in 0u16..Q, c in 0u16..Q) {
        prop_assert_eq!(fe(a) * (fe(b) + fe(c)), fe(a) * fe(b) + fe(a) * fe(c));
    }

    /// Subtraction equals addition of the negation.
    #[test]
    fn field_sub_is_add_neg(a in 0u16..Q, b in 0u16..Q) {
        prop_assert_eq!(fe(a) - fe(b), fe(a) + (-fe(b)));
    }

    /// The NTT and its inverse compose to the identity.
    #[test]
    fn ntt_roundtrip(coeffs in vec(0u16..Q, N)) {
        let f = rq(&coeffs);

        prop_assert_eq!(f.ntt().ntt_inverse(), f);
    }

    /// Multiplication in Tq matches schoolbook multiplication in Rq.
    #[test]
    fn ntt_mul_matches_schoolbook(f in vec(0u16..Q, N), g in vec(0u16..Q, N)) {
        let via_ntt = (rq(&f).ntt() * rq(&g).ntt()).ntt_inverse();

        prop_assert_eq!(via_ntt, schoolbook(&f, &g));
    }

    /// `ByteEncode_12` / `ByteDecode_12` round-trip on canonical coefficients.
    #[test]
    fn byte_encode_roundtrip(coeffs in vec(0u16..Q, N)) {
        let f = TqElement::new(core::array::from_fn(|i| fe(coeffs[i])));

        prop_assert_eq!(TqElement::byte_decode(&f.byte_encode()), f);
    }

    /// `Decompress_d` then `Compress_d` is the identity on `d`-bit values.
    #[test]
    fn compress_decompress_roundtrip(
        (d, y) in prop::sample::select(vec![1usize, 4, 5, 10, 11])
            .prop_flat_map(|d| (Just(d), 0u16..(1u16 << d))),
    ) {
        prop_assert_eq!(FieldElement::decompress(y, d).compress(d), y);
    }
}
