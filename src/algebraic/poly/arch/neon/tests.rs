//! Differential tests: the NEON kernels must agree with the portable scalar
//! backend on random inputs, across the full `i16` representative range.

use crate::{
    algebraic::{field::FieldElement, poly::arch::generic},
    parameters,
};

/// Fills a polynomial with pseudo-random `i16` representatives (not just
/// canonical values) from a xorshift32 stream, exercising the full input domain
/// of the Montgomery arithmetic.
fn random_poly(state: &mut u32) -> [FieldElement; parameters::N] {
    core::array::from_fn(|_| {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;

        FieldElement::from_montgomery_table((*state & 0xFFFF) as i16)
    })
}

/// `multiply` matches the scalar backend over many random input pairs.
#[test]
fn multiply_matches_generic() {
    let mut state = 0x1234_5678;

    for _ in 0..1000 {
        let f = random_poly(&mut state);
        let g = random_poly(&mut state);

        assert_eq!(super::multiply(&f, &g), generic::multiply(&f, &g));
    }
}
