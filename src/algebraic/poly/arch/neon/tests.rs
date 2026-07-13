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

/// Fills a polynomial with canonical coefficients (`0..q`). The forward NTT
/// accumulates across stages without per-stage reduction, so its inputs must be
/// bounded — unlike `multiply`, whose Montgomery reduction bounds every
/// product.
fn random_canonical_poly(state: &mut u32) -> [FieldElement; parameters::N] {
    core::array::from_fn(|_| {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;

        FieldElement::new((*state & 0xFFFF) as u16)
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

/// `ntt` matches the scalar backend over many random canonical inputs.
#[test]
fn ntt_matches_generic() {
    let mut state = 0x9E37_79B9;

    for _ in 0..1000 {
        let input = random_canonical_poly(&mut state);

        let mut vectorized = input;
        let mut scalar = input;
        super::ntt(&mut vectorized);
        generic::ntt(&mut scalar);

        assert_eq!(vectorized, scalar);
    }
}

/// Every [`crate::algebraic::poly::arch::ZETA_BARRETT`] entry matches `round(zeta * 2^15 / q)`
/// recomputed via an independent `u32` code path.
///
/// The Barrett-with-constant sequence self-corrects for `b_bar` errors of a
/// few units, so mutations that perturb the const initializer (rounding-
/// direction flips, operator swaps in the divisor) leave [`ntt`] output
/// unchanged modulo q and would slip through [`ntt_matches_generic`]. This
/// test compares the shipped table against a from-scratch recomputation so
/// any per-entry mismatch is caught directly.
#[test]
fn zeta_barrett_matches_reference() {
    let q: u32 = parameters::Q as u32;
    let pairs = crate::algebraic::poly::arch::ZETA_RAW
        .iter()
        .zip(crate::algebraic::poly::arch::ZETA_BARRETT.iter());

    for (i, (&zeta, &bar)) in pairs.enumerate() {
        let zeta_u = u32::from(zeta);
        let reference = ((zeta_u * 32_768 + q / 2) / q) as i16;

        assert_eq!(
            bar, reference,
            "ZETA_BARRETT[{i}] mismatch: got {bar}, expected {reference} (zeta={zeta})",
        );
    }
}

/// `ntt_inverse` matches the scalar backend over many random canonical inputs.
#[test]
fn ntt_inverse_matches_generic() {
    let mut state = 0xDEAD_BEEF;

    for _ in 0..1000 {
        let input = random_canonical_poly(&mut state);

        let mut vectorized = input;
        let mut scalar = input;
        super::ntt_inverse(&mut vectorized);
        generic::ntt_inverse(&mut scalar);

        assert_eq!(vectorized, scalar);
    }
}
