//! Differential tests: the NEON kernels must agree with the portable scalar
//! backend on random inputs, across the full `i16` representative range.

use proptest::prelude::*;

use crate::{
    algebraic::{field::FieldElement, poly::arch::generic},
    parameters,
};

/// A [`FieldElement; 256`] strategy over pseudo-random `i16` representatives
/// (not just canonical values), exercising the full input domain of the
/// Montgomery arithmetic that `multiply` accepts.
fn any_poly() -> impl Strategy<Value = [FieldElement; parameters::N]> {
    prop::collection::vec(any::<i16>(), parameters::N).prop_map(|values| {
        let mut poly = [FieldElement::ZERO; parameters::N];
        for (out, &v) in poly.iter_mut().zip(values.iter()) {
            *out = FieldElement::from_montgomery_table(v);
        }
        poly
    })
}

/// A [`FieldElement; 256`] strategy over canonical coefficients (`0..q`). The
/// forward NTT accumulates across stages without per-stage reduction, so its
/// inputs must be bounded — unlike `multiply`, whose Montgomery reduction
/// bounds every product.
fn canonical_poly() -> impl Strategy<Value = [FieldElement; parameters::N]> {
    prop::collection::vec(0u16..parameters::Q, parameters::N).prop_map(|values| {
        let mut poly = [FieldElement::ZERO; parameters::N];
        for (out, &v) in poly.iter_mut().zip(values.iter()) {
            *out = FieldElement::new(v);
        }
        poly
    })
}

proptest! {
    /// `multiply` matches the scalar backend on arbitrary Montgomery-domain
    /// inputs.
    #[test]
    fn multiply_matches_generic(f in any_poly(), g in any_poly()) {
        prop_assert_eq!(super::multiply(&f, &g), generic::multiply(&f, &g));
    }

    /// `ntt` matches the scalar backend on any canonical input polynomial.
    #[test]
    fn ntt_matches_generic(input in canonical_poly()) {
        let mut vectorized = input;
        let mut scalar = input;
        super::ntt(&mut vectorized);
        generic::ntt(&mut scalar);

        prop_assert_eq!(vectorized, scalar);
    }

    /// `ntt_inverse` matches the scalar backend on any canonical input.
    #[test]
    fn ntt_inverse_matches_generic(input in canonical_poly()) {
        let mut vectorized = input;
        let mut scalar = input;
        super::ntt_inverse(&mut vectorized);
        generic::ntt_inverse(&mut scalar);

        prop_assert_eq!(vectorized, scalar);
    }
}

/// Every [`crate::algebraic::poly::arch::ZETA_BARRETT`] entry matches
/// `round(zeta * 2^15 / q)` recomputed via an independent `u32` code path.
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
