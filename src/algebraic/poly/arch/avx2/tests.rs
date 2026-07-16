//! Differential tests: the AVX2 kernels must agree with the portable scalar
//! backend on random canonical inputs. These run only on an AVX2-capable
//! x86_64 target (CI), so the host that authored them (aarch64) cannot.

use proptest::prelude::*;

use crate::{
    algebraic::{field::FieldElement, poly::arch::generic},
    parameters,
};

/// A [`FieldElement; 256`] strategy over pseudo-random `i16` representatives;
/// `multiply`'s Montgomery reduction bounds every product, so inputs need not
/// be canonical.
fn any_poly() -> impl Strategy<Value = [FieldElement; parameters::N]> {
    prop::collection::vec(any::<i16>(), parameters::N).prop_map(|values| {
        let mut poly = [FieldElement::ZERO; parameters::N];
        for (out, &v) in poly.iter_mut().zip(values.iter()) {
            *out = FieldElement::from_montgomery_table(v);
        }
        poly
    })
}

/// A [`FieldElement; 256`] strategy over canonical coefficients (`0..q`).
/// The transforms accumulate without per-stage reduction, so inputs must be
/// bounded.
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
    // 4096 cases × 256 coefficients per poly hits the i16 boundary reliably
    // and covers a wider input distribution than proptest's default 256.
    // Failing inputs are persisted to `proptest-regressions/` and replayed on
    // subsequent runs (proptest's built-in `FileFailurePersistence`).
    #![proptest_config(ProptestConfig {
        cases: 4096,
        .. ProptestConfig::default()
    })]

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
/// few units, so mutations to the const initializer leave [`ntt`] output
/// unchanged modulo q and would slip through [`ntt_matches_generic`]. Mirror
/// of the NEON test, sharing the same table.
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
