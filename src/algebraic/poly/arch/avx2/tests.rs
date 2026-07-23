//! Differential tests: the AVX2 kernels must agree with the portable scalar
//! backend on random canonical inputs. These run only on an AVX2-capable
//! x86_64 target (CI), so the host that authored them (aarch64) cannot.

use core::array;

use proptest::prelude::*;

use crate::{
    algebraic::{
        field::FieldElement,
        poly::arch::{GAMMA_MONT, ProductAccumulator, generic},
    },
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

/// A [`FieldElement; 256`] strategy over the full `ntt_inverse` input
/// contract, `|x| ≤ 16383`: wide enough that a mis-scheduled reduction
/// overflows `i16` mid-transform where canonical inputs would not.
fn inverse_contract_poly() -> impl Strategy<Value = [FieldElement; parameters::N]> {
    prop::collection::vec(-16383i16..=16383, parameters::N).prop_map(|values| {
        let mut poly = [FieldElement::ZERO; parameters::N];
        for (out, &v) in poly.iter_mut().zip(values.iter()) {
            *out = FieldElement::from_montgomery_table(v);
        }
        poly
    })
}

/// The raw representatives of a polynomial. `FieldElement`'s `PartialEq`
/// compares canonical values; comparing representatives instead pins both
/// backends to the same reduction schedule, not just congruent outputs.
fn representatives(poly: &[FieldElement; parameters::N]) -> [i16; parameters::N] {
    poly.map(FieldElement::representative)
}

/// A [`FieldElement; 256`] strategy over representatives bounded by `3q/2`,
/// the accumulated base-multiplication input domain (see
/// `FieldElement::from_product_sum`); unbounded `i16` inputs would overflow
/// the `i32` accumulator at dot length 4.
fn accumulation_poly() -> impl Strategy<Value = [FieldElement; parameters::N]> {
    let bound = 3 * parameters::Q as i16 / 2;
    prop::collection::vec(-bound..=bound, parameters::N).prop_map(|values| {
        let mut poly = [FieldElement::ZERO; parameters::N];
        for (out, &v) in poly.iter_mut().zip(values.iter()) {
            *out = FieldElement::from_montgomery_table(v);
        }
        poly
    })
}

/// The asymmetric base-multiplication cache of `g`, computed with scalar field
/// arithmetic (the shared scalar path both backends consume).
// reason: indices 2i+1 and i are provably in bounds for i in 0..128, and the
// pairwise indexing matches `multiply`.
#[allow(clippy::indexing_slicing)]
fn mul_cache(g: &[FieldElement; parameters::N]) -> [FieldElement; parameters::N / 2] {
    array::from_fn(|i| g[2 * i + 1] * GAMMA_MONT[i])
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

    /// `multiply` matches the scalar backend, representative-exact, on
    /// arbitrary Montgomery-domain inputs.
    #[test]
    fn multiply_matches_generic(f in any_poly(), g in any_poly()) {
        prop_assert_eq!(
            representatives(&super::multiply(&f, &g)),
            representatives(&generic::multiply(&f, &g)),
        );
    }

    /// `ntt` matches the scalar backend, representative-exact, on any
    /// canonical input polynomial.
    #[test]
    fn ntt_matches_generic(input in canonical_poly()) {
        let mut vectorized = input;
        let mut scalar = input;
        super::ntt(&mut vectorized);
        generic::ntt(&mut scalar);

        prop_assert_eq!(representatives(&vectorized), representatives(&scalar));
    }

    /// `ntt_inverse` matches the scalar backend, representative-exact, on
    /// any canonical input.
    #[test]
    fn ntt_inverse_matches_generic(input in canonical_poly()) {
        let mut vectorized = input;
        let mut scalar = input;
        super::ntt_inverse(&mut vectorized);
        generic::ntt_inverse(&mut scalar);

        prop_assert_eq!(representatives(&vectorized), representatives(&scalar));
    }

    /// `ntt_inverse` matches the scalar backend across its full input
    /// contract, where a mis-placed reduction wraps `i16` mid-transform.
    #[test]
    fn ntt_inverse_matches_generic_on_contract(input in inverse_contract_poly()) {
        let mut vectorized = input;
        let mut scalar = input;
        super::ntt_inverse(&mut vectorized);
        generic::ntt_inverse(&mut scalar);

        prop_assert_eq!(representatives(&vectorized), representatives(&scalar));
    }

    /// The `basemul_accumulate` / `basemul_reduce` pair matches the scalar
    /// backend at every dot-product length the parameter sets use.
    // reason: j < k <= 4 indexes the length-4 strategy vectors; the loop
    // structure mirrors the dot product it tests.
    #[allow(clippy::indexing_slicing)]
    #[test]
    fn basemul_accumulate_reduce_matches_generic(
        f in prop::collection::vec(accumulation_poly(), 4),
        g in prop::collection::vec(accumulation_poly(), 4),
    ) {
        for k in 1..=4 {
            let mut vectorized = ProductAccumulator::default();
            let mut scalar = ProductAccumulator::default();

            for j in 0..k {
                let cache = mul_cache(&g[j]);
                super::basemul_accumulate(&mut vectorized, &f[j], &g[j], &cache);
                generic::basemul_accumulate(&mut scalar, &f[j], &g[j], &cache);
            }

            prop_assert_eq!(
                representatives(&super::basemul_reduce(&vectorized)),
                representatives(&generic::basemul_reduce(&scalar)),
            );
        }
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
