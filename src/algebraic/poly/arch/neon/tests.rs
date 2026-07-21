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

/// Thinned, software-pipelined stride-128 inverse-NTT stage: no sum-path
/// reduction, per the lazy len-2/len-16 schedule (this stage sees
/// `|x| ≤ 4q`). Test-only.
///
/// # Safety
///
/// `ptr` must point to a 256-`i16` window mutable for reads and writes, with
/// every representative in `[-4q, 4q]` so the unreduced sums fit `i16`.
unsafe fn intt_stride128_thin_asm(ptr: *mut i16, zeta: i16, zeta_bar: i16) {
    core::arch::asm!(
        // Prologue: broadcast constants and set the iteration count.
        "dup     v28.8h, {zeta:w}",
        "dup     v29.8h, {zbar:w}",
        "mov     w9,     #3329",
        "dup     v30.8h, w9",
        "mov     x9,     #8",

        // Preamble — seeds the in-flight iterations.
        "ldp q3, q16, [{ptr}, #0]",
        "ldp q7, q6, [{ptr}, #256]",
        "ldp q0, q1, [{ptr}, #32]",
        "sub v2.8H, v7.8H, v3.8H",
        "sub v20.8H, v6.8H, v16.8H",
        "add v21.8H, v16.8H, v6.8H",
        "sqrdmulh v22.8H, v2.8H, v29.8H",
        "mul v16.8H, v2.8H, v28.8H",
        "mul v6.8H, v20.8H, v28.8H",
        "sqrdmulh v18.8H, v20.8H, v29.8H",
        "add v7.8H, v3.8H, v7.8H",
        "ldp q2, q3, [{ptr}, #288]",
        "stp q7, q21, [{ptr}], #32",
        "mls v16.8H, v22.8H, v30.8H",
        "mls v6.8H, v18.8H, v30.8H",
        "sub x9, x9, #2",

        // Steady-state body — 6 cy/iter (IPC 2.33) on the M4 model.
    "2:",
        "sub v5.8H, v2.8H, v0.8H",
        "add v2.8H, v0.8H, v2.8H",
        "sub v7.8H, v3.8H, v1.8H",
        "add v3.8H, v1.8H, v3.8H",
        "ldp q0, q1, [{ptr}, #32]",
        "sqrdmulh v17.8H, v5.8H, v29.8H",
        "stp q2, q3, [{ptr}], #32",
        "stp q16, q6, [{ptr}, #192]",
        "mul v16.8H, v5.8H, v28.8H",
        "mul v6.8H, v7.8H, v28.8H",
        "sqrdmulh v7.8H, v7.8H, v29.8H",
        "ldp q2, q3, [{ptr}, #256]",
        "mls v16.8H, v17.8H, v30.8H",
        "mls v6.8H, v7.8H, v30.8H",
        "sub x9, x9, 1",
        "cbnz x9, 2b",

        // Postamble — drains the in-flight iterations.
        "add v5.8H, v0.8H, v2.8H",
        "sub v4.8H, v2.8H, v0.8H",
        "sub v0.8H, v3.8H, v1.8H",
        "stp q16, q6, [{ptr}, #224]",
        "add v16.8H, v1.8H, v3.8H",
        "sqrdmulh v17.8H, v0.8H, v29.8H",
        "mul v19.8H, v0.8H, v28.8H",
        "mul v2.8H, v4.8H, v28.8H",
        "sqrdmulh v0.8H, v4.8H, v29.8H",
        "stp q5, q16, [{ptr}], #32",
        "mls v2.8H, v0.8H, v30.8H",
        "mls v19.8H, v17.8H, v30.8H",
        "stp q2, q19, [{ptr}, #224]",

        ptr  = inout(reg) ptr => _,
        zeta = in(reg) zeta as u32,
        zbar = in(reg) zeta_bar as u32,
        out("x9")  _,
        out("v0")  _, out("v1")  _, out("v2")  _, out("v3")  _,
        out("v4")  _, out("v5")  _, out("v6")  _, out("v7")  _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v28") _,
        out("v29") _, out("v30") _,
        options(nostack),
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 4096,
        .. ProptestConfig::default()
    })]

    /// The thinned stage computes exact unreduced sums on the low half and
    /// congruent, `(-q, q)`-bounded Barrett products on the high half, for
    /// inputs in `[-4q, 4q]` — this stage's entry bound under the lazy
    /// schedule.
    #[test]
    fn intt_stride128_thin_matches_reference(
        window in prop::collection::vec(
            -4 * (parameters::Q as i16)..=4 * (parameters::Q as i16),
            256,
        )
    ) {
        let q = i64::from(parameters::Q);
        // The len=128 stage's zeta pair (k = 1 in `ntt_inverse`).
        let zeta = crate::algebraic::poly::arch::ZETA_RAW[1] as i16;
        let zbar = crate::algebraic::poly::arch::ZETA_BARRETT[1];

        let mut out = [0i16; 256];
        out.copy_from_slice(&window);
        // SAFETY: 256-i16 window; inputs bounded in [-4q, 4q] per the strategy.
        unsafe { intt_stride128_thin_asm(out.as_mut_ptr(), zeta, zbar) };

        let inputs = window.iter().take(128).zip(window.iter().skip(128));
        let outputs = out.iter().take(128).zip(out.iter().skip(128));
        for ((&vj, &vjl), (&lo, &hi)) in inputs.zip(outputs) {
            let (vj, vjl) = (i64::from(vj), i64::from(vjl));

            // Sum path: exact and unreduced — no i16 wrap by the entry bound.
            prop_assert_eq!(i64::from(lo), vj + vjl);

            // Diff path: barrett_const_mul — congruent to diff * zeta mod q
            // and bounded in (-q, q).
            let hi = i64::from(hi);
            prop_assert_eq!((hi - (vjl - vj) * i64::from(zeta)).rem_euclid(q), 0);
            prop_assert!(-q < hi && hi < q);
        }
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
