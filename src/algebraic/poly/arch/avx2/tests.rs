//! Differential tests: the AVX2 kernels must agree with the portable scalar
//! backend on random canonical inputs. These run only on an AVX2-capable
//! x86_64 target (CI), so the host that authored them (aarch64) cannot.

use crate::{
    algebraic::{field::FieldElement, poly::arch::generic},
    parameters,
};

/// Fills a polynomial with canonical coefficients (`0..q`) from a xorshift32
/// stream; the transforms accumulate without per-stage reduction, so inputs
/// must be bounded.
fn random_canonical_poly(state: &mut u32) -> [FieldElement; parameters::N] {
    core::array::from_fn(|_| {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;

        FieldElement::new((*state & 0xFFFF) as u16)
    })
}

/// Fills a polynomial with arbitrary `i16` representatives; `multiply`'s
/// Montgomery reduction bounds every product, so its inputs need not be
/// canonical.
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
