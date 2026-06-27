//! Unit tests for ring-element sampling.

use sha3::{
    Shake128,
    digest::{ExtendableOutput, Update},
};

use super::*;

/// All-zero CBD input yields the zero polynomial.
#[test]
fn sample_poly_cbd_zero_input() {
    for eta in [Eta::Two, Eta::Three] {
        let poly = RqElement::sample_cbd(eta, &vec![0u8; 64 * usize::from(eta)]);

        assert_eq!(poly, RqElement::ZERO);
    }
}

/// CBD coefficients land in the centered binomial range `{-eta, ..., eta}`,
/// represented modulo q.
#[test]
fn sample_poly_cbd_range() {
    let eta = Eta::Three;
    let n = usize::from(eta);

    // Alternating bits exercise both the positive and negative tails.
    let bytes: Vec<u8> = (0..64 * n).map(|i| (i as u8).wrapping_mul(37)).collect();

    let poly = RqElement::sample_cbd(eta, &bytes);

    for coeff in poly.coefficients() {
        let v = coeff.value();
        let centered = if v <= n as u16 {
            i32::from(v)
        } else {
            i32::from(v) - i32::from(Q)
        };

        assert!(
            (-(n as i32)..=(n as i32)).contains(&centered),
            "coefficient {v} out of CBD range",
        );
    }
}

/// `sample_ntt` is deterministic in its input stream and yields canonical
/// coefficients.
#[test]
fn sample_ntt_deterministic_and_canonical() {
    let seed = |label: u8| {
        let mut h = Shake128::default();
        h.update(&[label; 34]);
        h.finalize_xof()
    };

    let a = TqElement::sample_ntt(&mut seed(7));
    let b = TqElement::sample_ntt(&mut seed(7));

    assert_eq!(a, b);

    for coeff in a.coefficients() {
        assert!(coeff.value() < Q);
    }
}
