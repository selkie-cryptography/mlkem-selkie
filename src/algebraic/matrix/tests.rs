//! Unit tests for matrices of NTT-domain polynomials.

use super::*;
use crate::{
    algebraic::{
        FieldElement,
        poly::{PolynomialRingElement, TqElement},
    },
    parameters::MLKEM512,
};

/// Transposing a matrix twice is the identity.
#[test]
fn matrix_transpose_involution() {
    let mut counter = 0u16;
    let mut next = || {
        let coeffs = core::array::from_fn(|_| {
            counter = counter.wrapping_add(1);
            FieldElement::new(counter)
        });
        TqElement::new(coeffs)
    };

    let matrix = TqMatrix::<MLKEM512>::from_fn(|_| TqVector::<MLKEM512>::from_fn(|_| next()));

    assert_eq!(matrix.transpose().transpose(), matrix);
}
