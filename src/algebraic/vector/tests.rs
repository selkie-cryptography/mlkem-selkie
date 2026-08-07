//! Unit tests for vectors of polynomial ring elements.

use core::array;

use super::*;
use crate::{
    algebraic::{FieldElement, poly::PolynomialRingElement},
    parameters::MLKEM512,
};

/// The plain (uncached) dot product matches accumulating each pointwise
/// product.
#[test]
fn dot_product_matches_componentwise() {
    let mut counter = 0u16;
    let mut next = || {
        let coeffs = array::from_fn(|_| {
            counter = counter.wrapping_add(3);
            FieldElement::new(counter)
        });
        TqElement::new(coeffs)
    };

    let f = TqVector::<MLKEM512>::from_fn(|_| next());
    let g = TqVector::<MLKEM512>::from_fn(|_| next());

    let mut componentwise = TqElement::ZERO;
    for i in 0..2 {
        componentwise += &f[i] * &g[i];
    }

    assert_eq!(&f * &g, componentwise);
}
