//! Polynomial ring arithmetic over finite fields for [FIPS 203].
//!
//! Arithmetic in the rings Rq and Tq, whose elements are polynomials defined
//! over Zq, which are isomorphic to each other. The number-theoretic transform
//! (NTT) is a computationally efficient isomorphism between these rings that
//! allows for efficient arithmetic over matrices and vectors of ring elements
//! in Rq, split across the `field`, `poly`, `vector`, and `matrix` submodules.
//!
//! Byte serialization and compression of these types live in
//! [`crate::encoding`]; sampling lives in [`crate::sampling`].
//!
//! [FIPS 203]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf

mod field;
mod matrix;
mod poly;
mod vector;

pub use field::FieldElement;
pub use matrix::TqMatrix;
pub use poly::{PolynomialRingElement, RqElement, TqElement};
pub use vector::{RqVector, TqVector};
