//! Unit tests for the polynomial rings and the NTT.

use super::*;

/// `bit_rev_7` reverses the low 7 bits of its input.
#[test]
fn bit_rev_7_known_values() {
    assert_eq!(bit_rev_7(0), 0);
    assert_eq!(bit_rev_7(1), 64);
    assert_eq!(bit_rev_7(64), 1);
    assert_eq!(bit_rev_7(127), 127);
    // 0b0000001 -> 0b1000000, 0b0000011 -> 0b1100000 = 96.
    assert_eq!(bit_rev_7(3), 96);
}

/// The NTT and its inverse compose to the identity on Rq.
#[test]
fn ntt_roundtrip_identity() {
    let coeffs = core::array::from_fn(|i| FieldElement::new((7 * i as u16 + 1) % parameters::Q));
    let f = RqElement::new(coeffs);

    assert_eq!(f.ntt().ntt_inverse(), f);
}

/// Multiplication in Tq agrees with schoolbook multiplication in Rq modulo
/// `X^256 + 1`.
#[test]
fn ntt_multiplication_matches_schoolbook() {
    let f_coeffs = core::array::from_fn(|i| FieldElement::new((i as u16) % parameters::Q));
    let g_coeffs = core::array::from_fn(|i| FieldElement::new((2 * i as u16 + 3) % parameters::Q));
    let f = RqElement::new(f_coeffs);
    let g = RqElement::new(g_coeffs);

    let via_ntt = (f.ntt() * g.ntt()).ntt_inverse();

    assert_eq!(via_ntt, schoolbook_multiply(f, g));
}

/// Reference multiplication in `Zq[X] / (X^256 + 1)`.
// reason: i, j < N so i + j < 2N and (i + j) - N < N; every index into the
// length-N `result` is provably in bounds.
#[allow(clippy::indexing_slicing)]
fn schoolbook_multiply(f: RqElement, g: RqElement) -> RqElement {
    let mut result = [FieldElement::ZERO; parameters::N];

    for i in 0..parameters::N {
        for j in 0..parameters::N {
            let product = f[i] * g[j];
            let k = i + j;

            if k < parameters::N {
                result[k] = result[k] + product;
            } else {
                // X^256 = -1, so wrap with a sign flip.
                result[k - parameters::N] = result[k - parameters::N] - product;
            }
        }
    }

    RqElement::new(result)
}
