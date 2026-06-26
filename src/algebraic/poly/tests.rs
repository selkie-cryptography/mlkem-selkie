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

/// In signed-Montgomery form `ntt_inverse(ntt(f))` recovers `f` scaled by `R`:
/// the forward transform is exact, but the inverse leaves the standard-domain
/// `R` factor that base multiplication consumes (see
/// [`TqElement::to_montgomery`]).
#[test]
fn ntt_round_trip_scales_by_montgomery_r() {
    let coeffs = core::array::from_fn(|i| FieldElement::new((7 * i as u16 + 1) % parameters::Q));
    let f = RqElement::new(coeffs);

    let expected = RqElement::new(coeffs.map(FieldElement::to_montgomery));

    assert_eq!(f.ntt().ntt_inverse(), expected);
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

/// Reference multiplication in `Zq[X] / (X^256 + 1)`, computed in the canonical
/// value domain so it matches the standard-domain product the NTT path returns.
///
/// Uses [`FieldElement::mul_reference`] rather than the type's Montgomery `*`,
/// and Barrett-reduces each accumulation to keep the running sums inside `i16`
/// (an unreduced column sums up to `N` products and would overflow).
// reason: i, j < N so i + j < 2N and (i + j) - N < N; every index into the
// length-N `result` is provably in bounds.
#[allow(clippy::indexing_slicing)]
fn schoolbook_multiply(f: RqElement, g: RqElement) -> RqElement {
    let mut result = [FieldElement::ZERO; parameters::N];

    for i in 0..parameters::N {
        for j in 0..parameters::N {
            let product = f[i].mul_reference(g[j]);
            let k = i + j;

            if k < parameters::N {
                result[k] = (result[k] + product).reduce();
            } else {
                // X^256 = -1, so wrap with a sign flip.
                result[k - parameters::N] = (result[k - parameters::N] - product).reduce();
            }
        }
    }

    RqElement::new(result)
}
