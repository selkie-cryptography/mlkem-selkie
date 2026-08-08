//! Unit tests for the polynomial rings and the NTT.

use super::*;

/// Reverses the low 7 bits of an integer in `{0, ..., 127}`. Test-only oracle
/// used to spot-check the precomputed zeta table's `BitRev7` indexing (see
/// [FIPS 203 section 4.3]); the zetas themselves are compile-time literals, so
/// this helper never runs outside these tests.
///
/// [FIPS 203 section 4.3]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#subsection.4.3
fn bit_rev_7(i: u8) -> u8 {
    let mut reversed: u8 = 0;

    for bit in 0..7 {
        reversed <<= 1;
        reversed |= (i >> bit) & 1;
    }

    reversed
}

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
    let coeffs = array::from_fn(|i| FieldElement::new((7 * i as u16 + 1) % parameters::Q));
    let f = RqElement::new(coeffs);

    let expected = RqElement::new(coeffs.map(FieldElement::to_montgomery));

    assert_eq!(f.ntt().ntt_inverse(), expected);
}

/// Multiplication in Tq agrees with schoolbook multiplication in Rq modulo
/// `X^256 + 1`.
#[test]
fn ntt_multiplication_matches_schoolbook() {
    let f_coeffs = array::from_fn(|i| FieldElement::new((i as u16) % parameters::Q));
    let g_coeffs = array::from_fn(|i| FieldElement::new((2 * i as u16 + 3) % parameters::Q));
    let f = RqElement::new(f_coeffs);
    let g = RqElement::new(g_coeffs);

    let via_ntt = (f.ntt() * g.ntt()).ntt_inverse();

    assert_eq!(via_ntt, schoolbook_multiply(f, g));
}

/// [`TqElement::mul_cache`] holds `gamma_i * g_(2i+1)` per pair.
#[test]
fn mul_cache_matches_definition() {
    let g = TqElement::new(array::from_fn(|i| {
        FieldElement::new((5 * i as u16 + 2) % parameters::Q)
    }));

    let cache = g.mul_cache();

    for (i, (entry, &gamma)) in cache.0.iter().zip(arch::GAMMA_MONT.iter()).enumerate() {
        assert_eq!(entry.value(), (g[2 * i + 1] * gamma).value());
    }
}

/// [`TqElement::accumulated_dot`] agrees with the sum of per-component
/// products at every dot-product length the parameter sets use.
// reason: j < k <= 4 indexes the length-4 component arrays; the loop
// structure mirrors the dot product it tests.
#[allow(clippy::indexing_slicing)]
#[test]
fn accumulated_dot_matches_componentwise_sum() {
    let f: [TqElement; 4] = array::from_fn(|j| {
        TqElement::new(array::from_fn(|i| {
            FieldElement::new((3 * i as u16 + 7 * j as u16 + 1) % parameters::Q)
        }))
    });
    let g: [TqElement; 4] = array::from_fn(|j| {
        TqElement::new(array::from_fn(|i| {
            FieldElement::new((11 * i as u16 + 13 * j as u16 + 5) % parameters::Q)
        }))
    });
    let caches = array::from_fn::<_, 4, _>(|j| g[j].mul_cache());

    for k in 1..=4 {
        let accumulated = TqElement::accumulated_dot(&f[..k], &g[..k], &caches[..k]);

        let mut componentwise = TqElement::ZERO;
        for j in 0..k {
            componentwise += &f[j] * &g[j];
        }

        assert_eq!(accumulated, componentwise, "dot length {k}");
    }
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

/// `TqElement`'s public surfaces speak natural coefficient order over the
/// split evens-then-odds storage: construction round-trips through
/// `coefficients`, and
/// `Index` returns the same coefficient the natural-order input held.
#[test]
fn tq_split_storage_preserves_natural_order_semantics() {
    let natural: [FieldElement; parameters::N] =
        array::from_fn(|i| FieldElement::new((13 * i as u16 + 7) % parameters::Q));

    let tq = TqElement::new(natural);

    assert_eq!(tq.coefficients(), natural);
    for (i, &expected) in natural.iter().enumerate() {
        assert_eq!(tq[i], expected, "index {i}");
    }
}

/// `zeroize` clears every coefficient in both domains, and `Default` is the
/// zero polynomial.
#[test]
fn zeroize_and_default_are_zero() {
    let mut f = RqElement::new(array::from_fn(|i| FieldElement::new(i as u16 + 1)));
    f.zeroize();
    assert_eq!(f, RqElement::default());
    assert_eq!(
        RqElement::default().coefficients(),
        [FieldElement::ZERO; parameters::N]
    );

    let mut g = TqElement::new(array::from_fn(|i| FieldElement::new(i as u16 + 1)));
    let mut cache = g.mul_cache();
    g.zeroize();
    assert_eq!(g, TqElement::default());
    assert_eq!(
        TqElement::default().coefficients(),
        [FieldElement::ZERO; parameters::N]
    );

    cache.zeroize();
    assert_eq!(cache, TqMulCache::default());
}

/// The value `Add` forms delegate to the in-place assign ops.
#[test]
fn value_add_matches_assign() {
    let f = TqElement::new(array::from_fn(|i| FieldElement::new(i as u16)));
    let g = TqElement::new(array::from_fn(|i| FieldElement::new(2 * i as u16 + 1)));

    let mut assigned = f.clone();
    assigned += &g;

    assert_eq!(f + g, assigned);
}
