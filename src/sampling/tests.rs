//! Unit tests for ring-element sampling.

use super::*;
use crate::functions::XOF;

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
    let a = TqElement::sample_ntt(&[7u8; 32], 7, 7);
    let b = TqElement::sample_ntt(&[7u8; 32], 7, 7);

    assert_eq!(a, b);

    for coeff in a.coefficients() {
        assert!(coeff.value() < Q);
    }
}

/// `sample_ntt` matches a byte-for-byte manual replay of FIPS 203 Algorithm 7:
/// read the `XOF(rho, i, j)` stream, unpack each 3-byte chunk into two 12-bit
/// candidates via the documented masks and shifts, and keep those `< q`. Any
/// mutation to the bit-op or comparison logic in `sample_ntt` diverges from
/// this in-test replay (which uses independent literals for `0x0F`, `<< 8`,
/// `>> 4`, `<< 4`, and the two `< Q` / `< N` guards).
#[test]
fn sample_ntt_matches_manual_unpack() {
    let rho = [0x42u8; 32];
    let (i, j) = (3u8, 5u8);

    let sampled = TqElement::sample_ntt(&rho, i, j);

    let mut reader = XOF(&rho, i, j);
    let mut expected: Vec<u16> = Vec::with_capacity(parameters::N);
    let mut triple = [0u8; 3];
    while expected.len() < parameters::N {
        reader.read(&mut triple);

        let d1 = u16::from(triple[0]) | (u16::from(triple[1] & 0x0F) << 8);
        let d2 = (u16::from(triple[1]) >> 4) | (u16::from(triple[2]) << 4);

        if d1 < Q {
            expected.push(d1);
        }
        if d2 < Q && expected.len() < parameters::N {
            expected.push(d2);
        }
    }

    for (idx, (got, want)) in sampled
        .coefficients()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert_eq!(got.value(), *want, "coefficient {idx}");
    }
}

/// The batched matrix-expansion path `sample_ntt_x4` agrees with the serial
/// `sample_ntt` on the same `(rho, i, j)` seed. Guards against `reject_into`'s
/// bit-op / bounds mutations by pinning each lane's output to the scalar impl.
#[test]
fn sample_ntt_x4_matches_serial() {
    let rho = [0x91u8; 32];
    let seeds: [[u8; 34]; 4] = core::array::from_fn(|k| {
        let mut seed = [0u8; 34];
        seed[..32].copy_from_slice(&rho);
        seed[32] = k as u8; // matches `XOF`'s `i` byte
        seed[33] = 2 * k as u8; // matches `XOF`'s `j` byte
        seed
    });

    let batched = TqElement::sample_ntt_x4(&seeds);

    for (k, (lane, seed)) in batched.iter().zip(seeds.iter()).enumerate() {
        let serial = TqElement::sample_ntt(&rho, seed[32], seed[33]);
        assert_eq!(*lane, serial, "lane {k}");
    }
}
