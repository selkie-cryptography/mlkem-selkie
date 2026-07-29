//! Unit tests for the batched-Keccak primitives.

use super::*;
use crate::parameters::Eta;

/// Every [`PRF_x4`] lane matches the scalar [`PRF`] on the same domain
/// separator, at both `eta` values.
#[test]
fn prf_x4_matches_scalar() {
    let seed = [7u8; 32];

    for eta in [Eta::Two, Eta::Three] {
        for (lane, bytes) in PRF_x4(eta, &seed, 5).lanes().into_iter().enumerate() {
            assert_eq!(
                bytes,
                &*PRF(eta, &seed, 5 + lane as u8),
                "eta {eta:?}, lane {lane}"
            );
        }
    }
}

/// Both [`PRF_x2`] lanes match the scalar [`PRF`].
#[test]
fn prf_x2_matches_scalar() {
    let seed = [11u8; 32];

    for eta in [Eta::Two, Eta::Three] {
        for (lane, bytes) in PRF_x2(eta, &seed, 200).lanes().into_iter().enumerate() {
            assert_eq!(
                bytes,
                &*PRF(eta, &seed, 200 + lane as u8),
                "eta {eta:?}, lane {lane}"
            );
        }
    }
}

/// A batch's lanes are exactly `64 * eta` bytes.
#[test]
fn prf_batch_lanes_are_exactly_sized() {
    let seed = [3u8; 32];

    for (eta, len) in [(Eta::Two, 128), (Eta::Three, 192)] {
        assert!(PRF_x4(eta, &seed, 0).lanes().iter().all(|l| l.len() == len));
        assert!(PRF_x2(eta, &seed, 0).lanes().iter().all(|l| l.len() == len));
    }
}

/// Every [`XOF_x4`] lane matches the scalar [`XOF`] on the same `(i, j)`,
/// across a three-block squeeze and one further block — exercising whichever
/// batched backend was selected.
#[test]
fn xof_x4_matches_scalar() {
    let rho = [0x5Au8; 32];
    let indices = [(0u8, 0u8), (1, 0), (0, 1), (3, 2)];

    let mut state = XOF_x4(&rho, indices);

    let mut first = [[0u8; 3 * Shake128X4::RATE]; 4];
    let [f0, f1, f2, f3] = &mut first;
    state.squeeze([f0, f1, f2, f3]);

    let mut next = [[0u8; Shake128X4::RATE]; 4];
    let [n0, n1, n2, n3] = &mut next;
    state.squeeze([n0, n1, n2, n3]);

    for (lane, ((&(i, j), lane_first), lane_next)) in
        indices.iter().zip(&first).zip(&next).enumerate()
    {
        let mut reader = XOF(&rho, i, j);

        let mut scalar_first = [0u8; 3 * Shake128X4::RATE];
        let mut scalar_next = [0u8; Shake128X4::RATE];
        reader.read(&mut scalar_first);
        reader.read(&mut scalar_next);

        assert_eq!(lane_first, &scalar_first, "lane {lane}");
        assert_eq!(lane_next, &scalar_next, "lane {lane}");
    }
}
