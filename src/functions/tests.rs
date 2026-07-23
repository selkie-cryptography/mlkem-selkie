//! Unit tests for the batched-Keccak primitives.

use sha3_selkie::{Shake128, Shake256};

use super::*;

/// `shake256_x4` produces the same four outputs as four scalar SHAKE256
/// squeezes — exercising whichever batched backend `sha3_selkie` selected.
#[test]
fn shake256_x4_matches_scalar() {
    let inputs = [[10u8; 33], [20u8; 33], [30u8; 33], [40u8; 33]];

    let mut batched = [[0u8; 192]; 4];
    shake256_x4(
        inputs.each_ref().map(<[u8; 33]>::as_slice),
        batched.each_mut().map(<[u8; 192]>::as_mut_slice),
    );

    let scalar = inputs.map(|input| Shake256::digest::<192>(&input));

    assert_eq!(batched, scalar);
}

/// `Shake128X4`'s three-block then next-block squeezes match four scalar
/// SHAKE128 streams — exercising whichever batched backend was selected.
#[test]
fn shake128_x4_matches_scalar() {
    let seeds: [[u8; 34]; 4] =
        core::array::from_fn(|i| core::array::from_fn(|k| (i * 7 + k) as u8));

    let mut state = Shake128X4::absorb(&seeds);
    let first = state.squeeze_first_three_blocks();
    let next = state.squeeze_next_block();

    for ((seed, lane_first), lane_next) in seeds.iter().zip(&first).zip(&next) {
        let mut hasher = Shake128::new();
        hasher.update(seed);
        let mut reader = hasher.finalize_xof();

        let mut scalar_first = [0u8; SHAKE128_THREE_BLOCKS];
        let mut scalar_next = [0u8; SHAKE128_BLOCK];
        reader.read(&mut scalar_first);
        reader.read(&mut scalar_next);

        assert_eq!(lane_first, &scalar_first);
        assert_eq!(lane_next, &scalar_next);
    }
}
