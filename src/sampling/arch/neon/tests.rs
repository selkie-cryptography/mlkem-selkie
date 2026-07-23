//! Differential tests: the NEON rejection kernel must agree with the
//! portable scalar kernel on the outputs callers consume.

use proptest::prelude::*;

use crate::{
    parameters,
    sampling::arch::{EMPTY_REJECT_BUFFER, generic},
};

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 4096,
        .. ProptestConfig::default()
    })]

    /// Feeding the same SHAKE128-block-sized byte stream to both kernels
    /// yields identical accepted coefficients and counts, including streams
    /// long enough to fill the polynomial and spill into the slack.
    #[test]
    fn reject_matches_generic(
        blocks in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 168),
            1..=4,
        )
    ) {
        let mut vectorized = EMPTY_REJECT_BUFFER;
        let mut scalar = EMPTY_REJECT_BUFFER;
        let (mut vectorized_count, mut scalar_count) = (0, 0);

        for block in &blocks {
            super::reject(block, &mut vectorized, &mut vectorized_count);
            generic::reject(block, &mut scalar, &mut scalar_count);
        }

        prop_assert_eq!(vectorized_count, scalar_count);
        prop_assert_eq!(
            vectorized.get(..vectorized_count),
            scalar.get(..scalar_count),
        );
    }
}

/// A block of `0xFF` bytes yields only candidates of `0xFFF >= q`: everything
/// is rejected and the count stays put.
#[test]
fn reject_all_rejected() {
    let mut buffer = EMPTY_REJECT_BUFFER;
    let mut count = 0;

    super::reject(&[0xFF; 168], &mut buffer, &mut count);

    assert_eq!(count, 0);
}

/// A block of zero bytes yields only zero candidates: all `112` are accepted.
#[test]
fn reject_all_accepted() {
    let mut buffer = EMPTY_REJECT_BUFFER;
    let mut count = 0;

    super::reject(&[0u8; 168], &mut buffer, &mut count);

    assert_eq!(count, 112);
}

/// Once `N` coefficients are collected, later groups must not be processed:
/// the slack tail stays untouched. Fills to exactly `N` mid-block (two
/// all-accept blocks, then four accepting groups), with the block's
/// remaining groups holding accepted nonzero candidates (`0x111 < q`) that a
/// keep-sampling mutant would write into the slack.
#[test]
fn reject_stops_at_full() {
    let mut buffer = EMPTY_REJECT_BUFFER;
    let mut count = 0;

    super::reject(&[0u8; 168], &mut buffer, &mut count);
    super::reject(&[0u8; 168], &mut buffer, &mut count);

    let mut block = [0x11u8; 168];
    block[..48].fill(0);
    super::reject(&block, &mut buffer, &mut count);

    assert_eq!(count, parameters::N);
    assert!(
        buffer.iter().skip(parameters::N).all(|c| c.value() == 0),
        "slack written past N"
    );
}
