//! Architecture-dispatched rejection-sampling kernel for `SampleNTT`.
//!
//! Every backend exposes the same [`reject`] kernel: it scans a squeezed
//! SHAKE128 byte block for 12-bit candidates, appends those below q to the
//! output in stream order, and stops once `N` coefficients are collected.
//! The vectorized backend writes whole candidate groups, overshooting `N` by
//! up to [`REJECT_SLACK`] elements into the output's slack tail; callers use
//! only the first `N` elements, which match the scalar backend's exactly.

use crate::{algebraic::FieldElement, parameters};

#[cfg(mlkem_selkie_arch = "avx2")]
mod avx2;
#[cfg(mlkem_selkie_arch = "neon")]
mod neon;

mod generic;

#[cfg(mlkem_selkie_arch = "avx2")]
pub(crate) use avx2::reject;
#[cfg(not(any(mlkem_selkie_arch = "neon", mlkem_selkie_arch = "avx2")))]
pub(crate) use generic::reject;
#[cfg(mlkem_selkie_arch = "neon")]
pub(crate) use neon::reject;

/// Slack elements past `N` in a rejection buffer: the vectorized kernel
/// stores eight candidates at a time, so an acceptance landing at `N - 1`
/// writes up to seven elements beyond it.
pub(crate) const REJECT_SLACK: usize = 8;

/// A rejection-sampling output buffer: `N` coefficients plus the slack the
/// vectorized kernel may write into.
pub(crate) type RejectBuffer = [FieldElement; parameters::N + REJECT_SLACK];

/// An all-zero [`RejectBuffer`].
pub(crate) const EMPTY_REJECT_BUFFER: RejectBuffer =
    [FieldElement::ZERO; parameters::N + REJECT_SLACK];

/// Builds a 256-entry byte-shuffle table packing accepted `u16` lanes to the
/// front: entry `m` lists the byte pairs of `m`'s set bits in ascending
/// order, padded with `padding` (each backend passes the index its shuffle
/// instruction maps to zero).
// Skipped by cargo-mutants: the reachable mutations of the two loop bounds
// are equivalent (the extra iteration's bit test `mask & (1 << lane)` is
// always false for `mask == 256` or `lane == 8`, so the table is unchanged);
// the table's contents are pinned by the backends' differential and
// edge-case kernel tests.
#[cfg_attr(test, mutants::skip)]
#[cfg(any(mlkem_selkie_arch = "neon", mlkem_selkie_arch = "avx2"))]
const fn compact_table(padding: u8) -> [[u8; 16]; 256] {
    let mut table = [[0u8; 16]; 256];

    let mut mask = 0;
    while mask < 256 {
        let mut byte = 0;
        while byte < 16 {
            // reason: mask < 256 and byte < 16 by the loop guards, so every
            // index is in bounds (and any violation is a compile error in
            // this const evaluation).
            #[allow(clippy::indexing_slicing)]
            {
                table[mask][byte] = padding;
            }
            byte += 1;
        }

        let mut packed = 0;
        let mut lane = 0;
        while lane < 8 {
            if mask & (1 << lane) != 0 {
                // reason: as above — mask < 256 and packed < lane < 8.
                #[allow(clippy::indexing_slicing)]
                {
                    table[mask][2 * packed] = (2 * lane) as u8;
                    table[mask][2 * packed + 1] = (2 * lane + 1) as u8;
                }
                packed += 1;
            }
            lane += 1;
        }
        mask += 1;
    }

    table
}
