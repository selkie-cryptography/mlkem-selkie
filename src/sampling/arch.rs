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
