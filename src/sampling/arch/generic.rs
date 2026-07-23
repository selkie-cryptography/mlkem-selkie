//! Portable scalar rejection-sampling kernel: the always-available fallback
//! and the differential-test oracle for the vectorized backends.
#![allow(dead_code)]

use super::RejectBuffer;
use crate::{algebraic::FieldElement, parameters};

/// Appends the accepted 12-bit candidates of `bytes` to `out[*count..]`.
///
/// Every 3 bytes hold two little-endian 12-bit candidates (`d1` from the low
/// twelve bits, `d2` from the high twelve); each candidate below q is kept,
/// in stream order, until `*count` reaches `N`. `bytes` must be a whole
/// number of 3-byte groups (SHAKE128 blocks are).
// reason: `count < N` guards every write, so each index is in bounds.
#[allow(clippy::indexing_slicing)]
pub(crate) fn reject(bytes: &[u8], out: &mut RejectBuffer, count: &mut usize) {
    for chunk in bytes.chunks_exact(3) {
        if *count >= parameters::N {
            break;
        }

        if let [b0, b1, b2] = chunk {
            let d1 = u16::from(*b0) | (u16::from(*b1 & 0x0F) << 8);
            let d2 = (u16::from(*b1) >> 4) | (u16::from(*b2) << 4);

            if d1 < parameters::Q {
                out[*count] = FieldElement::new(d1);
                *count += 1;
            }
            if d2 < parameters::Q && *count < parameters::N {
                out[*count] = FieldElement::new(d2);
                *count += 1;
            }
        }
    }
}
