//! NEON rejection-sampling kernel: eight 12-bit candidates per vector, with
//! table-driven compaction of the accepted lanes.
//!
//! Twelve input bytes hold eight candidates. A `tbl` gathers each candidate's
//! byte pair into a `u16` lane, a per-lane shift aligns the odd candidates,
//! and a compare against q yields the acceptance mask; a 256-entry shuffle
//! table then packs the accepted lanes to the front in one more `tbl`, so
//! rejection never branches per candidate. Matches
//! [`super::generic::reject`] on the first `N` outputs (this kernel writes
//! whole eight-lane groups, spilling into the buffer's slack tail).
//!
//! The rejection loop's timing depends only on public XOF output (`rho` is
//! public), as the scalar kernel's does.
#![allow(unsafe_code)]

use core::arch::aarch64::{
    vaddv_u8, vand_u8, vandq_u16, vcltq_u16, vdupq_n_u16, vld1_u8, vld1q_s16, vld1q_u8, vmovn_u16,
    vqtbl1q_u8, vreinterpretq_u8_u16, vreinterpretq_u16_u8, vshlq_u16, vst1q_u8,
};

use super::RejectBuffer;
use crate::parameters;

#[cfg(test)]
mod tests;

/// `tbl` indices gathering each candidate's little-endian byte pair from a
/// 12-byte group: candidate `2i` spans bytes `(3i, 3i + 1)`, candidate
/// `2i + 1` spans `(3i + 1, 3i + 2)`.
const GATHER: [u8; 16] = [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11];

/// Per-lane shifts aligning the candidates: even lanes keep their low twelve
/// bits, odd lanes discard the shared nibble (negative = right shift).
const SHIFTS: [i16; 8] = [0, -4, 0, -4, 0, -4, 0, -4];

/// Lane-mask bit weights for narrowing an acceptance mask to one byte.
const BIT_WEIGHTS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Byte-shuffle table packing accepted `u16` lanes to the front: entry `m`
/// lists the byte pairs of `m`'s set bits in ascending order, padded with
/// `0xFF` (which `tbl` maps to zero).
const COMPACT: [[u8; 16]; 256] = {
    let mut table = [[0xFF; 16]; 256];

    let mut mask = 0;
    while mask < 256 {
        let mut packed = 0;
        let mut lane = 0;
        while lane < 8 {
            if mask & (1 << lane) != 0 {
                // reason: mask < 256 and packed < lane < 8 by the loop
                // guards, so every index is in bounds (and any violation is a
                // compile error in this const initializer).
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
};

/// Appends the accepted 12-bit candidates of `bytes` to `out[*count..]`,
/// eight candidates per iteration.
///
/// Matches [`super::generic::reject`] on `out[..N]` and the final (clamped)
/// `*count`; accepted groups straddling `N` spill into the buffer's
/// [`super::REJECT_SLACK`] tail. `bytes` must be a whole number of 12-byte
/// groups (SHAKE128 blocks are: `168 = 14 * 12`).
// reason: the final-group slice is provably 12 bytes (`len % 12 == 0`) and
// `mask` is a `u8` indexing the 256-entry table; the padded-copy rewrite the
// lint suggests is this code.
#[allow(clippy::indexing_slicing)]
pub(crate) fn reject(bytes: &[u8], out: &mut RejectBuffer, count: &mut usize) {
    debug_assert_eq!(bytes.len() % 12, 0);
    let groups = bytes.len() / 12;

    // SAFETY: group `i < groups` starts at byte `12 * i`. Full 16-byte loads
    // run only while `12 * i + 16 <= bytes.len()` (guaranteed for
    // `i < groups - 1` since blocks are at least 24 bytes); the final group
    // is copied into a padded stack buffer first. Each store writes eight
    // `i16` lanes at `out[*count]` with `*count < N` (loop guard), and
    // `N - 1 + 8` is the last slack index of `RejectBuffer`. `FieldElement`
    // is `repr(transparent)` over `i16`, so `out` reinterprets as `[u8]`.
    // NEON is baseline on aarch64.
    unsafe {
        let gather = vld1q_u8(GATHER.as_ptr());
        let shifts = vld1q_s16(SHIFTS.as_ptr());
        let weights = vld1_u8(BIT_WEIGHTS.as_ptr());
        let low12 = vdupq_n_u16(0x0FFF);
        let q = vdupq_n_u16(parameters::Q);

        let out_ptr = out.as_mut_ptr().cast::<u8>();

        let mut group = 0;
        while group < groups && *count < parameters::N {
            let data = if group + 1 < groups {
                vld1q_u8(bytes.as_ptr().add(12 * group))
            } else {
                let mut padded = [0u8; 16];
                padded[..12].copy_from_slice(&bytes[12 * group..]);
                vld1q_u8(padded.as_ptr())
            };

            // Eight 12-bit candidates in u16 lanes, stream order.
            let pairs = vreinterpretq_u16_u8(vqtbl1q_u8(data, gather));
            let candidates = vandq_u16(vshlq_u16(pairs, shifts), low12);

            // Acceptance mask -> one byte, one bit per lane.
            let accepted = vcltq_u16(candidates, q);
            let mask = vaddv_u8(vand_u8(vmovn_u16(accepted), weights));

            // Pack the accepted lanes to the front and append them.
            let packed = vqtbl1q_u8(
                vreinterpretq_u8_u16(candidates),
                vld1q_u8(COMPACT[usize::from(mask)].as_ptr()),
            );
            vst1q_u8(out_ptr.add(2 * *count), packed);
            *count += mask.count_ones() as usize;

            group += 1;
        }
    }

    *count = (*count).min(parameters::N);
}
