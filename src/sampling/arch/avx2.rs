//! AVX2 rejection-sampling kernel: eight 12-bit candidates per vector, with
//! table-driven compaction of the accepted lanes.
//!
//! The 128-bit sibling of the NEON kernel: a `pshufb` gathers each
//! candidate's byte pair into a `u16` lane, a blend of masked and shifted
//! copies aligns the odd candidates, and a signed compare against q yields
//! the acceptance mask; a 256-entry shuffle table then packs the accepted
//! lanes to the front in one more `pshufb`, so rejection never branches per
//! candidate. Matches [`super::generic::reject`] on the first `N` outputs
//! (this kernel writes whole eight-lane groups, spilling into the buffer's
//! slack tail).
//!
//! The rejection loop's timing depends only on public XOF output (`rho` is
//! public), as the scalar kernel's does.
#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, _mm_and_si128, _mm_blend_epi16, _mm_cmplt_epi16, _mm_loadu_si128, _mm_movemask_epi8,
    _mm_packs_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_shuffle_epi8, _mm_srli_epi16,
    _mm_storeu_si128,
};

use super::RejectBuffer;
use crate::parameters;

#[cfg(test)]
mod tests;

/// `pshufb` indices gathering each candidate's little-endian byte pair from a
/// 12-byte group: candidate `2i` spans bytes `(3i, 3i + 1)`, candidate
/// `2i + 1` spans `(3i + 1, 3i + 2)`.
const GATHER: [u8; 16] = [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11];

/// Byte-shuffle table packing accepted `u16` lanes to the front, padded with
/// `0x80` (which `pshufb` maps to zero).
const COMPACT: [[u8; 16]; 256] = super::compact_table(0x80);

/// Appends the accepted 12-bit candidates of `bytes` to `out[*count..]`,
/// eight candidates per iteration.
///
/// Matches [`super::generic::reject`] on `out[..N]` and the final (clamped)
/// `*count`; accepted groups straddling `N` spill into the buffer's
/// [`super::REJECT_SLACK`] tail. `bytes` must be a whole number of 12-byte
/// groups (SHAKE128 blocks are: `168 = 14 * 12`).
// reason: `padded[..12]` is a constant range of a `[u8; 16]` and the mask
// (8 bits) indexes the 256-entry table; both are provably in bounds.
#[allow(clippy::indexing_slicing)]
pub(crate) fn reject(bytes: &[u8], out: &mut RejectBuffer, count: &mut usize) {
    debug_assert_eq!(bytes.len() % 12, 0);
    let groups = bytes.len() / 12;

    // SAFETY: every 16-byte load reads a `bytes.get`-checked window or a
    // local padded buffer, so loads are in bounds by construction. Each
    // store writes eight `i16` lanes at `out[*count]` with `*count < N`
    // (loop guard), and `N - 1 + 8` is the last slack index of
    // `RejectBuffer`. `FieldElement` is `repr(transparent)` over `i16`, so
    // `out` reinterprets as `[u8]`. The avx2 module compiles only with AVX2
    // enabled, so every intrinsic is available.
    unsafe {
        let gather = _mm_loadu_si128(GATHER.as_ptr().cast::<__m128i>());
        let low12 = _mm_set1_epi16(0x0FFF);
        let q = _mm_set1_epi16(parameters::Q as i16);

        let out_ptr = out.as_mut_ptr().cast::<u8>();

        let mut group = 0;
        while group < groups && *count < parameters::N {
            let start = 12 * group;
            let data = if let Some(window) = bytes.get(start..start + 16) {
                _mm_loadu_si128(window.as_ptr().cast::<__m128i>())
            } else {
                let mut padded = [0u8; 16];
                if let Some(tail) = bytes.get(start..start + 12) {
                    padded[..12].copy_from_slice(tail);
                }
                _mm_loadu_si128(padded.as_ptr().cast::<__m128i>())
            };

            // Eight 12-bit candidates in u16 lanes, stream order: even lanes
            // keep their low twelve bits, odd lanes discard the shared
            // nibble.
            let pairs = _mm_shuffle_epi8(data, gather);
            let candidates = _mm_blend_epi16::<0b1010_1010>(
                _mm_and_si128(pairs, low12),
                _mm_srli_epi16::<4>(pairs),
            );

            // Acceptance mask -> one byte, one bit per lane (candidates fit
            // in twelve bits, so the signed compare is exact).
            let accepted = _mm_cmplt_epi16(candidates, q);
            let mask = _mm_movemask_epi8(_mm_packs_epi16(accepted, _mm_setzero_si128())) & 0xFF;

            // Pack the accepted lanes to the front and append them.
            let packed = _mm_shuffle_epi8(
                candidates,
                _mm_loadu_si128(COMPACT[mask as usize].as_ptr().cast::<__m128i>()),
            );
            _mm_storeu_si128(out_ptr.add(2 * *count).cast::<__m128i>(), packed);
            *count += mask.count_ones() as usize;

            group += 1;
        }
    }

    *count = (*count).min(parameters::N);
}
