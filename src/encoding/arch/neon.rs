//! NEON-vectorized bit-unpacking for aarch64: eight `ByteDecode_d` values per
//! vector via constant-index table lookups and per-lane shifts.
//!
//! Eight consecutive `d`-bit values occupy exactly `d` bytes, so each
//! iteration gathers the two (or, at `d = 11`, four) bytes containing every
//! output value with `tbl`, shifts each lane down to its bit offset, and
//! masks to `d` bits. Only fixed-index shuffles and public-length branches
//! touch the data, so the impls are constant-time; `ByteDecode_12` runs
//! over secret decryption-key bytes when parsing a decapsulation key.
#![allow(unsafe_code)]

use core::arch::aarch64::{
    uint8x16_t, vandq_u8, vandq_u16, vandq_u32, vcombine_u16, vdupq_n_u8, vdupq_n_u16, vdupq_n_u32,
    vget_low_u8, vld1q_s16, vld1q_s32, vld1q_u8, vmovl_high_u8, vmovl_u8, vmovn_u32, vqtbl1q_u8,
    vreinterpretq_u16_u8, vreinterpretq_u32_u8, vshlq_u16, vshlq_u32, vshrq_n_u8, vst1q_u16,
    vzip1q_u8, vzip2q_u8,
};

use super::generic;
use crate::parameters::N;

/// `tbl` indices gathering the two bytes of each 5-bit value.
const IDX_5: [u8; 16] = [0, 1, 0, 1, 1, 2, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5];

/// Per-lane right shifts (as negative left shifts) for the 5-bit offsets.
const SHIFTS_5: [i16; 8] = [0, -5, -2, -7, -4, -1, -6, -3];

/// `tbl` indices gathering the two bytes of each 10-bit value.
const IDX_10: [u8; 16] = [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9];

/// Per-lane right shifts for the 10-bit offsets.
const SHIFTS_10: [i16; 8] = [0, -2, -4, -6, 0, -2, -4, -6];

/// `tbl` indices gathering the two bytes of each 12-bit value.
const IDX_12: [u8; 16] = [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11];

/// Per-lane right shifts for the 12-bit offsets.
const SHIFTS_12: [i16; 8] = [0, -4, 0, -4, 0, -4, 0, -4];

/// `tbl` indices gathering the four bytes of 11-bit values 0..4; an 11-bit
/// value can span three bytes, so the lanes widen to `u32`.
const IDX_11_LOW: [u8; 16] = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 4, 5, 6, 7];

/// Per-lane right shifts for 11-bit values 0..4.
const SHIFTS_11_LOW: [i32; 4] = [0, -3, -6, -1];

/// `tbl` indices gathering the four bytes of 11-bit values 4..8.
const IDX_11_HIGH: [u8; 16] = [5, 6, 7, 8, 6, 7, 8, 9, 8, 9, 10, 11, 9, 10, 11, 12];

/// Per-lane right shifts for 11-bit values 4..8.
const SHIFTS_11_HIGH: [i32; 4] = [-4, -7, -2, -5];

/// Unpacks `N` `d`-bit values from bytes, least-significant bit first: the
/// NEON dispatch of [`generic::unpack`].
///
/// Widths without a kernel (`d = 1`) and the zero-padded short inputs take
/// the scalar path.
pub(in crate::encoding) fn unpack(bytes: &[u8], d: usize) -> [u16; N] {
    if bytes.len() != N * d / 8 {
        return generic::unpack(bytes, d);
    }

    match d {
        4 => unpack_4(bytes),
        5 => unpack_gathered::<5>(bytes, IDX_5, SHIFTS_5),
        10 => unpack_gathered::<10>(bytes, IDX_10, SHIFTS_10),
        11 => unpack_11(bytes),
        12 => unpack_gathered::<12>(bytes, IDX_12, SHIFTS_12),
        _ => generic::unpack(bytes, d),
    }
}

/// Loads the 16 bytes at `start`, zero-padding past the end of `bytes`; the
/// gathers never index past byte 13, so the padding is inert.
#[inline]
fn window(bytes: &[u8], start: usize) -> uint8x16_t {
    if let Some(full) = bytes.get(start..start + 16) {
        // SAFETY: `full` is exactly 16 bytes.
        unsafe { vld1q_u8(full.as_ptr()) }
    } else {
        let rest = bytes.get(start..).unwrap_or_default();
        let mut padded = [0u8; 16];
        padded.split_at_mut(rest.len()).0.copy_from_slice(rest);

        // SAFETY: `padded` is 16 bytes.
        unsafe { vld1q_u8(padded.as_ptr()) }
    }
}

/// Unpacks the widths whose values span at most two bytes (`d` in
/// `{5, 10, 12}`): one `tbl` gather into `u16` lanes, one shift, one mask.
fn unpack_gathered<const D: usize>(bytes: &[u8], idx: [u8; 16], shifts: [i16; 8]) -> [u16; N] {
    let mut out = [0u16; N];

    // SAFETY: pure value ops on 16-byte constants; the store writes the
    // 8-element chunk the loop hands out.
    unsafe {
        let idx = vld1q_u8(idx.as_ptr());
        let shifts = vld1q_s16(shifts.as_ptr());
        let mask = vdupq_n_u16(((1u32 << D) - 1) as u16);

        for (i, chunk) in out.chunks_exact_mut(8).enumerate() {
            let gathered = vreinterpretq_u16_u8(vqtbl1q_u8(window(bytes, D * i), idx));
            let values = vandq_u16(vshlq_u16(gathered, shifts), mask);
            vst1q_u16(chunk.as_mut_ptr(), values);
        }
    }

    out
}

/// Unpacks `d = 11`, whose values can span three bytes: two `tbl` gathers
/// into `u32` lanes, shifted, masked, and narrowed back to `u16`.
fn unpack_11(bytes: &[u8]) -> [u16; N] {
    let mut out = [0u16; N];

    // SAFETY: pure value ops on 16-byte constants; the store writes the
    // 8-element chunk the loop hands out.
    unsafe {
        let idx_low = vld1q_u8(IDX_11_LOW.as_ptr());
        let idx_high = vld1q_u8(IDX_11_HIGH.as_ptr());
        let shifts_low = vld1q_s32(SHIFTS_11_LOW.as_ptr());
        let shifts_high = vld1q_s32(SHIFTS_11_HIGH.as_ptr());
        let mask = vdupq_n_u32((1 << 11) - 1);

        for (i, chunk) in out.chunks_exact_mut(8).enumerate() {
            let data = window(bytes, 11 * i);

            let low = vreinterpretq_u32_u8(vqtbl1q_u8(data, idx_low));
            let low = vandq_u32(vshlq_u32(low, shifts_low), mask);
            let high = vreinterpretq_u32_u8(vqtbl1q_u8(data, idx_high));
            let high = vandq_u32(vshlq_u32(high, shifts_high), mask);

            vst1q_u16(
                chunk.as_mut_ptr(),
                vcombine_u16(vmovn_u32(low), vmovn_u32(high)),
            );
        }
    }

    out
}

/// Unpacks `d = 4`: each byte splits into its low and high nibble, zipped
/// back into value order and widened, thirty-two values per 16-byte load.
fn unpack_4(bytes: &[u8]) -> [u16; N] {
    let mut out = [0u16; N];

    // SAFETY: pure value ops; the loads and stores cover exactly the 16-byte
    // and 32-element chunks the zipped iterators hand out.
    unsafe {
        let mask = vdupq_n_u8(0xF);

        for (chunk, values) in bytes.chunks_exact(16).zip(out.chunks_exact_mut(32)) {
            let data = vld1q_u8(chunk.as_ptr());
            let low = vandq_u8(data, mask);
            let high = vshrq_n_u8(data, 4);

            let first = vzip1q_u8(low, high);
            let second = vzip2q_u8(low, high);

            let ptr = values.as_mut_ptr();
            vst1q_u16(ptr, vmovl_u8(vget_low_u8(first)));
            vst1q_u16(ptr.add(8), vmovl_high_u8(first));
            vst1q_u16(ptr.add(16), vmovl_u8(vget_low_u8(second)));
            vst1q_u16(ptr.add(24), vmovl_high_u8(second));
        }
    }

    out
}
