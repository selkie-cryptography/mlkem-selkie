//! AVX2-vectorized bit-packing for x86_64: eight `u32` lanes per `__m256i`
//! via constant-index shuffles and per-lane shifts, in both directions.
//!
//! Eight consecutive `d`-bit values occupy exactly `d` bytes, so each
//! iteration broadcasts a 16-byte window to both 128-bit lanes and gathers
//! with `pshufb`. Unpacking gathers the four bytes containing every output
//! value (an 11-bit value can span three bytes, so all widths share `u32`
//! lanes), shifts each lane down to its bit offset with `srlv`, masks to `d`
//! bits, and packs back to `u16`. Packing (`d >= 8`, where an output byte
//! draws from at most two values) gathers the two contributing values per
//! output byte, shifts them to the byte's bit positions with `srlv`/`sllv`,
//! and keeps the low byte of their OR. Only fixed-index shuffles and
//! public-length branches touch the data, so the impls are constant-time;
//! `ByteDecode_12` runs over secret decryption key bytes when parsing a
//! decapsulation key.
#![allow(unsafe_code)]

use core::{
    arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm_packus_epi16, _mm_packus_epi32, _mm_storel_epi64,
        _mm_storeu_si128, _mm256_and_si256, _mm256_broadcastsi128_si256, _mm256_castsi256_si128,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi32,
        _mm256_shuffle_epi8, _mm256_sllv_epi32, _mm256_srlv_epi32,
    },
    array,
};

use super::generic;
use crate::parameters::N;

/// Packs `N` `d`-bit values into the leading `N * d / 8` bytes of the widest
/// packing, least-significant bit first: the AVX2 dispatch of
/// [`generic::pack`].
///
/// Widths where an output byte can draw from three values (`d < 8`) take the
/// scalar path.
pub(crate) fn pack(values: &[u16; N], d: usize) -> [u8; 384] {
    match d {
        10 => pack_gathered::<10>(values),
        11 => pack_gathered::<11>(values),
        12 => pack_gathered::<12>(values),
        _ => generic::pack(values, d),
    }
}

/// The `pshufb` controls and shifts scattering eight values into the output
/// bytes `j0..j0 + 8`: byte `j` keeps the low byte of
/// `(v[i] >> s) | (v[i + 1] << (d - s))` for `i = 8 * j / d` and
/// `s = 8 * j - d * i`; control `0xFF` reads as zero for the lane past the
/// window.
fn pack_controls<const D: usize>(j0: usize) -> ([u8; 32], [u32; 8], [u8; 32], [u32; 8]) {
    let value_index = |k: usize| (8 * (j0 + k)) / D;
    let bit_offset = |k: usize| (8 * (j0 + k) - D * value_index(k)) as u32;

    let idx_value = array::from_fn(|p| {
        let (k, b) = (p / 4, p % 4);
        if b < 2 {
            (2 * value_index(k) + b) as u8
        } else {
            0xFF
        }
    });
    let shift_value = array::from_fn(bit_offset);

    let idx_next = array::from_fn(|p| {
        let (k, b) = (p / 4, p % 4);
        let next = value_index(k) + 1;
        if b < 2 && next < 8 {
            (2 * next + b) as u8
        } else {
            0xFF
        }
    });
    let shift_next = array::from_fn(|k| D as u32 - bit_offset(k));

    (idx_value, shift_value, idx_next, shift_next)
}

/// Packs the widths whose output bytes draw from at most two values
/// (`d >= 8`): two `pshufb` gathers per 8-byte group, shifted and
/// OR-combined, low bytes kept. The two groups per iteration overlap to
/// cover `d` bytes.
fn pack_gathered<const D: usize>(values: &[u16; N]) -> [u8; 384] {
    let low = pack_controls::<D>(0);
    let high = pack_controls::<D>(D - 8);
    let mut out = [0u8; 384];

    // SAFETY: pure value ops on 32-byte constants; each iteration `i` writes
    // the two 8-byte groups at `D * i` and `D * i + D - 8`, staying within
    // the `32 * D <= 384` bytes the loop tiles. `packus` saturation is
    // inert: every lane is masked to one byte first.
    unsafe {
        let controls = [low, high].map(|(idx_value, shift_value, idx_next, shift_next)| {
            (
                _mm256_loadu_si256(idx_value.as_ptr().cast()),
                _mm256_loadu_si256(shift_value.as_ptr().cast()),
                _mm256_loadu_si256(idx_next.as_ptr().cast()),
                _mm256_loadu_si256(shift_next.as_ptr().cast()),
            )
        });
        let mask = _mm256_set1_epi32(0xFF);

        for (i, chunk) in values.chunks_exact(8).enumerate() {
            let wide = _mm256_broadcastsi128_si256(_mm_loadu_si128(chunk.as_ptr().cast()));

            for (group, &(idx_value, shift_value, idx_next, shift_next)) in
                controls.iter().enumerate()
            {
                let value = _mm256_srlv_epi32(_mm256_shuffle_epi8(wide, idx_value), shift_value);
                let next = _mm256_sllv_epi32(_mm256_shuffle_epi8(wide, idx_next), shift_next);
                let bytes = _mm256_and_si256(_mm256_or_si256(value, next), mask);

                let packed = _mm_packus_epi32(
                    _mm256_castsi256_si128(bytes),
                    _mm256_extracti128_si256::<1>(bytes),
                );
                _mm_storel_epi64(
                    out.as_mut_ptr().add(D * i + group * (D - 8)).cast(),
                    _mm_packus_epi16(packed, packed),
                );
            }
        }
    }

    out
}

/// Unpacks `N` `d`-bit values from bytes, least-significant bit first: the
/// AVX2 dispatch of [`generic::unpack`].
///
/// Widths without a per-arch backend, (`d = 1`) take the scalar path. Short
/// inputs need no dispatch: the windowed loads zero-pad past the end, matching
/// the scalar form.
pub(in crate::encoding) fn unpack(bytes: &[u8], d: usize) -> [u16; N] {
    match d {
        4 => unpack_gathered::<4>(bytes),
        5 => unpack_gathered::<5>(bytes),
        10 => unpack_gathered::<10>(bytes),
        11 => unpack_gathered::<11>(bytes),
        12 => unpack_gathered::<12>(bytes),
        _ => generic::unpack(bytes, d),
    }
}

/// The `pshufb` control gathering the four bytes of value `i` into `u32` lane
/// `i`, and the per-lane bit offsets: value `i` starts at byte
/// `D * i / 8` with offset `D * i mod 8`. Control byte `k` addresses byte
/// `k % 4` of lane `k / 4`'s value, relative to the 16-byte window broadcast
/// to both halves.
fn gather_controls<const D: usize>() -> ([u8; 32], [u32; 8]) {
    let idx = array::from_fn(|k| ((D * (k / 4)) / 8 + k % 4) as u8);
    let shifts = array::from_fn(|i| ((D * i) % 8) as u32);

    (idx, shifts)
}

/// Loads the 16 bytes at `start`, zero-padding past the end of `bytes`; the
/// gathers never index past byte 13, so the padding is inert.
#[inline]
fn window(bytes: &[u8], start: usize) -> __m128i {
    if let Some(full) = bytes.get(start..start + 16) {
        // SAFETY: `full` is exactly 16 bytes; `loadu` has no alignment
        // requirement.
        unsafe { _mm_loadu_si128(full.as_ptr().cast()) }
    } else {
        let rest = bytes.get(start..).unwrap_or_default();
        let mut padded = [0u8; 16];
        padded.split_at_mut(rest.len()).0.copy_from_slice(rest);

        // SAFETY: `padded` is 16 bytes.
        unsafe { _mm_loadu_si128(padded.as_ptr().cast()) }
    }
}

/// Unpacks any supported width: gather to `u32` lanes, shift, mask, and pack
/// eight values back to `u16`.
fn unpack_gathered<const D: usize>(bytes: &[u8]) -> [u16; N] {
    let (idx, shifts) = gather_controls::<D>();
    let mut out = [0u16; N];

    // SAFETY: pure value ops on 32-byte constants; the store writes the
    // 8-element chunk the loop hands out. `*_packus_*` saturation is inert: every
    // masked value fits `d <= 12` bits.
    unsafe {
        let idx: __m256i = _mm256_loadu_si256(idx.as_ptr().cast());
        let shifts: __m256i = _mm256_loadu_si256(shifts.as_ptr().cast());
        let mask = _mm256_set1_epi32(((1u32 << D) - 1) as i32);

        for (i, chunk) in out.chunks_exact_mut(8).enumerate() {
            let wide = _mm256_broadcastsi128_si256(window(bytes, D * i));
            let gathered = _mm256_shuffle_epi8(wide, idx);
            let values = _mm256_and_si256(_mm256_srlv_epi32(gathered, shifts), mask);

            let packed = _mm_packus_epi32(
                _mm256_castsi256_si128(values),
                _mm256_extracti128_si256::<1>(values),
            );
            _mm_storeu_si128(chunk.as_mut_ptr().cast(), packed);
        }
    }

    out
}
