//! Portable scalar bit-unpacking, reading the input in 64-bit words.

use core::array;

use crate::parameters::N;

/// Unpacks `N` `d`-bit values from bytes, least-significant bit first.
///
/// Inverse of the packing in [`crate::encoding`]; implements the bit
/// recomposition of `ByteDecode_d` (Algorithm 6). Reads the leading
/// `N * d / 8` bytes in 64-bit words, each value in `0..2^d`; a short input
/// is zero-padded.
pub(crate) fn unpack(bytes: &[u8], d: usize) -> [u16; N] {
    let mask = (1u32 << d) - 1;

    let mut words = bytes.chunks_exact(8);
    let mut tail = Some(words.remainder()).filter(|rest| !rest.is_empty());
    let mut next_word = move || {
        if let Some(chunk) = words.next() {
            u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)"))
        } else if let Some(rest) = tail.take() {
            let mut padded = [0u8; 8];
            padded.split_at_mut(rest.len()).0.copy_from_slice(rest);

            u64::from_le_bytes(padded)
        } else {
            0
        }
    };

    let mut accumulator: u128 = 0;
    let mut bits = 0usize;

    array::from_fn(|_| {
        if bits < d {
            accumulator |= u128::from(next_word()) << bits;
            bits += 64;
        }

        let value = (accumulator as u32 & mask) as u16;
        accumulator >>= d;
        bits -= d;

        value
    })
}
