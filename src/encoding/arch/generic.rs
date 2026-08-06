//! Portable scalar bit-packing, moving 64-bit words in both directions.

use core::array;

use crate::parameters::N;

/// Packs `N` `d`-bit values into the leading `N * d / 8` bytes of the widest
/// packing (`d = 12`), least-significant bit first; the tail stays zero.
///
/// The bit at position `j` of `values[i]` lands at global bit `i * d + j`,
/// accumulated and stored in 64-bit words.
#[must_use]
pub(crate) fn pack(values: &[u16; N], d: usize) -> [u8; 384] {
    let mut out = [0u8; 384];
    let mut words = out.chunks_exact_mut(8);
    let mut accumulator: u128 = 0;
    let mut bits = 0usize;

    for &value in values {
        accumulator |= u128::from(value) << bits;
        bits += d;

        if bits >= 64 {
            if let Some(word) = words.next() {
                word.copy_from_slice(&(accumulator as u64).to_le_bytes());
            }
            accumulator >>= 64;
            bits -= 64;
        }
    }

    out
}

/// Unpacks `N` `d`-bit values from bytes, least-significant bit first.
///
/// Inverse of the packing in [`crate::encoding`]; implements the bit
/// recomposition of `ByteDecode_d` ([Algorithm 6]). Reads the leading
/// `N * d / 8` bytes in 64-bit words, each value in `0..2^d`; a short input
/// is zero-padded.
///
/// [Algorithm 6]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#algorithm.6
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
