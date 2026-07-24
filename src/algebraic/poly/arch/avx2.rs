//! AVX2-vectorized polynomial kernels for x86_64, processing sixteen `i16`
//! coefficients per `__m256i`.
//!
//! Mirrors the verified NEON backend: the stride-≥16 butterfly stages run
//! sixteen butterflies per vector under one broadcast zeta, while the narrow
//! stride-8/4/2 stages (and `ntt`'s final reduce) run scalar. `multiply`
//! separates even/odd coefficients with shuffles (AVX2 has no deinterleaving
//! load). The arithmetic matches the scalar backend (the `tests` module
//! cross-checks it), in the signed Montgomery convention of
//! [`crate::algebraic::field`].
//!
//! `unsafe` is confined to this module's intrinsic blocks. The module compiles
//! only when `mlkem_selkie_arch = "avx2"` is set, i.e. when the build enables
//! AVX2, so each intrinsic call is sound, as the per-block `SAFETY` notes
//! record.
#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_set1_epi64x, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256,
    _mm256_cvtepu16_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_mulhi_epi16, _mm256_mulhrs_epi16, _mm256_mullo_epi16, _mm256_mullo_epi32,
    _mm256_or_si256, _mm256_packs_epi32, _mm256_packus_epi32, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8,
    _mm256_shuffle_epi8, _mm256_sll_epi32, _mm256_slli_epi32, _mm256_slli_epi64, _mm256_srai_epi16,
    _mm256_srai_epi32, _mm256_srl_epi32, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi16,
    _mm256_sub_epi32, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

use crate::{algebraic::field::FieldElement, parameters};

#[cfg(test)]
mod tests;

/// The modulus q, mirroring `crate::algebraic::field`.
const Q: i16 = parameters::Q as i16;

/// `q^-1 mod 2^16`, signed, mirroring `crate::algebraic::field`.
const QINV: i16 = -3327;

/// `((2^26 + q/2) / q) = 20159`, the Barrett multiplier, mirroring `field`.
const BARRETT_V: i16 = 20159;

/// The final NTT⁻¹ scale `f = 128^-1 * R^2 mod q = 1441` in Montgomery form,
/// mirroring `generic`'s `NTT_INVERSE_SCALE`.
const NTT_INVERSE_SCALE: i16 = 1441;

/// Montgomery-multiplies sixteen coefficient lanes, `a * b -> a*b*R^-1 mod q`:
/// the vector form of the scalar `*` on `FieldElement`.
///
/// Uses the `mullo`/`mulhi` identity: with `lo`/`hi` the low/high halves of
/// `a*b` and `m = lo * QINV`, the reduction `(a*b - m*q) >> 16` equals
/// `hi - mulhi(m, q)` because `a*b - m*q` is a multiple of `2^16`.
#[inline]
fn fqmul(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe {
        let q = _mm256_set1_epi16(Q);
        let qinv = _mm256_set1_epi16(QINV);

        let lo = _mm256_mullo_epi16(a, b);
        let hi = _mm256_mulhi_epi16(a, b);
        let m = _mm256_mullo_epi16(lo, qinv);
        let t = _mm256_mulhi_epi16(m, q);

        _mm256_sub_epi16(hi, t)
    }
}

/// Multiplies `a` by a compile-time constant `b` using Barrett reduction with
/// the precomputed multiplier `b_bar = round(b * 2^15 / q)`, returning
/// `a * b mod q` per lane in four vector instructions.
///
/// `b` must be a canonical (non-Montgomery) representative for the result to
/// preserve the Montgomery-domain convention: given `a = a_true * R`,
/// `a * b mod q = (a_true * b) * R`. For NTT butterflies with the shared
/// [`super::ZETA_BARRETT`] table paired with [`super::ZETA_RAW`], this
/// replaces the [`fqmul`] path (`mullo` + `mulhi` + `mullo` + `mulhi` + `sub`,
/// five intrinsics) with four.
#[inline]
fn barrett_const_mul(a: __m256i, b: __m256i, b_bar: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe {
        let c_low = _mm256_mullo_epi16(a, b);
        let t = _mm256_mulhrs_epi16(a, b_bar);
        let tq = _mm256_mullo_epi16(t, _mm256_set1_epi16(Q));

        _mm256_sub_epi16(c_low, tq)
    }
}

/// Splits natural coefficient order into Tq's evens-then-odds storage,
/// sixteen base pairs at a time. Matches [`super::generic::pack`]; the one
/// surviving deinterleave shuffle, once per forward transform instead of in
/// every base multiplication.
pub(crate) fn pack(natural: &[FieldElement; parameters::N]) -> [FieldElement; parameters::N] {
    let mut halves = [FieldElement::ZERO; parameters::N];

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so both
    // arrays reinterpret as `[i16]`. Each iteration reads one interleaved
    // 32-`i16` window of `natural` and writes the two even/odd 16-`i16` halves
    // (offsets `i` and `128 + i`; 8 windows tile 256), all in bounds.
    // `loadu`/`storeu` are unaligned; the module is AVX2.
    unsafe {
        // Per-128-bit-lane byte mask gathering even `i16`s into the low half
        // and odds into the high half.
        let deinterleave = _mm256_setr_epi8(
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, //
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
        );

        let (even_half, odd_half) = halves.split_at_mut(parameters::N / 2);

        let windows = natural.chunks_exact(32).zip(
            even_half
                .chunks_exact_mut(16)
                .zip(odd_half.chunks_exact_mut(16)),
        );
        for (window, (even_out, odd_out)) in windows {
            let in_ptr = window.as_ptr().cast::<i16>();
            let v0 = _mm256_loadu_si256(in_ptr.cast::<__m256i>());
            let v1 = _mm256_loadu_si256(in_ptr.add(16).cast::<__m256i>());

            let d0 = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(v0, deinterleave));
            let d1 = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(v1, deinterleave));

            _mm256_storeu_si256(
                even_out.as_mut_ptr().cast::<__m256i>(),
                _mm256_permute2x128_si256::<0x20>(d0, d1),
            );
            _mm256_storeu_si256(
                odd_out.as_mut_ptr().cast::<__m256i>(),
                _mm256_permute2x128_si256::<0x31>(d0, d1),
            );
        }
    }

    halves
}

/// Re-interleaves Tq's evens-then-odds storage back to natural order,
/// sixteen base pairs at a time. Matches [`super::generic::unpack`].
pub(crate) fn unpack(halves: &[FieldElement; parameters::N]) -> [FieldElement; parameters::N] {
    let mut natural = [FieldElement::ZERO; parameters::N];

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so both
    // arrays reinterpret as `[i16]`. Each iteration reads the two halves
    // 16-`i16` halves (offsets `i` and `128 + i`) and writes one interleaved
    // 32-`i16` window of `natural` (8 windows tile 256), all in bounds.
    // `loadu`/`storeu` are unaligned; the module is AVX2.
    unsafe {
        let (even_half, odd_half) = halves.split_at(parameters::N / 2);

        let windows = even_half
            .chunks_exact(16)
            .zip(odd_half.chunks_exact(16))
            .zip(natural.chunks_exact_mut(32));
        for ((even_window, odd_window), out_window) in windows {
            let even = _mm256_loadu_si256(even_window.as_ptr().cast::<__m256i>());
            let odd = _mm256_loadu_si256(odd_window.as_ptr().cast::<__m256i>());

            let lo = _mm256_unpacklo_epi16(even, odd);
            let hi = _mm256_unpackhi_epi16(even, odd);

            let out_ptr = out_window.as_mut_ptr();
            _mm256_storeu_si256(
                out_ptr.cast::<__m256i>(),
                _mm256_permute2x128_si256::<0x20>(lo, hi),
            );
            _mm256_storeu_si256(
                out_ptr.add(16).cast::<__m256i>(),
                _mm256_permute2x128_si256::<0x31>(lo, hi),
            );
        }
    }

    natural
}

/// One Cooley-Tukey butterfly over sixteen lanes: returns `(a + t, a - t)`
/// for `t = b * zeta`, with the zeta given as its `(raw, Barrett)` broadcast
/// pair.
#[inline]
fn butterfly(a: __m256i, b: __m256i, zeta: (__m256i, __m256i)) -> (__m256i, __m256i) {
    let t = barrett_const_mul(b, zeta.0, zeta.1);

    // SAFETY: the avx2 module compiles only with AVX2 enabled; both
    // intrinsics are total.
    unsafe { (_mm256_add_epi16(a, t), _mm256_sub_epi16(a, t)) }
}

/// Barrett-reduces sixteen `i16` lanes to a representative in `(-q/2, q/2]`:
/// the vector form of `FieldElement::reduce`.
///
/// `t = (a*BARRETT_V + 2^25) >> 26` is computed as `(mulhi(a, BARRETT_V) + 2^9)
/// >> 10`, since `mulhi` is the `>> 16`.
#[inline]
fn barrett_reduce(a: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe {
        let mut t = _mm256_mulhi_epi16(a, _mm256_set1_epi16(BARRETT_V));
        t = _mm256_add_epi16(t, _mm256_set1_epi16(1 << 9));
        t = _mm256_srai_epi16::<10>(t);
        t = _mm256_mullo_epi16(t, _mm256_set1_epi16(Q));

        _mm256_sub_epi16(a, t)
    }
}

/// The multiplier for the vectorized `Compress_d` division:
/// `floor(a / q) = (a * 2580335) >> 33`, exact for every
/// `a = (x << d) + q/2` with canonical `x` and `d <= 12` — the same
/// `BARRETT_DIV_Q` the scalar field arithmetic const-asserts exact over this
/// range; the `tests` module re-checks every input per `d` exhaustively.
/// The 44-bit products need 64-bit lanes (`mul_epu32` over the even and odd
/// halves).
const COMPRESS_DIV: i64 = 2_580_335;

/// The exact `floor(a / q)` of eight `u32` lanes via `a * COMPRESS_DIV >>
/// 33`, with the 64-bit products taken over the even and odd lanes
/// separately and recombined.
#[inline]
fn divide_by_q(a: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe {
        let divisor = _mm256_set1_epi32(COMPRESS_DIV as i32);

        let even = _mm256_srli_epi64::<33>(_mm256_mul_epu32(a, divisor));
        let odd = _mm256_srli_epi64::<33>(_mm256_mul_epu32(_mm256_srli_epi64::<32>(a), divisor));

        _mm256_or_si256(even, _mm256_slli_epi64::<32>(odd))
    }
}

/// Canonicalizes sixteen lanes to `[0, q)`: Barrett-reduce, then a
/// branch-free conditional add of q on the negative lanes. The vector form of
/// `FieldElement::value`.
#[inline]
fn canonical_lanes(x: __m256i) -> __m256i {
    let reduced = barrett_reduce(x);

    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe {
        let negative = _mm256_srai_epi16::<15>(reduced);

        _mm256_add_epi16(reduced, _mm256_and_si256(negative, _mm256_set1_epi16(Q)))
    }
}

/// Packs two eight-lane `i32` vectors of sixteen-bit values back to one
/// sixteen-lane `i16` vector in coefficient order (`packus` interleaves the
/// 128-bit lanes; the `permute` restores them).
#[inline]
fn pack_lanes(lo: __m256i, hi: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe { _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(lo, hi)) }
}

/// `Compress_d` of every coefficient: canonicalizes and maps each to
/// `round((2^d / q) * x) mod 2^d`, sixteen sixteen-lane groups.
///
/// Matches [`super::generic::compress`]. Constant-time on secret-derived
/// inputs: every lane runs the same multiply/shift sequence, with no
/// data-dependent branches or lookups.
pub(crate) fn compress(
    coefficients: &[FieldElement; parameters::N],
    d: usize,
) -> [u16; parameters::N] {
    debug_assert!(d <= 12);

    let mut out = [0u16; parameters::N];

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16` and the `u16`
    // outputs are below `2^d`, so both element types reinterpret as `i16`.
    // Every load and store uses a pointer derived from a 16-element chunk of
    // the array it covers, so it is in bounds by construction.
    // `loadu`/`storeu` are unaligned; the module is AVX2.
    unsafe {
        let up = _mm_set1_epi64x(d as i64);
        let half_q = _mm256_set1_epi32(i32::from(parameters::Q / 2));
        let mask = _mm256_set1_epi32((1 << d) - 1);

        let divide = |x: __m128i| -> __m256i {
            // (x << d) + q/2, then the exact division by q.
            let a = _mm256_add_epi32(_mm256_sll_epi32(_mm256_cvtepu16_epi32(x), up), half_q);

            _mm256_and_si256(divide_by_q(a), mask)
        };

        let windows = coefficients.chunks_exact(16).zip(out.chunks_exact_mut(16));
        for (window, out_window) in windows {
            let x = canonical_lanes(_mm256_loadu_si256(window.as_ptr().cast::<__m256i>()));

            let lo = divide(_mm256_extracti128_si256::<0>(x));
            let hi = divide(_mm256_extracti128_si256::<1>(x));

            _mm256_storeu_si256(
                out_window.as_mut_ptr().cast::<__m256i>(),
                pack_lanes(lo, hi),
            );
        }
    }

    out
}

/// `Decompress_d` of every value: maps each `d`-bit value back into Zq via
/// `(q * y + 2^(d-1)) >> d`, sixteen sixteen-lane groups.
///
/// Matches [`super::generic::decompress`]. Constant-time: every lane runs
/// the same multiply/shift sequence.
pub(crate) fn decompress(values: &[u16; parameters::N], d: usize) -> [FieldElement; parameters::N] {
    let mut out = [FieldElement::ZERO; parameters::N];

    // SAFETY: the `u16` inputs are below `2^12` and the outputs canonical, so
    // both element types reinterpret as `i16` (`FieldElement` is
    // `repr(transparent)` over `i16`). Every load and store uses a pointer
    // derived from a 16-element chunk of the array it covers, so it is in
    // bounds by construction. `loadu`/`storeu` are unaligned; the module is
    // AVX2.
    unsafe {
        let down = _mm_set1_epi64x(d as i64);
        let half = _mm256_set1_epi32(1 << (d - 1));
        let q = _mm256_set1_epi32(i32::from(parameters::Q));

        let scale = |y: __m128i| -> __m256i {
            // (q * y + 2^(d-1)) >> d, exact in i32.
            let a = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_cvtepu16_epi32(y), q), half);

            _mm256_srl_epi32(a, down)
        };

        let windows = values.chunks_exact(16).zip(out.chunks_exact_mut(16));
        for (window, out_window) in windows {
            let y = _mm256_loadu_si256(window.as_ptr().cast::<__m256i>());

            let lo = scale(_mm256_extracti128_si256::<0>(y));
            let hi = scale(_mm256_extracti128_si256::<1>(y));

            _mm256_storeu_si256(
                out_window.as_mut_ptr().cast::<__m256i>(),
                pack_lanes(lo, hi),
            );
        }
    }

    out
}

/// The canonical representative in `[0, q)` of every coefficient
/// (`FieldElement::value`), re-interleaved to natural order for
/// serialization, eight sixteen-lane group pairs.
///
/// Matches [`super::generic::canonical`]. Constant-time: branch-free per
/// lane.
pub(crate) fn canonical(coefficients: &[FieldElement; parameters::N]) -> [u16; parameters::N] {
    let mut out = [0u16; parameters::N];

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16` and the
    // canonical outputs are below q, so both element types reinterpret as
    // `i16`. Every load and store uses a pointer derived from a chunk of the
    // half or array it covers (even and odd 16-element chunks in, one
    // 32-element chunk out), so it is in bounds by construction.
    // `loadu`/`storeu` are unaligned; the module is AVX2.
    unsafe {
        let (even_half, odd_half) = coefficients.split_at(parameters::N / 2);

        let windows = even_half
            .chunks_exact(16)
            .zip(odd_half.chunks_exact(16))
            .zip(out.chunks_exact_mut(32));
        for ((even_window, odd_window), out_window) in windows {
            let even = canonical_lanes(_mm256_loadu_si256(even_window.as_ptr().cast::<__m256i>()));
            let odd = canonical_lanes(_mm256_loadu_si256(odd_window.as_ptr().cast::<__m256i>()));

            // Interleave (e, o) pairs: unpack within 128-bit lanes, then
            // gather the lane halves into sequential order.
            let lo = _mm256_unpacklo_epi16(even, odd);
            let hi = _mm256_unpackhi_epi16(even, odd);

            let out_ptr = out_window.as_mut_ptr();
            _mm256_storeu_si256(
                out_ptr.cast::<__m256i>(),
                _mm256_permute2x128_si256::<0x20>(lo, hi),
            );
            _mm256_storeu_si256(
                out_ptr.add(16).cast::<__m256i>(),
                _mm256_permute2x128_si256::<0x31>(lo, hi),
            );
        }
    }

    out
}

/// Pointwise base multiplication of two NTT representations: 128 degree-one
/// products modulo the quadratics `X^2 - gamma`, computed sixteen pairs at a
/// time.
///
/// Matches [`super::generic::multiply`]. The evens-then-odds storage yields
/// each pair's halves as contiguous vectors: no shuffles. The result is scaled
/// by `R^-1`, which `ntt_inverse` later undoes.
///
/// `h` is stack-allocated `MaybeUninit`, not `[ZERO; N]`, so we skip the
/// 16-`vmovups` array wipe LLVM otherwise emits — it can't see through the
/// `unsafe` pointer stores to prove the loop overwrites every element.
pub(crate) fn multiply(
    f: &[FieldElement; parameters::N],
    g: &[FieldElement; parameters::N],
) -> [FieldElement; parameters::N] {
    let mut h = [FieldElement::ZERO; parameters::N];

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the three
    // length-256 arrays and the length-128 `GAMMA_MONT` reinterpret as
    // `[i16]`. Every load and store uses a pointer derived from a 16-element
    // chunk of the half it covers, so it is in bounds by construction.
    // `loadu`/`storeu` are unaligned; the module is AVX2.
    unsafe {
        let (f0_half, f1_half) = f.split_at(parameters::N / 2);
        let (g0_half, g1_half) = g.split_at(parameters::N / 2);
        let (h0_half, h1_half) = h.split_at_mut(parameters::N / 2);

        let windows = f0_half
            .chunks_exact(16)
            .zip(f1_half.chunks_exact(16))
            .zip(g0_half.chunks_exact(16).zip(g1_half.chunks_exact(16)))
            .zip(super::GAMMA_MONT.chunks_exact(16))
            .zip(
                h0_half
                    .chunks_exact_mut(16)
                    .zip(h1_half.chunks_exact_mut(16)),
            );
        for ((((f0_w, f1_w), (g0_w, g1_w)), gamma_w), (h0_w, h1_w)) in windows {
            // Even (degree-0) and odd (degree-1) halves.
            let a0 = _mm256_loadu_si256(f0_w.as_ptr().cast::<__m256i>());
            let a1 = _mm256_loadu_si256(f1_w.as_ptr().cast::<__m256i>());
            let b0 = _mm256_loadu_si256(g0_w.as_ptr().cast::<__m256i>());
            let b1 = _mm256_loadu_si256(g1_w.as_ptr().cast::<__m256i>());

            let gamma = _mm256_loadu_si256(gamma_w.as_ptr().cast::<__m256i>());

            // c0 = a0*b0 + a1*b1*gamma ; c1 = a0*b1 + a1*b0
            let c0 = _mm256_add_epi16(fqmul(a0, b0), fqmul(fqmul(a1, b1), gamma));
            let c1 = _mm256_add_epi16(fqmul(a0, b1), fqmul(a1, b0));

            _mm256_storeu_si256(h0_w.as_mut_ptr().cast::<__m256i>(), c0);
            _mm256_storeu_si256(h1_w.as_mut_ptr().cast::<__m256i>(), c1);
        }
    }

    h
}

/// Widens two 16-lane `i16` products into eight-lane `i32` sums-of-products,
/// low unpack half: `mullo`/`mulhi` recombined per lane.
///
/// [`_mm256_unpacklo_epi16`] interleaves within 128-bit lanes, so the result
/// holds pairs `{0..3, 8..11}` of the block; [`widening_mul_hi`] holds
/// `{4..7, 12..15}`. [`basemul_accumulate`] and [`basemul_reduce`] both use
/// this split, so the accumulator plane layout is self-consistent.
#[inline]
fn widening_mul_lo(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe { _mm256_unpacklo_epi16(_mm256_mullo_epi16(a, b), _mm256_mulhi_epi16(a, b)) }
}

/// Widens two 16-lane `i16` products into eight-lane `i32` sums-of-products,
/// high unpack half: pairs `{4..7, 12..15}` of the block.
#[inline]
fn widening_mul_hi(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe { _mm256_unpackhi_epi16(_mm256_mullo_epi16(a, b), _mm256_mulhi_epi16(a, b)) }
}

/// Montgomery-reduces eight `i32` lanes to `i16` values (in the low 16 bits of
/// each `i32` lane): the vector form of `FieldElement::from_product_sum`.
#[inline]
fn montgomery_reduce_wide(a: __m256i) -> __m256i {
    // SAFETY: the avx2 module compiles only with AVX2 enabled; every intrinsic
    // below is total.
    unsafe {
        let q = _mm256_set1_epi32(Q as i32);
        let qinv = _mm256_set1_epi32(QINV as i32);

        // m = sign-extended low 16 bits of a * QINV.
        let m = _mm256_mullo_epi32(a, qinv);
        let m = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(m));

        _mm256_srai_epi32::<16>(_mm256_sub_epi32(a, _mm256_mullo_epi32(m, q)))
    }
}

/// Accumulates one component of an asymmetric base-multiplication dot product
/// into `acc`: widening multiplies summed in `i32` lanes, no reduction.
///
/// Matches [`super::generic::basemul_accumulate`] mod q. The accumulator
/// planes hold each 16-pair block in the unpack order
/// `{0..3, 8..11, 4..7, 12..15}` (see [`widening_mul_lo`]); only
/// [`basemul_reduce`] reads them back, with the matching order.
pub(crate) fn basemul_accumulate(
    acc: &mut super::ProductAccumulator,
    f: &[FieldElement; parameters::N],
    g: &[FieldElement; parameters::N],
    cache: &[FieldElement; parameters::N / 2],
) {
    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so `f`/`g`/
    // `cache` reinterpret as `[i16]`. Each iteration reads the two halves
    // 16-`i16` halves of each base-pair window of `f`/`g` (offsets `pair` and
    // `128 + pair`; 8 windows tile 256), a 16-`i16` window of `cache`, and
    // read-modify-writes two 8-`i32` windows of each 128-`i32` accumulator
    // plane (pair and pair + 8, with pair + 16 <= 128), all in bounds.
    // `loadu`/`storeu` are unaligned; the module is AVX2.
    unsafe {
        let (f0_half, f1_half) = f.split_at(parameters::N / 2);
        let (g0_half, g1_half) = g.split_at(parameters::N / 2);

        let windows = f0_half
            .chunks_exact(16)
            .zip(f1_half.chunks_exact(16))
            .zip(g0_half.chunks_exact(16).zip(g1_half.chunks_exact(16)))
            .zip(cache.chunks_exact(16))
            .zip(
                acc.even
                    .chunks_exact_mut(16)
                    .zip(acc.odd.chunks_exact_mut(16)),
            );
        for ((((f0_w, f1_w), (g0_w, g1_w)), cache_w), (even_w, odd_w)) in windows {
            let a0 = _mm256_loadu_si256(f0_w.as_ptr().cast::<__m256i>());
            let a1 = _mm256_loadu_si256(f1_w.as_ptr().cast::<__m256i>());
            let b0 = _mm256_loadu_si256(g0_w.as_ptr().cast::<__m256i>());
            let b1 = _mm256_loadu_si256(g1_w.as_ptr().cast::<__m256i>());

            let c = _mm256_loadu_si256(cache_w.as_ptr().cast::<__m256i>());

            // even += a0*b0 + a1*cache, in i32 lanes.
            let even_gain_lo = _mm256_add_epi32(widening_mul_lo(a0, b0), widening_mul_lo(a1, c));
            let even_gain_hi = _mm256_add_epi32(widening_mul_hi(a0, b0), widening_mul_hi(a1, c));
            let even_lo_ptr = even_w.as_mut_ptr().cast::<__m256i>();
            let even_hi_ptr = even_w.as_mut_ptr().add(8).cast::<__m256i>();
            _mm256_storeu_si256(
                even_lo_ptr,
                _mm256_add_epi32(_mm256_loadu_si256(even_lo_ptr), even_gain_lo),
            );
            _mm256_storeu_si256(
                even_hi_ptr,
                _mm256_add_epi32(_mm256_loadu_si256(even_hi_ptr), even_gain_hi),
            );

            // odd += a0*b1 + a1*b0, in i32 lanes.
            let odd_gain_lo = _mm256_add_epi32(widening_mul_lo(a0, b1), widening_mul_lo(a1, b0));
            let odd_gain_hi = _mm256_add_epi32(widening_mul_hi(a0, b1), widening_mul_hi(a1, b0));
            let odd_lo_ptr = odd_w.as_mut_ptr().cast::<__m256i>();
            let odd_hi_ptr = odd_w.as_mut_ptr().add(8).cast::<__m256i>();
            _mm256_storeu_si256(
                odd_lo_ptr,
                _mm256_add_epi32(_mm256_loadu_si256(odd_lo_ptr), odd_gain_lo),
            );
            _mm256_storeu_si256(
                odd_hi_ptr,
                _mm256_add_epi32(_mm256_loadu_si256(odd_hi_ptr), odd_gain_hi),
            );
        }
    }
}

/// Montgomery-reduces the accumulated product sums into the evens-then-odds
/// storage halves, sixteen pairs at a time.
///
/// Matches [`super::generic::basemul_reduce`] mod q. `packs_epi32` packs
/// within 128-bit lanes, which exactly inverts the unpack order the
/// accumulator planes were stored in (see [`basemul_accumulate`]); the packed
/// values are Montgomery residues bounded well inside `i16`, so the
/// saturation in `packs` never fires.
pub(crate) fn basemul_reduce(acc: &super::ProductAccumulator) -> [FieldElement; parameters::N] {
    let mut h = [FieldElement::ZERO; parameters::N];

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so `h`
    // reinterprets as `[i16]`. Each iteration reads two 8-`i32` windows of
    // each 128-`i32` accumulator plane and writes the two even/odd 16-`i16`
    // halves of `h` (offsets `pair` and `128 + pair`; 8 windows tile 256),
    // all in bounds. `loadu`/`storeu` are unaligned; the module is AVX2.
    unsafe {
        let even_ptr = acc.even.as_ptr();
        let odd_ptr = acc.odd.as_ptr();
        let h_ptr = h.as_mut_ptr().cast::<i16>();

        let mut pair = 0;
        while pair < 128 {
            let even_lo =
                montgomery_reduce_wide(_mm256_loadu_si256(even_ptr.add(pair).cast::<__m256i>()));
            let even_hi = montgomery_reduce_wide(_mm256_loadu_si256(
                even_ptr.add(pair + 8).cast::<__m256i>(),
            ));
            let odd_lo =
                montgomery_reduce_wide(_mm256_loadu_si256(odd_ptr.add(pair).cast::<__m256i>()));
            let odd_hi =
                montgomery_reduce_wide(_mm256_loadu_si256(odd_ptr.add(pair + 8).cast::<__m256i>()));

            // Pack the i32 residues back to pair order: the halves halves.
            let c0 = _mm256_packs_epi32(even_lo, even_hi);
            let c1 = _mm256_packs_epi32(odd_lo, odd_hi);

            _mm256_storeu_si256(h_ptr.add(pair).cast::<__m256i>(), c0);
            _mm256_storeu_si256(h_ptr.add(128 + pair).cast::<__m256i>(), c1);

            pair += 16;
        }
    }

    h
}

/// Forward NTT in place, then Barrett-reduce. The stride-≥16 butterfly
/// stages run in two register-resident trips (stride 128/64 merged, then
/// 32/16 merged), so each stage pair costs one memory round-trip instead of
/// one per stage; the narrow stride-8/4/2 stages and the final reduction run
/// scalar.
///
/// Matches [`super::generic::ntt`]: the merged trips compute the same
/// butterflies as the stage-at-a-time loop, grouped by data slice instead of
/// stage, and butterflies within a stage are independent.
// reason: scalar-tail butterfly/zeta indices are provably in 0..256, as in the
// generic backend.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt(coefficients: &mut [FieldElement; parameters::N]) {
    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the array
    // reinterprets as `[i16]`. The head trip reads/writes vectors
    // `16 * (i + 4b)` for `i < 4`, `b < 4`; the quarter trip vectors
    // `16 * (4q + w)` for `q < 4`, `w < 4` (each tiles the 16 vector windows
    // of 256 coefficients once). Zeta indices stay below 16.
    // `loadu`/`storeu` are unaligned; the module is AVX2.
    unsafe {
        let ptr = coefficients.as_mut_ptr().cast::<i16>();
        let zeta_raw_ptr = super::ZETA_RAW.as_ptr().cast::<i16>();
        let zeta_bar_ptr = super::ZETA_BARRETT.as_ptr();
        let zeta = |k: usize| {
            (
                _mm256_set1_epi16(*zeta_raw_ptr.add(k)),
                _mm256_set1_epi16(*zeta_bar_ptr.add(k)),
            )
        };
        let load = |v: usize| _mm256_loadu_si256(ptr.add(16 * v).cast::<__m256i>());
        let store =
            |v: usize, x: __m256i| _mm256_storeu_si256(ptr.add(16 * v).cast::<__m256i>(), x);

        // Stride-128/64 trip: each iteration holds four vectors (one per
        // 64-coefficient quarter, at the same offset) through both butterfly
        // levels, one load and one store per vector instead of one per stage.
        let z1 = zeta(1);
        let (z2, z3) = (zeta(2), zeta(3));

        for i in 0..4 {
            let mut v0 = load(i);
            let mut v1 = load(i + 4);
            let mut v2 = load(i + 8);
            let mut v3 = load(i + 12);

            // Stride 128: halves, one zeta.
            (v0, v2) = butterfly(v0, v2, z1);
            (v1, v3) = butterfly(v1, v3, z1);

            // Stride 64: quarters, one zeta per half.
            (v0, v1) = butterfly(v0, v1, z2);
            (v2, v3) = butterfly(v2, v3, z3);

            store(i, v0);
            store(i + 4, v1);
            store(i + 8, v2);
            store(i + 12, v3);
        }

        // Stride-32/16 trip: each iteration holds one 64-coefficient
        // quarter's four vectors through both levels.
        for quarter in 0..4 {
            let mut v0 = load(4 * quarter);
            let mut v1 = load(4 * quarter + 1);
            let mut v2 = load(4 * quarter + 2);
            let mut v3 = load(4 * quarter + 3);

            // Stride 32: one zeta per quarter.
            let z = zeta(4 + quarter);
            (v0, v2) = butterfly(v0, v2, z);
            (v1, v3) = butterfly(v1, v3, z);

            // Stride 16: one zeta per 32-coefficient block.
            let (za, zb) = (zeta(8 + 2 * quarter), zeta(9 + 2 * quarter));
            (v0, v1) = butterfly(v0, v1, za);
            (v2, v3) = butterfly(v2, v3, zb);

            store(4 * quarter, v0);
            store(4 * quarter + 1, v1);
            store(4 * quarter + 2, v2);
            store(4 * quarter + 3, v3);
        }
    }

    // Scalar tail: stride-8, stride-4, and stride-2 groups are narrower than a
    // vector.
    let mut k = 16;
    for len in [8usize, 4, 2] {
        let mut start = 0;
        while start < 256 {
            let zeta = super::ZETA_RAW[k] as i16;
            let zeta_bar = super::ZETA_BARRETT[k];
            k += 1;

            for j in start..start + len {
                let t = coefficients[j + len].barrett_const_mul(zeta, zeta_bar);
                coefficients[j + len] = coefficients[j] - t;
                coefficients[j] = coefficients[j] + t;
            }

            start += 2 * len;
        }
    }

    for coefficient in coefficients.iter_mut() {
        *coefficient = coefficient.reduce();
    }
}

/// Inverse NTT in place, followed by the scale back to the standard domain.
/// Vectorizes the stride-≥16 Gentleman-Sande stages and the final scale; the
/// narrow stride-2/4/8 stages run scalar. Reductions follow the lazy len-2 /
/// len-16 schedule of [`super::generic::ntt_inverse`].
///
/// Matches [`super::generic::ntt_inverse`].
// reason: scalar-head butterfly/zeta indices are provably in 0..256, as in the
// generic backend.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt_inverse(coefficients: &mut [FieldElement; parameters::N]) {
    let mut k = 127;

    // Scalar head: stride-2, stride-4, and stride-8 groups are narrower than a
    // vector.
    for len in [2usize, 4, 8] {
        let mut start = 0;
        while start < 256 {
            let zeta = super::ZETA_RAW[k] as i16;
            let zeta_bar = super::ZETA_BARRETT[k];
            k -= 1;

            for j in start..start + len {
                let t = coefficients[j];
                let sum = t + coefficients[j + len];
                // Lazy reduction: len-2 only (len-16 reduces in the vector loop).
                coefficients[j] = if len == 2 { sum.reduce() } else { sum };
                coefficients[j + len] =
                    (coefficients[j + len] - t).barrett_const_mul(zeta, zeta_bar);
            }

            start += 2 * len;
        }
    }

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the array
    // reinterprets as `[i16]`. Each `__m256i` window stays within the
    // 256-element array, and the descending `k` indexes `ZETA_RAW` /
    // `ZETA_BARRETT` (it reaches 0 only after the last use). `loadu`/`storeu`
    // are unaligned; the module is AVX2.
    unsafe {
        let ptr = coefficients.as_mut_ptr().cast::<i16>();
        let zeta_raw_ptr = super::ZETA_RAW.as_ptr().cast::<i16>();
        let zeta_bar_ptr = super::ZETA_BARRETT.as_ptr();

        for len in [16usize, 32, 64, 128] {
            let mut start = 0;
            while start < 256 {
                let zeta = _mm256_set1_epi16(*zeta_raw_ptr.add(k));
                let zeta_bar = _mm256_set1_epi16(*zeta_bar_ptr.add(k));
                k -= 1;

                let mut j = start;
                while j < start + len {
                    let vj = _mm256_loadu_si256(ptr.add(j).cast::<__m256i>());
                    let vjl = _mm256_loadu_si256(ptr.add(j + len).cast::<__m256i>());

                    let sum = _mm256_add_epi16(vj, vjl);
                    // Lazy reduction: len-16 only.
                    _mm256_storeu_si256(
                        ptr.add(j).cast::<__m256i>(),
                        if len == 16 { barrett_reduce(sum) } else { sum },
                    );
                    _mm256_storeu_si256(
                        ptr.add(j + len).cast::<__m256i>(),
                        barrett_const_mul(_mm256_sub_epi16(vjl, vj), zeta, zeta_bar),
                    );

                    j += 16;
                }

                start += 2 * len;
            }
        }

        // Final scale: every coefficient times `NTT_INVERSE_SCALE`.
        let scale = _mm256_set1_epi16(NTT_INVERSE_SCALE);
        let mut j = 0;
        while j < 256 {
            let scaled = fqmul(scale, _mm256_loadu_si256(ptr.add(j).cast::<__m256i>()));
            _mm256_storeu_si256(ptr.add(j).cast::<__m256i>(), scaled);
            j += 16;
        }
    }
}
