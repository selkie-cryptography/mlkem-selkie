//! NEON-vectorized polynomial kernels for aarch64, processing eight `i16`
//! coefficients per vector.
//!
//! All three kernels are vectorized. The transforms vectorize the stride-≥8
//! butterfly stages (eight butterflies per vector under one broadcast zeta) and
//! leave the narrow stride-4/2 stages scalar; `ntt`'s final Barrett reduction
//! is also scalar, while `ntt_inverse`'s lazy per-schedule reduction (len-2
//! and len-16 stages only) and final scale follow the stage they land in.
//! The arithmetic matches the scalar backend — the
//! `tests` module cross-checks every kernel — in the signed Montgomery
//! convention of [`crate::algebraic::field`].
//!
//! The only `unsafe` in the crate lives in this module (and the future `avx2`
//! sibling): NEON is baseline on aarch64, so each intrinsic call is sound, as
//! the per-block `SAFETY` notes record.
#![allow(unsafe_code)]

use core::arch::aarch64::{
    int16x4_t, int16x8_t, int16x8x2_t, int32x4_t, vaddq_s16, vcombine_s16, vdup_n_s16, vdupq_n_s16,
    vdupq_n_s32, vget_low_s16, vld1q_s16, vld1q_s32, vld2q_s16, vmlal_high_s16, vmlal_s16,
    vmlaq_n_s32, vmlsq_s16, vmovl_high_s16, vmovl_s16, vmovn_s32, vmul_s16, vmull_high_s16,
    vmull_s16, vmulq_n_s32, vmulq_s16, vqrdmulhq_s16, vshrn_n_s32, vshrq_n_s32, vst1q_s16,
    vst1q_s32, vst2q_s16, vsubq_s16, vsubq_s32,
};

use crate::{algebraic::field::FieldElement, parameters};

#[cfg(test)]
mod tests;

/// The modulus q, mirroring `crate::algebraic::field`.
const Q: i16 = parameters::Q as i16;

/// `q^-1 mod 2^16`, signed, mirroring `crate::algebraic::field`.
const QINV: i16 = -3327;

/// `((2^26 + q/2) / q)`, the Barrett multiplier, mirroring `field`'s
/// `BARRETT_V`.
const BARRETT_V: i32 = ((1 << 26) + (parameters::Q as i32) / 2) / (parameters::Q as i32);

/// The final NTT⁻¹ scale `f = 128^-1 * R^2 mod q = 1441` in Montgomery form,
/// mirroring `generic`'s `NTT_INVERSE_SCALE`.
const NTT_INVERSE_SCALE: i16 = 1441;

/// Montgomery-reduces four `i32` products to `i16`, returning `a * R^-1 mod q`
/// in `(-q, q)` per lane: the vector form of `FieldElement::montgomery_reduce`.
#[inline]
fn montgomery_reduce(a: int32x4_t, q: int16x4_t, qinv: int16x4_t) -> int16x4_t {
    // SAFETY: NEON is baseline on aarch64; every intrinsic below is total.
    unsafe {
        let a_low = vmovn_s32(a); // a as i16 (low 16 bits)
        let m = vmul_s16(a_low, qinv); // m = (a as i16) * QINV
        let m_q = vmull_s16(m, q); // m * q, widened to i32
        let difference = vsubq_s32(a, m_q); // a - m*q

        vshrn_n_s32::<16>(difference) // (a - m*q) >> 16, narrowed to i16
    }
}

/// Montgomery-multiplies eight coefficient lanes, `a * b -> a*b*R^-1 mod q`:
/// the vector form of the scalar `*` on `FieldElement`.
#[inline]
fn fqmul(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is baseline on aarch64; every intrinsic below is total.
    unsafe {
        let q = vdup_n_s16(Q);
        let qinv = vdup_n_s16(QINV);

        let product_low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        let product_high = vmull_high_s16(a, b);

        vcombine_s16(
            montgomery_reduce(product_low, q, qinv),
            montgomery_reduce(product_high, q, qinv),
        )
    }
}

/// Multiplies `a` by a compile-time constant `b` using Barrett reduction with
/// the precomputed multiplier `b_bar = round(b * 2^15 / q)`, returning
/// `a * b mod q` per lane in three vector instructions.
///
/// `b` must be a canonical (non-Montgomery) representative for the result to
/// preserve the Montgomery-domain convention: given `a = a_true * R`,
/// `a * b mod q = (a_true * b) * R`. For NTT butterflies with zeta table
/// [`super::ZETA_BARRETT`] paired with [`super::ZETA_RAW`], this replaces the
/// [`fqmul`] path (~12 intrinsics per lane) with 3.
#[inline]
fn barrett_const_mul(a: int16x8_t, b: int16x8_t, b_bar: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is baseline on aarch64; every intrinsic below is total.
    unsafe {
        let c_low = vmulq_s16(a, b);
        let t = vqrdmulhq_s16(a, b_bar);

        vmlsq_s16(c_low, t, vdupq_n_s16(Q))
    }
}

/// Stride-64 forward-NTT butterfly group, software-pipelined for the M4.
///
/// Runs 8 vector butterflies at `[ptr, ptr+128)` paired with
/// `[ptr+128, ptr+256)`, all sharing one zeta pair. Called twice per
/// forward NTT (once per outer start position) with the two different
/// zetas.
///
/// Same schedule shape as [`ntt_stride128_asm`], with a shorter loop
/// (4 iters → 6 cy/iter steady state → ~35 cy stage total, vs. 52 cy for
/// the previous hand-scheduled 2× interleave).
///
/// # Safety
///
/// `ptr` must point to a 256-byte window mutable for reads and writes.
/// Clobbers `v0`-`v7` and `v16`-`v30` (all caller-saved; `v8`-`v15` stay
/// reserved), general register `x9`, and the input `ptr` (post-increments
/// through the low half).
#[inline]
unsafe fn ntt_stride64_group_asm(ptr: *mut i16, zeta: i16, zeta_bar: i16) {
    // SAFETY: `ptr` covers 128 i16s lower + 128 i16s upper = 256 bytes.
    // The preamble seeds iterations 0 and 1, the body handles the
    // remaining pipelined steady state, and the postamble drains the last
    // two in-flight iterations. All writes stay within `[ptr, ptr+256)`
    // and the schedule tiles the full stage exactly once. No stack use.
    core::arch::asm!(
        // Prologue: broadcast constants and set the iteration count.
        "dup     v28.8h, {zeta:w}",
        "dup     v29.8h, {zbar:w}",
        "mov     w9,     #3329",
        "dup     v30.8h, w9",
        "mov     x9,     #4",

        // Preamble — seeds the in-flight iterations.
        "ldp q6, q27, [{ptr}, #0]",
        "ldp q19, q3, [{ptr}, #128]",
        "sqrdmulh v7.8H, v3.8H, v29.8H",
        "sqrdmulh v4.8H, v19.8H, v29.8H",
        "mul v18.8H, v3.8H, v28.8H",
        "mul v24.8H, v19.8H, v28.8H",
        "mls v24.8H, v4.8H, v30.8H",
        "ldp q19, q4, [{ptr}, #160]",
        "mls v18.8H, v7.8H, v30.8H",
        "sub x9, x9, #2",

        // Steady-state body — 6 cy/iter (IPC 2.33) on the M4 model.
    "2:",
        "add v1.8H, v27.8H, v18.8H",
        "sub v22.8H, v27.8H, v18.8H",
        "mul v18.8H, v4.8H, v28.8H",
        "sqrdmulh v7.8H, v4.8H, v29.8H",
        "add v21.8H, v6.8H, v24.8H",
        "sub v17.8H, v6.8H, v24.8H",
        "ldp q6, q27, [{ptr}, #32]",
        "mul v24.8H, v19.8H, v28.8H",
        "sqrdmulh v3.8H, v19.8H, v29.8H",
        "ldp q19, q4, [{ptr}, #192]",
        "stp q17, q22, [{ptr}, #128]",
        "mls v18.8H, v7.8H, v30.8H",
        "mls v24.8H, v3.8H, v30.8H",
        "stp q21, q1, [{ptr}], #32",
        "sub x9, x9, 1",
        "cbnz x9, 2b",

        // Postamble — drains the in-flight iterations.
        "mul v7.8H, v4.8H, v28.8H",
        "sqrdmulh v26.8H, v4.8H, v29.8H",
        "ldp q17, q23, [{ptr}, #32]",
        "mul v4.8H, v19.8H, v28.8H",
        "sqrdmulh v20.8H, v19.8H, v29.8H",
        "add v22.8H, v27.8H, v18.8H",
        "sub v2.8H, v27.8H, v18.8H",
        "add v3.8H, v6.8H, v24.8H",
        "mls v7.8H, v26.8H, v30.8H",
        "mls v4.8H, v20.8H, v30.8H",
        "sub v0.8H, v6.8H, v24.8H",
        "stp q3, q22, [{ptr}], #32",
        "stp q0, q2, [{ptr}, #96]",
        "add v16.8H, v23.8H, v7.8H",
        "sub v5.8H, v23.8H, v7.8H",
        "add v25.8H, v17.8H, v4.8H",
        "sub v6.8H, v17.8H, v4.8H",
        "stp q6, q5, [{ptr}, #128]",
        "stp q25, q16, [{ptr}], #32",

        ptr  = inout(reg) ptr => _,
        zeta = in(reg) zeta as u32,
        zbar = in(reg) zeta_bar as u32,
        out("x9")  _,
        out("v0")  _, out("v1")  _, out("v2")  _, out("v3")  _,
        out("v4")  _, out("v5")  _, out("v6")  _, out("v7")  _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
        out("v28") _, out("v29") _, out("v30") _,
        options(nostack),
    );
}

/// Stride-128 forward-NTT stage, software-pipelined for the M4.
///
/// Runs all sixteen vector butterflies of the stride-128 stage in place:
///
///   - **Preamble** (9 instrs): kicks off iterations 0 and 1.
///   - **Steady-state body** (14 instrs, 6 cy/iter on the M4 model): each pass
///     finishes iteration N-2's `add` / `sub` / `stp` while starting iteration
///     N's loads and Barrett chain.
///   - **Postamble** (19 instrs): drains the final two in-flight iterations.
///
/// Pipelining across the loop back-edge buys ~49% over the previous
/// hand-written 2× interleave (13 → 6 cy/iter steady state); the hand
/// schedule was already optimal without it.
///
/// # Safety
///
/// `ptr` must point to the base of a 256-`i16` (`FieldElement::N`) buffer,
/// mutable for reads and writes. Clobbers `v0`-`v7` and `v16`-`v30` (all
/// caller-saved; the callee-saved `v8`-`v15` bank stays reserved so LLVM
/// does not emit a `stp d, d, [sp, #-N]!` prologue), general register `x9`,
/// and the input `ptr` (post-increments through the low half).
#[inline]
unsafe fn ntt_stride128_asm(ptr: *mut i16, zeta: i16, zeta_bar: i16) {
    // SAFETY: `ptr` is a 256-`i16` buffer; the schedule reads and writes
    // exactly `[ptr, ptr+512)` once — the preamble seeds iters 0/1, the loop
    // body advances `ptr` by 32 bytes per pass across the lower half, and the
    // postamble drains the last two iterations. No stack use beyond LLVM's
    // callee-save prologue for `v8`-`v15`. All working registers are declared
    // as clobbers; the `poly/arch/neon/tests.rs` differential proptest covers
    // correctness.
    core::arch::asm!(
        // Prologue: broadcast constants and set the iteration count.
        "dup     v28.8h, {zeta:w}",
        "dup     v29.8h, {zbar:w}",
        "mov     w9,     #3329",
        "dup     v30.8h, w9",
        "mov     x9,     #8",

        // Preamble — seeds the in-flight iterations.
        "ldp q16, q17, [{ptr}, #288]",
        "ldp q4, q25, [{ptr}, #256]",
        "mul v21.8H, v4.8H, v28.8H",
        "sqrdmulh v23.8H, v25.8H, v29.8H",
        "sqrdmulh v0.8H, v4.8H, v29.8H",
        "mul v26.8H, v25.8H, v28.8H",
        "mls v26.8H, v23.8H, v30.8H",
        "mls v21.8H, v0.8H, v30.8H",
        "ldp q0, q25, [{ptr}, #0]",
        "sub x9, x9, #2",

        // Steady-state body — 6 cy/iter (IPC 2.33) on the M4 model.
    "2:",
        "sqrdmulh v23.8H, v17.8H, v29.8H",
        "sub v18.8H, v25.8H, v26.8H",
        "add v4.8H, v25.8H, v26.8H",
        "mul v26.8H, v17.8H, v28.8H",
        "sqrdmulh v2.8H, v16.8H, v29.8H",
        "sub v5.8H, v0.8H, v21.8H",
        "add v22.8H, v0.8H, v21.8H",
        "mul v21.8H, v16.8H, v28.8H",
        "ldp q16, q17, [{ptr}, #320]",
        "ldp q0, q25, [{ptr}, #32]",
        "mls v26.8H, v23.8H, v30.8H",
        "mls v21.8H, v2.8H, v30.8H",
        "stp q5, q18, [{ptr}, #256]",
        "stp q22, q4, [{ptr}], #32",
        "sub x9, x9, 1",
        "cbnz x9, 2b",

        // Postamble — drains the in-flight iterations.
        "sqrdmulh v18.8H, v17.8H, v29.8H",
        "mul v23.8H, v17.8H, v28.8H",
        "sqrdmulh v19.8H, v16.8H, v29.8H",
        "mul v17.8H, v16.8H, v28.8H",
        "sub v7.8H, v0.8H, v21.8H",
        "add v21.8H, v0.8H, v21.8H",
        "ldp q0, q27, [{ptr}, #32]",
        "sub v24.8H, v25.8H, v26.8H",
        "mls v23.8H, v18.8H, v30.8H",
        "mls v17.8H, v19.8H, v30.8H",
        "add v3.8H, v25.8H, v26.8H",
        "stp q7, q24, [{ptr}, #256]",
        "stp q21, q3, [{ptr}], #32",
        "sub v6.8H, v27.8H, v23.8H",
        "add v1.8H, v27.8H, v23.8H",
        "add v20.8H, v0.8H, v17.8H",
        "sub v0.8H, v0.8H, v17.8H",
        "stp q0, q6, [{ptr}, #256]",
        "stp q20, q1, [{ptr}], #32",

        ptr  = inout(reg) ptr => _,
        zeta = in(reg) zeta as u32,
        zbar = in(reg) zeta_bar as u32,
        out("x9")  _,
        out("v0")  _, out("v1")  _, out("v2")  _, out("v3")  _,
        out("v4")  _, out("v5")  _, out("v6")  _, out("v7")  _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
        out("v28") _, out("v29") _, out("v30") _,
        options(nostack),
    );
}

/// Barrett-reduces eight `i16` lanes to a representative in `(-q/2, q/2]`: the
/// vector form of `FieldElement::reduce`.
#[inline]
fn barrett_reduce(a: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is baseline on aarch64; every intrinsic below is total.
    unsafe {
        let bias = vdupq_n_s32(1 << 25);
        let q = Q as i32;

        // t = (BARRETT_V * a + 2^25) >> 26 ; result = a - t*q, per i32 lane.
        let low = vmovl_s16(vget_low_s16(a));
        let t_low = vshrq_n_s32::<26>(vmlaq_n_s32(bias, low, BARRETT_V));
        let result_low = vmovn_s32(vsubq_s32(low, vmulq_n_s32(t_low, q)));

        let high = vmovl_high_s16(a);
        let t_high = vshrq_n_s32::<26>(vmlaq_n_s32(bias, high, BARRETT_V));
        let result_high = vmovn_s32(vsubq_s32(high, vmulq_n_s32(t_high, q)));

        vcombine_s16(result_low, result_high)
    }
}

/// Pointwise base multiplication of two NTT representations: 128 degree-one
/// products modulo the quadratics `X^2 - gamma`, computed eight pairs at a
/// time.
///
/// Matches [`super::generic::multiply`]. The result is scaled by `R^-1`
/// (Montgomery convention), which `ntt_inverse` later undoes.
///
/// `h` is stack-allocated `MaybeUninit`, not `[ZERO; N]`, so we skip the
/// 16-`stp` array wipe LLVM otherwise emits — it can't see through the
/// `unsafe` pointer stores to prove the loop overwrites every element.
pub(crate) fn multiply(
    f: &[FieldElement; parameters::N],
    g: &[FieldElement; parameters::N],
) -> [FieldElement; parameters::N] {
    let mut h = core::mem::MaybeUninit::<[FieldElement; parameters::N]>::uninit();

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the three
    // length-256 arrays and the length-128 `GAMMA_MONT` reinterpret as `[i16]`.
    // Each iteration reads a 16-`i16` window of `f`/`g` and writes a 16-`i16`
    // window of `h` (16 windows tile 256), and reads an 8-`i16` window of the
    // gammas (16 windows tile 128), all in bounds. The `vst2q` writes tile the
    // full 256-element `h`, so the `assume_init` at the end reads only
    // initialized bytes. `vld1q`/`vld2q`/`vst2q` are unaligned. NEON is
    // baseline on aarch64.
    unsafe {
        let f_ptr = f.as_ptr().cast::<i16>();
        let g_ptr = g.as_ptr().cast::<i16>();
        let h_ptr = h.as_mut_ptr().cast::<i16>();
        let gamma_ptr = super::GAMMA_MONT.as_ptr().cast::<i16>();

        let mut pair = 0;
        while pair < 128 {
            // Deinterleave eight pairs: `.0` is the even (degree-0) coefficient,
            // `.1` the odd (degree-1).
            let f_de = vld2q_s16(f_ptr.add(2 * pair));
            let g_de = vld2q_s16(g_ptr.add(2 * pair));
            let gamma = vld1q_s16(gamma_ptr.add(pair));

            let (a0, a1) = (f_de.0, f_de.1);
            let (b0, b1) = (g_de.0, g_de.1);

            // c0 = a0*b0 + a1*b1*gamma ; c1 = a0*b1 + a1*b0
            let c0 = vaddq_s16(fqmul(a0, b0), fqmul(fqmul(a1, b1), gamma));
            let c1 = vaddq_s16(fqmul(a0, b1), fqmul(a1, b0));

            vst2q_s16(h_ptr.add(2 * pair), int16x8x2_t(c0, c1));

            pair += 8;
        }

        h.assume_init()
    }
}

/// Accumulates one component of an asymmetric base-multiplication dot product
/// into `acc`: widening multiply-accumulates, no reduction.
///
/// Matches [`super::generic::basemul_accumulate`].
pub(crate) fn basemul_accumulate(
    acc: &mut super::ProductAccumulator,
    f: &[FieldElement; parameters::N],
    g: &[FieldElement; parameters::N],
    cache: &[FieldElement; parameters::N / 2],
) {
    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so `f`/`g`/
    // `cache` reinterpret as `[i16]`. Each iteration reads 16-`i16` windows of
    // `f`/`g` (16 windows tile 256), an 8-`i16` window of `cache`, and
    // read-modify-writes two 4-`i32` windows of each 128-`i32` accumulator
    // plane (pair and pair + 4, with pair + 8 <= 128), all in bounds.
    // `vld1q`/`vld2q`/`vst1q` are unaligned. NEON is baseline on aarch64.
    unsafe {
        let f_ptr = f.as_ptr().cast::<i16>();
        let g_ptr = g.as_ptr().cast::<i16>();
        let cache_ptr = cache.as_ptr().cast::<i16>();
        let even_ptr = acc.even.as_mut_ptr();
        let odd_ptr = acc.odd.as_mut_ptr();

        let mut pair = 0;
        while pair < 128 {
            let f_de = vld2q_s16(f_ptr.add(2 * pair));
            let g_de = vld2q_s16(g_ptr.add(2 * pair));
            let c = vld1q_s16(cache_ptr.add(pair));

            let (f0, f1) = (f_de.0, f_de.1);
            let (g0, g1) = (g_de.0, g_de.1);

            // even += f0*g0 + f1*cache, in i32 lanes.
            let mut even_lo = vld1q_s32(even_ptr.add(pair));
            let mut even_hi = vld1q_s32(even_ptr.add(pair + 4));
            even_lo = vmlal_s16(even_lo, vget_low_s16(f0), vget_low_s16(g0));
            even_hi = vmlal_high_s16(even_hi, f0, g0);
            even_lo = vmlal_s16(even_lo, vget_low_s16(f1), vget_low_s16(c));
            even_hi = vmlal_high_s16(even_hi, f1, c);
            vst1q_s32(even_ptr.add(pair), even_lo);
            vst1q_s32(even_ptr.add(pair + 4), even_hi);

            // odd += f0*g1 + f1*g0, in i32 lanes.
            let mut odd_lo = vld1q_s32(odd_ptr.add(pair));
            let mut odd_hi = vld1q_s32(odd_ptr.add(pair + 4));
            odd_lo = vmlal_s16(odd_lo, vget_low_s16(f0), vget_low_s16(g1));
            odd_hi = vmlal_high_s16(odd_hi, f0, g1);
            odd_lo = vmlal_s16(odd_lo, vget_low_s16(f1), vget_low_s16(g0));
            odd_hi = vmlal_high_s16(odd_hi, f1, g0);
            vst1q_s32(odd_ptr.add(pair), odd_lo);
            vst1q_s32(odd_ptr.add(pair + 4), odd_hi);

            pair += 8;
        }
    }
}

/// Montgomery-reduces the accumulated product sums to interleaved
/// coefficients, eight pairs at a time.
///
/// Matches [`super::generic::basemul_reduce`].
pub(crate) fn basemul_reduce(acc: &super::ProductAccumulator) -> [FieldElement; parameters::N] {
    let mut h = [FieldElement::ZERO; parameters::N];

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so `h`
    // reinterprets as `[i16]`. Each iteration reads two 4-`i32` windows of
    // each 128-`i32` accumulator plane and writes a 16-`i16` window of `h`
    // (16 windows tile 256), all in bounds. `vld1q`/`vst2q` are unaligned.
    // NEON is baseline on aarch64.
    unsafe {
        let even_ptr = acc.even.as_ptr();
        let odd_ptr = acc.odd.as_ptr();
        let h_ptr = h.as_mut_ptr().cast::<i16>();

        let q = vdup_n_s16(Q);
        let qinv = vdup_n_s16(QINV);

        let mut pair = 0;
        while pair < 128 {
            let c0 = vcombine_s16(
                montgomery_reduce(vld1q_s32(even_ptr.add(pair)), q, qinv),
                montgomery_reduce(vld1q_s32(even_ptr.add(pair + 4)), q, qinv),
            );
            let c1 = vcombine_s16(
                montgomery_reduce(vld1q_s32(odd_ptr.add(pair)), q, qinv),
                montgomery_reduce(vld1q_s32(odd_ptr.add(pair + 4)), q, qinv),
            );

            vst2q_s16(h_ptr.add(2 * pair), int16x8x2_t(c0, c1));

            pair += 8;
        }
    }

    h
}

/// Forward NTT in place, then Barrett-reduce. Vectorizes the stride-≥8
/// butterfly stages (eight butterflies per vector under one broadcast zeta);
/// the stride-4 and stride-2 stages and the final reduction run scalar.
///
/// Matches [`super::generic::ntt`].
// reason: scalar-tail butterfly/zeta indices are provably in 0..256, as in the
// generic backend.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt(coefficients: &mut [FieldElement; parameters::N]) {
    let mut k = 1;

    // SAFETY: `FieldElement` is `repr(transparent)` over `i16`, so the array
    // reinterprets as `[i16]`. Each vector window `[j, j + 8)` and
    // `[j + len, j + len + 8)` stays within the 256-element array (the loops
    // bound `j + len + 8 <= start + 2*len <= 256`), and `k < 32 < 128` indexes
    // `ZETA_RAW` / `ZETA_BARRETT`. `vld1q`/`vst1q` are unaligned. NEON is
    // baseline on aarch64.
    unsafe {
        let ptr = coefficients.as_mut_ptr().cast::<i16>();
        let zeta_raw_ptr = super::ZETA_RAW.as_ptr().cast::<i16>();
        let zeta_bar_ptr = super::ZETA_BARRETT.as_ptr();

        // Stride-128 stage: one zeta pair spans all 16 vector butterflies,
        // so the whole stage runs from a single hand-scheduled `asm!` block
        // that interleaves adjacent butterflies for the M1 vector-mul units.
        ntt_stride128_asm(ptr, *zeta_raw_ptr.add(k), *zeta_bar_ptr.add(k));
        k += 1;

        // Stride-64 stage: two independent butterfly groups (one per outer
        // start position, each with its own zeta pair). Same asm kernel as
        // stride-128 with a shorter loop and a `+128` upper offset.
        ntt_stride64_group_asm(ptr, *zeta_raw_ptr.add(k), *zeta_bar_ptr.add(k));
        k += 1;
        ntt_stride64_group_asm(ptr.add(128), *zeta_raw_ptr.add(k), *zeta_bar_ptr.add(k));
        k += 1;

        for len in [32usize, 16, 8] {
            let mut start = 0;
            while start < 256 {
                let zeta = vdupq_n_s16(*zeta_raw_ptr.add(k));
                let zeta_bar = vdupq_n_s16(*zeta_bar_ptr.add(k));
                k += 1;

                let mut j = start;
                while j < start + len {
                    let vj = vld1q_s16(ptr.add(j));
                    let vjl = vld1q_s16(ptr.add(j + len));
                    let t = barrett_const_mul(vjl, zeta, zeta_bar);

                    vst1q_s16(ptr.add(j + len), vsubq_s16(vj, t));
                    vst1q_s16(ptr.add(j), vaddq_s16(vj, t));

                    j += 8;
                }

                start += 2 * len;
            }
        }
    }

    // Scalar tail: stride-4 and stride-2 groups are narrower than a vector.
    for len in [4usize, 2] {
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
///
/// The stride-≥8 Gentleman-Sande stages run eight butterflies per vector, and
/// the final scale is vectorized; the narrower stride-4 and stride-2 stages run
/// scalar.
///
/// Matches [`super::generic::ntt_inverse`].
// reason: scalar-head butterfly/zeta indices are provably in 0..256, as in the
// generic backend.
#[allow(clippy::indexing_slicing)]
pub(crate) fn ntt_inverse(coefficients: &mut [FieldElement; parameters::N]) {
    let mut k = 127;

    // Scalar head: stride-2 and stride-4 groups are narrower than a vector.
    for len in [2usize, 4] {
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
    // reinterprets as `[i16]`. Each vector window `[j, j + 8)` and
    // `[j + len, j + len + 8)` stays within the 256-element array, and the
    // descending `k` indexes `ZETA_RAW` / `ZETA_BARRETT` (it reaches 0 only
    // after the last use). `vld1q`/`vst1q` are unaligned. NEON is baseline
    // on aarch64.
    unsafe {
        let ptr = coefficients.as_mut_ptr().cast::<i16>();
        let zeta_raw_ptr = super::ZETA_RAW.as_ptr().cast::<i16>();
        let zeta_bar_ptr = super::ZETA_BARRETT.as_ptr();

        for len in [8usize, 16, 32, 64, 128] {
            let mut start = 0;
            while start < 256 {
                let zeta = vdupq_n_s16(*zeta_raw_ptr.add(k));
                let zeta_bar = vdupq_n_s16(*zeta_bar_ptr.add(k));
                k -= 1;

                let mut j = start;
                while j < start + len {
                    let vj = vld1q_s16(ptr.add(j));
                    let vjl = vld1q_s16(ptr.add(j + len));

                    let sum = vaddq_s16(vj, vjl);
                    // Lazy reduction: len-16 only.
                    vst1q_s16(
                        ptr.add(j),
                        if len == 16 { barrett_reduce(sum) } else { sum },
                    );
                    vst1q_s16(
                        ptr.add(j + len),
                        barrett_const_mul(vsubq_s16(vjl, vj), zeta, zeta_bar),
                    );

                    j += 8;
                }

                start += 2 * len;
            }
        }

        // Final scale: every coefficient times `NTT_INVERSE_SCALE`.
        let scale = vdupq_n_s16(NTT_INVERSE_SCALE);
        let mut j = 0;
        while j < 256 {
            vst1q_s16(ptr.add(j), fqmul(scale, vld1q_s16(ptr.add(j))));
            j += 8;
        }
    }
}
