///
/// Extracted from src/algebraic/poly/arch/neon.rs::ntt_stride128_asm
///
/// Hand-scheduled 2-butterfly interleave over 8 iterations. Prologue sets up
/// zeta / zeta_bar / Q broadcasts and the iteration counter. The loop body is
/// what SLOTHY optimises.
///
///     x0  = ptr to coefficients (in-place, post-increments through lower half)
///     w1  = zeta                (as u32; low 16 bits used)
///     w2  = zeta_bar            (as u32; low 16 bits used)
///     v28 = zeta broadcast      (established by prologue)
///     v29 = zeta_bar broadcast
///     v30 = Q broadcast
///     x9  = loop counter
///

.text
.global ntt_stride128
.type ntt_stride128, %function
ntt_stride128:
        dup     v28.8h, w1
        dup     v29.8h, w2
        mov     w9,     #3329
        dup     v30.8h, w9
        mov     x9,     #8

stride128_start:
        ldp     q0,  q1,  [x0, #256]
        ldp     q18, q19, [x0, #0]
        mul       v2.8h, v0.8h, v28.8h
        mul       v3.8h, v1.8h, v28.8h
        sqrdmulh  v4.8h, v0.8h, v29.8h
        sqrdmulh  v5.8h, v1.8h, v29.8h
        mls       v2.8h, v4.8h, v30.8h
        mls       v3.8h, v5.8h, v30.8h
        sub     v6.8h,  v18.8h, v2.8h
        add     v18.8h, v18.8h, v2.8h
        sub     v7.8h,  v19.8h, v3.8h
        add     v19.8h, v19.8h, v3.8h
        stp     q18, q19, [x0], #32
        stp     q6,  q7,  [x0, #224]
        sub     x9, x9, #1
        cbnz    x9, stride128_start

        ret
