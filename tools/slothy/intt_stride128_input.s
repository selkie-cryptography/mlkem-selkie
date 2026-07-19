///
/// Stride-128 inverse-NTT (Gentleman-Sande) butterfly stage.
///
/// The last vectorised stage of ntt_inverse. Runs 16 vector butterflies at
/// [ptr, ptr+256) paired with [ptr+256, ptr+512), all sharing one zeta pair.
///
/// Per butterfly:
///   sum  = vj + vjl
///   diff = vjl - vj
///   out_lo = barrett_reduce(sum)                              (12 vec ops)
///   out_hi = barrett_const_mul(diff, zeta, zeta_bar)          (3 vec ops)
///
/// 2-butterfly interleave over 8 iterations. Loop body advances ptr by 32
/// bytes per pass; upper half stays at +256.
///
///   x0  = ptr to 512-byte window
///   w1  = zeta
///   w2  = zeta_bar
///   w3  = BARRETT_V (20159, fits in u16)
///
/// Constant registers set by prologue:
///   v28 = zeta broadcast (i16 x 8)
///   v29 = zeta_bar broadcast (i16 x 8)
///   v30 = Q (i16 x 8) — used by barrett_const_mul's mls
///   v25 = BARRETT_V broadcast (i32 x 4) — barrett_reduce mla input
///   v26 = bias 1 << 25 (i32 x 4) — barrett_reduce mla accumulator seed
///   v24 = Q (i32 x 4) — barrett_reduce mls input
///

.text
.global intt_stride128
.type intt_stride128, %function
intt_stride128:
        dup     v28.8h, w1
        dup     v29.8h, w2
        mov     w9,     #3329
        dup     v30.8h, w9

        dup     v24.4s, w9                     // Q as i32
        movi    v26.4s, #2, lsl #24            // bias = 2 << 24 = 2^25
        dup     v25.4s, w3                     // BARRETT_V as i32

        mov     x9,     #8

intt128_start:
        ldp     q0,  q1,  [x0, #0]             // vj_a,  vj_b
        ldp     q2,  q3,  [x0, #256]           // vjl_a, vjl_b

        // sum = vj + vjl ; diff = vjl - vj
        add     v4.8h, v0.8h, v2.8h            // sum_a
        sub     v6.8h, v2.8h, v0.8h            // diff_a
        add     v5.8h, v1.8h, v3.8h            // sum_b
        sub     v7.8h, v3.8h, v1.8h            // diff_b

        // barrett_reduce(sum_a) -> v4
        sxtl    v16.4s, v4.4h
        sxtl2   v17.4s, v4.8h
        mov     v18.16b, v26.16b
        mov     v19.16b, v26.16b
        mla     v18.4s, v16.4s, v25.4s
        mla     v19.4s, v17.4s, v25.4s
        sshr    v18.4s, v18.4s, #26
        sshr    v19.4s, v19.4s, #26
        mls     v16.4s, v18.4s, v24.4s
        mls     v17.4s, v19.4s, v24.4s
        xtn     v4.4h, v16.4s
        xtn2    v4.8h, v17.4s

        // barrett_reduce(sum_b) -> v5
        sxtl    v16.4s, v5.4h
        sxtl2   v17.4s, v5.8h
        mov     v18.16b, v26.16b
        mov     v19.16b, v26.16b
        mla     v18.4s, v16.4s, v25.4s
        mla     v19.4s, v17.4s, v25.4s
        sshr    v18.4s, v18.4s, #26
        sshr    v19.4s, v19.4s, #26
        mls     v16.4s, v18.4s, v24.4s
        mls     v17.4s, v19.4s, v24.4s
        xtn     v5.4h, v16.4s
        xtn2    v5.8h, v17.4s

        // barrett_const_mul(diff_a, zeta, zbar) -> v6
        mul     v20.8h, v6.8h, v28.8h          // t   = diff * zeta
        sqrdmulh v21.8h, v6.8h, v29.8h         // s   = qrdmulh(diff, zbar)
        mls     v20.8h, v21.8h, v30.8h         // t  -= s * Q
        mov     v6.16b, v20.16b

        // barrett_const_mul(diff_b, zeta, zbar) -> v7
        mul     v22.8h, v7.8h, v28.8h
        sqrdmulh v23.8h, v7.8h, v29.8h
        mls     v22.8h, v23.8h, v30.8h
        mov     v7.16b, v22.16b

        stp     q4, q5, [x0], #32              // out_lo pair, post-increment
        stp     q6, q7, [x0, #224]             // out_hi pair at (base+256-32)

        sub     x9, x9, #1
        cbnz    x9, intt128_start

        ret
