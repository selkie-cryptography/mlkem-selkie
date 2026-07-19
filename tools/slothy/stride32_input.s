///
/// Stride-32 forward-NTT butterfly group, 2-butterfly interleave.
///
/// Called four times per forward NTT (once per outer start position) with
/// its own zeta pair over 4 vector butterflies. Upper half at +64 bytes.
/// The loop runs 2 iterations of 2 butterflies each.
///
///     x0  = ptr to 128-byte window (in-place, post-increments through lower)
///     w1  = zeta                (as u32; low 16 bits used)
///     w2  = zeta_bar            (as u32; low 16 bits used)
///

.text
.global ntt_stride32_group
.type ntt_stride32_group, %function
ntt_stride32_group:
        dup     v28.8h, w1
        dup     v29.8h, w2
        mov     w9,     #3329
        dup     v30.8h, w9
        mov     x9,     #2

stride32_start:
        ldp     q0,  q1,  [x0, #64]
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
        stp     q6,  q7,  [x0, #32]
        sub     x9, x9, #1
        cbnz    x9, stride32_start

        ret
