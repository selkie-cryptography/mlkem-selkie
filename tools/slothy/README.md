# SLOTHY integration workspace

Off-tree artifacts driving the SW-pipelined `asm!` blocks in
`src/algebraic/poly/arch/neon.rs`. Nothing in this directory is compiled or
shipped in the crate — it's the input/output of the SLOTHY runs plus the
model patch that upstream needs before those runs can happen against an
Apple M4 target.

## Files

- `apple-m4-model.patch` — adds `Apple_M4_everest_experimental` to upstream
  [SLOTHY](https://github.com/slothy-optimizer/slothy). Fork of
  `Apple_M1_firestorm_experimental` with `issue_rate = 8 → 10` (M4 P-core
  widened decode, per Chips-and-Cheese) and paired-Q `ldp`/`stp` coverage
  added (a genuine gap in upstream — the shipped M1 model only knew about
  single-Q `Ldr_Q`/`Str_Q`). Per-opcode latencies are still at Firestorm
  values and need real M4 silicon measurement before upstream will merge.

- `stride128_input.s` — the extracted stride-128 forward-NTT kernel with
  our hand-written 2-butterfly interleave, ready to feed SLOTHY. Loop
  labelled `stride128_start`.

- `stride128_output_m4.s` — SLOTHY's SW-pipelined output for `stride128_input.s`
  against the `Apple_M4_everest_experimental` model. This is what got
  transcribed into `ntt_stride128_asm` in `neon.rs`.

## Reproducing

```
git clone https://github.com/slothy-optimizer/slothy.git /tmp/slothy
cd /tmp/slothy && git apply /path/to/slothy/apple-m4-model.patch
python3 -m venv /tmp/slothy-venv
/tmp/slothy-venv/bin/pip install -e . ortools==9.15.6755 sympy==1.14.0

/tmp/slothy-venv/bin/python3 /tmp/slothy/slothy-cli \
    Arm_AArch64 Apple_M4_everest_experimental \
    stride128_input.s \
    -c sw_pipelining.enabled=true \
    -c inputs_are_outputs \
    -c sw_pipelining.allow_post=true \
    -c variable_size \
    -c constraints.stalls_first_attempt=8 \
    -c "reserved_regs=[v8,v9,v10,v11,v12,v13,v14,v15]" \
    -l stride128_start \
    -o stride128_output_m4.s
```

## Head-to-head

`SLOTHY-M4` model, stride-128 forward-NTT stage, 8 iterations:

| schedule                    | cy/iter (steady) | stage total |
|-----------------------------|:----------------:|:-----------:|
| hand-written 2× interleave  |       13         |     104     |
| SLOTHY-M4, no SW pipelining |       13         |     104     |
| SLOTHY-M4, SW pipelining    |    **6**         |   **~53**   |

The hand schedule was already SLOTHY-optimal at the same problem shape.
The remaining ~49% comes entirely from software pipelining across the loop
back-edge — the transformation you can't do inside a single `asm!` block
without a three-phase preamble/body/postamble restructure.

`llvm-mca` against the shipped `apple-m1` model reads this schedule
differently (+6 cy on whole-NTT vs. the hand schedule) because mca does not
credit SW pipelining the way SLOTHY's constraint solver does. Real M4 or
M1 hardware PMU measurement resolves the disagreement; both are
approximations.

## Stages we tried and skipped

- `stride32_input.s` / `stride32_output_m4.s` — 4 outer groups × 4
  butterflies each with distinct zetas. The 2-iteration loop is too
  short for SW pipelining to amortize (SLOTHY folds all iters into
  preamble+postamble, no steady state), and 4 separate `asm!` calls each
  replay the 3-instruction `dup zeta/zbar/Q` prologue. Total cost
  exceeds the intrinsic baseline. **Not wired into `neon.rs`.**

- `intt_stride128_input.s` — the biggest remaining vectorized target
  (inverse NTT stride-128 with `barrett_reduce` + `barrett_const_mul`
  per butterfly). Blocked on the SLOTHY aarch64 parser: `sxtl` / `xtn2`
  / `mov v.dt, v.dt` (same-dt copy) / `movi v.dt, #x, lsl #y` / the
  scalar-broadcast `mla.4s` form all produce `FatalParsingException:
  Inconsistent dt: <dt1>`. Partial parser additions live in
  `apple-m4-model.patch` (search for `class vxtn2`, `class vsxtl`,
  `class vsshr`, `class vmov_reg`, `class vmovi_shift`) but need the
  datatype-constraint tables filled in to be sound. Real semantic-layer
  work — worth doing to unlock inverse NTT, out of scope for a
  drive-by session.

## Stages worth trying next, in order

1. **Whole forward NTT via extracted compiled asm.** Compile `RqElement::
   ntt`, pull the compiled asm between prologue and epilogue with
   `cargo asm`, wrap it as a single SLOTHY region, and let SLOTHY
   schedule the whole thing (including our existing stride-128/-64
   `asm!` blocks and the intrinsic stride-32/-16/-8 tails). This is the
   mlkem-native workflow — they run SLOTHY on `layer123`/`layer4567`
   fused regions. Bigger scheduling window = bigger wins.
2. **`ntt_inverse` after parser fix.** Same shape as forward NTT plus
   the Barrett reduce, once the parser blocker above is resolved.
3. **`multiply` (base pointwise mul).** Uses `fqmul` (Montgomery
   reduction), already SLOTHY-friendly opcodes.
