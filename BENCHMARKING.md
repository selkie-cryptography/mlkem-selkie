# Benchmarking

Build both binaries first, run them alternated in one sitting, compare
medians, and trust ordering consistency over magnitude.

## A/B Measurement

Builds the baseline in a temporary detached worktree (uncommitted changes
cannot leak into it), alternates the runs, discards warmup, and prints
per-op median deltas:

```sh
scripts/ab-bench.sh main                    # baseline main vs the working tree
scripts/ab-bench.sh main my-branch          # baseline main vs a ref
scripts/ab-bench.sh main '' MLKEM1024 5     # filter and round count
```

The manual procedure it encodes:

1. Build the baseline binary.

   ```sh
   git switch main
   cargo bench --bench mlkem --no-run   # prints the binary path
   cp target/release/deps/mlkem-<hash> /tmp/bench_a
   ```

2. Build the candidate binary the same way on your branch.

   ```sh
   git switch my-change
   cargo bench --bench mlkem --no-run
   cp target/release/deps/mlkem-<hash> /tmp/bench_b
   ```

3. Confirm the binaries differ.

   ```sh
   md5 /tmp/bench_a /tmp/bench_b
   ```

   Identical checksums mean the A/B measures nothing. The usual cause:
   uncommitted changes ride along on `git switch`, so "main" built with the
   candidate code. Park work with a temporary WIP commit first. Do not use
   `git stash`: the stash stack is shared across worktrees.

4. Run alternated, in one terminal session, back to back.

   ```sh
   for run in a b b a a b; do
     /tmp/bench_$run --bench MLKEM768
   done
   ```

   `--bench` is required when invoking the binary directly; without it,
   divan runs in test mode and prints no timings. The mirrored pattern
   (`a b b a a b`, not `a b a b`) shows each binary both the early and the
   late thermal state, so machine drift cancels instead of biasing one side.

5. Read the median column. Discard each binary's first run as warmup; it
   routinely runs 5-10% slow. Means are polluted by outliers, and "fastest"
   rewards lucky runs.

6. Decide by ordering consistency. A real effect shows the same winner in
   every stable round. If the winner flips between rounds, it is noise: run
   more rounds or accept "no measurable difference". A single wild outlier
   means background load; close things and rerun rather than averaging it in.

## Rules learned the hard way

- Never rebuild between timing runs. Rebuild-per-round interleaving produced
  contradictory verdicts twice on the same machine before this document
  existed. Build everything, then only run.
- Compare like with like. Wall-clock numbers from different harnesses (divan
  vs a hand-written C loop) carry different overheads; treat cross-harness
  deltas under ~5% as unresolved.
- Per-byte closures and branchy collectors can defeat memcpy codegen in ways
  that only show up under measurement. Staging into one contiguous buffer
  and collecting from it has repeatedly measured faster than "clever"
  branch-per-index alternatives. Measure; do not reason your way to a
  verdict.

## Profiling

1. Pick the operation with the divan per-operation benchmarks. `keygen_derand`,
   `encaps_derand`, `decaps`, `parse_*`, and `serialize_*` isolate each
   phase, so a gap against a competitor or an expectation already names the
   neighborhood before any profiler runs.

2. Profile that operation with the repo driver:

   ```sh
   rustc -O scripts/profile.rs -o /tmp/mlkem-profile
   /tmp/mlkem-profile samply decaps 20000    # CPU profile -> Firefox Profiler
   /tmp/mlkem-profile dhat keygen            # heap profile -> dhat-heap.json
   /tmp/mlkem-profile flamegraph decaps      # flamegraph.svg
   ```

   `mode` is `keygen | encaps | decaps | all`; the number is iterations.

3. Read the inverted call tree (self time per function), not the top-down
   view. Things that have mattered here:

   - `memmove`/`memcpy` self time is data movement: staging buffers,
     serialization, typed-value placement. Compare its share against the
     arithmetic.
   - Iterator adapters (`from_fn`, `flat_map`, `chain` frames) with real
     self time mean a per-byte chain did not compile to block copies.
   - Keccak share is the floor set by hashing; arithmetic wins cannot
     shrink it.
   - Attribution error is real: a hot leaf often bills its cost to an
     inlined caller. Treat percentages as pointers, not verdicts.

4. For a single hot loop (not where time goes, but why it is slow),
   `scripts/mca.rs` runs llvm-mca over its release asm and reports
   throughput bottlenecks and port pressure.

samply runs on macOS, Linux, and Windows; install with `cargo install
samply` (likewise `cargo install cargo-instruments flamegraph` for the
alternatives below). Native alternatives when samply does not fit:

- macOS: Instruments, via `cargo instruments -t time` (cargo-instruments).
- Linux: `perf record --call-graph dwarf` then `perf report`, or
  cargo-flamegraph for a one-shot flame graph.
- Windows: Windows Performance Analyzer over an ETW trace.

## CI benchmarks

CI runs divan wall-time, gungraun instruction counts, and rdtsc megacycles
per backend on every PR, with a PR comment for deltas against the base.
Instruction counts are stable across runners; wall-time on shared runners is
indicative only. A local A/B on quiet hardware outranks both.
