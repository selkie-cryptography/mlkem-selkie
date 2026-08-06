#!/usr/bin/env bash
# A/B benchmark driver: builds two bench binaries up front, runs them
# alternated in a mirrored order, discards the warmup round, and compares
# per-benchmark medians.
#
# The baseline builds in a temporary detached worktree, so uncommitted
# changes in the working tree never leak into it.
#
# Usage:  scripts/ab-bench.sh <baseline-ref> [<candidate-ref>] [<filter>] [<rounds>]
#   baseline-ref   git ref for side A (e.g. main)
#   candidate-ref  git ref for side B; default: the working tree as-is
#   filter         divan filter, default MLKEM768
#   rounds         timing rounds per side after warmup, default 3

set -euo pipefail

baseline_ref=${1:?usage: scripts/ab-bench.sh <baseline-ref> [<candidate-ref>] [<filter>] [<rounds>]}
candidate_ref=${2:-}
filter=${3:-MLKEM768}
rounds=${4:-3}

repo=$(git rev-parse --show-toplevel)
workdir=$(mktemp -d)
cleanup() {
  git worktree remove --force "$workdir/baseline" 2>/dev/null || true
  git worktree remove --force "$workdir/candidate" 2>/dev/null || true
  rm -rf "$workdir"
}
trap cleanup EXIT

# Builds the mlkem bench binary in $1 and copies it to $2.
build_bench() {
  local dir=$1 out=$2 path
  path=$( (cd "$dir" && cargo bench --bench mlkem --no-run 2>&1) |
    grep -oE 'target/release/deps/mlkem-[a-f0-9]+' | head -1)
  [ -n "$path" ] || { echo "error: no bench binary path from cargo in $dir" >&2; exit 1; }
  cp "$dir/$path" "$out"
}

echo "building baseline ($baseline_ref)..."
git worktree add --quiet --detach "$workdir/baseline" "$baseline_ref"
build_bench "$workdir/baseline" "$workdir/bench_a"

if [ -n "$candidate_ref" ]; then
  echo "building candidate ($candidate_ref)..."
  git worktree add --quiet --detach "$workdir/candidate" "$candidate_ref"
  build_bench "$workdir/candidate" "$workdir/bench_b"
else
  echo "building candidate (working tree)..."
  build_bench "$repo" "$workdir/bench_b"
fi

if cmp -s "$workdir/bench_a" "$workdir/bench_b"; then
  echo "error: baseline and candidate binaries are identical; the A/B measures nothing" >&2
  exit 1
fi

# Warmup round for both sides, then mirrored timing rounds (a b | b a a b ...)
# so each binary sees both the early and the late thermal state.
order=(a b)
for ((r = 1; r <= rounds; r++)); do
  if ((r % 2)); then order+=(b a); else order+=(a b); fi
done

log="$workdir/ab.log"
for ((i = 0; i < ${#order[@]}; i++)); do
  side=${order[$i]}
  round=$((i / 2))
  echo "round $round side $side..."
  # Group names live on their own `├─ name` lines; each result row carries
  # the columns fastest │ slowest │ median │ mean │ samples │ iters, so the
  # median is always the fourth field from the end.
  "$workdir/bench_$side" --bench "$filter" 2>&1 |
    awk -v side="$side" -v round="$round" -F '│' '
      /^(├─ |╰─ )/ { split($1, parts, "─ "); group = parts[2]; sub(/ +$/, "", group) }
      NF >= 5 && /MLKEM/ {
        median = $(NF - 3)
        gsub(/^ +| +$/, "", median)
        split(median, m, " ")
        ns = m[1]
        if (m[2] == "µs") ns *= 1000
        else if (m[2] == "ms") ns *= 1000000
        else if (m[2] == "s") ns *= 1000000000
        if (round > 0) print side, group, ns
      }' >>"$log"
done

echo
awk '
  { samples[$1 "," $2] = samples[$1 "," $2] " " $3; groups[$2] = 1 }
  END {
    printf "%-28s %12s %12s %9s\n", "benchmark", "baseline", "candidate", "delta"
    for (g in groups) {
      a = med(samples["a," g]); b = med(samples["b," g])
      if (a == "" || b == "") continue
      printf "%-28s %10.0f ns %10.0f ns %+8.1f%%\n", g, a, b, 100 * (b - a) / a
    }
  }
  function med(list, n, arr, i, j, tmp) {
    n = split(list, arr, " ")
    for (i = 2; i <= n; i++)
      for (j = i; j > 1 && arr[j - 1] + 0 > arr[j] + 0; j--) {
        tmp = arr[j]; arr[j] = arr[j - 1]; arr[j - 1] = tmp
      }
    return n ? arr[int((n + 1) / 2)] : ""
  }' "$log" | sort
echo
echo "deltas are medians of $rounds rounds; trust ordering consistency over magnitude"
