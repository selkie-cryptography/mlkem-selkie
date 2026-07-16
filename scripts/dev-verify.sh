#!/usr/bin/env bash
# Local pre-push checks that mirror CI. Runs on the developer's host, catches
# what would fail CI without waiting for the round trip.
#
# Usage:  scripts/dev-verify.sh                        # default: fast checks (~30s)
#         DEV_VERIFY_X86_RUN=1  scripts/dev-verify.sh  # + AVX2 tests under rosetta
#         DEV_VERIFY_FEATURES=1 scripts/dev-verify.sh  # + feature-combo build matrix
#         DEV_VERIFY_VECTORS=1  scripts/dev-verify.sh  # + KAT / wycheproof / xtest vectors
#         DEV_VERIFY_ALL=1      scripts/dev-verify.sh  # all optional gates on
#
# One-time setup for x86 cross-check on aarch64 macOS hosts:
#   rustup target add x86_64-apple-darwin

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ "${DEV_VERIFY_ALL:-0}" = "1" ]; then
    DEV_VERIFY_X86_RUN=1
    DEV_VERIFY_FEATURES=1
    DEV_VERIFY_VECTORS=1
fi

step() { printf '\n=== %s ===\n' "$1"; }

step "cargo +nightly fmt --check"
cargo +nightly fmt --all --check

step "cargo clippy --all-targets (lib + tests + benches)"
cargo clippy --all-targets --all-features -- -D warnings

step "cargo doc --document-private-items --all-features (deny warnings)"
RUSTDOCFLAGS="-D warnings" \
    cargo doc --no-deps --document-private-items --all-features

step "cargo bench --no-run (bench compile)"
cargo bench --no-run --features expose-internals

step "cargo test --doc (doctests)"
cargo test --doc --features expose-internals

step "cargo test --features expose-internals (lib + integration, dev profile)"
cargo test --features expose-internals

step "cargo test --profile release-checked --features expose-internals (overflow-checks on)"
cargo test --profile release-checked --features expose-internals

# Cross-arch AVX2 compile check on aarch64 macOS.
if [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] \
        && rustup target list --installed | grep -q x86_64-apple-darwin; then
    step "cargo check --target x86_64-apple-darwin (AVX2 backend)"
    RUSTFLAGS="-C target-cpu=x86-64-v3" \
        cargo check --target x86_64-apple-darwin --all-targets --all-features
fi

# Optional: AVX2 test suite under rosetta.
if [ "${DEV_VERIFY_X86_RUN:-0}" = "1" ]; then
    if rustup target list --installed | grep -q x86_64-apple-darwin; then
        step "cargo test --target x86_64-apple-darwin (AVX2 backend, rosetta)"
        RUSTFLAGS="-C target-cpu=x86-64-v3" \
            cargo test --target x86_64-apple-darwin --features expose-internals
    else
        echo
        echo "note: DEV_VERIFY_X86_RUN=1 set but x86_64-apple-darwin target is not installed."
        echo "      run: rustup target add x86_64-apple-darwin"
    fi
fi

# Optional: minimum-feature and per-parameter-set builds. Mirrors CI's
# "Feature combinations build" step.
if [ "${DEV_VERIFY_FEATURES:-0}" = "1" ]; then
    step "cargo build --no-default-features (minimum feature set)"
    cargo build --no-default-features

    for set in "mlkem512" "mlkem768" "mlkem1024" \
               "mlkem768,expose-internals" "mlkem768,fips"; do
        step "cargo build --no-default-features --features $set"
        cargo build --no-default-features --features "$set"
    done
fi

# Optional: KAT / wycheproof / cross-implementation vectors. Slow (~1min);
# CI runs these under a dedicated `vectors` profile.
if [ "${DEV_VERIFY_VECTORS:-0}" = "1" ]; then
    step "cargo test --profile release-checked --features expose-internals --test kats"
    cargo test --profile release-checked --features expose-internals --test kats
    step "cargo test --profile release-checked --features expose-internals --test wycheproof"
    cargo test --profile release-checked --features expose-internals --test wycheproof
fi

echo
echo "All dev checks passed."
