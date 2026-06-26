#!/usr/bin/env bash
#
# Compile the BoringSSL ML-KEM-768 interop oracle against a prebuilt BoringSSL.
#
# BoringSSL is not vendored or fetched here (it needs cmake + go to build);
# point this script at an existing checkout that has been CMake-built:
#
#   git clone https://boringssl.googlesource.com/boringssl
#   cmake -S boringssl -B boringssl/build -DCMAKE_BUILD_TYPE=Release && \
#       cmake --build boringssl/build --target crypto
#
# Then:
#   ORACLE=$(tools/boringssl_xtest/build.sh)
#   MLKEM_BSSL_ORACLE="$ORACLE" cargo test --test boringssl_xtest -- --ignored
#
# Env overrides:
#   BORINGSSL_ROOT  — checkout root (default: ~/src/github.com/google/boringssl)
#   BORINGSSL_BUILD — CMake build dir (default: $BORINGSSL_ROOT/build)
#
# The only stdout line is the path to the built oracle, so callers can do
# ORACLE=$(...). Diagnostics go to stderr.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${BORINGSSL_ROOT:-$HOME/src/github.com/google/boringssl}"
build="${BORINGSSL_BUILD:-$root/build}"
out="$here/oracle"

if [[ ! -d "$root/include/openssl" ]]; then
    echo "error: BoringSSL headers not found at $root/include/openssl" >&2
    echo "       set BORINGSSL_ROOT to your BoringSSL checkout" >&2
    exit 1
fi

# BoringSSL's static crypto archive lands in different places across versions.
libcrypto=""
for cand in \
    "$build/crypto/libcrypto.a" \
    "$build/libcrypto.a" \
    "$build/crypto/libcrypto.a"; do
    if [[ -f "$cand" ]]; then
        libcrypto="$cand"
        break
    fi
done
if [[ -z "$libcrypto" ]]; then
    echo "error: libcrypto.a not found under $build; CMake-build the 'crypto' target first" >&2
    exit 1
fi

cc -O2 -std=c11 -Wall -Wextra \
    -I"$root/include" \
    "$here/oracle.c" -o "$out" \
    "$libcrypto" -lpthread

echo "$out"
