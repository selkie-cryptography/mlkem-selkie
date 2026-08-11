# mlkem-selkie [![](https://buildstats.info/crate/mlkem-selkie)](https://crates.io/crates/mlkem-selkie) [![](https://img.shields.io/docsrs/mlkem-selkie)](https://docs.rs/mlkem-selkie) [![CI](https://github.com/selkie-cryptography/mlkem-selkie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/selkie-cryptography/mlkem-selkie/actions/workflows/ci.yml)

Rust ML-KEM (FIPS 203) implementation for beautiful, secure code.

### Example

Each parameter set has its own module of type aliases — `mlkem512`, `mlkem768`,
and `mlkem1024` — so the generic `DecapsulationKey<MLKEM768>` reads as `mlkem768::DecapsulationKey`:

```rust
use mlkem_selkie::mlkem768;

// Generate an ML-KEM-768 decapsulation key from OS entropy (`getrandom`).
let decaps_key = mlkem768::DecapsulationKey::generate();

// The sender borrows the corresponding encapsulation key and encapsulates
// a fresh shared secret; the ciphertext travels to the decapsulator.
let (sender_secret, ciphertext) = decaps_key.encapsulation_key().encapsulate();

// The holder of the decapsulation key recovers the same shared secret.
let receiver_secret = decaps_key.decapsulate(&ciphertext);

assert_eq!(sender_secret.as_bytes(), receiver_secret.as_bytes());
```

Keys and ciphertexts serialize with `to_bytes` / `as_bytes` and parse back with
`from_bytes`, which applies the FIPS 203 §7 input validation.

### Zeroization

`DecapsulationKey`, `DecryptionKey`, `RejectionSeed`, `KeyGenRandomnessSeed`,
`SharedSecret`, `Aes256CtrDrbg`, and the `RqVector`/`TqVector` secret-noise
containers derive `ZeroizeOnDrop`; the named decaps/keygen stack transients
are zeroized explicitly. Best-effort: the compiler may spill bytes to
slots we cannot reach, and a copy of `*SharedSecret::as_bytes()` into a
plain `[u8; 32]` does not inherit the zeroization.

### Constant-time

Every operation on secret-derived data runs in constant time. Field
arithmetic uses a branchless Barrett mul-shift for reduction and a
branch-free conditional add for canonicalization; `ML-KEM.Decaps`'s
implicit-rejection compare and shared-secret selection use `subtle`'s
`ct_eq` and `conditional_select` over the full ciphertext bytes rather
than a plain `==`, so neither the equality result nor which secret is
returned leaks through a branch or an early exit.

Two CT harnesses — **tacet** and **ctgrind** — run in CI on every push
to `main` and fail the build on regression.
`SampleNTT`'s rejection-sampling loop is variable-time by construction,
but its iteration count depends only on the public matrix seed `rho`,
not on any secret.

### Backends

Vectorized backends are selected at compile time from the target's
features — no cargo features, no runtime detection. aarch64 gets the
NEON polynomial backend unconditionally (NEON is baseline); x86_64 gets
the AVX2 backend when the target features include it, e.g.
`-C target-cpu=x86-64-v3`. The Keccak backends inside
[`sha3-selkie`](https://github.com/selkie-cryptography/sha3-selkie)
select the same way: the Arm SHA-3 extension on Apple silicon (baseline
there), AVX2 on x86_64, and a batched scalar/NEON hybrid kernel on
other aarch64. On non-Apple aarch64 with `FEAT_SHA3` (Graviton3 and
later), opt in with `RUSTFLAGS="-C target-feature=+sha3"`.

### Backend override

`MLKEM_SELKIE_BACKEND` overrides the automatic selection, for A/B
benchmarking and testing the portable backend on SIMD hardware:

```sh
# force the portable scalar backend
MLKEM_SELKIE_BACKEND=scalar cargo bench

# also keep the Keccak in sha3-selkie scalar
MLKEM_SELKIE_BACKEND=scalar SHA3_SELKIE_BACKEND=scalar cargo bench

# fail the build unless a SIMD backend is selected, guarding bench
# scripts against silently measuring the scalar backend
MLKEM_SELKIE_BACKEND=simd cargo bench
```

Unset (or empty) selects automatically. Changing the variable triggers
a rebuild.

### About

<img width="27%" align="right" src="https://user-images.githubusercontent.com/552961/197638905-f5144be3-a2f2-48c2-9ecb-26e4e34d8d8a.svg#gh-light-mode-only"/>
<img width="27%" align="right" src="https://user-images.githubusercontent.com/552961/197640007-f3f05dd1-c61c-4c16-bd04-d1813937ad47.svg#gh-dark-mode-only"/>


*"In very ancient times some of the Clan Coneely, one of the early septs of the county, were changed
by “art magick” into seals; since then no Coneely can kill a seal without afterwards having bad
luck." - Connemara Folk-Lore*

