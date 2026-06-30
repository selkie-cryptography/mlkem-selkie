# mlkem-selkie [![](https://buildstats.info/crate/mlkem-selkie)](https://crates.io/crates/mlkem-selkie) [![](https://img.shields.io/docsrs/mlkem-selkie)](https://docs.rs/mlkem-selkie) [![CI](https://github.com/selkie-cryptography/mlkem-selkie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/selkie-cryptography/mlkem-selkie/actions/workflows/ci.yml)

Rust ML-KEM (FIPS 203) implementation for beautiful, secure code.

### Example

Each parameter set has its own module of type aliases — `mlkem512`, `mlkem768`,
and `mlkem1024` — so the generic `KeyPair<MLKEM768>` reads as `mlkem768::KeyPair`:

```rust
use mlkem_selkie::mlkem768;
use rand::rngs::OsRng;

// Generate an ML-KEM-768 key pair.
let mlkem768::KeyPair {
    encapsulation_key: encaps_key,
    decapsulation_key: decaps_key,
} = mlkem768::KeyPair::generate(&mut OsRng);

// The sender encapsulates a fresh shared secret under the encapsulation key,
// and transmits the ciphertext.
let (sender_secret, ciphertext) = encaps_key.encapsulate(&mut OsRng);

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
(`m'`, `r'`, `K'`, `K_bar`, `g_input`, `sigma`, the seeds) are zeroized
explicitly. Best-effort: the compiler may spill bytes to slots we cannot
reach, and a copy of `*SharedSecret::as_bytes()` into a plain `[u8; 32]`
does not inherit the zeroization.

### About

<img width="27%" align="right" src="https://user-images.githubusercontent.com/552961/197638905-f5144be3-a2f2-48c2-9ecb-26e4e34d8d8a.svg#gh-light-mode-only"/>
<img width="27%" align="right" src="https://user-images.githubusercontent.com/552961/197640007-f3f05dd1-c61c-4c16-bd04-d1813937ad47.svg#gh-dark-mode-only"/>


*"In very ancient times some of the Clan Coneely, one of the early septs of the county, were changed
by “art magick” into seals; since then no Coneely can kill a seal without afterwards having bad
luck." - Connemara Folk-Lore*

