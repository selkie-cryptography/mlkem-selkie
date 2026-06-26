//! Emits the `mlkem_selkie_arch` cfg that selects the vectorized NTT backend.
//!
//! `neon` on aarch64 (NEON is baseline on the architecture) and `avx2` on
//! x86_64 when the target features include it (raise the baseline with
//! `-C target-cpu=...` in `.cargo/config.toml`). Absent the cfg, the portable
//! scalar backend in `src/algebraic/poly/arch/generic.rs` is used.
//!
//! The cfg is a reserved selector: it is registered and emitted here, while the
//! dispatch in `src/algebraic/poly/arch.rs` switches the active kernels onto it
//! as each vectorized backend lands.

use std::env;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(mlkem_selkie_arch, values(\"neon\", \"avx2\"))");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has_feature = |name: &str| target_features.split(',').any(|feature| feature == name);

    match target_arch.as_str() {
        "aarch64" if has_feature("neon") => {
            println!("cargo::rustc-cfg=mlkem_selkie_arch=\"neon\"");
        }
        "x86_64" if has_feature("avx2") => {
            println!("cargo::rustc-cfg=mlkem_selkie_arch=\"avx2\"");
        }
        _ => {}
    }
}
