//! Emits the cfgs that select the vectorized NTT backend and its tuning.
//!
//! Three tiers on aarch64:
//!
//! - `mlkem_selkie_arch = "neon"` — any little-endian aarch64 (NEON is baseline
//!   on the architecture): the vectorized intrinsics backend.
//! - `mlkem_selkie_neon_asm` — Apple cores: additionally enables the
//!   software-pipelined `asm!` kernels, which are scheduled for Apple's wide
//!   NEON pipes and regress on narrower cores (measured on Graviton-class CI
//!   runners).
//! - `mlkem_selkie_neon_tune = "apple-m<N>"` (or `"apple"` when the exact core
//!   is unknown) — which Apple core the build targets, so kernels can
//!   specialize per-core as schedules accumulate.
//!
//! The tune is resolved from, in priority order: the `MLKEM_SELKIE_NEON_TUNE`
//! env var (`generic` disables the asm tier entirely — useful for running the
//! intrinsics path's tests on an Apple host), a `-C target-cpu=apple-m<N>` in
//! `RUSTFLAGS`, then the target vendor.
//!
//! On x86_64, `mlkem_selkie_arch = "avx2"` when the target features include
//! AVX2 (raise the baseline with `-C target-cpu=...` in
//! `.cargo/config.toml`). Absent any arch cfg, the portable scalar backend in
//! `src/algebraic/poly/arch/generic.rs` is used.
//!
//! The cfgs are reserved selectors: they are registered and emitted here,
//! while the dispatch in `src/algebraic/poly/arch.rs` (and the stage
//! selection in `src/algebraic/poly/arch/neon.rs`) switches the active
//! kernels onto them.

use std::env;

/// Resolves the Apple-core tune tier: `None` for the plain intrinsics
/// backend, or the `mlkem_selkie_neon_tune` value to emit.
///
/// # Panics
///
/// On an unrecognized `MLKEM_SELKIE_NEON_TUNE` value, so a typo fails the
/// build instead of silently selecting the wrong kernels.
fn neon_tune(target_vendor: &str) -> Option<String> {
    if let Ok(tune) = env::var("MLKEM_SELKIE_NEON_TUNE") {
        return match tune.as_str() {
            "generic" => None,
            "apple" => Some("apple".to_string()),
            other
                if other
                    .strip_prefix("apple-m")
                    .is_some_and(|n| n.parse::<u32>().is_ok()) =>
            {
                Some(other.to_string())
            }
            other => panic!(
                "MLKEM_SELKIE_NEON_TUNE must be `generic`, `apple`, or `apple-m<N>`, got `{other}`"
            ),
        };
    }

    if let Some(cpu) = rustflags_target_cpu() {
        if cpu
            .strip_prefix("apple-m")
            .is_some_and(|n| n.parse::<u32>().is_ok())
        {
            return Some(cpu);
        }
    }

    (target_vendor == "apple").then(|| "apple".to_string())
}

/// The last `-C target-cpu=` value in the build's `RUSTFLAGS`, if any,
/// handling both the fused (`-Ctarget-cpu=x`) and split (`-C target-cpu=x`)
/// spellings of the 0x1F-separated `CARGO_ENCODED_RUSTFLAGS` encoding.
fn rustflags_target_cpu() -> Option<String> {
    let encoded = env::var("CARGO_ENCODED_RUSTFLAGS").ok()?;
    let mut cpu = None;

    let mut flags = encoded.split('\u{1f}').peekable();
    while let Some(flag) = flags.next() {
        let value = if flag == "-C" {
            flags.peek().copied()
        } else {
            flag.strip_prefix("-C")
        };
        if let Some(rest) = value.and_then(|v| v.strip_prefix("target-cpu=")) {
            cpu = Some(rest.to_string());
        }
    }

    cpu
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(mlkem_selkie_arch, values(\"neon\", \"avx2\"))");
    println!("cargo::rustc-check-cfg=cfg(mlkem_selkie_neon_asm)");
    println!("cargo::rustc-check-cfg=cfg(mlkem_selkie_neon_tune, values(any()))");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ENDIAN");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_VENDOR");
    println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo::rerun-if-env-changed=MLKEM_SELKIE_NEON_TUNE");

    match std::env::var("CARGO_CFG_MLKEM_SELKIE_BACKEND").as_deref() {
        Ok("serial") => {} // Set no flags
        Ok("simd") => select_backend(true),
        Ok(e) => panic!("Unknown `mlkem_selkie_backend` value `{e}`"),
        _ => select_backend(false),
    };
}

/// Reads the target arch info and sets the appropriate config flags. If `expect_simd` is
/// set, then some sort of SIMD backend must be chosen, otherwise it panics.
fn select_backend(expect_simd: bool) {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has_feature = |name: &str| target_features.split(',').any(|feature| feature == name);

    // The NEON kernels reinterpret vectors between lane widths, which assumes
    // little-endian lane layout; on `aarch64_be` they would silently
    // miscompute, so big-endian targets keep the scalar backend. x86_64 has
    // no big-endian variant.
    let little_endian = env::var("CARGO_CFG_TARGET_ENDIAN").as_deref() == Ok("little");

    match target_arch.as_str() {
        "aarch64" if has_feature("neon") && little_endian => {
            println!("cargo::rustc-cfg=mlkem_selkie_arch=\"neon\"");

            if let Some(tune) = neon_tune(&target_vendor) {
                println!("cargo::rustc-cfg=mlkem_selkie_neon_asm");
                println!("cargo::rustc-cfg=mlkem_selkie_neon_tune=\"{tune}\"");
            }
        }
        "x86_64" if has_feature("avx2") => {
            println!("cargo::rustc-cfg=mlkem_selkie_arch=\"avx2\"");
        }
        _ => {
            if expect_simd {
                panic!(
                    "`mlkem_selkies_backend=\"simd\"` used on an arch with no supported SIMD backend"
                )
            } else {
                // Nothing. We just use scalar
            }
        }
    }
}
