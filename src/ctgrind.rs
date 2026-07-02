//! Valgrind declassification hooks for the constant-time test harness.
//!
//! [`declassify`] marks a byte slice as "defined" for memcheck, erasing any
//! taint that would otherwise propagate from a secret input through the hash
//! or PRF that produced it. Use at the point where a value is public per
//! spec but memcheck sees it as tainted — e.g. `rho`, the first 32 bytes of
//! `G(d ‖ k)` in `K-PKE.KeyGen` — so `sample_ntt`'s rejection-sampling
//! branches don't read as secret-dependent leaks.
//!
//! Compiles to nothing outside `--features ctgrind,crabgrind` on Linux (the
//! Constant-time workflow); no production codegen impact.

/// Marks `bytes` as "defined" for Valgrind memcheck. A no-op unless built
/// under `--features ctgrind,crabgrind` on Linux.
//
// Skipped by cargo-mutants: the function is only observable through Valgrind
// under `--features ctgrind,crabgrind`, which the `cargo test` baseline does
// not enable, so mutants can freely replace its body without any test noticing.
#[cfg_attr(test, mutants::skip)]
#[inline]
pub(crate) fn declassify(_bytes: &[u8]) {
    #[cfg(all(feature = "crabgrind", target_os = "linux"))]
    {
        let _ = crabgrind::memcheck::mark_memory(
            _bytes.as_ptr() as *const core::ffi::c_void,
            _bytes.len(),
            crabgrind::memcheck::MemState::Defined,
        );
    }
}
