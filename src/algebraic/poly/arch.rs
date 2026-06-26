//! Architecture-dispatched NTT and base-multiplication kernels.
//!
//! Every backend exposes the same three kernels — [`ntt`], [`ntt_inverse`], and
//! [`multiply`] — operating on the 256-coefficient array, so
//! [`super::RqElement`] and [`super::TqElement`] stay architecture-agnostic.
//! The active backend is chosen at compile time from the `mlkem_selkie_arch`
//! cfg that `build.rs` emits; absent it, the portable scalar [`generic`]
//! backend is used.
//!
//! The vectorized `neon`/`avx2` backends are not yet implemented, so every
//! target currently resolves to [`generic`]. As each lands, a
//! `#[cfg(mlkem_selkie_arch = "...")]` arm will re-export its kernels in place
//! of the `generic` re-export below.

mod generic;

pub(crate) use generic::{multiply, ntt, ntt_inverse};
