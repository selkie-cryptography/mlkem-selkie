//! Architecture-dispatched polynomial kernels.
//!
//! Every backend exposes the same three kernels — [`ntt`], [`ntt_inverse`], and
//! [`multiply`] — operating on the 256-coefficient array, so
//! [`super::RqElement`] and [`super::TqElement`] stay architecture-agnostic.
//! The active backend is chosen at compile time from the `mlkem_selkie_arch`
//! cfg that `build.rs` emits; absent it, the portable scalar [`generic`]
//! backend is used.
//!
//! The shared zeta tables live here so every backend reads the same constants.
//! [`ZETA_RAW`] holds the canonical values from FIPS 203 Appendix A;
//! [`ZETA_BARRETT`] pairs each with `round(zeta · 2^15 / q)`, consumed by the
//! Barrett-with-constant butterflies in every backend. [`GAMMA_MONT`] holds
//! the base-multiplication tweaks in Montgomery form.

use crate::algebraic::field::FieldElement;

#[cfg(mlkem_selkie_arch = "avx2")]
mod avx2;
mod generic;
#[cfg(mlkem_selkie_arch = "neon")]
mod neon;

#[cfg(mlkem_selkie_arch = "avx2")]
pub(crate) use avx2::{multiply, ntt, ntt_inverse};
#[cfg(not(any(mlkem_selkie_arch = "neon", mlkem_selkie_arch = "avx2")))]
pub(crate) use generic::{multiply, ntt, ntt_inverse};
#[cfg(mlkem_selkie_arch = "neon")]
pub(crate) use neon::{multiply, ntt, ntt_inverse};

/// The canonical values `ζ^BitRev7(i) mod q` for `i` in `{0, ..., 127}`,
/// [FIPS 203 Appendix A].
///
/// [FIPS 203 Appendix A]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#appendix.A
pub(super) const ZETA_RAW: [u16; 128] = [
    1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 3260, 569, 1746, 296,
    2447, 1339, 1476, 3046, 56, 2240, 1333, 1426, 2094, 535, 2882, 2393, 2879, 1974, 821, 289, 331,
    3253, 1756, 1197, 2304, 2277, 2055, 650, 1977, 2513, 632, 2865, 33, 1320, 1915, 2319, 1435,
    807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648, 2474, 3110, 1227, 910, 17, 2761, 583,
    2649, 1637, 723, 2288, 1100, 1409, 2662, 3281, 233, 756, 2156, 3015, 3050, 1703, 1651, 2789,
    1789, 1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641, 1584, 2298, 2037,
    3220, 375, 2549, 2090, 1645, 1063, 319, 2773, 757, 2099, 561, 2466, 2594, 2804, 1092, 403,
    1026, 1143, 2150, 2775, 886, 1722, 1212, 1874, 1029, 2110, 2935, 885, 2154,
];

/// Barrett multipliers `round(zeta * 2^15 / q)` paired with [`ZETA_RAW`],
/// consumed by the SIMD backends' Barrett-with-constant NTT butterflies
/// (`neon::barrett_const_mul` via `vqrdmulhq_s16`,
/// `avx2::barrett_const_mul` via `_mm256_mulhrs_epi16`). The `2^15` (not
/// `2^16`) pre-divides by 2 to compensate for the doubling in both
/// instructions.
pub(super) const ZETA_BARRETT: [i16; 128] = {
    let q = crate::parameters::Q as i32;
    let mut table = [0i16; 128];
    let mut i = 0;
    while i < 128 {
        #[allow(clippy::indexing_slicing)] // reason: i < 128 by the loop guard.
        {
            let zeta = ZETA_RAW[i] as i32;
            table[i] = ((zeta * (1 << 15) + q / 2) / q) as i16;
        }
        i += 1;
    }
    table
};

/// The Montgomery-form values `ζ^(2 BitRev7(i) + 1) * R mod q` for `i` in
/// `{0, ..., 127}`, derived from the canonical [FIPS 203 Appendix A] table (the
/// modular reduction applied, matching BoringSSL).
///
/// [FIPS 203 Appendix A]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf#appendix.A
const GAMMA_MONT: [FieldElement; 128] = FieldElement::montgomery_table([
    17, 3312, 2761, 568, 583, 2746, 2649, 680, 1637, 1692, 723, 2606, 2288, 1041, 1100, 2229, 1409,
    1920, 2662, 667, 3281, 48, 233, 3096, 756, 2573, 2156, 1173, 3015, 314, 3050, 279, 1703, 1626,
    1651, 1678, 2789, 540, 1789, 1540, 1847, 1482, 952, 2377, 1461, 1868, 2687, 642, 939, 2390,
    2308, 1021, 2437, 892, 2388, 941, 733, 2596, 2337, 992, 268, 3061, 641, 2688, 1584, 1745, 2298,
    1031, 2037, 1292, 3220, 109, 375, 2954, 2549, 780, 2090, 1239, 1645, 1684, 1063, 2266, 319,
    3010, 2773, 556, 757, 2572, 2099, 1230, 561, 2768, 2466, 863, 2594, 735, 2804, 525, 1092, 2237,
    403, 2926, 1026, 2303, 1143, 2186, 2150, 1179, 2775, 554, 886, 2443, 1722, 1607, 1212, 2117,
    1874, 1455, 1029, 2300, 2110, 1219, 2935, 394, 885, 2444, 2154, 1175,
]);
