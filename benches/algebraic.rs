//! Wall-clock benchmarks for the internal hot paths (NTT, sampling, encoding).
//!
//! `cargo bench --bench algebraic --features expose-internals`.

use divan::{Bencher, black_box};
use mlkem_selkie::{
    algebraic::{
        CachedTqVector, FieldElement, PolynomialRingElement, RqElement, TqElement, TqVector,
    },
    parameters::{Eta, MLKEM768},
};

fn main() {
    divan::main();
}

/// A deterministic, non-trivial Rq element to transform.
fn sample_poly() -> RqElement {
    RqElement::new(core::array::from_fn(|i| {
        FieldElement::new(7 * i as u16 + 1)
    }))
}

/// `NTT`: standard domain to NTT domain.
#[divan::bench]
fn ntt(bencher: Bencher<'_, '_>) {
    let f = sample_poly();

    bencher.bench(|| black_box(f).ntt());
}

/// `NTT⁻¹`: NTT domain back to standard domain.
#[divan::bench]
fn ntt_inverse(bencher: Bencher<'_, '_>) {
    let f = sample_poly().ntt();

    bencher.bench(|| black_box(f).ntt_inverse());
}

/// `MultiplyNTTs`: pointwise product of two NTT-domain polynomials.
#[divan::bench]
fn multiply(bencher: Bencher<'_, '_>) {
    let a = sample_poly().ntt();
    let b = sample_poly().ntt();

    bencher.bench(|| black_box(a) * black_box(b));
}

/// Asymmetric base-multiplication cache of one NTT-domain polynomial.
#[divan::bench]
fn mul_cache(bencher: Bencher<'_, '_>) {
    let g = sample_poly().ntt();

    bencher.bench(|| black_box(&g).mul_cache());
}

/// Cached, accumulated dot product of two `K = 3` vectors (the hot inner
/// product of encryption and decryption at ML-KEM-768).
#[divan::bench]
fn accumulated_dot(bencher: Bencher<'_, '_>) {
    let f = TqVector::<MLKEM768>::from_fn(|_| sample_poly().ntt());
    let g = CachedTqVector::from(TqVector::<MLKEM768>::from_fn(|_| sample_poly().ntt()));

    bencher.bench(|| black_box(&f) * black_box(&g));
}

/// `SamplePolyCBD_eta` at the largest noise parameter (eta = 3).
#[divan::bench]
fn sample_poly_cbd(bencher: Bencher<'_, '_>) {
    let bytes = vec![0xA5u8; 64 * 3];

    bencher.bench(|| RqElement::sample_cbd(Eta::Three, black_box(&bytes)));
}

/// `SampleNTT`: rejection sampling a uniform Tq element from a SHAKE128 stream.
#[divan::bench]
fn sample_ntt(bencher: Bencher<'_, '_>) {
    bencher.bench(|| TqElement::sample_ntt(black_box(&[7u8; 32]), 0, 0));
}

/// `ByteEncode_12` of an NTT-domain polynomial.
#[divan::bench]
fn byte_encode(bencher: Bencher<'_, '_>) {
    let f = sample_poly().ntt();

    bencher.bench(|| black_box(f).byte_encode().collect::<Vec<u8>>());
}

/// `ByteDecode_12` back into an NTT-domain polynomial.
#[divan::bench]
fn byte_decode(bencher: Bencher<'_, '_>) {
    let bytes: Vec<u8> = sample_poly().ntt().byte_encode().collect();

    bencher.bench(|| TqElement::byte_decode(black_box(&bytes)));
}
