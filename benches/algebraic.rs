//! Wall-clock benchmarks for the internal hot paths (NTT, sampling, encoding).
//!
//! `cargo bench --bench algebraic --features expose-internals`.

use divan::{Bencher, black_box};
use mlkem_selkie::{
    Eta,
    algebraic::{FieldElement, PolynomialRingElement, RqElement, TqElement},
    functions::XOF,
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

/// `SamplePolyCBD_eta` at the largest noise parameter (eta = 3).
#[divan::bench]
fn sample_poly_cbd(bencher: Bencher<'_, '_>) {
    let bytes = vec![0xA5u8; 64 * 3];

    bencher.bench(|| RqElement::sample_cbd(Eta::Three, black_box(&bytes)));
}

/// `SampleNTT`: rejection sampling a uniform Tq element from a SHAKE128 stream.
#[divan::bench]
fn sample_ntt(bencher: Bencher<'_, '_>) {
    bencher
        .with_inputs(|| XOF(&[7u8; 32], 0, 0))
        .bench_refs(TqElement::sample_ntt);
}

/// `ByteEncode_12` of an NTT-domain polynomial.
#[divan::bench]
fn byte_encode(bencher: Bencher<'_, '_>) {
    let f = sample_poly().ntt();

    bencher.bench(|| black_box(f).byte_encode());
}

/// `ByteDecode_12` back into an NTT-domain polynomial.
#[divan::bench]
fn byte_decode(bencher: Bencher<'_, '_>) {
    let bytes = sample_poly().ntt().byte_encode();

    bencher.bench(|| TqElement::byte_decode(black_box(&bytes)));
}
