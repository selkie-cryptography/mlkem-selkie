//! rdtsc cycle benchmark: keygen / encaps / decaps in CPU cycles.
//!
//! x86_64-only. Cycle counts come from `rdtsc` via Criterion's
//! [`criterion_cycles_per_byte::CyclesPerByte`] measurement. There is no
//! unprivileged cycle-counter read on aarch64, so on every non-x86_64 target
//! this bench compiles to a skip stub.
//!
//! Deterministic off fixed seeds, so the cycle counts are reproducible run to
//! run. `cargo bench --bench cycles`.

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::hint::black_box;

    use criterion::{Criterion, criterion_group};
    use criterion_cycles_per_byte::CyclesPerByte;
    use mlkem_selkie::{KeyPair, MLKEM768};

    /// A fixed `d ‖ z` key-generation seed.
    const SEED: [u8; 64] = [0x42; 64];

    /// A fixed encapsulation message.
    const MESSAGE: [u8; 32] = [0x17; 32];

    /// Benchmarks keygen, encaps, and decaps under the rdtsc measurement.
    fn ops(c: &mut Criterion<CyclesPerByte>) {
        let mut group = c.benchmark_group("mlkem768");
        group.sample_size(50);

        group.bench_function("keygen", |b| {
            b.iter(|| KeyPair::<MLKEM768>::generate_derand(black_box(&SEED)))
        });

        let keypair = KeyPair::<MLKEM768>::generate_derand(&SEED);
        group.bench_function("encaps", |b| {
            b.iter(|| {
                keypair
                    .encapsulation_key
                    .encapsulate_derand(black_box(&MESSAGE))
            })
        });

        let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&MESSAGE);
        group.bench_function("decaps", |b| {
            b.iter(|| {
                keypair
                    .decapsulation_key
                    .decapsulate(black_box(&ciphertext))
            })
        });

        group.finish();
    }

    criterion_group!(
        name = benches;
        config = Criterion::default().with_measurement(CyclesPerByte);
        targets = ops
    );

    /// Runs the Criterion group (it configures itself from the bench CLI args).
    pub fn run() {
        benches();
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    imp::run();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("cycles bench: x86_64-only (rdtsc); skipped on this target");
}
