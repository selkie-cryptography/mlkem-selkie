//! Profiling workload for ML-KEM-768 keygen / encaps / decaps.
//!
//! Built under `[profile.profiling]` (release + debuginfo) and driven by
//! `scripts/profile.rs` for flamegraphs, samply, and heap profiling.
//!
//! ```text
//! cargo run --profile profiling --example profile -- all 5000      # CPU profile
//! cargo run --profile profiling --features dhat-heap --example profile -- all
//! ```
//!
//! Modes: `keygen`, `encaps`, `decaps`, `all` (default). The second argument is
//! the iteration count (forced to 1 under `dhat-heap`, since dhat reports
//! totals).

use std::hint::black_box;

use mlkem_selkie::{KeyPair, MLKEM768};

// dhat replaces the global allocator only when explicitly profiling the heap,
// so CPU/time profiles keep the real allocator and its costs.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// A fixed `d ‖ z` key-generation seed.
const SEED: [u8; 64] = [0x42; 64];

/// A fixed encapsulation message.
const MESSAGE: [u8; 32] = [0x17; 32];

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map_or("all", String::as_str);
    let iters = if cfg!(feature = "dhat-heap") {
        1
    } else {
        args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000)
    };

    for _ in 0..iters {
        match mode {
            "keygen" => {
                black_box(KeyPair::<MLKEM768>::generate_derand(black_box(&SEED)));
            }
            "encaps" => {
                let keypair = KeyPair::<MLKEM768>::generate_derand(&SEED);
                black_box(
                    keypair
                        .encapsulation_key
                        .encapsulate_derand(black_box(&MESSAGE)),
                );
            }
            "decaps" => {
                let keypair = KeyPair::<MLKEM768>::generate_derand(&SEED);
                let (_, ciphertext) = keypair.encapsulation_key.encapsulate_derand(&MESSAGE);
                black_box(
                    keypair
                        .decapsulation_key
                        .decapsulate(black_box(&ciphertext)),
                );
            }
            _ => {
                let keypair = KeyPair::<MLKEM768>::generate_derand(black_box(&SEED));
                let (_, ciphertext) = keypair
                    .encapsulation_key
                    .encapsulate_derand(black_box(&MESSAGE));
                black_box(
                    keypair
                        .decapsulation_key
                        .decapsulate(black_box(&ciphertext)),
                );
            }
        }
    }
}
