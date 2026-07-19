//! Microarchitectural analysis driver for `mlkem-selkie` — the closest thing to
//! SLOTHY-style feedback we can get without owning the register file.
//!
//! Emits the release asm for a chosen kernel (via `cargo-show-asm`, which drives
//! `rustc --emit=asm` and demangles the result) and feeds it to `llvm-mca`
//! against the vendor scheduling model for the target CPU: Apple's own model
//! ships with LLVM for `apple-m1`/`m2`/`m3`, the Skylake/Cascadelake/Zen models
//! for x86_64. `llvm-mca` reports the port-pressure / latency / throughput split
//! and the critical dependency chain; you then iterate the intrinsics to
//! relieve whatever it flags.
//!
//! ```text
//! rustc -O scripts/mca.rs -o /tmp/mlkem-mca
//! /tmp/mlkem-mca ntt                    # apple-m1 by default; ntt kernel
//! /tmp/mlkem-mca ntt_inverse apple-m2
//! /tmp/mlkem-mca mul znver3             # AVX2 backend on Zen 3
//! /tmp/mlkem-mca ntt cascadelake        # AVX2 backend on Skylake-Server
//! /tmp/mlkem-mca list                   # list known kernels and CPUs
//! /tmp/mlkem-mca asm ntt apple-m1       # just dump the demangled asm
//! /tmp/mlkem-mca ntt apple-m1 --timeline  # full pipeline timeline (large)
//! ```
//!
//! `--iterations=<N>` and any other `llvm-mca` flag pass through unchanged
//! after the kernel + CPU: `mca ntt apple-m1 --all-stats --iterations=1000`.
//!
//! The scoring is a whole-function analysis (prologue + hot loops + epilogue).
//! `llvm-mca`'s region markers (`# LLVM-MCA-BEGIN` / `# LLVM-MCA-END`) narrow it
//! to a single loop body: run `mca asm <kernel> <cpu> > kernel.s`, hand-annotate
//! the hot butterfly stage, and rerun `llvm-mca -mcpu=<cpu> kernel.s`.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::{exit, Command, Stdio},
};

/// Kernel selectors → (demangled symbol, one-line description).
///
/// Kept short and pointed at the arithmetic hot path: the NTT round-trip and
/// pointwise multiply exercise every intrinsic in the arch backend, and are
/// what any perf regression here shows up in first.
fn kernels() -> BTreeMap<&'static str, (&'static str, &'static str)> {
    BTreeMap::from([
        (
            "ntt",
            (
                "mlkem_selkie::algebraic::poly::RqElement::ntt",
                "Forward NTT (Rq → Tq), Cooley-Tukey butterflies with a final Barrett reduce",
            ),
        ),
        (
            "ntt_inverse",
            (
                "mlkem_selkie::algebraic::poly::TqElement::ntt_inverse",
                "Inverse NTT (Tq → Rq), Gentleman-Sande butterflies + f=1441 scale",
            ),
        ),
        (
            "mul",
            (
                "<mlkem_selkie::algebraic::poly::TqElement as core::ops::arith::Mul>::mul",
                "Pointwise base multiply of two NTT reprs (128 degree-1 products mod X^2-γ)",
            ),
        ),
        (
            "sample_cbd",
            (
                "mlkem_selkie::sampling::<impl mlkem_selkie::algebraic::poly::RqElement>::sample_cbd",
                "SampleCBD_η: word-parallel centered-binomial sampling from PRF output",
            ),
        ),
        (
            "sample_ntt_x4",
            (
                "mlkem_selkie::sampling::<impl mlkem_selkie::algebraic::poly::TqElement>::sample_ntt_x4",
                "SampleNTT ×4: rejection-sample 4 Tq streams (public, variable-time)",
            ),
        ),
    ])
}

/// CPU selectors → (rust target triple, `-C target-cpu`, `llvm-mca -mcpu`).
///
/// The rust triple picks the register file and calling convention; the
/// `target-cpu` unlocks NEON / AVX2 in codegen; the `llvm-mca` mcpu is the
/// scheduling model. Keep the three consistent — a mismatch (say, Zen 3 asm
/// scored against Skylake) still runs but the port pressure numbers are noise.
fn cpus() -> BTreeMap<&'static str, (&'static str, &'static str, &'static str)> {
    BTreeMap::from([
        // aarch64 → NEON backend. Apple ships scheduling models in LLVM for
        // every M-series core; earlier ones map to the equivalent A-series.
        ("apple-m1", ("aarch64-apple-darwin", "apple-m1", "apple-m1")),
        ("apple-m2", ("aarch64-apple-darwin", "apple-m2", "apple-m2")),
        ("apple-m3", ("aarch64-apple-darwin", "apple-m3", "apple-m3")),
        // x86_64 → AVX2 backend. Skylake / Cascadelake / Ice Lake share the
        // client Skylake model; Zen 3+ have their own.
        (
            "skylake",
            ("x86_64-unknown-linux-gnu", "skylake", "skylake"),
        ),
        (
            "cascadelake",
            ("x86_64-unknown-linux-gnu", "cascadelake", "cascadelake"),
        ),
        (
            "znver3",
            ("x86_64-unknown-linux-gnu", "znver3", "znver3"),
        ),
        (
            "znver4",
            ("x86_64-unknown-linux-gnu", "znver4", "znver4"),
        ),
    ])
}

/// The default llvm-mca flags: keep the summary + bottleneck breakdown
/// (resource pressure vs data deps vs latency) and the critical dep chain, and
/// skip the per-cycle timeline unless the caller opts in with `--timeline`.
///
/// Iterations = 1 keeps `llvm-mca`'s "Instructions:" honest (one full pass of
/// the kernel), so the per-line resource numbers add up to something you can
/// reason about; scaling up mostly changes the Block-RThroughput asymptote.
const DEFAULT_MCA_ARGS: &[&str] = &[
    "-bottleneck-analysis",
    "-resource-pressure=true",
    "-iterations=1",
];

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();

    let (command, rest) = raw
        .split_first()
        .map_or(("help", &[][..]), |(c, r)| (c.as_str(), r));

    match command {
        "help" | "-h" | "--help" => {
            print_help();
        }
        "list" => {
            list();
        }
        "asm" => {
            let (kernel, cpu, extra) = parse_target(rest);
            let (kernel_sym, _) = resolve_kernel(kernel);
            let (triple, target_cpu, _) = resolve_cpu(cpu);
            run_asm(&crate_root(), triple, target_cpu, kernel_sym, extra);
        }
        _ => {
            // command is the kernel name.
            let (kernel, cpu, extra) = parse_target(&raw);
            let (kernel_sym, _) = resolve_kernel(kernel);
            let (triple, target_cpu, mcpu) = resolve_cpu(cpu);
            run_mca(&crate_root(), triple, target_cpu, mcpu, kernel_sym, extra);
        }
    }
}

/// Splits `[kernel, cpu, extra...]` into its three pieces, defaulting the
/// second slot to `apple-m1` when omitted.
fn parse_target<'a>(args: &'a [String]) -> (&'a str, &'a str, Vec<&'a str>) {
    let kernel = args.first().map_or("ntt", String::as_str);
    let cpu = args.get(1).map_or("apple-m1", String::as_str);
    let extra: Vec<&str> = args.iter().skip(2).map(String::as_str).collect();

    (kernel, cpu, extra)
}

/// Runs `cargo-show-asm` in `--mca` mode: emit release asm for `symbol` under
/// the given target/cpu, hand it to `llvm-mca` for `mcpu`, and stream the
/// analysis. Extra args pass through as `--mca-arg=...` to `llvm-mca`; a
/// bare `--timeline` shorthand toggles the (large) per-cycle timeline on.
fn run_mca(
    root: &Path,
    triple: &str,
    target_cpu: &str,
    mcpu: &str,
    symbol: &str,
    extra: Vec<&str>,
) {
    let mut cargo = cargo(root);
    cargo.args([
        "asm",
        "--lib",
        "--target",
        triple,
        "--target-cpu",
        target_cpu,
        "--features",
        "expose-internals",
        "--mca",
        "--simplify",
    ]);

    for arg in DEFAULT_MCA_ARGS {
        cargo.arg(format!("--mca-arg={arg}"));
    }
    cargo.arg(format!("--mca-arg=-mcpu={mcpu}"));

    for arg in extra {
        if arg == "--timeline" {
            cargo.arg("--mca-arg=-timeline");
        } else if arg.starts_with("--iterations=") {
            let n = &arg["--iterations=".len()..];
            cargo.arg(format!("--mca-arg=-iterations={n}"));
        } else if let Some(rest) = arg.strip_prefix("--mca=") {
            cargo.arg(format!("--mca-arg={rest}"));
        } else {
            cargo.arg(format!("--mca-arg={arg}"));
        }
    }

    cargo.arg(symbol);

    // The two rustflags that keep the vectorized backend selected on x86_64
    // (aarch64's NEON is baseline). Applied narrowly to only the x86_64 triple.
    if triple.starts_with("x86_64") {
        cargo.env(
            "RUSTFLAGS",
            format!("-C target-cpu={target_cpu} -C target-feature=+avx2"),
        );
    }

    eprintln!(
        "$ cargo asm --lib --target {triple} --target-cpu {target_cpu} --features expose-internals \\\n\
         \t--mca [-mcpu={mcpu}] {symbol}"
    );

    exec(cargo);
}

/// Runs `cargo-show-asm` in `--asm` mode: emit the demangled release asm to
/// stdout so it can be redirected to a file and hand-annotated with
/// `# LLVM-MCA-BEGIN` / `# LLVM-MCA-END` markers around a specific loop body,
/// then fed to `llvm-mca` directly for a scoped analysis.
fn run_asm(root: &Path, triple: &str, target_cpu: &str, symbol: &str, extra: Vec<&str>) {
    let mut cargo = cargo(root);
    cargo.args([
        "asm",
        "--lib",
        "--target",
        triple,
        "--target-cpu",
        target_cpu,
        "--features",
        "expose-internals",
        "--asm",
        "--simplify",
    ]);
    for arg in extra {
        cargo.arg(arg);
    }
    cargo.arg(symbol);

    if triple.starts_with("x86_64") {
        cargo.env(
            "RUSTFLAGS",
            format!("-C target-cpu={target_cpu} -C target-feature=+avx2"),
        );
    }

    exec(cargo);
}

/// Prints the kernel and CPU tables (also the docstring at the top of `--help`).
fn list() {
    println!("kernels:");
    for (name, (sym, desc)) in kernels() {
        println!("  {name:<16} {desc}");
        println!("  {:16} → {sym}", "");
    }
    println!("\nCPU targets  (`triple` / `-C target-cpu` / `llvm-mca -mcpu`):");
    for (name, (triple, target, mcpu)) in cpus() {
        println!("  {name:<16} {triple:<28} {target:<14} {mcpu}");
    }
}

fn print_help() {
    eprintln!(
        "usage: mca <kernel> [cpu] [llvm-mca args...]\n\
         \t   mca asm <kernel> [cpu] [cargo-asm args...]\n\
         \t   mca list\n\
         \n\
         defaults: kernel = ntt, cpu = apple-m1.\n\
         examples:\n\
         \tmca ntt                          # summary + bottleneck on apple-m1\n\
         \tmca mul znver3                   # AVX2 backend, Zen 3 scheduler\n\
         \tmca ntt cascadelake --timeline   # full per-cycle timeline\n\
         \tmca asm ntt apple-m1 > ntt.s     # dump asm to hand-annotate"
    );
}

fn resolve_kernel(name: &str) -> (&'static str, &'static str) {
    let table = kernels();
    match table.get(name).copied() {
        Some(entry) => entry,
        None => {
            eprintln!("unknown kernel {name:?}; try `mca list`");
            exit(2);
        }
    }
}

fn resolve_cpu(name: &str) -> (&'static str, &'static str, &'static str) {
    let table = cpus();
    match table.get(name).copied() {
        Some(entry) => entry,
        None => {
            eprintln!("unknown cpu {name:?}; try `mca list`");
            exit(2);
        }
    }
}

fn cargo(root: &Path) -> Command {
    let mut command = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    command.current_dir(root);
    command.stdout(Stdio::inherit());
    command.stderr(Stdio::inherit());

    command
}

fn exec(mut command: Command) {
    match command.status() {
        Ok(status) if status.success() => {}
        Ok(status) => exit(status.code().unwrap_or(1)),
        Err(error) => {
            eprintln!("failed to run {command:?}: {error}");
            exit(1);
        }
    }
}

fn crate_root() -> PathBuf {
    let mut dir = std::env::current_dir().expect("current dir");
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("src/lib.rs").is_file() {
            return dir;
        }
        if !dir.pop() {
            eprintln!("could not locate the crate root (Cargo.toml + src/lib.rs)");
            exit(1);
        }
    }
}
