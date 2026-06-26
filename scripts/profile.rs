//! Local profiling driver for `mlkem-selkie`.
//!
//! A thin dispatcher over CPU and heap profilers, all pointed at the
//! `examples/profile` workload built under the `profiling` profile (release
//! codegen + full debuginfo). Run it with `rustc`, not cargo:
//!
//! ```text
//! rustc -O scripts/profile.rs -o /tmp/mlkem-profile
//! /tmp/mlkem-profile samply all 20000     # CPU profile -> Firefox Profiler
//! /tmp/mlkem-profile dhat keygen          # heap profile -> dhat-heap.json
//! /tmp/mlkem-profile flamegraph decaps     # flamegraph.svg (needs cargo-flamegraph)
//! /tmp/mlkem-profile build                  # just build the example
//! ```
//!
//! `mode` is one of `keygen | encaps | decaps | all` (default `all`); the
//! trailing number is the iteration count (ignored under `dhat`).

use std::{
    path::{Path, PathBuf},
    process::{exit, Command},
};

/// The cargo profile and example the driver always targets.
const PROFILE: &str = "profiling";
/// The example workload binary name.
const EXAMPLE: &str = "profile";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = args.split_first().map_or(("samply", &[][..]), |(c, r)| (c.as_str(), r));

    let root = crate_root();

    match command {
        "build" => {
            build(&root, &[]);
        }
        "dhat" => {
            let mode = rest.first().map_or("all", String::as_str);
            let mut cargo = cargo(&root);
            cargo.args([
                "run",
                "--profile",
                PROFILE,
                "--features",
                "dhat-heap",
                "--example",
                EXAMPLE,
                "--",
                mode,
            ]);
            run(cargo);
            eprintln!("wrote {}/dhat-heap.json (open at https://nnethercote.github.io/dh_view/dh_view.html)", root.display());
        }
        "flamegraph" => {
            // cargo-flamegraph builds and records in one shot.
            let mode = rest.first().map_or("all", String::as_str);
            let mut cargo = cargo(&root);
            cargo.args([
                "flamegraph",
                "--profile",
                PROFILE,
                "--example",
                EXAMPLE,
                "--",
                mode,
            ]);
            run(cargo);
        }
        "samply" => {
            build(&root, &[]);
            let bin = root
                .join("target")
                .join(PROFILE)
                .join("examples")
                .join(EXAMPLE);
            let mut samply = Command::new("samply");
            samply.arg("record").arg(&bin).args(rest);
            run(samply);
        }
        other => {
            eprintln!("unknown command {other:?}; expected build | dhat | flamegraph | samply");
            exit(2);
        }
    }
}

/// Builds the profiling example, forwarding any extra cargo args.
fn build(root: &Path, extra: &[&str]) {
    let mut cargo = cargo(root);
    cargo.args(["build", "--profile", PROFILE, "--example", EXAMPLE]);
    cargo.args(extra);
    run(cargo);
}

/// A `cargo` command rooted at the crate directory.
fn cargo(root: &Path) -> Command {
    let mut command = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    command.current_dir(root);

    command
}

/// Runs a command, inheriting stdio, and exits on failure.
fn run(mut command: Command) {
    match command.status() {
        Ok(status) if status.success() => {}
        Ok(status) => exit(status.code().unwrap_or(1)),
        Err(error) => {
            eprintln!("failed to run {command:?}: {error}");
            exit(1);
        }
    }
}

/// Walks up from the current directory to the crate root (a directory holding
/// both `Cargo.toml` and `src/lib.rs`).
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
