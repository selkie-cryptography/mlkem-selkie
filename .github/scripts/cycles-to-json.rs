//! Convert criterion cycle-count estimates to JSON for the CI site.
//!
//! Usage: cycles-to-json <criterion-root-dir> <sha>
//!
//! `<criterion-root-dir>` is criterion's output directory
//! (`target/criterion`). The `cycles` bench (criterion +
//! criterion-cycles-per-byte, rdtsc) lays results out as
//! `<root>/<group>/<bench>/new/estimates.json`, whose point estimates are
//! in the measurement's unit — CPU cycles. This walks the root, reads
//! every `new/estimates.json`, and emits the payload shape the
//! dashboard's `cycles` kind consumes:
//!
//! ```json
//! { "sha": "...", "updated_at": "<iso8601>", "total": <n>,
//!   "results": [ {"name": "<group>::<bench>", "cycles": <u64 median>,
//!     "mean_cycles": <u64>, "std_dev_cycles": <u64>}, ... ] }
//! ```
//!
//! Compile: `rustc -O cycles-to-json.rs -o cycles-to-json`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// One benchmark's cycle-count estimates.
struct BenchResult {
    /// `group::bench`, e.g. `mlkem768::keygen`, from the directory layout.
    name: String,
    /// Median cycles (`median.point_estimate`) — the headline value.
    cycles: u64,
    /// Mean cycles (`mean.point_estimate`).
    mean_cycles: u64,
    /// Standard deviation in cycles (`std_dev.point_estimate`).
    std_dev_cycles: u64,
}

impl BenchResult {
    /// Builds a result from one `new/estimates.json`, or `None` when the
    /// file has no median estimate (e.g. an unrelated criterion artifact).
    fn from_estimates(path: &Path) -> Option<BenchResult> {
        let contents = fs::read_to_string(path).ok()?;
        let median = point_estimate(&contents, "median")?;

        // The bench directory is `<group>/<bench>/new/estimates.json`.
        let bench_dir = path.parent()?.parent()?;
        let bench = bench_dir.file_name()?.to_str()?;
        let group = bench_dir.parent()?.file_name()?.to_str()?;

        Some(BenchResult {
            name: format!("{group}::{bench}"),
            cycles: median.round() as u64,
            mean_cycles: point_estimate(&contents, "mean")
                .unwrap_or(median)
                .round() as u64,
            std_dev_cycles: point_estimate(&contents, "std_dev")
                .unwrap_or(0.0)
                .round() as u64,
        })
    }
}

/// Reads `<section>.point_estimate` out of an estimates.json.
///
/// Good enough for criterion's flat layout: each statistic is a top-level
/// object keyed by its name, holding a scalar `point_estimate`.
fn point_estimate(json: &str, section: &str) -> Option<f64> {
    let needle = format!("\"{section}\"");
    let idx = json.find(&needle)?;
    let rest = &json[idx..];
    let pe = rest.find("\"point_estimate\"")?;
    let after = &rest[pe + "\"point_estimate\"".len()..];
    let colon = after.find(':')?;
    let val = after[colon + 1..].trim_start();
    let end = val
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E')
        .unwrap_or(val.len());
    val[..end].parse().ok()
}

/// Collects every `new/estimates.json` under `root`, recursing into
/// subdirectories. Baseline (`base/`) and report artifacts are skipped so
/// only the current run's estimates are counted.
fn find_estimates(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("estimates.json")
                && path
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    == Some("new")
            {
                found.push(path);
            }
        }
    }

    found
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn iso8601_now() -> String {
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let secs = dur.as_secs();
    let (h, m, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    let mut y = 1970i64;
    let mut rem = (secs / 86400) as i64;
    loop {
        let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if rem < yd { break }
        rem -= yd;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let md = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0;
    for &d in &md { if rem < d { break } rem -= d; mo += 1; }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo + 1, rem + 1, h, m, s)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: cycles-to-json <criterion-root-dir> <sha>");
        std::process::exit(1);
    }

    let root = Path::new(&args[1]);
    let sha = &args[2];

    let mut results: Vec<BenchResult> = find_estimates(root)
        .iter()
        .filter_map(|path| BenchResult::from_estimates(path))
        .collect();

    // Sort by name for deterministic output across runs and filesystems.
    results.sort_by(|a, b| a.name.cmp(&b.name));

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"total\": {},", results.len())?;
    writeln!(w, "  \"results\": [")?;

    for (i, r) in results.iter().enumerate() {
        write!(
            w,
            "    {{\"name\": {}, \"cycles\": {}, \"mean_cycles\": {}, \"std_dev_cycles\": {}}}",
            json_str(&r.name),
            r.cycles,
            r.mean_cycles,
            r.std_dev_cycles
        )?;
        if i + 1 < results.len() { writeln!(w, ",")?; } else { writeln!(w)?; }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}
