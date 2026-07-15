//! Emit a markdown bench-diff between two directories of dashboard payloads.
//!
//! Usage: `bench-diff <baseline-dir> <head-dir> <baseline-sha>`
//!
//! Reads `{baseline,head}/{cycles,instructions,bench}-{backend}.json` for
//! every payload kind the [`bench.yml`] workflow produces. A missing file for
//! any (kind, backend) pair is silently skipped so a stale baseline (e.g.
//! before neon divan was added) still renders a partial table instead of
//! failing the job.
//!
//! `cycles` (rdtsc) and `instructions` (callgrind Ir) are deterministic and
//! head the report. `bench` (divan wall-clock) is noisy on shared GH runners,
//! so it lives in a collapsed `<details>` block that reviewers can open when
//! they want the reference.
//!
//! Compile: `rustc -O bench-diff.rs -o bench-diff`

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// A backend the workflow runs against (`generic`, `avx2`, `neon`). Not every
/// kind covers every backend — see [`Kind::backends`].
type Backend = &'static str;

/// A payload kind produced by the [`bench.yml`] workflow.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `cycles-{backend}.json`, dashboard `cycles` kind. rdtsc medians.
    Cycles,
    /// `instructions-{backend}.json`, dashboard `instructions` kind. Callgrind `Ir`.
    Instructions,
    /// `bench-{backend}.json`, dashboard `bench` kind. Divan median wall-clock ns.
    Wallclock,
}

impl Kind {
    /// The filename prefix under the payload directory. The full path is
    /// `<dir>/<prefix>-<backend>.json`.
    fn prefix(self) -> &'static str {
        match self {
            Kind::Cycles => "cycles",
            Kind::Instructions => "instructions",
            Kind::Wallclock => "bench",
        }
    }

    /// The section heading rendered in the markdown output.
    fn heading(self) -> &'static str {
        match self {
            Kind::Cycles => "Cycles (rdtsc, median)",
            Kind::Instructions => "Instructions (callgrind Ir)",
            Kind::Wallclock => "Wall-clock ns (divan, noisy)",
        }
    }

    /// The backends the perf workflow runs this kind against.
    fn backends(self) -> &'static [Backend] {
        match self {
            // rdtsc / callgrind are x86_64 only.
            Kind::Cycles | Kind::Instructions => &["generic", "avx2"],
            // Divan runs on every backend the ci.yml matrix covers.
            Kind::Wallclock => &["generic", "avx2", "neon"],
        }
    }

    /// Whether this kind's deltas should be collapsed under a `<details>`
    /// tag (noisy → yes; deterministic → no).
    fn is_noisy(self) -> bool {
        matches!(self, Kind::Wallclock)
    }

    /// Parses this kind's payload into a flat list of benchmark readings.
    /// `Cycles` also pulls the payload's `std_dev_cycles` so the diff can
    /// hold its delta to a 2·σ noise floor; `Instructions` is exact and
    /// `Wallclock` doesn't emit a stdev today, so both fall back to
    /// point-only classification.
    fn parse(self, json: &str) -> Vec<Bench> {
        match self {
            Kind::Cycles => parse_results(json, "cycles", Some("std_dev_cycles")),
            Kind::Instructions => parse_results(json, "instructions", None),
            Kind::Wallclock => parse_wallclock(json),
        }
    }
}

/// One benchmark reading. `value` carries whichever scalar the kind is
/// tracking (cycles / instructions / ns); `stdev` is populated when the
/// payload includes a companion standard-deviation field for the same unit.
struct Bench {
    /// `group::name` — matches the layout the dashboard-JSON scripts emit.
    name: String,
    /// The scalar metric, `u64` since every source rounds before serializing.
    value: u64,
    /// Absolute standard deviation in the same unit as `value`, when the
    /// payload carries one. `None` on deterministic metrics (instructions)
    /// and on payloads that don't emit a stdev field (divan wall-clock).
    stdev: Option<u64>,
}

/// Extracts every `results[].{ name, <field>, <stdev_field?> }` entry from a
/// dashboard payload shaped `{ results: [{ name, cycles|instructions, ... }, ...] }`.
fn parse_results(json: &str, field: &str, stdev_field: Option<&str>) -> Vec<Bench> {
    let mut out = Vec::new();

    for obj in split_top_level_objects(json, "\"results\"") {
        let Some(name) = string_field(&obj, "name") else {
            continue;
        };
        let Some(value) = number_field(&obj, field) else {
            continue;
        };
        let stdev = stdev_field.and_then(|f| number_field(&obj, f));

        out.push(Bench { name, value, stdev });
    }

    out
}

/// Extracts every `groups[].benchmarks[].{ name, median_ns }` entry from a
/// divan payload and joins the group name back into the bench name so the
/// output matches the shape used by `parse_results`.
fn parse_wallclock(json: &str) -> Vec<Bench> {
    let mut out = Vec::new();

    for group in split_top_level_objects(json, "\"groups\"") {
        let Some(group_name) = string_field(&group, "name") else {
            continue;
        };
        let Some(benchmarks_body) = section(&group, "\"benchmarks\"") else {
            continue;
        };

        for bench in split_objects_in(benchmarks_body) {
            let Some(bench_name) = string_field(&bench, "name") else {
                continue;
            };
            let Some(value) = number_field(&bench, "median_ns") else {
                continue;
            };

            out.push(Bench {
                name: format!("{group_name}::{bench_name}"),
                value,
                stdev: None,
            });
        }
    }

    out
}

/// Returns the substring inside the outermost brackets that follow `key` —
/// used to isolate the `results` / `groups` / `benchmarks` arrays before
/// object-splitting. Bracket-nesting-aware; ignores brackets inside strings.
fn section<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let start = json.find(key)? + key.len();
    let bracket = json[start..].find('[')?;
    let body_start = start + bracket + 1;

    let mut depth = 1usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut i = body_start;
    let bytes = json.as_bytes();

    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
        } else if c == b'"' {
            in_string = true;
        } else if c == b'[' {
            depth += 1;
        } else if c == b']' {
            depth -= 1;
            if depth == 0 {
                return Some(&json[body_start..i]);
            }
        }
        i += 1;
    }

    None
}

/// Splits the array under `key` into top-level object substrings. Returns
/// an empty vector when `key` is absent or the array is empty.
fn split_top_level_objects<'a>(json: &'a str, key: &str) -> Vec<&'a str> {
    section(json, key).map(split_objects_in).unwrap_or_default()
}

/// Splits a comma-separated sequence of top-level `{ ... }` objects (with
/// no outer array brackets) into slices. Bracket-nesting-aware; ignores
/// braces inside strings.
fn split_objects_in(body: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0usize;
    let mut obj_start: Option<usize> = None;

    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
        } else if c == b'"' {
            in_string = true;
        } else if c == b'{' {
            if depth == 0 {
                obj_start = Some(i);
            }
            depth += 1;
        } else if c == b'}' {
            depth -= 1;
            if depth == 0 {
                if let Some(s) = obj_start.take() {
                    out.push(&body[s..=i]);
                }
            }
        }
        i += 1;
    }

    out
}

/// Extracts a string-valued field from a JSON object substring. Assumes the
/// value has no embedded escapes past the simple `\"` case; the dashboard
/// payloads only carry ASCII bench names.
fn string_field(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let key_at = obj.find(&pat)?;
    let after = &obj[key_at + pat.len()..];
    let colon = after.find(':')?;
    let quote = after[colon..].find('"')? + colon + 1;
    let rest = &after[quote..];
    let end = rest.find('"')?;

    Some(rest[..end].to_string())
}

/// Extracts a numeric field from a JSON object substring, rounding fractional
/// values into `u64`. Returns `None` when the field is absent or `null`.
fn number_field(obj: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\"");
    let key_at = obj.find(&pat)?;
    let after = &obj[key_at + pat.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();

    if rest.starts_with("null") {
        return None;
    }

    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+')
        .unwrap_or(rest.len());
    let num = &rest[..end];

    num.parse::<f64>().ok().map(|v| v.round() as u64)
}

/// Regression threshold (%) — Δ at or beyond this classifies as regressed
/// or improved. Below it, the delta is unchanged. Deterministic metrics
/// (instructions, cycles) can meaningfully trigger at ±2%; wall-clock
/// stays visible in [`Kind::Wallclock`]'s collapsed section so its noise
/// doesn't crowd out the deterministic tables.
const REGRESSION_THRESHOLD_PCT: f64 = 2.0;

/// Deltas at or beyond this magnitude are rendered `**bold**` for visual
/// weight, so a scanner picks the largest movers out immediately.
const BOLD_THRESHOLD_PCT: f64 = 5.0;

/// A bench's classification against its baseline. Drives the per-section
/// counters and the top-of-report summary.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Change {
    /// Present in both, Δ ≥ +[`REGRESSION_THRESHOLD_PCT`]%.
    Regressed,
    /// Present in both, Δ ≤ -[`REGRESSION_THRESHOLD_PCT`]%.
    Improved,
    /// Present in both, |Δ| < [`REGRESSION_THRESHOLD_PCT`]%.
    Unchanged,
    /// In head only.
    New,
    /// In baseline only.
    Removed,
}

/// One rendered row: name, both values (option per side), the % delta and
/// its propagated stdev (when the source payload carries one), and the
/// classification. Rows sort by |delta| descending so the biggest movers
/// land at the top of each section table.
struct Row {
    name: String,
    baseline: Option<u64>,
    head: Option<u64>,
    delta_pct: Option<f64>,
    /// Combined delta stdev in percentage points. `None` when the source
    /// payload has no stdev field (instructions, wall-clock).
    delta_stdev_pct: Option<f64>,
    change: Change,
}

impl Row {
    /// Sort key: larger |Δ| first, then New/Removed at the bottom.
    fn sort_key(&self) -> f64 {
        self.delta_pct.map_or(f64::NEG_INFINITY, f64::abs)
    }

    /// Formats the `| Δ | Bench | Baseline | Head |` markdown row.
    fn to_markdown(&self) -> String {
        let delta_str = match (self.delta_pct, self.change) {
            (Some(d), _) => format_delta(d, self.delta_stdev_pct),
            (None, Change::New) => String::from("(new)"),
            (None, Change::Removed) => String::from("(removed)"),
            (None, _) => String::from("n/a"),
        };
        let baseline_str = self
            .baseline
            .map(thousands)
            .unwrap_or_else(|| String::from("—"));
        let head_str = self
            .head
            .map(thousands)
            .unwrap_or_else(|| String::from("—"));

        format!(
            "| {delta_str} | `{}` | {baseline_str} | {head_str} |\n",
            self.name,
        )
    }
}

/// One (kind, backend) pair's rendered content plus its counter tallies.
struct Section {
    kind: Kind,
    backend: Backend,
    rows: Vec<Row>,
    regressed: usize,
    improved: usize,
    unchanged: usize,
}

impl Section {
    /// Loads baseline + head payloads for one (kind, backend) pair and builds
    /// the sorted, classified rows. Returns `None` when either payload is
    /// missing so the caller can skip the section entirely.
    fn build(
        kind: Kind,
        backend: Backend,
        baseline_dir: &Path,
        head_dir: &Path,
    ) -> Option<Section> {
        let path = |dir: &Path| dir.join(format!("{}-{}.json", kind.prefix(), backend));

        let baseline_json = fs::read_to_string(path(baseline_dir)).ok()?;
        let head_json = fs::read_to_string(path(head_dir)).ok()?;

        let mut rows = build_rows(&kind.parse(&baseline_json), &kind.parse(&head_json));
        rows.sort_by(|a, b| {
            b.sort_key()
                .partial_cmp(&a.sort_key())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let regressed = rows.iter().filter(|r| r.change == Change::Regressed).count();
        let improved = rows.iter().filter(|r| r.change == Change::Improved).count();
        let unchanged = rows.iter().filter(|r| r.change == Change::Unchanged).count();

        Some(Section {
            kind,
            backend,
            rows,
            regressed,
            improved,
            unchanged,
        })
    }

    /// Renders the per-backend count line and the rows table.
    fn to_markdown(&self) -> String {
        let mut out = format!(
            "**{}** — {} regressed, {} improved, {} unchanged\n\n",
            self.backend, self.regressed, self.improved, self.unchanged,
        );

        if self.rows.is_empty() {
            out.push_str("_No baseline vs head bench pairs._\n");
            return out;
        }

        out.push_str("| Δ | Bench | Baseline | Head |\n|---:|---|---:|---:|\n");
        for row in &self.rows {
            out.push_str(&row.to_markdown());
        }
        out
    }
}

/// The full report — one section per (kind, backend) pair that resolved.
struct Report {
    baseline_sha: String,
    sections: Vec<Section>,
}

impl Report {
    /// Walks every (kind, backend) pair and collects the resolvable sections.
    fn build(baseline_dir: &Path, head_dir: &Path, baseline_sha: String) -> Report {
        let mut sections = Vec::new();

        for kind in [Kind::Cycles, Kind::Instructions, Kind::Wallclock] {
            for backend in kind.backends() {
                if let Some(section) = Section::build(kind, backend, baseline_dir, head_dir) {
                    sections.push(section);
                }
            }
        }

        Report {
            baseline_sha,
            sections,
        }
    }

    /// Renders the report as markdown, bracketed by the sticky HTML-comment
    /// marker so `bench.yml`'s `github-script` step updates the same comment
    /// on subsequent PR pushes.
    fn to_markdown(&self) -> String {
        let mut out = String::from("<!-- bench-diff-marker -->\n");

        if self.sections.is_empty() {
            out.push_str("### Benchmarks · no baseline\n\n");
            out.push_str(&format!(
                "_No matching baseline payloads for `{}` — first run on this branch, or the baseline cache expired._\n",
                short_sha(&self.baseline_sha),
            ));
            return out;
        }

        let total_regressed: usize = self.sections.iter().map(|s| s.regressed).sum();
        let total_improved: usize = self.sections.iter().map(|s| s.improved).sum();
        let total_unchanged: usize = self.sections.iter().map(|s| s.unchanged).sum();
        let total = total_regressed + total_improved + total_unchanged;

        out.push_str(&format!(
            "### Benchmarks{}\n\n",
            title_verdict(total_regressed, total_improved),
        ));

        out.push_str(&format!(
            "Compared {total} benchmarks vs `{}` · {total_unchanged} unchanged · threshold ±{}%\n\n",
            short_sha(&self.baseline_sha),
            REGRESSION_THRESHOLD_PCT as u32,
        ));

        let mut current_kind: Option<Kind> = None;
        for section in &self.sections {
            if current_kind != Some(section.kind) {
                if current_kind.is_some_and(Kind::is_noisy) {
                    out.push_str("</details>\n\n");
                }
                if section.kind.is_noisy() {
                    out.push_str(&format!(
                        "<details><summary><b>{}</b></summary>\n\n",
                        section.kind.heading(),
                    ));
                } else {
                    out.push_str(&format!("#### {}\n\n", section.kind.heading()));
                }
                current_kind = Some(section.kind);
            }

            out.push_str(&section.to_markdown());
            out.push('\n');
        }

        if current_kind.is_some_and(Kind::is_noisy) {
            out.push_str("</details>\n");
        }

        out
    }
}

/// Joins baseline + head bench lists into per-name [`Row`]s. Benches present
/// on only one side become `New` / `Removed`.
fn build_rows(baseline: &[Bench], head: &[Bench]) -> Vec<Row> {
    let baseline_map: std::collections::BTreeMap<&str, &Bench> =
        baseline.iter().map(|b| (b.name.as_str(), b)).collect();
    let head_map: std::collections::BTreeMap<&str, &Bench> =
        head.iter().map(|b| (b.name.as_str(), b)).collect();

    let all_names: std::collections::BTreeSet<&str> = baseline_map
        .keys()
        .copied()
        .chain(head_map.keys().copied())
        .collect();

    all_names
        .into_iter()
        .map(|name| {
            let base = baseline_map.get(name).copied();
            let head_val = head_map.get(name).copied();
            let (delta_pct, delta_stdev_pct, change) = classify(base, head_val);
            Row {
                name: name.to_string(),
                baseline: base.map(|b| b.value),
                head: head_val.map(|b| b.value),
                delta_pct,
                delta_stdev_pct,
                change,
            }
        })
        .collect()
}

/// Classifies one bench's baseline + head against [`REGRESSION_THRESHOLD_PCT`],
/// using propagated stdev (when available) to require the delta to clear
/// the 2·σ noise floor before counting as a regression or improvement.
fn classify(baseline: Option<&Bench>, head: Option<&Bench>) -> (Option<f64>, Option<f64>, Change) {
    match (baseline, head) {
        (Some(b), Some(h)) if b.value > 0 => {
            let delta = (h.value as f64 - b.value as f64) / b.value as f64 * 100.0;
            let delta_stdev = delta_stdev_pct(b, h);

            // Noise floor: 2·σ (≈ 95% CI). Applied only when both sides
            // carry stdev; otherwise fall through to the pure threshold gate.
            let noise = delta_stdev.map_or(0.0, |s| 2.0 * s);
            let change = if delta >= REGRESSION_THRESHOLD_PCT + noise {
                Change::Regressed
            } else if delta <= -(REGRESSION_THRESHOLD_PCT + noise) {
                Change::Improved
            } else {
                Change::Unchanged
            };
            (Some(delta), delta_stdev, change)
        }
        (Some(_), None) => (None, None, Change::Removed),
        (None, Some(_)) => (None, None, Change::New),
        _ => (None, None, Change::Unchanged),
    }
}

/// Propagates baseline + head absolute stdevs into a combined stdev of the
/// percentage delta. Returns `None` when either side lacks a stdev field.
/// Uses `hypot(sb/b, sh/h) · 100` as the approximation — valid when the two
/// measurements are independent (they are; separate runs).
fn delta_stdev_pct(baseline: &Bench, head: &Bench) -> Option<f64> {
    let (sb, sh) = (baseline.stdev?, head.stdev?);
    if baseline.value == 0 || head.value == 0 {
        return None;
    }
    let frac_b = sb as f64 / baseline.value as f64;
    let frac_h = sh as f64 / head.value as f64;
    Some((frac_b.powi(2) + frac_h.powi(2)).sqrt() * 100.0)
}

/// Formats a `%` delta with a sign, optionally appending `±X%` when a
/// propagated stdev is available. Wraps the point estimate in `**` when
/// `|Δ| ≥ BOLD_THRESHOLD_PCT`.
fn format_delta(delta: f64, stdev: Option<f64>) -> String {
    let abs = delta.abs();
    if abs < 0.5 {
        return String::from("~0%");
    }
    let sign = if delta > 0.0 { "+" } else { "" };
    let point = format!("{sign}{delta:.1}%");
    let bolded = if abs >= BOLD_THRESHOLD_PCT {
        format!("**{point}**")
    } else {
        point
    };

    // Only show `±σ` when it's meaningful (>0.05%); a stdev pinned to zero
    // just clutters the column.
    match stdev {
        Some(s) if s >= 0.05 => format!("{bolded} ±{s:.1}%"),
        _ => bolded,
    }
}

/// Builds the ` · N regressed · M improved` verdict trailer for the report
/// title. Suppresses zero counts so the title reads cleanly in the PR
/// timeline (no `0 regressed` clutter on a clean run).
fn title_verdict(regressed: usize, improved: usize) -> String {
    match (regressed, improved) {
        (0, 0) => String::from(" · no changes"),
        (r, 0) => format!(" · {r} regressed"),
        (0, i) => format!(" · {i} improved"),
        (r, i) => format!(" · {r} regressed · {i} improved"),
    }
}

/// Renders `n` with `,` thousands separators. `1234567` → `"1,234,567"`.
fn thousands(n: u64) -> String {
    let digits: Vec<char> = n.to_string().chars().collect();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.iter().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*c);
    }
    out
}

/// Shortens a 40-char SHA to its first 7 chars for compact display in the
/// report header. Non-40-char inputs are echoed verbatim.
fn short_sha(sha: &str) -> String {
    if sha.len() >= 7 {
        sha[..7].to_string()
    } else {
        sha.to_string()
    }
}

/// Entry point. Fails when the argv shape is wrong or the directories are
/// unreadable; a merely missing payload for a single (kind, backend) is not
/// a failure — see [`Section::build`].
fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let [_, baseline_dir, head_dir, baseline_sha] = args.as_slice() else {
        eprintln!("Usage: bench-diff <baseline-dir> <head-dir> <baseline-sha>");
        return ExitCode::from(2);
    };

    let report = Report::build(
        &PathBuf::from(baseline_dir),
        &PathBuf::from(head_dir),
        baseline_sha.clone(),
    );

    print!("{}", report.to_markdown());
    ExitCode::SUCCESS
}
