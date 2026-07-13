//! Emit a markdown perf-diff between two directories of dashboard payloads.
//!
//! Usage: `perf-diff <baseline-dir> <head-dir> <baseline-sha>`
//!
//! Reads `{baseline,head}/{cycles,instructions,bench}-{backend}.json` for
//! every payload kind the [`perf.yml`] workflow produces. A missing file for
//! any (kind, backend) pair is silently skipped so a stale baseline (e.g.
//! before neon divan was added) still renders a partial table instead of
//! failing the job.
//!
//! `cycles` (rdtsc) and `instructions` (callgrind Ir) are deterministic and
//! head the report. `bench` (divan wall-clock) is noisy on shared GH runners,
//! so it lives in a collapsed `<details>` block that reviewers can open when
//! they want the reference.
//!
//! Compile: `rustc -O perf-diff.rs -o perf-diff`

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// A backend the workflow runs against (`generic`, `avx2`, `neon`). Not every
/// kind covers every backend — see [`Kind::backends`].
type Backend = &'static str;

/// A payload kind produced by the [`perf.yml`] workflow.
#[derive(Clone, Copy)]
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
    fn parse(self, json: &str) -> Vec<Bench> {
        match self {
            Kind::Cycles => parse_results(json, "cycles"),
            Kind::Instructions => parse_results(json, "instructions"),
            Kind::Wallclock => parse_wallclock(json),
        }
    }
}

/// One benchmark reading. `value` carries whichever scalar the kind is
/// tracking (cycles / instructions / ns).
struct Bench {
    /// `group::name` — matches the layout the dashboard-JSON scripts emit.
    name: String,
    /// The scalar metric, `u64` since every source rounds before serializing.
    value: u64,
}

/// Extracts every `results[].{ name, <field> }` entry from a dashboard payload
/// shaped `{ results: [{ name, cycles|instructions, ... }, ...] }`.
fn parse_results(json: &str, field: &str) -> Vec<Bench> {
    let mut out = Vec::new();

    for obj in split_top_level_objects(json, "\"results\"") {
        let Some(name) = string_field(&obj, "name") else {
            continue;
        };
        let Some(value) = number_field(&obj, field) else {
            continue;
        };

        out.push(Bench { name, value });
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
        let Some(benchmarks) = section(&group, "\"benchmarks\"") else {
            continue;
        };

        for bench in split_top_level_objects(benchmarks, "\"benchmarks\"") {
            let Some(bench_name) = string_field(&bench, "name") else {
                continue;
            };
            let Some(value) = number_field(&bench, "median_ns") else {
                continue;
            };

            out.push(Bench {
                name: format!("{group_name}::{bench_name}"),
                value,
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

/// Splits the array under `key` into top-level object substrings. Each
/// returned slice is one `{ ... }` element with matched braces. Returns an
/// empty vector when `key` is absent or the array is empty.
fn split_top_level_objects<'a>(json: &'a str, key: &str) -> Vec<&'a str> {
    let Some(body) = section(json, key) else {
        return Vec::new();
    };

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

/// A rendered section for one (kind, backend) pair. `body` is the markdown
/// table (or a one-line note when the diff is empty).
struct Section {
    kind: Kind,
    backend: Backend,
    body: String,
}

impl Section {
    /// Loads baseline + head payloads for one (kind, backend) pair and renders
    /// the delta table. Returns `None` when either payload is missing so the
    /// caller can skip the section entirely.
    fn build(kind: Kind, backend: Backend, baseline_dir: &Path, head_dir: &Path) -> Option<Section> {
        let path = |dir: &Path| dir.join(format!("{}-{}.json", kind.prefix(), backend));

        let baseline_json = fs::read_to_string(path(baseline_dir)).ok()?;
        let head_json = fs::read_to_string(path(head_dir)).ok()?;

        let baseline = kind.parse(&baseline_json);
        let head = kind.parse(&head_json);

        Some(Section {
            kind,
            backend,
            body: render_table(&baseline, &head),
        })
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

    /// Renders the report as markdown. The output is bracketed by a sticky
    /// HTML-comment marker so `perf.yml`'s `github-script` step can update
    /// the same comment on subsequent PR pushes.
    fn to_markdown(&self) -> String {
        let mut out = String::from("<!-- perf-diff-marker -->\n### Perf diff vs main\n\n");
        out.push_str(&format!(
            "Baseline: `{}`. Deltas below 0.5% are elided as `~0%`. Deltas at or above ±5% are flagged with `**`.\n\n",
            short_sha(&self.baseline_sha)
        ));

        if self.sections.is_empty() {
            out.push_str("_No matching baseline payloads found — first run on this branch, or the baseline cache expired._\n");
            return out;
        }

        let mut current_kind: Option<&'static str> = None;
        for section in &self.sections {
            let heading = section.kind.heading();
            if current_kind != Some(heading) {
                if section.kind.is_noisy() {
                    out.push_str("<details><summary><b>");
                    out.push_str(heading);
                    out.push_str("</b></summary>\n\n");
                } else {
                    out.push_str("#### ");
                    out.push_str(heading);
                    out.push('\n');
                    out.push('\n');
                }
                current_kind = Some(heading);
            }

            out.push_str(&format!("<b>{}</b>\n\n", section.backend));
            out.push_str(&section.body);
            out.push('\n');
        }

        // Close the trailing `<details>` if the last-rendered kind was noisy.
        if self
            .sections
            .last()
            .is_some_and(|s| s.kind.is_noisy())
        {
            out.push_str("</details>\n");
        }

        out
    }
}

/// Emits a `| bench | baseline | head | Δ% |` markdown table joining baseline
/// and head by name. Benches present in only one side are listed at the
/// bottom as `(removed)` / `(new)`.
fn render_table(baseline: &[Bench], head: &[Bench]) -> String {
    let mut baseline_map: std::collections::BTreeMap<&str, u64> =
        baseline.iter().map(|b| (b.name.as_str(), b.value)).collect();
    let mut head_map: std::collections::BTreeMap<&str, u64> =
        head.iter().map(|b| (b.name.as_str(), b.value)).collect();

    let shared: Vec<&str> = baseline_map
        .keys()
        .copied()
        .filter(|k| head_map.contains_key(k))
        .collect();

    if shared.is_empty() && baseline_map.is_empty() && head_map.is_empty() {
        return String::from("_No shared benches._\n");
    }

    let mut out = String::from("| bench | baseline | head | Δ |\n|---|---:|---:|---:|\n");

    for name in &shared {
        let b = baseline_map.remove(name).expect("shared implies present");
        let h = head_map.remove(name).expect("shared implies present");
        out.push_str(&format!("| `{}` | {} | {} | {} |\n", name, thousands(b), thousands(h), format_delta(b, h)));
    }

    for (name, value) in baseline_map {
        out.push_str(&format!("| `{}` | {} | — | (removed) |\n", name, thousands(value)));
    }
    for (name, value) in head_map {
        out.push_str(&format!("| `{}` | — | {} | (new) |\n", name, thousands(value)));
    }

    out
}

/// Formats the percentage change from `baseline` to `head`. Deltas within
/// ±0.5% render as `~0%`; deltas at or above ±5% are wrapped in `**` so the
/// markdown table calls them out visually.
fn format_delta(baseline: u64, head: u64) -> String {
    if baseline == 0 {
        return String::from("n/a");
    }

    let delta = (head as f64 - baseline as f64) / baseline as f64 * 100.0;
    let abs = delta.abs();

    if abs < 0.5 {
        return String::from("~0%");
    }

    let sign = if delta > 0.0 { "+" } else { "" };
    let formatted = format!("{sign}{delta:.1}%");

    if abs >= 5.0 {
        format!("**{formatted}**")
    } else {
        formatted
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
        eprintln!("Usage: perf-diff <baseline-dir> <head-dir> <baseline-sha>");
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
