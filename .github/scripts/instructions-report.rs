//! Convert gungraun (`--save-summary`) benchmark summaries to JSON for the
//! CI site.
//!
//! Usage: `instructions-report <summary-root-dir> <sha>`
//!
//! `<summary-root-dir>` is the directory gungraun writes its per-benchmark
//! `summary.json` files under (e.g. `target/gungraun`). gungraun lays them
//! out as `<root>/<pkg>/<crate>/<group>/<bench>/summary.json`, one per
//! benchmark: it splits the summary's `module_path` (`<crate>::<group>`) into
//! directory components and appends the `function_name` (`<bench>`) leaf. This
//! walks the root recursively, reads every `summary.json`, and emits the
//! payload shape the dashboard's `instructions` kind consumes:
//!
//! ```json
//! { "sha": "...", "updated_at": "<iso8601>", "total": <u64>,
//!   "results": [ {"name": "<group>::<bench>", "instructions": <u64>,
//!     "l1_misses": <u64>, "l2_misses": <u64>, "branch_misses": <u64>,
//!     "estimated_cycles": <u64>, "flamegraph": "<url>"}, ... ] }
//! ```
//!
//! Fields are optional and omitted when the underlying callgrind events are
//! absent; `total` is the number of results, not a sum.
//!
//! Compile: `rustc -O instructions-report.rs -o instructions-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Dashboard / data host (same Fly app serves the static site and JSON).
/// Flamegraphs uploaded under `/data/instructions/flamegraphs/<sha>/` are
/// served here via nginx's `/instructions/` alias.
const SITE: &str = "https://mlkem-selkie-ci.fly.dev";

/// Benches a flamegraphs job records sampled call-stack SVGs for. No
/// workflow produces flamegraphs yet; list benches here (and upload the
/// SVGs via `ci-upload --assets`) to light up the dashboard's flame links.
const FLAMEGRAPH_BENCHES: [&str; 0] = [];

/// One benchmark's whole-run totals, mapped onto the dashboard's field set.
struct BenchResult {
    /// `group::bench` identifier, derived from the summary's `module_path`.
    name: String,
    /// Callgrind `Ir` (instructions executed).
    instructions: Option<u64>,
    /// L1 misses: `I1mr + D1mr + D1mw` (any access that missed L1).
    /// Present only with `--cache-sim=yes`.
    l1_misses: Option<u64>,
    /// Last-level (LL) misses: `ILmr + DLmr + DLmw` (any access that missed
    /// the last-level cache). Kept under the `l2_misses` key for pipeline
    /// compatibility. Present only with `--cache-sim=yes`.
    l2_misses: Option<u64>,
    /// Branch mispredictions: `Bcm + Bim`. Present only with
    /// `--branch-sim=yes`.
    branch_misses: Option<u64>,
    /// Callgrind's `EstimatedCycles` (instructions + memory penalties).
    /// Present only with `--cache-sim=yes`.
    estimated_cycles: Option<u64>,
    /// Public URL of this benchmark's sampled flamegraph on the CI site, set
    /// for the benches in [`FLAMEGRAPH_BENCHES`]. The Profile workflow's
    /// `flamegraphs` job pushes each SVG to the matching `/data/...` path.
    flamegraph: Option<String>,
}

impl BenchResult {
    /// Builds a result from one parsed `summary.json`, or `None` if it has no
    /// callgrind totals worth reporting (e.g. a non-callgrind tool run, or a
    /// summary without an extractable name).
    fn from_summary(summary: &Json, dir: &Path, sha: &str) -> Option<BenchResult> {
        let name = summary_name(dir)?;
        let metrics = callgrind_totals(summary)?;

        // The flamegraphs job uploads `<group>__<bench>.svg` for the fixed
        // bench set; emit the matching URL. The jobs run concurrently, so
        // the link may 404 briefly until the SVG lands.
        let flamegraph = FLAMEGRAPH_BENCHES.contains(&name.as_str()).then(|| {
            format!("{SITE}/instructions/flamegraphs/{sha}/{}.svg", name.replace("::", "__"))
        });

        // L1 miss = any access that didn't hit L1: I1 read + D1 read/write
        // misses. LL miss = any access that didn't hit the last-level cache:
        // IL read + DL read/write misses. branch miss = conditional +
        // indirect mispredictions. Each derived field is emitted only when
        // all of its constituent events are present.
        let l1_misses = sum_events(&metrics, &["I1mr", "D1mr", "D1mw"]);
        let l2_misses = sum_events(&metrics, &["ILmr", "DLmr", "DLmw"]);
        let branch_misses = sum_events(&metrics, &["Bcm", "Bim"]);

        Some(BenchResult {
            name,
            instructions: metrics_lookup(&metrics, "Ir"),
            l1_misses,
            l2_misses,
            branch_misses,
            estimated_cycles: metrics_lookup(&metrics, "EstimatedCycles"),
            flamegraph,
        })
    }
}

/// Derives the `group::bench` name from the summary's directory, which
/// gungraun lays out as `.../<group>/<bench>/summary.json`.
///
/// The directory is the reliable source: the summary's `module_path` ends in
/// the function name, not the group, so parsing it yields `bench::bench`. The
/// directory's last two components are `<group>/<bench>`, matching the
/// `<group>__<bench>.svg` file names the flamegraphs job uploads — so the
/// URLs emitted here line up with the hosted files.
fn summary_name(dir: &Path) -> Option<String> {
    let components: Vec<&str> = dir
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    match components.as_slice() {
        [.., group, bench] => Some(format!("{group}::{bench}")),
        [.., bench] => Some((*bench).to_string()),
        _ => None,
    }
}

/// Extracts the callgrind whole-benchmark totals as an `EventKind -> u64` map.
///
/// The summary's `profiles` array holds one `Profile` per valgrind tool; the
/// callgrind profile's `summaries.total.summary` is a `ToolMetricSummary`
/// tagged `{"Callgrind": {<EventKind>: MetricsDiff, ...}}`. Each `MetricsDiff`
/// carries the metric as `metrics: {"Left"|"Right"|"Both": ...}`, where the
/// "new" run is `Left` (no baseline) or `Both[0]` (with baseline). Returns
/// `None` if no callgrind profile with integer totals is present.
fn callgrind_totals(summary: &Json) -> Option<Vec<(String, u64)>> {
    let profiles = summary.get("profiles").and_then(Json::as_array)?;

    for profile in profiles {
        if profile.get("tool").and_then(Json::as_str) != Some("Callgrind") {
            continue;
        }
        let callgrind = profile
            .get("summaries")
            .and_then(|s| s.get("total"))
            .and_then(|t| t.get("summary"))
            .and_then(|s| s.get("Callgrind"))?;
        let Json::Obj(pairs) = callgrind else {
            continue;
        };

        let mut metrics = Vec::with_capacity(pairs.len());
        for (event, diff) in pairs {
            if let Some(value) = metric_diff_new(diff) {
                metrics.push((event.clone(), value));
            }
        }
        return Some(metrics);
    }

    None
}

/// Reads the "new" (current-run) integer metric out of a `MetricsDiff`.
///
/// `metrics` is an externally tagged `EitherOrBoth`: `{"Left": M}` (new
/// only), `{"Right": M}` (old only), or `{"Both": [M_new, M_old]}`. Prefer
/// the new value (`Both[0]` or `Left`), falling back to `Right`. A `Metric`
/// is `{"Int": u64}` or `{"Float": f64}`; only integer metrics map to our
/// `u64` fields, so float-valued events (rates) are dropped.
fn metric_diff_new(diff: &Json) -> Option<u64> {
    let metrics = diff.get("metrics")?;
    let metric = metrics
        .get("Both")
        .and_then(Json::as_array)
        .and_then(|both| both.first())
        .or_else(|| metrics.get("Left"))
        .or_else(|| metrics.get("Right"))?;
    metric.get("Int").and_then(Json::as_u64)
}

/// Looks up a single event's value in the totals map.
fn metrics_lookup(metrics: &[(String, u64)], event: &str) -> Option<u64> {
    metrics
        .iter()
        .find(|(k, _)| k == event)
        .map(|(_, v)| *v)
}

/// Sums a set of events, returning `None` unless every one is present so a
/// derived field is only emitted when fully measured.
fn sum_events(metrics: &[(String, u64)], events: &[&str]) -> Option<u64> {
    let mut total = 0u64;
    for &event in events {
        total = total.saturating_add(metrics_lookup(metrics, event)?);
    }
    Some(total)
}

/// Collects every `summary.json` path under `root`, recursing into
/// subdirectories. Unreadable directories are skipped rather than fatal so a
/// partial benchmark run still reports what it produced.
fn find_summaries(root: &Path) -> Vec<PathBuf> {
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
            } else if path.file_name().and_then(|n| n.to_str()) == Some("summary.json") {
                found.push(path);
            }
        }
    }

    found
}

/// Escapes a string as a JSON string literal.
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

/// Formats the current time as `YYYY-MM-DDThh:mm:ssZ` (UTC), walking the
/// civil calendar from the Unix epoch.
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
        eprintln!("usage: instructions-report <summary-root-dir> <sha>");
        std::process::exit(1);
    }

    let root = Path::new(&args[1]);
    let sha = &args[2];

    let mut results: Vec<BenchResult> = find_summaries(root)
        .iter()
        .filter_map(|path| {
            let summary = Json::from_file(path.to_str()?)?;
            let dir = path.parent().unwrap_or(root);
            BenchResult::from_summary(&summary, dir, sha)
        })
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
        write!(w, "    {{\"name\": {}", json_str(&r.name))?;
        if let Some(v) = r.instructions { write!(w, ", \"instructions\": {}", v)?; }
        if let Some(v) = r.l1_misses { write!(w, ", \"l1_misses\": {}", v)?; }
        if let Some(v) = r.l2_misses { write!(w, ", \"l2_misses\": {}", v)?; }
        if let Some(v) = r.branch_misses { write!(w, ", \"branch_misses\": {}", v)?; }
        if let Some(v) = r.estimated_cycles { write!(w, ", \"estimated_cycles\": {}", v)?; }
        if let Some(ref u) = r.flamegraph { write!(w, ", \"flamegraph\": {}", json_str(u))?; }
        write!(w, "}}")?;
        if i + 1 < results.len() { writeln!(w, ",")?; } else { writeln!(w)?; }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}

/// A minimal JSON value, enough to read gungraun summaries without a serde
/// dependency (CI scripts are dependency-free, compiled with `rustc`).
enum Json {
    /// A JSON number.
    Num(f64),
    /// A JSON string.
    Str(String),
    /// A JSON array.
    Arr(Vec<Json>),
    /// A JSON object, as ordered key/value pairs.
    Obj(Vec<(String, Json)>),
    /// `null`, `true`, or `false` (never read individually).
    Other,
}

impl Json {
    /// Parses the file at `path`, or `None` if absent/empty/malformed.
    fn from_file(path: &str) -> Option<Json> {
        let text = fs::read_to_string(path).ok()?;
        let bytes = text.as_bytes();
        let mut pos = 0;

        Json::parse_value(bytes, &mut pos)
    }

    /// Looks up `key` in an object, returning `None` otherwise.
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Returns the array elements, or `None` for non-arrays.
    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    /// Returns the string value, or `None` for non-strings.
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the value as a non-negative integer, or `None` for non-numbers
    /// and negatives. gungraun serializes metric integers as JSON numbers, so
    /// reinterpret the parsed `f64` as `u64` when it is whole and in range.
    fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 && *n <= u64::MAX as f64 => {
                Some(*n as u64)
            }
            _ => None,
        }
    }

    /// Parses one value at `*pos`, advancing past it.
    fn parse_value(bytes: &[u8], pos: &mut usize) -> Option<Json> {
        skip_ws(bytes, pos);

        match bytes.get(*pos)? {
            b'{' => Json::parse_object(bytes, pos),
            b'[' => Json::parse_array(bytes, pos),
            b'"' => parse_string(bytes, pos).map(Json::Str),
            b't' | b'f' | b'n' => {
                while bytes.get(*pos).is_some_and(u8::is_ascii_alphabetic) {
                    *pos += 1;
                }
                Some(Json::Other)
            }
            _ => parse_number(bytes, pos).map(Json::Num),
        }
    }

    /// Parses an object body starting at `{`.
    fn parse_object(bytes: &[u8], pos: &mut usize) -> Option<Json> {
        *pos += 1;
        let mut pairs = Vec::new();

        loop {
            skip_ws(bytes, pos);
            match bytes.get(*pos)? {
                b'}' => {
                    *pos += 1;
                    return Some(Json::Obj(pairs));
                }
                b',' => *pos += 1,
                b'"' => {
                    let key = parse_string(bytes, pos)?;
                    skip_ws(bytes, pos);
                    if bytes.get(*pos)? != &b':' {
                        return None;
                    }
                    *pos += 1;
                    pairs.push((key, Json::parse_value(bytes, pos)?));
                }
                _ => return None,
            }
        }
    }

    /// Parses an array body starting at `[`.
    fn parse_array(bytes: &[u8], pos: &mut usize) -> Option<Json> {
        *pos += 1;
        let mut items = Vec::new();

        loop {
            skip_ws(bytes, pos);
            match bytes.get(*pos)? {
                b']' => {
                    *pos += 1;
                    return Some(Json::Arr(items));
                }
                b',' => *pos += 1,
                _ => items.push(Json::parse_value(bytes, pos)?),
            }
        }
    }
}

/// Advances `*pos` past JSON whitespace.
fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while bytes.get(*pos).is_some_and(u8::is_ascii_whitespace) {
        *pos += 1;
    }
}

/// Parses a JSON string starting at the opening quote, decoding escapes
/// loosely (enough for keys and ASCII values).
fn parse_string(bytes: &[u8], pos: &mut usize) -> Option<String> {
    *pos += 1;
    let mut s = String::new();

    while let Some(&b) = bytes.get(*pos) {
        match b {
            b'"' => {
                *pos += 1;
                return Some(s);
            }
            b'\\' => {
                *pos += 1;
                let esc = *bytes.get(*pos)?;
                if esc == b'u' {
                    *pos += 4;
                } else {
                    s.push(esc as char);
                }
                *pos += 1;
            }
            _ => {
                s.push(b as char);
                *pos += 1;
            }
        }
    }

    None
}

/// Parses a JSON number at `*pos`, advancing past it.
fn parse_number(bytes: &[u8], pos: &mut usize) -> Option<f64> {
    let start = *pos;

    while let Some(&b) = bytes.get(*pos) {
        if b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E') {
            *pos += 1;
        } else {
            break;
        }
    }

    std::str::from_utf8(&bytes[start..*pos]).ok()?.parse().ok()
}
