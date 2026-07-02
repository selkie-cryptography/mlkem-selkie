//! Parse divan benchmark output into a JSON array of per-benchmark records.
//!
//! Usage: divan-to-json (reads stdin, writes stdout)
//!
//! Divan prints a group header followed by a tree of result rows. Plain
//! benches are one level deep; type-parameterized benches (`types = [...]`,
//! e.g. the `keygen`/`encaps`/`decaps` benches over `MLKEM512`/`MLKEM768`/
//! `MLKEM1024`) nest the parameter set one level below the bench name:
//!
//! ```text
//!   mlkem             fastest  │ slowest  │ median  │ mean   │ samples │ iters
//!   ├─ decaps                  │          │         │        │         │
//!   │  ├─ MLKEM512    2.6 µs   │ 2.7 µs   │ 2.6 µs  │ 2.6 µs │ 100     │ 204800
//! ```
//!
//! Rows are named by their full tree path (`mlkem::decaps::MLKEM512`);
//! parent rows with no timings only extend the path. We capture every
//! timing column so the dashboard and PR comparison can show median, the
//! fast/slow spread, and sample counts. `value` is the median — more
//! robust to CI noise than the mean — and `range` is the half-spread,
//! so consumers that want a single number plus an error bar have both.
//!
//! Compile: `rustc -O divan-to-json.rs -o divan-to-json`

use std::io::{self, BufRead, Write};

/// One parsed benchmark row, with all timing columns in nanoseconds.
///
/// `mean` is the only always-present timing (every divan row has it);
/// the rest are `Option` because a malformed or truncated row may omit
/// later columns, and historical divan layouts vary.
struct Bench {
    /// The `::`-joined tree path, e.g. `mlkem::decaps::MLKEM512` or
    /// `algebraic::ntt`.
    full_name: String,
    /// Shortest observed sample (the `fastest` column).
    fastest: Option<f64>,
    /// Longest observed sample (the `slowest` column).
    slowest: Option<f64>,
    /// Median sample — the headline `value`.
    median: Option<f64>,
    /// Arithmetic mean of the samples.
    mean: f64,
    /// Number of samples divan collected.
    samples: Option<u64>,
    /// Total iterations across all samples.
    iters: Option<u64>,
}

impl Bench {
    /// Parses the timing columns of one divan result row, or `None` when
    /// the row carries no timings (parent rows of a nested tree only name
    /// their subtree).
    ///
    /// `rest` is the row with its tree prefix stripped: the leaf name
    /// followed by the `fastest` timing, then the `│`-delimited
    /// `slowest │ median │ mean │ samples │ iters` columns.
    fn from_columns(full_name: String, rest: &str) -> Option<Bench> {
        let columns: Vec<&str> = rest.split('│').collect();
        if columns.len() < 4 {
            return None;
        }

        // First column: the leaf name is the first token and `fastest`
        // follows it.
        let mut head_tokens = columns[0].split_whitespace();
        head_tokens.next()?;

        let fastest = parse_time(&head_tokens.collect::<Vec<_>>().join(" "));
        let slowest = parse_time(columns[1]);
        let median = parse_time(columns[2]);
        let mean = parse_time(columns[3])?;

        let samples = columns.get(4).and_then(|c| parse_count(c));
        let iters = columns.get(5).and_then(|c| parse_count(c));

        Some(Bench {
            full_name,
            fastest,
            slowest,
            median,
            mean,
            samples,
            iters,
        })
    }

    /// The headline value: median when divan reported it, else the mean.
    fn value(&self) -> f64 {
        self.median.unwrap_or(self.mean)
    }

    /// The half-spread `(slowest - fastest) / 2`, when both bounds exist.
    fn half_spread(&self) -> Option<f64> {
        match (self.fastest, self.slowest) {
            (Some(f), Some(s)) if s >= f => Some((s - f) / 2.0),
            _ => None,
        }
    }

    /// Writes this record as one object of the output JSON array.
    ///
    /// Always emits every key; absent timings/counts are `null` so the
    /// consumer schema is uniform.
    fn write_json(&self, w: &mut impl Write) -> io::Result<()> {
        let range = match self.half_spread() {
            Some(h) => format!("\"\\u00b1 {h} ns/iter\""),
            None => "null".to_string(),
        };

        write!(
            w,
            "  {{\"name\": \"{}\", \"unit\": \"ns/iter\", \"value\": {}, \"range\": {}, \
             \"median_ns\": {}, \"fastest_ns\": {}, \"slowest_ns\": {}, \"mean_ns\": {}, \
             \"samples\": {}, \"iters\": {}}}",
            self.full_name,
            self.value(),
            range,
            num_or_null(self.median),
            num_or_null(self.fastest),
            num_or_null(self.slowest),
            self.mean,
            count_or_null(self.samples),
            count_or_null(self.iters),
        )
    }
}

/// Parses a divan timing field (`"2.6 ns"`, `"1.2 µs"`) into nanoseconds.
fn parse_time(field: &str) -> Option<f64> {
    let mut parts = field.split_whitespace();
    let value: f64 = parts.next()?.parse().ok()?;
    let factor = match parts.next()? {
        "ps" => 0.001,
        "ns" => 1.0,
        "µs" | "us" => 1_000.0,
        "ms" => 1_000_000.0,
        "s" => 1_000_000_000.0,
        _ => return None,
    };

    Some(value * factor)
}

/// Parses a divan count field into an absolute count.
///
/// Divan abbreviates large iteration counts with a `K`/`M`/`G`/`T`
/// magnitude suffix that may be attached (`12.8M`) or space-separated
/// (`12.8 M`), and may use thousands separators (`204,800`). We scan the
/// numeric prefix and the first non-numeric character as the suffix,
/// wherever it falls, so both layouts parse.
fn parse_count(field: &str) -> Option<u64> {
    let mut digits = String::new();
    let mut suffix = None;

    for c in field.trim().chars() {
        if c.is_ascii_digit() || c == '.' {
            digits.push(c);
        } else if c == ',' || c.is_whitespace() {
            continue;
        } else {
            suffix = Some(c);
            break;
        }
    }

    let scale = match suffix {
        None => 1.0,
        Some('K' | 'k') => 1e3,
        Some('M') => 1e6,
        Some('G' | 'B') => 1e9,
        Some('T') => 1e12,
        Some(_) => return None,
    };

    let value: f64 = digits.parse().ok()?;

    Some((value * scale).round() as u64)
}

/// Splits a divan tree row into its depth and the content after the
/// branch glyph, or `None` if `line` is not a tree row.
///
/// Divan renders one 3-char column (`│  ` or `   `) per ancestor level,
/// then `├─ ` / `╰─ ` before the leaf name, so the depth is the number of
/// characters before the branch glyph divided by the column width, plus
/// one for the leaf itself.
fn split_tree_prefix(line: &str) -> Option<(usize, &str)> {
    let mut chars_before_glyph = 0usize;
    for (i, c) in line.char_indices() {
        match c {
            '├' | '╰' => {
                let rest = line[i..]
                    .trim_start_matches(['├', '╰', '─'])
                    .trim_start();
                return Some((chars_before_glyph / 3 + 1, rest));
            }
            '│' | ' ' => chars_before_glyph += 1,
            _ => return None,
        }
    }
    None
}

/// Renders an optional nanosecond timing as a JSON number or `null`.
fn num_or_null(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

/// Renders an optional count as a JSON number or `null`.
fn count_or_null(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

/// Reads divan output from stdin and writes the JSON array to stdout.
fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    let mut group = String::new();
    // Tree path of the current row's ancestors (path[0] is depth 1).
    let mut path: Vec<String> = Vec::new();
    let mut first = true;

    writeln!(w, "[")?;

    for line in stdin.lock().lines() {
        let line = line?;

        // A group header is a bare word followed by the `fastest` column
        // label; it sets the group for the tree that follows.
        if let Some(word) = line.split_whitespace().next() {
            if word.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && line.contains("fastest")
            {
                group = word.to_string();
                path.clear();
                continue;
            }
        }

        let Some((depth, rest)) = split_tree_prefix(&line) else {
            continue;
        };

        // The leaf name is the first token after the branch glyph.
        let Some(name) = rest.split_whitespace().next() else {
            continue;
        };
        let leading = name.chars().next().unwrap_or(' ');
        if !leading.is_ascii_alphanumeric() && leading != '_' {
            continue;
        }

        path.truncate(depth - 1);
        path.push(name.to_string());

        let mut segments = Vec::with_capacity(path.len() + 1);
        if !group.is_empty() {
            segments.push(group.as_str());
        }
        segments.extend(path.iter().map(String::as_str));
        let full_name = segments.join("::");

        // Parent rows carry no timings; they only extend the path.
        let Some(bench) = Bench::from_columns(full_name, rest) else {
            continue;
        };

        if !first {
            writeln!(w, ",")?;
        }
        first = false;

        bench.write_json(&mut w)?;
    }

    writeln!(w)?;
    writeln!(w, "]")?;

    Ok(())
}
