//! Convert LCOV coverage data to JSON for the CI site.
//!
//! Usage: lcov-to-json <lcov-file> <sha>
//!
//! Produces per-module, per-file, per-line coverage data grouped
//! by Rust module (first path component under `src/`).
//!
//! Compile: `rustc -O lcov-to-json.rs -o lcov-to-json`

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::SystemTime;

struct FileCoverage {
    abspath: String,
    relpath: String,
    module: String,
    /// (line_number, execution_count)
    lines: Vec<(u32, u64)>,
    hit: u32,
    found: u32,
}

fn parse_lcov(contents: &str) -> Vec<FileCoverage> {
    let mut files = Vec::new();
    let mut current: Option<FileCoverage> = None;

    for line in contents.lines() {
        if let Some(path) = line.strip_prefix("SF:") {
            // Extract path relative to src/ (use last occurrence,
            // since the absolute path may contain /src/ earlier).
            let Some(idx) = path.rfind("/src/") else {
                current = None;
                continue;
            };
            let relpath = &path[idx + 5..];

            let parts: Vec<&str> = relpath.splitn(2, '/').collect();
            let module = if parts.len() == 1 {
                parts[0].strip_suffix(".rs").unwrap_or(parts[0])
            } else {
                parts[0]
            };

            current = Some(FileCoverage {
                abspath: path.to_string(),
                relpath: relpath.to_string(),
                module: module.to_string(),
                lines: Vec::new(),
                hit: 0,
                found: 0,
            });
        } else if let Some(da) = line.strip_prefix("DA:") {
            if let Some(ref mut file) = current {
                let mut parts = da.splitn(2, ',');
                if let (Some(lineno_s), Some(count_s)) = (parts.next(), parts.next()) {
                    if let (Ok(lineno), Ok(count)) = (lineno_s.parse::<u32>(), count_s.parse::<u64>())
                    {
                        file.lines.push((lineno, count));
                        file.found += 1;
                        if count > 0 {
                            file.hit += 1;
                        }
                    }
                }
            }
        } else if line == "end_of_record" {
            if let Some(file) = current.take() {
                files.push(file);
            }
        }
    }

    files
}

fn pct(hit: u32, found: u32) -> f64 {
    if found == 0 {
        0.0
    } else {
        hit as f64 * 100.0 / found as f64
    }
}

fn iso8601_now() -> String {
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    let secs = dur.as_secs();
    // Minimal UTC formatting without chrono.
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let h = time_secs / 3600;
    let m = (time_secs % 3600) / 60;
    let s = time_secs % 60;

    // Days since epoch to Y-M-D (simplified Gregorian).
    let mut y = 1970i64;
    let mut remaining = days as i64;
    loop {
        let year_days = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
        if remaining < year_days {
            break;
        }
        remaining -= year_days;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut mo = 0;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        mo += 1;
    }

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        mo + 1,
        remaining + 1,
        h,
        m,
        s
    )
}

/// JSON-escape a string (handles the minimal set needed for file paths).
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
            c if (c as u32) < 0x20 => {
                // RFC 8259: escape all control characters U+0000–U+001F.
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: lcov-to-json <lcov-file> <sha>");
        std::process::exit(1);
    }

    let contents = fs::read_to_string(&args[1])?;
    let sha = &args[2];
    let files = parse_lcov(&contents);

    let total_hit: u32 = files.iter().map(|f| f.hit).sum();
    let total_found: u32 = files.iter().map(|f| f.found).sum();

    // Read source files for line text.
    let mut source_cache: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in &files {
        if !source_cache.contains_key(&f.abspath) {
            if let Ok(src) = fs::read_to_string(&f.abspath) {
                source_cache.insert(
                    f.abspath.clone(),
                    src.lines().map(|l| l.to_string()).collect(),
                );
            }
        }
    }

    // Group by module, preserving order.
    let mut modules: BTreeMap<&str, Vec<&FileCoverage>> = BTreeMap::new();
    for f in &files {
        modules.entry(&f.module).or_default().push(f);
    }

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(
        w,
        "  \"total\": {{\"covered\": {}, \"total\": {}, \"percent\": {:.4}}},",
        total_hit,
        total_found,
        pct(total_hit, total_found)
    )?;

    writeln!(w, "  \"modules\": [")?;
    let mod_count = modules.len();
    for (mi, (module, mod_files)) in modules.iter().enumerate() {
        let mhit: u32 = mod_files.iter().map(|f| f.hit).sum();
        let mfound: u32 = mod_files.iter().map(|f| f.found).sum();

        writeln!(
            w,
            "    {{\"name\": {}, \"covered\": {}, \"total\": {}, \"percent\": {:.4},",
            json_str(module),
            mhit,
            mfound,
            pct(mhit, mfound)
        )?;
        writeln!(w, "     \"files\": [")?;

        for (fi, file) in mod_files.iter().enumerate() {
            write!(
                w,
                "      {{\"name\": {}, \"covered\": {}, \"total\": {}, \"percent\": {:.4}, \"lines\": [",
                json_str(&file.relpath),
                file.hit,
                file.found,
                pct(file.hit, file.found)
            )?;

            // Emit every line of the source file.
            // Instrumented lines get their count; non-instrumented lines
            // (comments, blanks, declarations) get -1.
            let cov_map: std::collections::HashMap<u32, u64> =
                file.lines.iter().copied().collect();
            if let Some(src_lines) = source_cache.get(&file.abspath) {
                for (i, line_text) in src_lines.iter().enumerate() {
                    let lineno = (i + 1) as u32;
                    let count = cov_map.get(&lineno).copied().unwrap_or_default();
                    let instrumented = cov_map.contains_key(&lineno);
                    if i > 0 {
                        write!(w, ",")?;
                    }
                    // [line_number, count (-1 = not instrumented), "source"]
                    let ct: i64 = if instrumented { count as i64 } else { -1 };
                    write!(w, "[{},{},{}]", lineno, ct, json_str(line_text))?;
                }
            } else {
                // Fallback: only instrumented lines.
                for (li, &(lineno, count)) in file.lines.iter().enumerate() {
                    if li > 0 {
                        write!(w, ",")?;
                    }
                    write!(w, "[{},{},\"\"]", lineno, count)?;
                }
            }

            write!(w, "]}}")?;
            if fi + 1 < mod_files.len() {
                writeln!(w, ",")?;
            } else {
                writeln!(w)?;
            }
        }

        write!(w, "    ]}}")?;
        if mi + 1 < mod_count {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;

    Ok(())
}
