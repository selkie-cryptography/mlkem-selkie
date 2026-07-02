//! Convert dudect-bencher output to JSON for the CI site.
//!
//! Usage: dudect-to-json <dudect-output.txt> <sha>
//!
//! Parses lines like:
//!   bench fp_mul ... : n == +0.100M, max t = +0.42345, max tau = +0.00134, (5/tau)^2 = 13913043
//!
//! Compile: `rustc -O dudect-to-json.rs -o dudect-to-json`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::SystemTime;

const THRESHOLD: f64 = 4.5;

struct Result {
    name: String,
    max_t: f64,
    max_tau: Option<f64>,
    samples_m: f64,
    samples_needed: Option<u64>,
    status: &'static str,
}

fn parse_output(contents: &str) -> Vec<Result> {
    let mut results = Vec::new();

    for line in contents.lines() {
        // Match: bench <name> ... : n == +X.XXXM, max t = +X.XXXXX, max tau = +X.XXXXX, (5/tau)^2 = N
        if !line.contains("bench ") || !line.contains("max t =") {
            continue;
        }

        // Extract name: "bench <name> "
        let name = match line.split_whitespace().nth(1) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Extract n (samples in millions)
        let samples_m = extract_after(line, "n == ")
            .and_then(|s| s.trim_start_matches('+').replace('M', "").parse::<f64>().ok())
            .unwrap_or(0.0);

        // Extract max t
        let max_t = match extract_after(line, "max t = ") {
            Some(s) => match s.trim_start_matches('+').parse::<f64>() {
                Ok(v) => v,
                Err(_) => continue,
            },
            None => continue,
        };

        // Extract max tau (optional)
        let max_tau = extract_after(line, "max tau = ")
            .and_then(|s| s.trim_start_matches('+').parse::<f64>().ok());

        // Extract (5/tau)^2 (optional)
        let samples_needed = extract_after(line, "(5/tau)^2 = ")
            .and_then(|s| s.parse::<u64>().ok());

        let status = if max_t.abs() < THRESHOLD { "pass" } else { "fail" };

        results.push(Result {
            name,
            max_t,
            max_tau,
            samples_m,
            samples_needed,
            status,
        });
    }

    results
}

/// Extract the value string after a marker, up to the next comma or end of line.
fn extract_after<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    let idx = line.find(marker)?;
    let rest = &line[idx + marker.len()..];
    let end = rest.find(',').or_else(|| rest.find(')')).unwrap_or(rest.len());
    let val = rest[..end].trim();
    if val.is_empty() { None } else { Some(val) }
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
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn iso8601_now() -> String {
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    let secs = dur.as_secs();
    let time_secs = secs % 86400;
    let h = time_secs / 3600;
    let m = (time_secs % 3600) / 60;
    let s = time_secs % 60;

    let mut y = 1970i64;
    let mut remaining = (secs / 86400) as i64;
    loop {
        let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if remaining < yd { break; }
        remaining -= yd;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let md = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0;
    for &d in &md {
        if remaining < d { break; }
        remaining -= d;
        mo += 1;
    }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo + 1, remaining + 1, h, m, s)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: dudect-to-json <dudect-output.txt> <sha>");
        std::process::exit(1);
    }

    let contents = fs::read_to_string(&args[1])?;
    let sha = &args[2];
    let results = parse_output(&contents);

    let pass_count = results.iter().filter(|r| r.status == "pass").count();
    let fail_count = results.iter().filter(|r| r.status == "fail").count();

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"pass_count\": {},", pass_count)?;
    writeln!(w, "  \"fail_count\": {},", fail_count)?;
    writeln!(w, "  \"total\": {},", results.len())?;
    writeln!(w, "  \"results\": [")?;

    for (i, r) in results.iter().enumerate() {
        write!(w, "    {{\"name\": {}, \"max_t\": {:.5}, \"samples_m\": {:.3}, \"status\": {}",
            json_str(&r.name), r.max_t, r.samples_m, json_str(r.status))?;
        if let Some(tau) = r.max_tau {
            write!(w, ", \"max_tau\": {:.5}", tau)?;
        }
        if let Some(needed) = r.samples_needed {
            write!(w, ", \"samples_needed\": {}", needed)?;
        }
        write!(w, "}}")?;
        if i + 1 < results.len() {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;

    Ok(())
}
