//! Convert tacet bench output to JSON for the CI site.
//!
//! Usage: tacet-to-json <tacet-output.txt> <sha>
//!
//! Parses lines like:
//!   PASS  fp_mul               [shared_hw   ] leak_prob=0.0123
//!   FAIL  scalar_mul           [pq_sentinel ] leak_prob=0.9200 exploit=High(45.2ns)
//!   SKIP  keygen               [adjacent    ] inconclusive: TooFewSamples
//!
//! Compile: `rustc -O tacet-to-json.rs -o tacet-to-json`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::SystemTime;

struct Result {
    name: String,
    model: String,
    status: String,
    leak_prob: Option<f64>,
    detail: String,
}

fn parse_output(contents: &str) -> Vec<Result> {
    let mut results = Vec::new();

    for line in contents.lines() {
        let line = line.trim();
        // Match: STATUS  name  [model] details
        let (status, rest) = if line.starts_with("PASS") {
            ("pass", &line[4..])
        } else if line.starts_with("FAIL") {
            ("fail", &line[4..])
        } else if line.starts_with("SKIP") {
            ("skip", &line[4..])
        } else {
            continue;
        };

        let rest = rest.trim();

        // Extract name (first word).
        let name = match rest.split_whitespace().next() {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Extract model from [brackets].
        let model = if let Some(start) = rest.find('[') {
            if let Some(end) = rest[start..].find(']') {
                rest[start + 1..start + end].trim().to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Extract leak_prob if present.
        let leak_prob = if let Some(idx) = rest.find("leak_prob=") {
            let after = &rest[idx + 10..];
            let end = after.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(after.len());
            after[..end].parse().ok()
        } else {
            None
        };

        // Everything after the ] bracket.
        let detail = rest.find(']').map(|i| rest[i + 1..].trim().to_string()).unwrap_or_default();

        results.push(Result {
            name,
            model,
            status: status.to_string(),
            leak_prob,
            detail,
        });
    }

    results
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
        eprintln!("usage: tacet-to-json <tacet-output.txt> <sha>");
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
        write!(w, "    {{\"name\": {}, \"model\": {}, \"status\": {}",
            json_str(&r.name), json_str(&r.model), json_str(&r.status))?;
        if let Some(lp) = r.leak_prob {
            write!(w, ", \"leak_prob\": {:.4}", lp)?;
        }
        if !r.detail.is_empty() {
            write!(w, ", \"detail\": {}", json_str(&r.detail))?;
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
