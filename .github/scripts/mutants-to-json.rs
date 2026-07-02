//! Convert cargo-mutants outcomes.json to the CI site JSON format.
//!
//! Usage: mutants-to-json <outcomes.json> <sha>
//!
//! Compile: `rustc -O mutants-to-json.rs -o mutants-to-json`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: mutants-to-json <outcomes.json> <sha>");
        std::process::exit(1);
    }

    let contents = fs::read_to_string(&args[1])?;
    let sha = &args[2];

    // Count outcomes by summary type.
    let mut caught = 0u64;
    let mut missed = 0u64;
    let mut timeout = 0u64;
    let mut unviable = 0u64;

    // Also collect survivors (MissedMutant).
    struct Survivor {
        name: String,
        file: String,
        function: String,
        line: u64,
    }
    let mut survivors: Vec<Survivor> = Vec::new();

    // Parse by finding "summary" fields in each outcome object.
    // The outcomes array is at .outcomes[].
    // We split on `"summary":` to find each outcome's type.
    let outcomes_start = contents.find("\"outcomes\"").unwrap_or(0);
    let rest = &contents[outcomes_start..];

    // Find each outcome object by splitting on boundaries.
    // Each outcome has "summary": "CaughtMutant"|"MissedMutant"|etc.
    for chunk in rest.split("\"summary\"").skip(1) {
        let summary = extract_quoted_after(chunk, ":");
        match summary.as_str() {
            "CaughtMutant" => caught += 1,
            "MissedMutant" => {
                missed += 1;
                // Extract mutant info from the "scenario" that precedes this summary.
                // We need to look backwards, but it's easier to find the Mutant block
                // in the same chunk by looking for "name", "file", "function_name", "line".
                // The Mutant data is in the enclosing object.
            }
            "Timeout" => timeout += 1,
            "Unviable" => unviable += 1,
            "Success" => {} // baseline, skip
            _ => {}
        }
    }

    // Second pass: extract survivor details from MissedMutant outcomes.
    // Split on each outcome object boundary more carefully.
    // Each outcome is a JSON object within the "outcomes" array.
    // We look for objects containing "MissedMutant".
    let mut pos = 0;
    while let Some(idx) = contents[pos..].find("\"MissedMutant\"") {
        let abs_idx = pos + idx;
        // Find the enclosing object by searching backwards for the nearest "Mutant" block.
        // The "name" field of the Mutant is before the summary.
        let obj_start = contents[..abs_idx].rfind("\"Mutant\"").unwrap_or(abs_idx);
        let obj_region = &contents[obj_start..std::cmp::min(abs_idx + 200, contents.len())];

        let name = extract_nested_string(obj_region, "name");
        let file = extract_nested_string(obj_region, "file");
        let function = extract_nested_string(obj_region, "function_name");
        let line = extract_nested_num(obj_region, "line");

        if !file.is_empty() {
            survivors.push(Survivor { name, file, function, line });
        }

        pos = abs_idx + 14; // skip past "MissedMutant"
    }

    // Sort survivors by file, line.
    survivors.sort_by(|a, b| a.file.cmp(&b.file).then(a.line.cmp(&b.line)));

    let total = caught + missed + timeout + unviable;
    let updated_at = iso8601_now();

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&updated_at))?;
    writeln!(w, "  \"summary\": {{")?;
    writeln!(w, "    \"caught\": {},", caught)?;
    writeln!(w, "    \"missed\": {},", missed)?;
    writeln!(w, "    \"timeout\": {},", timeout)?;
    writeln!(w, "    \"unviable\": {},", unviable)?;
    writeln!(w, "    \"total\": {}", total)?;
    writeln!(w, "  }},")?;
    writeln!(w, "  \"survivors\": [")?;

    for (i, s) in survivors.iter().enumerate() {
        write!(w, "    {{\"name\": {}, \"file\": {}, \"function\": {}, \"line\": {}}}",
            json_str(&s.name), json_str(&s.file), json_str(&s.function), s.line)?;
        if i + 1 < survivors.len() {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;

    Ok(())
}

/// Extract the first quoted string after a marker.
fn extract_quoted_after(text: &str, marker: &str) -> String {
    let Some(idx) = text.find(marker) else { return String::new() };
    let rest = &text[idx + marker.len()..];
    let rest = rest.trim_start();
    if !rest.starts_with('"') { return String::new() }
    let start = 1;
    let mut end = start;
    let bytes = rest.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' && bytes[end - 1] != b'\\' { break }
        end += 1;
    }
    rest[start..end].to_string()
}

/// Extract a string value for a key within a region.
fn extract_nested_string(region: &str, key: &str) -> String {
    let needle = format!("\"{}\"", key);
    let Some(idx) = region.find(&needle) else { return String::new() };
    extract_quoted_after(&region[idx + needle.len()..], ":")
}

/// Extract a numeric value for a key within a region.
fn extract_nested_num(region: &str, key: &str) -> u64 {
    let needle = format!("\"{}\"", key);
    let Some(idx) = region.find(&needle) else { return 0 };
    let rest = &region[idx + needle.len()..];
    let Some(colon) = rest.find(':') else { return 0 };
    let after = rest[colon + 1..].trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
    after[..end].parse().unwrap_or(0)
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
