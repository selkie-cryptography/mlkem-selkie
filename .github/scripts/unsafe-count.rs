//! Count `unsafe` usage in src/ and output JSON.
//!
//! Usage: unsafe-count <sha>
//!
//! Compile: `rustc -O unsafe-count.rs -o unsafe-count`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::SystemTime;

fn count_unsafe_in_dir(dir: &Path) -> u32 {
    let mut count = 0;
    let Ok(entries) = fs::read_dir(dir) else { return 0 };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            count += count_unsafe_in_dir(&path);
        } else if path.extension().map_or(false, |e| e == "rs") {
            if let Ok(contents) = fs::read_to_string(&path) {
                for line in contents.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("//") { continue; }
                    if trimmed.contains("mod tests") { continue; }
                    if trimmed.contains("unsafe") { count += 1; }
                }
            }
        }
    }
    count
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 { eprintln!("usage: unsafe-count <sha>"); std::process::exit(1); }
    let sha = &args[1];

    let src = Path::new("src");
    let total = count_unsafe_in_dir(src);

    // Per-module breakdown.
    let mut modules = Vec::new();
    if let Ok(entries) = fs::read_dir(src) {
        let mut dirs: Vec<_> = entries.flatten()
            .filter(|e| e.path().is_dir())
            .collect();
        dirs.sort_by_key(|e| e.file_name());
        for entry in &dirs {
            let name = entry.file_name().to_string_lossy().to_string();
            let count = count_unsafe_in_dir(&entry.path());
            modules.push((name, count));
        }
    }
    // Root files.
    let root_count = fs::read_dir(src).ok()
        .map(|entries| {
            entries.flatten()
                .filter(|e| e.path().is_file() && e.path().extension().map_or(false, |x| x == "rs"))
                .map(|e| {
                    fs::read_to_string(e.path()).ok()
                        .map(|c| c.lines().filter(|l| !l.trim().starts_with("//") && l.contains("unsafe")).count() as u32)
                        .unwrap_or(0)
                })
                .sum::<u32>()
        })
        .unwrap_or(0);
    modules.push(("root".to_string(), root_count));

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"total\": {},", total)?;
    writeln!(w, "  \"modules\": [")?;
    for (i, (name, count)) in modules.iter().enumerate() {
        write!(w, "    {{\"name\": {}, \"count\": {}}}", json_str(name), count)?;
        if i + 1 < modules.len() { writeln!(w, ",")?; } else { writeln!(w)?; }
    }
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len()+2); o.push('"');
    for c in s.chars() { match c { '"'=>o.push_str("\\\""), '\\'=>o.push_str("\\\\"),
        '\n'=>o.push_str("\\n"), '\r'=>o.push_str("\\r"), '\t'=>o.push_str("\\t"),
        c if (c as u32)<0x20 => o.push_str(&format!("\\u{:04x}",c as u32)), _=>o.push(c) } }
    o.push('"'); o
}
fn iso8601_now() -> String {
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let s = dur.as_secs(); let (h,m,sc)=((s%86400)/3600,(s%3600)/60,s%60);
    let mut y=1970i64; let mut r=(s/86400) as i64;
    loop { let yd=if y%4==0&&(y%100!=0||y%400==0){366}else{365}; if r<yd{break} r-=yd; y+=1; }
    let lp=y%4==0&&(y%100!=0||y%400==0);
    let md=[31,if lp{29}else{28},31,30,31,30,31,31,30,31,30,31];
    let mut mo=0; for &d in &md { if r<d{break} r-=d; mo+=1; }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",y,mo+1,r+1,h,m,sc)
}
