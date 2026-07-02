//! Collect platform test matrix results and output JSON.
//!
//! Usage: platform-report <run-id> <sha>
//!
//! curl + jq (rather than the `gh` CLI) so this works on any runner
//! whose image isn't expected to carry `gh`.
//!
//! Compile: `rustc -O platform-report.rs -o platform-report`

use std::env;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

struct Platform {
    name: String,
    backend: String,
    target: String,
    status: String,
}

impl Platform {
    /// Maps a `test (<backend>)` job to the target its runner + RUSTFLAGS
    /// pin (see ci.yml): the polynomial backend is a build.rs cfg driven by
    /// the target CPU, not a cargo feature.
    fn from_job(name: &str, conclusion: &str) -> Option<Platform> {
        let backend = name
            .find('(')
            .and_then(|start| name[start..].find(')').map(|end| &name[start + 1..start + end]))?
            .trim()
            .to_string();

        let target = match backend.as_str() {
            "portable" => "x86_64-unknown-linux-gnu",
            "avx2" => "x86_64-unknown-linux-gnu +x86-64-v3",
            "neon" => "aarch64-apple-darwin",
            _ => "unknown",
        }
        .to_string();

        let status = match conclusion {
            "success" => "pass",
            "failure" => "fail",
            "skipped" => "skip",
            _ => "unknown",
        };

        Some(Platform {
            name: name.to_string(),
            backend,
            target,
            status: status.to_string(),
        })
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: platform-report <run-id> <sha>");
        std::process::exit(1);
    }
    let run_id = &args[1];
    let sha = &args[2];

    let repo = env::var("GITHUB_REPOSITORY").unwrap_or_default();
    if repo.is_empty() {
        eprintln!("GITHUB_REPOSITORY is not set");
        std::process::exit(1);
    }
    // Accept either GH_TOKEN (what `gh` reads) or GITHUB_TOKEN (default
    // in GHA jobs) so this works in either env shape.
    let token = env::var("GH_TOKEN")
        .or_else(|_| env::var("GITHUB_TOKEN"))
        .unwrap_or_default();
    if token.is_empty() {
        eprintln!("no GH_TOKEN / GITHUB_TOKEN in env");
        std::process::exit(1);
    }

    // per_page=100 to avoid paginating; if matrix ever grows past
    // that, follow the Link header.
    let url = format!(
        "https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
    );
    let auth = format!("Authorization: Bearer {token}");
    let jq_filter = r#".jobs[] | select(.name | startswith("test (")) | "\(.name)|\(.conclusion)""#;

    // `set -o pipefail` so a curl failure (e.g. 403 because the
    // workflow's GITHUB_TOKEN lacks `actions: read`) surfaces as a
    // pipeline failure. Without it, curl's non-zero exit on HTTP
    // error is masked by jq's clean exit on empty input, and the
    // script silently emits an empty `platforms` array.
    let pipeline = format!(
        "set -o pipefail; curl -fsSL -H 'Accept: application/vnd.github+json' -H \"$AUTH\" {url} | jq -r {filter}",
        url = shell_escape(&url),
        filter = shell_escape(jq_filter),
    );
    let output = Command::new("bash")
        .arg("-c")
        .arg(&pipeline)
        .env("AUTH", &auth)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to spawn bash for curl|jq pipeline");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("::error::curl|jq pipeline failed (status {}): {}", output.status, stderr);
        std::process::exit(1);
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let line_count = text.lines().filter(|l| !l.trim().is_empty()).count();
    eprintln!("REST API returned {line_count} matching job line(s)");
    if line_count == 0 {
        eprintln!(
            "::error::platform-report: zero jobs matched `startswith(\"test (\")` in run {run_id}. \
             Check that the test / test-neon / test-avx2 jobs actually ran (and that this job has `actions: read`)."
        );
        std::process::exit(1);
    }
    let mut platforms = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let parts: Vec<&str> = line.splitn(2, '|').collect();
        if parts.len() != 2 { continue; }

        if let Some(platform) = Platform::from_job(parts[0].trim(), parts[1].trim()) {
            platforms.push(platform);
        }
    }

    let pass = platforms.iter().filter(|p| p.status == "pass").count();
    let fail = platforms.iter().filter(|p| p.status == "fail").count();

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"pass\": {},", pass)?;
    writeln!(w, "  \"fail\": {},", fail)?;
    writeln!(w, "  \"total\": {},", platforms.len())?;
    writeln!(w, "  \"platforms\": [")?;

    for (i, p) in platforms.iter().enumerate() {
        write!(
            w,
            "    {{\"name\": {}, \"backend\": {}, \"target\": {}, \"status\": {}}}",
            json_str(&p.name),
            json_str(&p.backend),
            json_str(&p.target),
            json_str(&p.status)
        )?;
        if i + 1 < platforms.len() { writeln!(w, ",")?; } else { writeln!(w)?; }
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}

// Single-quote-wrap with `'\''` escaping. Our inputs (REST URL,
// jq filter) never contain `'` today; we quote rigorously anyway so
// future inputs can't surprise us.
fn shell_escape(s: &str) -> String {
    let escaped = s.replace('\'', "'\\''");
    format!("'{}'", escaped)
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
