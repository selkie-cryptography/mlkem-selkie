//! Signal build state (running/done) to the CI dashboard.
//!
//! Usage: ci-signal <kind> <state> <sha>
//!
//! Writes `{"state":"<state>","sha":"<sha>"}` to `/data/<kind>/status.json`
//! on the Fly.io CI site. Used at the start of CI jobs to signal "running"
//! so the dashboard can animate the favicon.
//!
//! Compile: `rustc -O ci-signal.rs -o ci-signal`

use std::env;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};

const APP: &str = "mlkem-selkie-ci";

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: ci-signal <kind> <state> <sha>");
        std::process::exit(1);
    }

    let kind = &args[1];
    let state = &args[2];
    let sha = &args[3];
    let dir = format!("/data/{kind}");

    // The CI VM is configured with min_machines_running=1 and
    // auto_stop_machines=false, so we can ssh in directly without
    // a wake-up dance.

    // Write status.
    let status = format!("{{\"state\":{},\"sha\":{}}}", json_str(state), json_str(sha));
    let path = "/tmp/ci-status.json";
    fs::write(path, &status).expect("failed to write status file");

    // Upload.
    ssh_cmd(&format!("rm -f {dir}/status.json"));
    sftp_put(path, &format!("{dir}/status.json"));

    // Also update /data/head.json so the dashboard can show the
    // new commit immediately, before any data uploads finish.
    if state == "running" {
        let subject = commit_subject(sha);
        let head = format!(
            "{{\"sha\":{},\"subject\":{},\"updated_at\":{}}}",
            json_str(sha),
            json_str(&subject),
            json_str(&iso8601_now())
        );
        let head_path = "/tmp/ci-head.json";
        fs::write(head_path, &head).expect("failed to write head.json");
        ssh_cmd("rm -f /data/head.json");
        sftp_put(head_path, "/data/head.json");
    }

    eprintln!("[ci-signal] {kind}: {state} ({sha})");
}

fn run(cmd: &str, args: &[&str]) {
    let status = Command::new(cmd)
        .args(args)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("failed to run {cmd}: {e}");
            std::process::exit(1);
        });
    if !status.success() {
        eprintln!("warning: {cmd} {:?} exited with {status}", args);
    }
}

fn ssh_cmd(cmd: &str) {
    run("flyctl", &["ssh", "console", "-a", APP, "-C", cmd]);
}

fn sftp_put(local: &str, remote: &str) {
    let input = format!("put {local} {remote}\n");
    let mut child = Command::new("flyctl")
        .args(["ssh", "sftp", "shell", "-a", APP])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| {
            eprintln!("sftp spawn failed: {e}");
            std::process::exit(1);
        });
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    if !out.status.success() {
        eprintln!(
            "warning: sftp put {local} → {remote} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

fn commit_subject(sha: &str) -> String {
    let output = Command::new("git")
        .args(["log", "-1", "--format=%s", sha])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok();
    output
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn iso8601_now() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    let secs = dur.as_secs();
    let (h, m, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    let mut y = 1970i64;
    let mut rem = (secs / 86400) as i64;
    loop {
        let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if rem < yd { break; }
        rem -= yd;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let md = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0;
    for &d in &md {
        if rem < d { break; }
        rem -= d;
        mo += 1;
    }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo + 1, rem + 1, h, m, s)
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
