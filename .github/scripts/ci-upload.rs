//! Upload CI data (coverage, bench, mutants, tacet, ctgrind) to the Fly.io CI site.
//!
//! Usage: ci-upload [--dry-run] [--pr-only] <kind> <json-file> <sha>
//!        ci-upload --assets <local-root> <remote-subdir>   (flamegraph SVGs)
//!
//! - Stores per-commit data at /data/<kind>/<sha>.json
//! - Updates /data/<kind>/latest.json
//! - Maintains /data/<kind>/index.json (last 50 summaries)
//! - Prunes per-commit files beyond 30 entries
//! - Writes status.json before/after for the site's indicator
//!
//! `--dry-run` runs the structural + per-kind floor validation and exits
//! with the result. No ssh, sftp, manifest update, prune, or signal
//! write happens — used by scripts-compile.yml to gate `validate_floor`
//! regressions at PR time rather than at next-cron time.
//!
//! `--pr-only` writes only the per-sha file (skips status, latest, index,
//! prune, manifest). Used by PR runs that need per-commit data uploaded
//! for cross-domain reports without clobbering main's baseline state.
//!
//! Compile: `rustc -O ci-upload.rs -o ci-upload`

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

const APP: &str = "mlkem-selkie-ci";
const SITE: &str = "https://mlkem-selkie-ci.fly.dev";
const MAX_INDEX: usize = 50;
const MAX_FILES: usize = 30;

/// Every data kind the site serves. Backend-matrixed kinds (`bench-*`,
/// `instructions-*`, `cycles-*`) and the mutants arch split are distinct
/// kinds — each gets its own per-commit files, index, and manifest entry.
const ALL_KINDS: &[&str] = &[
    "coverage",
    "bench-generic",
    "bench-avx2",
    "bench-neon",
    "instructions-generic",
    "instructions-avx2",
    "cycles-generic",
    "cycles-avx2",
    "mutants",
    "mutants-aarch64",
    "dudect",
    "tacet",
    "ctgrind",
    "deny",
    "unsafe",
    "platform",
    "kat",
];

/// Collapses a backend/arch-suffixed kind to its family for index and
/// floor dispatch: `bench-neon` → `bench`, `mutants-aarch64` → `mutants`.
fn kind_family(kind: &str) -> &str {
    kind.split_once('-').map_or(kind, |(family, _)| family)
}

fn main() {
    // Asset mode: `ci-upload --assets <local-root> <remote-subdir>` hosts
    // flamegraph SVGs. Separate from the JSON-payload path below (no floor,
    // index, or manifest) and run ungated so PR runs can host their graphs.
    let raw: Vec<String> = env::args().skip(1).collect();
    if raw.first().map(String::as_str) == Some("--assets") {
        upload_assets(&raw[1..]);
        return;
    }

    let mut dry_run = false;
    let mut pr_only = false;
    let positional: Vec<String> = env::args()
        .skip(1)
        .filter(|a| match a.as_str() {
            "--dry-run" => { dry_run = true; false }
            "--pr-only" => { pr_only = true; false }
            _ => true,
        })
        .collect();
    if positional.len() != 3 {
        eprintln!("usage: ci-upload [--dry-run] [--pr-only] <kind> <json-file> <sha>");
        std::process::exit(1);
    }

    let kind = &positional[0];
    let json_path = &positional[1];
    let sha = &positional[2];
    let dir = format!("/data/{kind}");

    let mut json_contents = fs::read_to_string(json_path)
        .unwrap_or_else(|e| { eprintln!("cannot read {json_path}: {e}"); std::process::exit(1); });

    validate_payload(kind, json_path, &json_contents, sha);

    if dry_run {
        eprintln!("[ci-upload] dry-run ok: {kind} {json_path} {sha}");
        return;
    }

    // Inject run_id from GITHUB_RUN_ID so the dashboard can link
    // directly to the Actions run. Inserted after the opening `{`.
    if let Ok(run_id) = env::var("GITHUB_RUN_ID") {
        if json_contents.starts_with('{') {
            json_contents = format!(
                "{{\"run_id\":{},{}",
                json_str(&run_id),
                &json_contents[1..]
            );
            // Rewrite the local file so the per-sha and latest copies include it.
            let _ = fs::write(json_path, &json_contents);
        }
    }

    if pr_only {
        // Per-sha upload only. status/latest/index/prune/manifest are
        // main-baseline state PRs must not clobber; the data is fetched
        // by name (kind/<sha>.json) for cross-domain PR reports.
        ssh_cmd(&format!("rm -f {dir}/{sha}.json"));
        sftp_put(json_path, &format!("{dir}/{sha}.json"));
        eprintln!("[ci-upload] PR-only: uploaded {sha}.json");
        return;
    }

    // The CI VM is configured with min_machines_running=1 and
    // auto_stop_machines=false, so we can ssh in directly without
    // a wake-up dance.

    // Signal "running".
    let status_running = format!("{{\"state\":\"running\",\"sha\":{}}}", json_str(sha));
    write_tmp("status.json", &status_running);
    sftp_put("/tmp/status.json", &format!("{dir}/status.json"));
    eprintln!("[ci-upload] signaled running");

    // Upload per-commit and latest.
    ssh_cmd(&format!("rm -f {dir}/latest.json {dir}/{sha}.json"));
    sftp_put(json_path, &format!("{dir}/{sha}.json"));
    sftp_put(json_path, &format!("{dir}/latest.json"));
    eprintln!("[ci-upload] uploaded {sha}.json and latest.json");

    // Update index.
    let existing_index = fetch_url(&format!("{SITE}/{kind}/index.json"))
        .unwrap_or_else(|| "[]".to_string());
    let new_index = build_index(kind, sha, &json_contents, &existing_index);
    write_tmp("index-new.json", &new_index);
    ssh_cmd(&format!("rm -f {dir}/index.json"));
    sftp_put("/tmp/index-new.json", &format!("{dir}/index.json"));
    eprintln!("[ci-upload] updated index");

    // Prune old per-commit files.
    let keep_shas = extract_index_shas(&new_index, MAX_FILES);
    let files = ssh_ls(&dir);
    let mut pruned = 0;
    for f in &files {
        if !f.ends_with(".json") { continue }
        match f.as_str() {
            "latest.json" | "index.json" | "status.json" => continue,
            _ => {}
        }
        let file_sha = f.trim_end_matches(".json");
        if !keep_shas.contains(&file_sha.to_string()) {
            ssh_cmd(&format!("rm -f {dir}/{f}"));
            pruned += 1;
        }
    }
    if pruned > 0 { eprintln!("[ci-upload] pruned {pruned} old files"); }

    // Signal "done".
    let status_done = format!("{{\"state\":\"done\",\"sha\":{}}}", json_str(sha));
    write_tmp("status.json", &status_done);
    ssh_cmd(&format!("rm -f {dir}/status.json"));
    sftp_put("/tmp/status.json", &format!("{dir}/status.json"));
    eprintln!("[ci-upload] done");

    // Update the global manifest so the dashboard can poll one file.
    update_manifest(kind, sha);
}

/// Updates `/data/manifest.json` — a single object mapping each data
/// kind to its latest SHA and timestamp. The dashboard polls this one
/// file instead of 20+ individual files.
fn update_manifest(kind: &str, sha: &str) {
    let manifest_url = format!("{SITE}/manifest.json");
    let existing = fetch_url(&manifest_url).unwrap_or_else(|| "{}".to_string());

    // Parse existing entries (simple key extraction).
    let mut entries = Vec::new();
    for k in ALL_KINDS {
        if *k == kind {
            // Replace with the new SHA.
            entries.push(format!("  {}: {}", json_str(k), json_str(sha)));
        } else {
            // Preserve the existing value.
            let existing_sha = extract_string(&existing, k);
            if !existing_sha.is_empty() {
                entries.push(format!("  {}: {}", json_str(k), json_str(&existing_sha)));
            }
        }
    }

    let manifest = format!("{{\n{}\n}}", entries.join(",\n"));
    write_tmp("manifest.json", &manifest);
    ssh_cmd("rm -f /data/manifest.json");
    sftp_put("/tmp/manifest.json", "/data/manifest.json");
    eprintln!("[ci-upload] updated manifest");
}

/// Gets the commit subject line via `git log`, or empty string if unavailable.
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

/// Build a new index JSON array by prepending this commit's entry
/// and capping at MAX_INDEX entries.
/// Extracts a compact `{"<name>": <value>, ...}` map from a payload's
/// `results` array, for the dashboard's per-bench trend sparklines. Each
/// result object is `{"name": ..., "<field>": N, ...}` (`instructions` for
/// gungraun, `cycles` for the rdtsc bench); the array runs to the end of the
/// payload, so splitting on `{` yields one chunk per result (the trailing
/// `]}` has no name and is skipped).
fn extract_results_map(json: &str, field: &str) -> String {
    let Some(start) = json.find("\"results\"") else {
        return "{}".to_string();
    };

    let mut pairs = Vec::new();
    for chunk in json[start..].split('{').skip(1) {
        let obj = format!("{{{chunk}");
        let name = extract_string(&obj, "name");
        if name.is_empty() {
            continue;
        }
        pairs.push(format!("{}:{}", json_str(&name), extract_num_u64(&obj, field)));
    }

    format!("{{{}}}", pairs.join(","))
}

fn build_index(kind: &str, sha: &str, json: &str, existing: &str) -> String {
    let updated_at = extract_string(json, "updated_at");
    let subject = commit_subject(sha);
    let subject_field = if subject.is_empty() {
        String::new()
    } else {
        format!(",\"subject\":{}", json_str(&subject))
    };

    // Build the new entry based on the kind family (backend-suffixed kinds
    // share their family's summary shape).
    let entry = match kind_family(kind) {
        "coverage" => {
            let pct = extract_num_f64(json, "percent");
            format!("{{\"sha\":{},\"percent\":{:.4},\"updated_at\":{}{subject_field}}}",
                json_str(sha), pct, json_str(&updated_at))
        }
        "mutants" => {
            let caught = extract_num_in_section(json, "summary", "caught");
            let missed = extract_num_in_section(json, "summary", "missed");
            let timeout = extract_num_in_section(json, "summary", "timeout");
            format!("{{\"sha\":{},\"caught\":{},\"missed\":{},\"timeout\":{},\"updated_at\":{}{subject_field}}}",
                json_str(sha), caught, missed, timeout, json_str(&updated_at))
        }
        "dudect" => {
            let pass = extract_num_u64(json, "pass_count");
            let fail = extract_num_u64(json, "fail_count");
            format!("{{\"sha\":{},\"pass_count\":{},\"fail_count\":{},\"updated_at\":{}{subject_field}}}",
                json_str(sha), pass, fail, json_str(&updated_at))
        }
        "bench" => {
            // Headline library-bench medians (ns, ML-KEM-768) for the
            // dashboard trend charts; `summary.*` keys are unique to the
            // payload root.
            let keygen = extract_num_f64(json, "keygen_ns");
            let encaps = extract_num_f64(json, "encaps_ns");
            let decaps = extract_num_f64(json, "decaps_ns");
            format!("{{\"sha\":{},\"keygen_ns\":{},\"encaps_ns\":{},\"decaps_ns\":{},\"updated_at\":{}{subject_field}}}",
                json_str(sha), keygen, encaps, decaps, json_str(&updated_at))
        }
        "instructions" => {
            // Per-bench instruction counts as a compact {name: instructions}
            // map, for the dashboard's per-row trend sparklines.
            let results = extract_results_map(json, "instructions");
            format!("{{\"sha\":{},\"results\":{},\"updated_at\":{}{subject_field}}}",
                json_str(sha), results, json_str(&updated_at))
        }
        "cycles" => {
            // Per-bench median cycle counts as a compact {name: cycles} map,
            // same shape as the instructions index.
            let results = extract_results_map(json, "cycles");
            format!("{{\"sha\":{},\"results\":{},\"updated_at\":{}{subject_field}}}",
                json_str(sha), results, json_str(&updated_at))
        }
        _ => {
            format!("{{\"sha\":{},\"updated_at\":{}{subject_field}}}", json_str(sha), json_str(&updated_at))
        }
    };

    let mut entries = Vec::new();
    entries.push(entry);

    // Walk `existing` and slice out each top-level `{...}` entry by tracking
    // brace depth (skipping braces inside string literals). Naïve split-on-`{`
    // loses entries whose values contain nested objects — e.g. instructions'
    // `"results": {…}` map — which silently truncates the index on every
    // upload. Skip the entry whose sha matches the new one (dedupe).
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in existing.char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start.take() {
                        let obj = &existing[s..=i];
                        let obj_sha = extract_string(obj, "sha");
                        if obj_sha != sha && !obj_sha.is_empty() {
                            entries.push(obj.to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Cap at MAX_INDEX.
    entries.truncate(MAX_INDEX);

    // Serialize.
    let mut out = String::from("[\n");
    for (i, e) in entries.iter().enumerate() {
        out.push_str("  ");
        out.push_str(e);
        if i + 1 < entries.len() { out.push(',') }
        out.push('\n');
    }
    out.push(']');
    out
}

/// Extract sha values from the first `n` index entries.
fn extract_index_shas(index_json: &str, n: usize) -> Vec<String> {
    let mut shas = Vec::new();
    for chunk in index_json.split('{').skip(1) {
        let obj = format!("{{{chunk}");
        let s = extract_string(&obj, "sha");
        if !s.is_empty() { shas.push(s) }
        if shas.len() >= n { break }
    }
    shas
}

// --- Shell helpers ---

fn run(cmd: &str, args: &[&str]) {
    let status = Command::new(cmd).args(args)
        .status()
        .unwrap_or_else(|e| { eprintln!("failed to run {cmd}: {e}"); std::process::exit(1); });
    if !status.success() {
        eprintln!("warning: {cmd} {:?} exited with {status}", args);
    }
}

/// Uploads flamegraph SVGs to the CI site under `/data/<remote-subdir>/`.
///
/// Scans `<local-root>` for `.svg` files, already named for their public
/// URL (`<group>__<bench>.svg`, the scheme `instructions-report` builds
/// links from), and `sftp_put`s each. Ungated by design — PR runs host
/// their flamegraphs so the report can embed them.
fn upload_assets(args: &[String]) {
    if args.len() != 2 {
        eprintln!("usage: ci-upload --assets <local-root> <remote-subdir>");
        std::process::exit(1);
    }

    let root = Path::new(&args[0]);
    let remote_subdir = &args[1];
    let svgs = find_flamegraphs(root);
    if svgs.is_empty() {
        eprintln!("[ci-upload] no .svg found under {}", root.display());
        return;
    }

    // sftp `put` doesn't create remote directories; make the sha dir first.
    ssh_cmd(&format!("mkdir -p /data/{remote_subdir}"));
    for (local, asset) in &svgs {
        sftp_put(local, &format!("/data/{remote_subdir}/{asset}"));
    }
    eprintln!("[ci-upload] uploaded {} flamegraph(s) to /data/{remote_subdir}", svgs.len());
}

/// Collects `(local-path, asset-name)` for every `.svg` under `root`,
/// recursively. The asset name is the file name — the producing workflow
/// names each SVG for its public URL before calling `--assets`.
fn find_flamegraphs(root: &Path) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "svg") {
                if let (Some(local), Some(asset)) = (
                    path.to_str().map(String::from),
                    path.file_name().and_then(|n| n.to_str()).map(String::from),
                ) {
                    found.push((local, asset));
                }
            }
        }
    }

    found.sort();
    found
}

fn ssh_cmd(cmd: &str) {
    run("flyctl", &["ssh", "console", "-a", APP, "-C", cmd]);
}

fn ssh_ls(dir: &str) -> Vec<String> {
    let output = Command::new("flyctl")
        .args(["ssh", "console", "-a", APP, "-C", &format!("ls {dir}")])
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .output().unwrap_or_else(|e| { eprintln!("ssh ls failed: {e}"); std::process::exit(1); });
    String::from_utf8_lossy(&output.stdout)
        .lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
}

fn sftp_put(local: &str, remote: &str) {
    let input = format!("put {local} {remote}\n");
    let mut child = Command::new("flyctl")
        .args(["ssh", "sftp", "shell", "-a", APP])
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped())
        .spawn().unwrap_or_else(|e| { eprintln!("sftp spawn failed: {e}"); std::process::exit(1); });
    child.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    if !out.status.success() {
        eprintln!("warning: sftp put {local} → {remote} failed: {}",
            String::from_utf8_lossy(&out.stderr));
    }
}

fn fetch_url(url: &str) -> Option<String> {
    let output = Command::new("curl").args(["-sf", url])
        .stdout(Stdio::piped()).stderr(Stdio::null())
        .output().ok()?;
    if output.status.success() { Some(String::from_utf8_lossy(&output.stdout).to_string()) }
    else { None }
}

fn write_tmp(name: &str, contents: &str) {
    let path = format!("/tmp/{name}");
    fs::write(&path, contents).unwrap_or_else(|e| { eprintln!("write {path}: {e}"); std::process::exit(1); });
}

/// Reject obviously-broken payloads before we touch the dashboard.
///
/// Two layers:
/// - **Structural** — file is JSON-shaped, has required common keys, sha
///   matches the argv sha (catches workflow uploads of stale or wrong-
///   commit data).
/// - **Per-kind floor** — a kind-specific "did the measurement actually
///   produce data?" sanity check. Catches the class of bugs where the
///   workflow runs to completion, the publish script accepts the file,
///   and the dashboard renders a green "0 platforms / 0 runs / 0 tests"
///   panel — which lies about the underlying property. Workflows often
///   have a `jq -e` belt that does the same check inline; this is the
///   suspenders, so a missed inline check can't ship a misleading panel.
fn validate_payload(kind: &str, path: &str, json: &str, expected_sha: &str) {
    let trimmed = json.trim();
    if trimmed.is_empty() {
        die(kind, path, "file is empty");
    }
    if !trimmed.starts_with('{') {
        die(kind, path, "file does not start with `{` (not a JSON object)");
    }
    if !trimmed.ends_with('}') {
        die(kind, path, "file does not end with `}` (truncated or malformed)");
    }
    if trimmed.len() < 30 {
        die(kind, path, &format!("file is suspiciously small ({} bytes)", trimmed.len()));
    }
    if !trimmed.contains("\"sha\"") {
        die(kind, path, "missing required `sha` field");
    }
    if !trimmed.contains("\"updated_at\"") {
        die(kind, path, "missing required `updated_at` field");
    }
    let in_file_sha = extract_string(trimmed, "sha");
    if in_file_sha != expected_sha {
        die(kind, path, &format!(
            "sha in file ({}) does not match argv sha ({}) — likely stale or wrong-commit data",
            in_file_sha, expected_sha,
        ));
    }

    validate_floor(kind, path, trimmed);
}

/// Per-kind data-shape sanity. Each arm asserts "the measurement
/// produced at least *some* data" — not "the data is good", which is the
/// dashboard's job. The floors are deliberately permissive (e.g. zero
/// failed mutants is fine; zero *total* mutants means cargo-mutants
/// produced no outcomes at all).
fn validate_floor(kind: &str, path: &str, json: &str) {
    // Good enough for our shapes: no nested top-level arrays at the
    // levels we check and no string values containing `[` or `]`.
    let array_len = |key: &str| -> usize {
        let needle = format!("\"{}\"", key);
        let Some(idx) = json.find(&needle) else { return 0 };
        let rest = &json[idx + needle.len()..];
        let Some(colon) = rest.find(':') else { return 0 };
        let after = rest[colon + 1..].trim_start();
        if !after.starts_with('[') { return 0 };
        let mut depth = 0i32;
        let mut end = 0;
        for (i, b) in after.bytes().enumerate() {
            match b {
                b'[' => depth += 1,
                b']' => { depth -= 1; if depth == 0 { end = i; break } }
                _ => {}
            }
        }
        if end == 0 { return 0 }
        let inside = after[1..end].trim();
        if inside.is_empty() { return 0 }
        let mut count = 1usize;
        let mut depth = 0i32;
        for b in inside.bytes() {
            match b {
                b'[' | b'{' => depth += 1,
                b']' | b'}' => depth -= 1,
                b',' if depth == 0 => count += 1,
                _ => {}
            }
        }
        count
    };

    let require = |cond: bool, reason: &str| {
        if !cond { die(kind, path, &format!("per-kind floor failed: {reason}")) }
    };

    match kind_family(kind) {
        "coverage"  => require(extract_num_in_section(json, "total", "total") > 0,
                               "coverage.total.total == 0 (no lines measured)"),
        "bench"     => require(array_len("groups") > 0,
                               "bench.groups is empty (no benchmarks produced output)"),
        "mutants"   => require(extract_num_in_section(json, "summary", "total") > 0,
                               "mutants.summary.total == 0 (cargo-mutants produced no outcomes)"),
        "dudect"    => require(extract_num_u64(json, "total") > 0,
                               "dudect.total == 0 (no t-tests recorded)"),
        "tacet"     => require(extract_num_u64(json, "total") > 0,
                               "tacet.total == 0 (no leak-model results)"),
        "ctgrind"   => require(extract_num_u64(json, "tests") > 0,
                               "ctgrind.tests == 0 (Valgrind taint produced no test results)"),
        "instructions" => require(extract_num_u64(json, "total") > 0 || array_len("results") > 0,
                                  "instructions has no instruction-count results"),
        "cycles"    => require(array_len("results") > 0,
                               "cycles.results is empty (no cycle counts recorded)"),
        "platform"  => require(array_len("platforms") > 0,
                               "platform.platforms is empty (no platform matrix jobs matched)"),
        "kat"       => require(array_len("suites") > 0,
                               "kat.suites is empty (no test-vector suites ran)"),
        // Pure-status payloads — `validate_payload`'s structural check is
        // already enough. A missing `status` would have failed `sha` /
        // `updated_at` already in practice.
        "deny" | "unsafe" => {}
        // Unknown kind: don't crash, but log that we didn't get to apply
        // a floor so the gap is visible in CI output.
        other => eprintln!("[ci-upload] note: no per-kind floor for `{other}`; skipping"),
    }
}

fn die(kind: &str, path: &str, reason: &str) -> ! {
    eprintln!("::error::ci-upload[{kind}]: refusing to upload {path}: {reason}");
    std::process::exit(1);
}

// --- JSON helpers ---

fn extract_string(json: &str, key: &str) -> String {
    let needle = format!("\"{}\"", key);
    let Some(idx) = json.find(&needle) else { return String::new() };
    let rest = &json[idx + needle.len()..];
    let Some(colon) = rest.find(':') else { return String::new() };
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('"') { return String::new() }
    let mut end = 1;
    let bytes = after.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' && bytes[end - 1] != b'\\' { break }
        end += 1;
    }
    after[1..end].to_string()
}

fn extract_num_u64(json: &str, key: &str) -> u64 {
    let needle = format!("\"{}\"", key);
    let Some(idx) = json.find(&needle) else { return 0 };
    let rest = &json[idx + needle.len()..];
    let Some(colon) = rest.find(':') else { return 0 };
    let after = rest[colon + 1..].trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
    after[..end].parse().unwrap_or(0)
}

fn extract_num_f64(json: &str, key: &str) -> f64 {
    let needle = format!("\"{}\"", key);
    let Some(idx) = json.find(&needle) else { return 0.0 };
    let rest = &json[idx + needle.len()..];
    let Some(colon) = rest.find(':') else { return 0.0 };
    let after = rest[colon + 1..].trim_start();
    let end = after.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(after.len());
    after[..end].parse().unwrap_or(0.0)
}

fn extract_num_in_section(json: &str, section: &str, key: &str) -> u64 {
    let needle = format!("\"{}\"", section);
    let Some(idx) = json.find(&needle) else { return 0 };
    let rest = &json[idx..];
    let Some(brace) = rest.find('{') else { return 0 };
    let Some(end) = rest[brace..].find('}') else { return 0 };
    extract_num_u64(&rest[brace..brace + end], key)
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
