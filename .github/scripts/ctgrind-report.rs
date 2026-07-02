//! Run ctgrind tests under Valgrind memcheck and output JSON.
//!
//! Builds the ctgrind test binary, then runs each test function
//! individually under Valgrind so errors map to specific functions.
//!
//! Usage: ctgrind-report <sha>
//!
//! Compile: `rustc -O ctgrind-report.rs -o ctgrind-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: ctgrind-report <sha>");
        std::process::exit(1);
    }
    let sha = &args[1];

    // Build the ctgrind test binary. Release: debug-mode overflow checks
    // insert conditional jumps that Valgrind flags as secret-dependent
    // (e.g. `v * BARRETT_DIV_Q` in `FieldElement::new`). Release matches
    // production and tests the actual generated code.
    eprintln!("[ctgrind-report] building...");
    // `crabgrind` is a Linux-only optional dep gating the Valgrind client
    // requests behind `feature = "crabgrind"`; without it the `declassify`
    // hook (see `src/ctgrind.rs`) is a compile-time no-op and memcheck sees
    // `rho` as tainted, lighting up ~1k branches per K-PKE.KeyGen.
    let status = Command::new("cargo")
        .args([
            "test",
            "--test",
            "ctgrind",
            "--features",
            "ctgrind,crabgrind",
            "--release",
            "--no-run",
        ])
        .status()
        .expect("failed to build");
    if !status.success() {
        eprintln!("[ctgrind-report] build failed");
        std::process::exit(1);
    }

    let bin = match find_binary("ctgrind") {
        Some(b) => b,
        None => {
            eprintln!("[ctgrind-report] binary not found");
            std::process::exit(1);
        }
    };

    // Discover test names by running with --list.
    let list_output = Command::new(&bin)
        .args(["--list", "--format=terse"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .expect("failed to list tests");
    let test_names: Vec<String> = String::from_utf8_lossy(&list_output.stdout)
        .lines()
        .filter_map(|l| l.strip_suffix(": test").map(|s| s.trim().to_string()))
        .collect();

    eprintln!(
        "[ctgrind-report] found {} tests: {:?}",
        test_names.len(),
        test_names
    );

    // Run each test individually under Valgrind. Classify from three
    // independent signals so a broken harness cannot masquerade as a clean
    // pass (the historical failure mode: every test reported "0 errors"
    // because no valgrind XML was produced, while a known variable-time
    // path showed nothing):
    //   * xml_ok      -- valgrind actually produced output (the XML root is
    //                    present). Without it there is no CT verdict, only
    //                    a vacuous "0 errors" from a missing file.
    //   * test_ran_ok -- the test's own exit status. With
    //                    `--error-exitcode=0` valgrind leaves the child exit
    //                    code untouched, so this is the test pass/fail,
    //                    independent of any memcheck findings.
    //   * errors      -- count of memcheck `<error>` records whose `<kind>`
    //                    is `UninitCondition` or `UninitValue` (branch or use
    //                    on tainted / secret-derived data). Leak kinds and
    //                    other memcheck findings are ignored: they are memory-
    //                    hygiene signals, not CT signals, and consistently
    //                    add one baseline error per test on the Rust runtime.
    // On anything but a clean pass we dump the captured valgrind stderr and
    // test stdout: no other tool runs valgrind on Apple Silicon, so the CI
    // log is the only window into what actually happened.
    let mut results = Vec::new();
    let mut total_errors = 0;
    let mut harness_broken = false;

    for name in &test_names {
        eprintln!("[ctgrind-report] running {name}...");
        let xml_file = format!("/tmp/ctgrind-{name}.xml");
        let _ = fs::remove_file(&xml_file);

        let output = Command::new("valgrind")
            .args([
                "--tool=memcheck",
                "--error-exitcode=0",
                "--xml=yes",
                &format!("--xml-file={xml_file}"),
                &bin,
                "--exact",
                name,
                "--test-threads=1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("failed to run valgrind");

        let xml = fs::read_to_string(&xml_file).unwrap_or_default();
        let xml_ok = xml.contains("<valgrindoutput>");
        // Only secret-dependent findings count: `UninitCondition` (branch on
        // tainted data) and `UninitValue` (use of tainted data in a copy or
        // syscall). Other `<error>` kinds — leaks, heap errors — are memory
        // hygiene, not CT.
        let errors = xml.matches("<kind>UninitCondition</kind>").count()
            + xml.matches("<kind>UninitValue</kind>").count();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let test_ran_ok = output.status.success() && stdout.contains("test result: ok");

        let detail = if !xml_ok {
            harness_broken = true;
            "no-analysis"
        } else if !test_ran_ok {
            harness_broken = true;
            "test-error"
        } else if errors > 0 {
            "leak"
        } else {
            "pass"
        };
        let status = if detail == "pass" { "pass" } else { "fail" };

        total_errors += errors;
        results.push((name.clone(), status.to_string(), errors));

        eprintln!(
            "[ctgrind-report]   {name}: {detail} (errors={errors}, xml_ok={xml_ok}, test_exit={:?})",
            output.status.code()
        );
        if detail != "pass" {
            eprintln!(
                "---- {name}: valgrind stderr (tail) ----\n{}",
                tail(&stderr, 40)
            );
            eprintln!(
                "---- {name}: test stdout (tail) ----\n{}",
                tail(&stdout, 20)
            );
            eprintln!("----");
        }
    }

    // Verdict. A broken harness (no valgrind analysis, or a test that did
    // not run cleanly) fails, and so does any memcheck error: the CT rework
    // has landed, so a secret-dependent branch or memory access here is a
    // regression, not known debt.
    let secret_leaks: Vec<&str> = results
        .iter()
        .filter(|(name, _, errs)| *errs > 0 && name.ends_with("_secret_independent"))
        .map(|(name, _, _)| name.as_str())
        .collect();
    let overall = if harness_broken || total_errors > 0 {
        "fail"
    } else {
        "pass"
    };

    // Write JSON.
    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"status\": {},", json_str(overall))?;
    writeln!(w, "  \"errors\": {},", total_errors)?;
    writeln!(w, "  \"tests\": {},", results.len())?;
    writeln!(w, "  \"results\": [")?;
    for (i, (name, status, errors)) in results.iter().enumerate() {
        write!(
            w,
            "    {{\"name\": {}, \"status\": {}, \"errors\": {}}}",
            json_str(name),
            json_str(status),
            errors
        )?;
        if i + 1 < results.len() {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    // `process::exit` skips the BufWriter's drop, so flush the JSON first.
    w.flush()?;
    drop(w);

    // A harness that cannot analyze -- no valgrind output, or a test that
    // did not run cleanly -- must not pass silently as a vacuous "0
    // errors"; and any memcheck error is a constant-time regression.
    if harness_broken {
        eprintln!(
            "[ctgrind-report] FAIL: a test produced no valgrind analysis or did not run cleanly"
        );
    }
    if !secret_leaks.is_empty() {
        eprintln!("[ctgrind-report] FAIL: secret-dependent findings in {secret_leaks:?}");
    }
    if overall == "fail" {
        std::process::exit(2);
    }
    Ok(())
}

/// Returns the last `n` lines of `s`, for surfacing captured output.
fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

fn find_binary(prefix: &str) -> Option<String> {
    for entry in fs::read_dir("target/release/deps").ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(prefix) && !name.contains('.') {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() && meta.len() > 0 {
                    return Some(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }
    None
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
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    let secs = dur.as_secs();
    let (h, m, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    let mut y = 1970i64;
    let mut rem = (secs / 86400) as i64;
    loop {
        let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
        if rem < yd {
            break;
        }
        rem -= yd;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let md = [
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
    for &d in &md {
        if rem < d {
            break;
        }
        rem -= d;
        mo += 1;
    }
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        mo + 1,
        rem + 1,
        h,
        m,
        s
    )
}
