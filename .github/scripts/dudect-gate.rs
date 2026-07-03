//! Gate dudect readings on the per-bench median |t| across several runs.
//!
//! Usage: dudect-gate <median-out> <report>...
//!
//! dudect-bencher prints one `bench <name> ... : ..., max t = <signed float>,
//! ...` line per bench and always exits 0, so pass/fail lives here. Shared CI
//! runners have heavy-tailed timing noise, and dudect's statistic is a max
//! over t-tests at many crop percentiles, so a single reading can spike far
//! past the gate on a true null (observed |t| 3.06 -> 10.72 across identical
//! runs minutes apart). Each report contributes one reading per bench and the
//! gate fails a bench only if its median |t| is at or above [`THRESHOLD`]: a
//! real leak fails every run and still trips the gate, a one-run noise
//! excursion does not.
//!
//! The median reading's report line is written to `<median-out>` so the
//! dashboard tracks the same reading the gate enforced. Exits nonzero if any
//! bench's median is at or above [`THRESHOLD`], a bench is missing a reading
//! in some report, or no readings parse at all.
//!
//! Compile: `rustc -O dudect-gate.rs -o dudect-gate`
//! Self-test: `rustc --test dudect-gate.rs -o dudect-gate-test &&
//! ./dudect-gate-test`

use std::{collections::BTreeMap, env, fs, process::ExitCode};

/// Pass/fail cutoff: a bench fails when its median |t| is at or above this.
/// Keep in sync with `THRESHOLD` in `dudect-to-json.rs`.
const THRESHOLD: f64 = 5.0;

/// One `bench <name> ... max t = <t>` line from a dudect report.
struct Reading {
    /// Absolute value of the reading's `max t`.
    abs_t: f64,
    /// The full report line, reproduced in the median output file.
    line: String,
}

impl Reading {
    /// Parses a report line into `(bench name, reading)`; returns `None` for
    /// lines that are not `max t` readings (seed announcements, progress).
    fn parse(line: &str) -> Option<(String, Reading)> {
        if !line.contains("bench ") || !line.contains("max t = ") {
            return None;
        }

        let name = line.split_whitespace().nth(1)?.to_string();

        let idx = line.find("max t = ")? + "max t = ".len();
        let rest = &line[idx..];
        let end = rest.find(',').unwrap_or(rest.len());
        let abs_t = rest[..end].trim().parse::<f64>().ok()?.abs();

        let reading = Reading {
            abs_t,
            line: line.to_string(),
        };
        Some((name, reading))
    }
}

/// Every reading from a set of dudect reports, grouped by bench name.
struct Benches {
    /// Number of reports parsed; every bench must have this many readings.
    expected: usize,
    /// Readings per bench in report order, keyed by bench name (`BTreeMap`
    /// for deterministic output order).
    readings: BTreeMap<String, Vec<Reading>>,
}

/// The gate's verdict over all benches.
struct GateOutcome {
    /// Human-readable verdict, one line per bench plus `FAIL:` lines.
    log: String,
    /// The median reading's report line per complete bench, in
    /// dudect-bencher's own format for the dashboard converter.
    medians: String,
    /// Whether any bench breached [`THRESHOLD`] or was missing readings.
    failed: bool,
}

impl Benches {
    /// Collects readings from the given report contents, one report per run.
    fn from_reports<S: AsRef<str>>(reports: &[S]) -> Benches {
        let mut readings: BTreeMap<String, Vec<Reading>> = BTreeMap::new();

        for report in reports {
            for line in report.as_ref().lines() {
                if let Some((name, reading)) = Reading::parse(line) {
                    readings.entry(name).or_default().push(reading);
                }
            }
        }

        Benches {
            expected: reports.len(),
            readings,
        }
    }

    /// Gates every bench on the median of its |t| readings.
    fn gate(self) -> GateOutcome {
        let mut outcome = GateOutcome {
            log: String::new(),
            medians: String::new(),
            failed: false,
        };

        if self.readings.is_empty() {
            outcome.log.push_str("FAIL: no dudect readings found\n");
            outcome.failed = true;
            return outcome;
        }

        for (name, readings) in &self.readings {
            if readings.len() != self.expected {
                outcome.log.push_str(&format!(
                    "FAIL: {name}: {} readings, expected {}\n",
                    readings.len(),
                    self.expected
                ));
                outcome.failed = true;
                continue;
            }

            // Sort an index permutation so the log can still list the
            // readings in run order.
            let mut by_t: Vec<usize> = (0..readings.len()).collect();
            by_t.sort_by(|&a, &b| readings[a].abs_t.total_cmp(&readings[b].abs_t));
            let median = &readings[by_t[(self.expected - 1) / 2]];

            let runs: Vec<String> = readings.iter().map(|r| format!("{:.5}", r.abs_t)).collect();
            outcome.log.push_str(&format!(
                "  {name} -> |t| runs: {}, median = {:.5}\n",
                runs.join(" "),
                median.abs_t
            ));

            outcome.medians.push_str(&median.line);
            outcome.medians.push('\n');

            if median.abs_t >= THRESHOLD {
                outcome
                    .log
                    .push_str(&format!("FAIL: {name} median |t| >= {THRESHOLD}\n"));
                outcome.failed = true;
            }
        }

        outcome
    }
}

/// Reads the report files named on the command line, gates them, and writes
/// the median lines for the dashboard.
fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dudect-gate <median-out> <report>...");
        return ExitCode::from(2);
    }

    let mut reports = Vec::new();
    for path in &args[2..] {
        match fs::read_to_string(path) {
            Ok(contents) => reports.push(contents),
            Err(err) => {
                eprintln!("FAIL: cannot read {path}: {err}");
                return ExitCode::FAILURE;
            }
        }
    }

    let outcome = Benches::from_reports(&reports).gate();
    print!("{}", outcome.log);

    if let Err(err) = fs::write(&args[1], &outcome.medians) {
        eprintln!("FAIL: cannot write {}: {err}", args[1]);
        return ExitCode::FAILURE;
    }

    if outcome.failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dudect report with the given decaps and encaps `max t` readings.
    fn report(decaps_t: f64, encaps_t: f64) -> String {
        format!(
            "bench decaps seeded with 0x428b8e5da2d1aca4\n\
             bench decaps ... : n == +0.100M, max t = {decaps_t:+.5}, max tau = +0.00491, (5/tau)^2 = 1038661\n\
             bench encaps seeded with 0xd8c5546c4574dd1d\n\
             bench encaps ... : n == +0.026M, max t = {encaps_t:+.5}, max tau = +0.01237, (5/tau)^2 = 163303\n"
        )
    }

    #[test]
    fn parses_reading_line() {
        let line = "bench decaps ... : n == +0.100M, max t = -1.54877, max tau = -0.00491, (5/tau)^2 = 1038661";

        let (name, reading) = Reading::parse(line).expect("reading line parses");

        assert_eq!(name, "decaps");
        assert_eq!(reading.abs_t, 1.54877);
        assert_eq!(reading.line, line);
    }

    #[test]
    fn ignores_non_reading_lines() {
        assert!(Reading::parse("bench decaps seeded with 0x428b8e5da2d1aca4").is_none());
        assert!(Reading::parse("running 2 benches").is_none());
        assert!(Reading::parse("").is_none());
    }

    #[test]
    fn one_run_noise_spike_passes() {
        let reports = [
            report(10.71542, 1.98935),
            report(-1.54877, 2.36055),
            report(-1.24036, -1.65129),
        ];

        let outcome = Benches::from_reports(&reports).gate();

        assert!(!outcome.failed, "log:\n{}", outcome.log);
        assert!(outcome.medians.contains("max t = -1.54877"));
        assert!(outcome.medians.contains("max t = +1.98935"));
    }

    #[test]
    fn persistent_leak_fails() {
        let reports = [
            report(10.71542, 1.98935),
            report(9.80000, 2.36055),
            report(-1.24036, -1.65129),
        ];

        let outcome = Benches::from_reports(&reports).gate();

        assert!(outcome.failed);
        assert!(outcome.log.contains("FAIL: decaps median |t| >= 5"));
        assert!(!outcome.log.contains("FAIL: encaps"));
    }

    #[test]
    fn median_at_threshold_fails() {
        let reports = [
            report(5.00000, 1.0),
            report(5.00000, 1.0),
            report(1.00000, 1.0),
        ];

        let outcome = Benches::from_reports(&reports).gate();

        assert!(outcome.failed);
        assert!(outcome.log.contains("FAIL: decaps median |t| >= 5"));
    }

    #[test]
    fn missing_reading_fails() {
        let reports = [
            report(1.0, 1.0),
            report(1.0, 1.0),
            "bench encaps ... : n == +0.026M, max t = +1.00000, max tau = +0.01237, (5/tau)^2 = 163303\n"
                .to_string(),
        ];

        let outcome = Benches::from_reports(&reports).gate();

        assert!(outcome.failed);
        assert!(outcome.log.contains("FAIL: decaps: 2 readings, expected 3"));
    }

    #[test]
    fn no_readings_fails() {
        let reports = ["".to_string(), "".to_string(), "".to_string()];

        let outcome = Benches::from_reports(&reports).gate();

        assert!(outcome.failed);
        assert!(outcome.log.contains("no dudect readings found"));
        assert!(outcome.medians.is_empty());
    }
}
