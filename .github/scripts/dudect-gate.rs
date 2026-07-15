//! Gate dudect readings on a paired sign test against per-bench null-inputs
//! calibration, falling back to a median gate for benches without a null pair.
//!
//! Usage: dudect-gate <median-out> <report>...
//!
//! dudect-bencher prints one `bench <name> ... : ..., max t = <signed float>,
//! ...` line per bench and always exits 0, so pass/fail lives here. Shared CI
//! runners have heavy-tailed timing noise, and dudect's statistic is a max
//! over t-tests at many crop percentiles, so a single reading can spike far
//! past the raw-|t| gate on a true null.
//!
//! # Paired sign test
//!
//! For every bench `X` that also has a companion `X_null` bench (same code
//! path, byte-identical inputs for both classes, so zero possible information
//! leak), the gate compares them per-run: `d_i = |t_{X,i}| - |t_{X_null,i}|`.
//! Under the null hypothesis "code is constant-time" the two benches share
//! the same distribution and `sign(d_i)` is 50/50; a real leak elevates
//! `|t_X|` above `|t_{X_null}|` and drives all diffs positive. Reject the
//! null when the one-sided binomial p-value falls below [`ALPHA`]. This is
//! nonparametric, uses the per-run pairing so runner-noise cancels within
//! each pair, and reports a real p-value instead of a hand-tuned threshold.
//!
//! # Fallback median gate
//!
//! Benches without an `X_null` sibling use the legacy median-|t| gate at
//! [`MEDIAN_THRESHOLD`]. This preserves existing signals for benches whose
//! calibration companion hasn't landed yet.
//!
//! # Null benches
//!
//! Benches whose name ends in `_null` are informational only — their raw |t|
//! measures runner noise, which is a machine condition rather than a code
//! failure. Their median is reported so a reader can eyeball the noise
//! floor.
//!
//! # Magnitude floor
//!
//! A 5/5-positive sign test rejects at `p = 0.031` even on trivial-magnitude
//! persistent bias. [`MAGNITUDE_FLOOR`] requires the median diff to also
//! exceed a small threshold so noise doesn't cross the gate.
//!
//! # Informational benches
//!
//! Benches in [`INFORMATIONAL_BENCHES`] carry accepted µarch signals: the
//! paired sign test is skipped for gating, but [`CATASTROPHIC_MEDIAN`]
//! backstops an outright regression.
//!
//! # Output
//!
//! The median reading's report line is written to `<median-out>` for the
//! dashboard converter. Exits nonzero if any paired sign test rejects H0,
//! any unpaired bench crosses [`MEDIAN_THRESHOLD`], a bench is missing
//! readings, or no readings parse at all.
//!
//! Compile: `rustc -O dudect-gate.rs -o dudect-gate`
//! Self-test: `rustc --test dudect-gate.rs -o dudect-gate-test &&
//! ./dudect-gate-test`

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    process::ExitCode,
};

/// One-sided significance level for the paired sign test.
const ALPHA: f64 = 0.05;

/// Fallback pass/fail cutoff for unpaired benches. Keep in sync with
/// `THRESHOLD` in `dudect-to-json.rs`.
const MEDIAN_THRESHOLD: f64 = 5.0;

/// Minimum median paired diff (|t| units) for a rejected sign test to fail.
/// Filters trivial-magnitude persistent bias that 5/5-positive alone flags.
const MAGNITUDE_FLOOR: f64 = 1.0;

/// Catastrophic median-|t| cutoff for [`INFORMATIONAL_BENCHES`]. Set at 3×
/// the observed µarch envelope on our runners (max ~17), which also sits at
/// the lower bound of the "known-leaky" band in the dudect paper (Reparaz
/// et al. 2017) for n=100k samples on real crypto bugs.
const CATASTROPHIC_MEDIAN: f64 = 50.0;

/// Suffix that identifies a null-calibration bench.
const NULL_SUFFIX: &str = "_null";

/// Benches with accepted µarch signals: skip the paired sign gate, keep the
/// [`CATASTROPHIC_MEDIAN`] backstop.
const INFORMATIONAL_BENCHES: &[&str] = &["decaps"];

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
    /// Whether any bench breached its gate or was missing readings.
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

    /// Gates every bench, dispatching by whether an `_null` sibling exists.
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

        let names: BTreeSet<String> = self.readings.keys().cloned().collect();

        for name in &names {
            let readings = &self.readings[name];

            if readings.len() != self.expected {
                outcome.log.push_str(&format!(
                    "FAIL: {name}: {} readings, expected {}\n",
                    readings.len(),
                    self.expected
                ));
                outcome.failed = true;
                continue;
            }

            // Always emit the median reading for the dashboard, regardless
            // of gate shape.
            let median_line = median_reading(readings).line.clone();
            outcome.medians.push_str(&median_line);
            outcome.medians.push('\n');

            if name.ends_with(NULL_SUFFIX) {
                report_null_bench(&mut outcome, name, readings);
                continue;
            }

            let null_name = format!("{name}{NULL_SUFFIX}");
            let null_readings = self
                .readings
                .get(&null_name)
                .filter(|r| r.len() == self.expected)
                .map(Vec::as_slice);

            if INFORMATIONAL_BENCHES.contains(&name.as_str()) {
                informational_gate(&mut outcome, name, readings, null_readings);
                continue;
            }

            if let Some(null) = null_readings {
                paired_sign_gate(&mut outcome, name, readings, null);
                continue;
            }

            median_gate(&mut outcome, name, readings);
        }

        outcome
    }
}

/// Returns the median reading (lower of the two middle for even N).
fn median_reading(readings: &[Reading]) -> &Reading {
    let mut idx: Vec<usize> = (0..readings.len()).collect();
    idx.sort_by(|&a, &b| readings[a].abs_t.total_cmp(&readings[b].abs_t));
    &readings[idx[(readings.len() - 1) / 2]]
}

/// Paired sign test: fails when the p-value is below [`ALPHA`] and the
/// median diff exceeds [`MAGNITUDE_FLOOR`].
fn paired_sign_gate(outcome: &mut GateOutcome, name: &str, real: &[Reading], null: &[Reading]) {
    let n = real.len();
    let diffs: Vec<f64> = real.iter().zip(null).map(|(r, z)| r.abs_t - z.abs_t).collect();
    let positive = diffs.iter().filter(|&&d| d > 0.0).count();
    let p_value = binomial_upper_tail(positive, n);

    let real_median = median_reading(real).abs_t;
    let null_median = median_reading(null).abs_t;
    let diff_median = median_of_diffs(&diffs);

    let diffs_str: Vec<String> = diffs.iter().map(|d| format!("{d:+.5}")).collect();
    outcome.log.push_str(&format!(
        "  {name} vs {name}{NULL_SUFFIX}: diffs {}, {positive}/{n} positive (p = {p_value:.4}); \
         real median = {real_median:.5}, null median = {null_median:.5}, diff median = {diff_median:+.5}\n",
        diffs_str.join(" "),
    ));

    if p_value < ALPHA && diff_median.abs() > MAGNITUDE_FLOOR {
        outcome.log.push_str(&format!(
            "FAIL: {name} paired sign test rejects H0 (p = {p_value:.4} < {ALPHA}, \
             |diff median| = {abs:.4} > {MAGNITUDE_FLOOR})\n",
            abs = diff_median.abs(),
        ));
        outcome.failed = true;
    } else if p_value < ALPHA {
        outcome.log.push_str(&format!(
            "  {name}: sign test would reject H0 (p = {p_value:.4}) but |diff median| = \
             {abs:.4} <= {MAGNITUDE_FLOOR} — noise-floor magnitude, informational\n",
            abs = diff_median.abs(),
        ));
    }
}

/// Skips the paired sign test, gates only on [`CATASTROPHIC_MEDIAN`]. Still
/// logs paired diffs when a null sibling exists.
fn informational_gate(
    outcome: &mut GateOutcome,
    name: &str,
    readings: &[Reading],
    null: Option<&[Reading]>,
) {
    let real_median = median_reading(readings).abs_t;

    if let Some(null) = null {
        let diffs: Vec<f64> = readings
            .iter()
            .zip(null)
            .map(|(r, z)| r.abs_t - z.abs_t)
            .collect();
        let positive = diffs.iter().filter(|&&d| d > 0.0).count();
        let p_value = binomial_upper_tail(positive, readings.len());
        let null_median = median_reading(null).abs_t;
        let diff_median = median_of_diffs(&diffs);

        let diffs_str: Vec<String> = diffs.iter().map(|d| format!("{d:+.5}")).collect();
        outcome.log.push_str(&format!(
            "  {name} (informational, accepted µarch signal) vs {name}{NULL_SUFFIX}: \
             diffs {}, {positive}/{n} positive (p = {p_value:.4}); real median = {real_median:.5}, \
             null median = {null_median:.5}, diff median = {diff_median:+.5}\n",
            diffs_str.join(" "),
            n = readings.len(),
        ));
    } else {
        let runs: Vec<String> = readings.iter().map(|r| format!("{:.5}", r.abs_t)).collect();
        outcome.log.push_str(&format!(
            "  {name} (informational, accepted µarch signal) -> |t| runs: {}, \
             median = {real_median:.5}\n",
            runs.join(" "),
        ));
    }

    if real_median >= CATASTROPHIC_MEDIAN {
        outcome.log.push_str(&format!(
            "FAIL: {name} median |t| = {real_median:.5} >= {CATASTROPHIC_MEDIAN} \
             (catastrophic — outside the accepted µarch envelope)\n"
        ));
        outcome.failed = true;
    }
}

/// Returns the median of a slice of floats; lower-of-two-middle for even N,
/// matching [`median_reading`].
fn median_of_diffs(diffs: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = diffs.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[(sorted.len() - 1) / 2]
}

/// Legacy median-|t| gate for benches without a null sibling.
fn median_gate(outcome: &mut GateOutcome, name: &str, readings: &[Reading]) {
    let median = median_reading(readings).abs_t;
    let runs: Vec<String> = readings.iter().map(|r| format!("{:.5}", r.abs_t)).collect();

    outcome.log.push_str(&format!(
        "  {name} -> |t| runs: {}, median = {median:.5} (no null pair)\n",
        runs.join(" "),
    ));

    if median >= MEDIAN_THRESHOLD {
        outcome.log.push_str(&format!(
            "FAIL: {name} median |t| >= {MEDIAN_THRESHOLD}\n"
        ));
        outcome.failed = true;
    }
}

/// Emits a null-bench median as calibration info; never fails the gate on it.
fn report_null_bench(outcome: &mut GateOutcome, name: &str, readings: &[Reading]) {
    let median = median_reading(readings).abs_t;
    let runs: Vec<String> = readings.iter().map(|r| format!("{:.5}", r.abs_t)).collect();

    outcome.log.push_str(&format!(
        "  {name} (null calibration) -> |t| runs: {}, median = {median:.5}\n",
        runs.join(" "),
    ));
}

/// Returns `P[X >= k]` where `X ~ Binomial(n, 0.5)`. Symmetric coin.
fn binomial_upper_tail(k: usize, n: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let denom = 2u128.pow(u32::try_from(n).expect("n fits in u32"));
    let mut num: u128 = 0;
    for i in k..=n {
        num += binomial_coefficient(n, i);
    }
    (num as f64) / (denom as f64)
}

/// `C(n, k)` for small `n` — u128 for headroom up to n <= 60ish. dudect run
/// counts are single digits in practice.
fn binomial_coefficient(n: usize, k: usize) -> u128 {
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 0..k {
        result = result * (n - i) as u128 / (i + 1) as u128;
    }
    result
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

    /// A dudect report line for one bench with the given `max t`.
    fn line(name: &str, t: f64) -> String {
        format!(
            "bench {name} ... : n == +0.100M, max t = {t:+.5}, max tau = +0.00491, (5/tau)^2 = 1038661"
        )
    }

    fn report(lines: &[String]) -> String {
        lines.join("\n") + "\n"
    }

    #[test]
    fn parses_reading_line() {
        let raw = "bench decaps ... : n == +0.100M, max t = -1.54877, max tau = -0.00491, (5/tau)^2 = 1038661";

        let (name, reading) = Reading::parse(raw).expect("reading line parses");

        assert_eq!(name, "decaps");
        assert_eq!(reading.abs_t, 1.54877);
        assert_eq!(reading.line, raw);
    }

    #[test]
    fn ignores_non_reading_lines() {
        assert!(Reading::parse("bench decaps seeded with 0x428b8e5da2d1aca4").is_none());
        assert!(Reading::parse("running 2 benches").is_none());
        assert!(Reading::parse("").is_none());
    }

    #[test]
    fn binomial_coefficient_matches_pascals_triangle() {
        assert_eq!(binomial_coefficient(5, 0), 1);
        assert_eq!(binomial_coefficient(5, 5), 1);
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(8, 3), 56);
    }

    #[test]
    fn binomial_upper_tail_matches_table() {
        // P[X = 5 | n=5] = 1/32
        assert!((binomial_upper_tail(5, 5) - 1.0 / 32.0).abs() < 1e-9);
        // P[X >= 4 | n=5] = 6/32
        assert!((binomial_upper_tail(4, 5) - 6.0 / 32.0).abs() < 1e-9);
        // P[X >= 7 | n=8] = 9/256
        assert!((binomial_upper_tail(7, 8) - 9.0 / 256.0).abs() < 1e-9);
    }

    /// 5/5 positive diffs with meaningful magnitude — sign test rejects H0
    /// and the diff median exceeds [`MAGNITUDE_FLOOR`].
    #[test]
    fn paired_all_positive_fails() {
        let reports: Vec<String> = [
            (10.0, 2.0),
            (11.0, 1.5),
            (9.5, 2.1),
            (12.0, 1.8),
            (10.5, 2.3),
        ]
        .into_iter()
        .map(|(d, n)| report(&[line("encaps", d), line("encaps_null", n)]))
        .collect();

        let outcome = Benches::from_reports(&reports).gate();

        assert!(outcome.failed, "log:\n{}", outcome.log);
        assert!(outcome.log.contains("paired sign test rejects H0"));
        assert!(outcome.log.contains("5/5 positive"));
    }

    /// 3/5 positive diffs (sign-balanced noise) — sign test passes.
    #[test]
    fn paired_balanced_signs_passes() {
        let reports: Vec<String> = [
            (10.0, 2.0),  // +
            (11.0, 12.5), // -
            (9.5, 2.1),   // +
            (12.0, 14.8), // -
            (10.5, 2.3),  // +
        ]
        .into_iter()
        .map(|(d, n)| report(&[line("encaps", d), line("encaps_null", n)]))
        .collect();

        let outcome = Benches::from_reports(&reports).gate();

        assert!(!outcome.failed, "log:\n{}", outcome.log);
        assert!(outcome.log.contains("3/5 positive"));
    }

    /// A null bench with sky-high `|t|` does not fail the gate on its own;
    /// its sibling still gates via the paired test.
    #[test]
    fn null_bench_is_informational_only() {
        let reports: Vec<String> = [
            (2.0, 40.0),
            (1.8, 35.0),
            (2.1, 42.0),
            (1.9, 38.0),
            (2.0, 41.0),
        ]
        .into_iter()
        .map(|(d, n)| report(&[line("encaps", d), line("encaps_null", n)]))
        .collect();

        let outcome = Benches::from_reports(&reports).gate();

        // encaps < null every run, so 0/5 positive → paired test passes.
        assert!(!outcome.failed, "log:\n{}", outcome.log);
        assert!(outcome.log.contains("(null calibration)"));
        assert!(outcome.log.contains("0/5 positive"));
    }

    /// Fallback median gate: an un-paired bench still fails when its
    /// median crosses the fixed threshold.
    #[test]
    fn unpaired_bench_falls_back_to_median_gate() {
        let reports: Vec<String> = [10.0, 11.0, 9.5, 12.0, 10.5]
            .into_iter()
            .map(|t| report(&[line("encaps", t)]))
            .collect();

        let outcome = Benches::from_reports(&reports).gate();

        assert!(outcome.failed, "log:\n{}", outcome.log);
        assert!(outcome.log.contains("no null pair"));
        assert!(outcome.log.contains("encaps median |t| >="));
    }

    #[test]
    fn missing_reading_fails() {
        let reports = [
            report(&[line("decaps", 1.0), line("decaps_null", 1.0)]),
            report(&[line("decaps", 1.0), line("decaps_null", 1.0)]),
            report(&[line("decaps_null", 1.0)]),
        ];

        let outcome = Benches::from_reports(&reports).gate();

        assert!(outcome.failed);
        assert!(outcome.log.contains("FAIL: decaps: 2 readings, expected 3"));
    }

    /// 5/5 positive diffs at noise-floor magnitude: [`MAGNITUDE_FLOOR`]
    /// suppresses the failure.
    #[test]
    fn tiny_magnitude_paired_signal_passes() {
        let reports: Vec<String> = [
            (2.5, 2.0),  // diff +0.5
            (2.6, 2.1),  // diff +0.5
            (2.7, 2.2),  // diff +0.5
            (2.4, 1.9),  // diff +0.5
            (2.55, 2.05), // diff +0.5
        ]
        .into_iter()
        .map(|(d, n)| report(&[line("encaps", d), line("encaps_null", n)]))
        .collect();

        let outcome = Benches::from_reports(&reports).gate();

        assert!(!outcome.failed, "log:\n{}", outcome.log);
        assert!(outcome.log.contains("sign test would reject H0"));
        assert!(outcome.log.contains("noise-floor magnitude"));
    }

    /// An [`INFORMATIONAL_BENCHES`] entry never fails the paired sign test,
    /// even on all-positive diffs of meaningful magnitude.
    #[test]
    fn informational_bench_never_fails_paired_sign_test() {
        let reports: Vec<String> = [
            (10.0, 2.0),
            (11.0, 1.5),
            (9.5, 2.1),
            (12.0, 1.8),
            (10.5, 2.3),
        ]
        .into_iter()
        .map(|(d, n)| report(&[line("decaps", d), line("decaps_null", n)]))
        .collect();

        let outcome = Benches::from_reports(&reports).gate();

        assert!(!outcome.failed, "log:\n{}", outcome.log);
        assert!(outcome.log.contains("accepted µarch signal"));
        assert!(outcome.log.contains("5/5 positive"));
    }

    /// An informational bench still fails when its median passes
    /// [`CATASTROPHIC_MEDIAN`].
    #[test]
    fn informational_bench_fails_on_catastrophic_median() {
        let reports: Vec<String> = [
            (70.0, 2.0),
            (75.0, 1.5),
            (68.0, 2.1),
            (72.0, 1.8),
            (69.0, 2.3),
        ]
        .into_iter()
        .map(|(d, n)| report(&[line("decaps", d), line("decaps_null", n)]))
        .collect();

        let outcome = Benches::from_reports(&reports).gate();

        assert!(outcome.failed, "log:\n{}", outcome.log);
        assert!(outcome.log.contains("catastrophic"));
        assert!(outcome.log.contains("outside the accepted µarch envelope"));
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
