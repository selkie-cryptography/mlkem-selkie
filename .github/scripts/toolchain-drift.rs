//! Guards the release-toolchain pin invariant chain: `rust-version`
//! (Cargo.toml, once declared) <= `RELEASE_TOOLCHAIN` (release.yml)
//! == every pinned `toolchain: "X.Y.Z"` leg in ci.yml <= current stable.
//! Also guards the dated fmt and clippy gate pins in ci.yml and mutants.yml.
//!
//! Modes:
//! - `toolchain-drift consistency` — offline; checks the checked-out tree. Run
//!   on pull requests (devops.yml).
//! - `curl -s .../channel-rust-stable.toml | toolchain-drift staleness` — reads
//!   the channel manifest on stdin and fails when the release pin lags stable
//!   by two or more minor releases, or a dated gate pin predates the current
//!   stable release. Run weekly (toolchain-canary.yml).

use std::{fmt, fs, io::Read, process::ExitCode, str::FromStr};

/// Path to the release workflow holding the `RELEASE_TOOLCHAIN` pin.
const RELEASE_YML: &str = ".github/workflows/release.yml";
/// Path to the CI workflow holding the pinned test-matrix leg and the dated
/// fmt and clippy pins.
const CI_YML: &str = ".github/workflows/ci.yml";
/// Path to the mutants workflow, whose nightly pin must equal ci.yml's.
const MUTANTS_YML: &str = ".github/workflows/mutants.yml";
/// Path to the crate manifest holding `rust-version` (the MSRV).
const CARGO_TOML: &str = "Cargo.toml";

/// A `major.minor.patch` Rust toolchain version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}

impl Version {
    /// Extracts the `RELEASE_TOOLCHAIN: "X.Y.Z"` pin from release.yml text.
    fn release_pin(text: &str) -> Option<Version> {
        let line = text.lines().find(|l| l.contains("RELEASE_TOOLCHAIN:"))?;

        quoted(line)?.parse().ok()
    }

    /// Extracts every pinned (quoted, numeric) `toolchain: "X.Y.Z"` value
    /// from workflow text. Channels, dated (`nightly-2026-10-01`) or not,
    /// and `${{ env.* }}` references don't parse as versions and are skipped.
    fn workflow_pins(text: &str) -> Vec<Version> {
        text.lines()
            .filter(|l| l.trim_start().starts_with("toolchain: \""))
            .filter_map(quoted)
            .filter_map(|v| v.parse().ok())
            .collect()
    }

    /// Extracts `rust-version = "X.Y[.Z]"` from Cargo.toml text; a missing
    /// patch reads as 0. `None` when the key is absent.
    fn msrv(text: &str) -> Option<Result<Version, String>> {
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with("rust-version"))?;
        let value = quoted(line).unwrap_or_default();
        let full = match value.matches('.').count() {
            1 => format!("{value}.0"),
            _ => value.to_string(),
        };

        Some(full.parse())
    }

    /// Extracts the `[pkg.rust]` version from a rustup channel manifest.
    fn stable_channel(manifest: &str) -> Option<Version> {
        let pkg_rust = manifest.split("[pkg.rust]").nth(1)?;
        let line = pkg_rust
            .lines()
            .find(|l| l.trim_start().starts_with("version"))?;

        // The manifest value is `"X.Y.Z (hash date)"`; the version is the
        // first token.
        quoted(line)?.split_whitespace().next()?.parse().ok()
    }

    /// Whether `self` lags `stable` by two or more minor releases (or any
    /// major release).
    fn lags(&self, stable: &Version) -> bool {
        stable.major > self.major || stable.minor >= self.minor + 2
    }
}

impl FromStr for Version {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split('.').map(|p| p.parse::<u32>());
        let mut next = || {
            parts
                .next()
                .ok_or_else(|| format!("missing component in {s:?}"))?
                .map_err(|e| format!("bad component in {s:?}: {e}"))
        };

        Ok(Version {
            major: next()?,
            minor: next()?,
            patch: next()?,
        })
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// A dated rustup channel such as `nightly-2026-10-01`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DatedChannel {
    channel: String,
    /// `YYYY-MM-DD`, so string order is date order.
    date: String,
}

impl DatedChannel {
    /// Reads the `key: <channel>-YYYY-MM-DD` line from workflow `env:` text
    /// and checks the channel name.
    fn env_pin(text: &str, key: &str, channel: &str) -> Result<DatedChannel, String> {
        let prefix = format!("{key}:");
        let value = text
            .lines()
            .find_map(|l| l.trim_start().strip_prefix(&prefix))
            .ok_or_else(|| format!("no {key} pin"))?
            .trim()
            .trim_matches('"');
        let pin: DatedChannel = value.parse().map_err(|e| format!("{key}: {e}"))?;
        if pin.channel != channel {
            return Err(format!("{key} is {pin}; expected a dated {channel}"));
        }

        Ok(pin)
    }
}

impl FromStr for DatedChannel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let malformed = || format!("{s:?} is not <channel>-YYYY-MM-DD");
        let (channel, date) = s.split_once('-').ok_or_else(malformed)?;
        if !is_date(date) {
            return Err(malformed());
        }

        Ok(DatedChannel {
            channel: channel.to_string(),
            date: date.to_string(),
        })
    }
}

impl fmt::Display for DatedChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.channel, self.date)
    }
}

/// Whether `s` has the shape `YYYY-MM-DD`.
fn is_date(s: &str) -> bool {
    s.len() == 10
        && s.bytes().enumerate().all(|(i, b)| match i {
            4 | 7 => b == b'-',
            _ => b.is_ascii_digit(),
        })
}

/// Extracts the top-level `date = "YYYY-MM-DD"` from a channel manifest.
fn manifest_date(manifest: &str) -> Option<&str> {
    let line = manifest
        .lines()
        .find(|l| l.starts_with("date ") || l.starts_with("date="))?;

    quoted(line).filter(|d| is_date(d))
}

/// Returns the text between the first pair of double quotes on a line.
fn quoted(line: &str) -> Option<&str> {
    let start = line.find('"')? + 1;
    let end = start + line.get(start..)?.find('"')?;

    line.get(start..end)
}

/// Reads the release pin, exiting with a diagnostic when absent.
fn read_release_pin() -> Result<Version, String> {
    let text = fs::read_to_string(RELEASE_YML).map_err(|e| format!("{RELEASE_YML}: {e}"))?;

    Version::release_pin(&text).ok_or_else(|| format!("no RELEASE_TOOLCHAIN pin in {RELEASE_YML}"))
}

/// Reads ci.yml's dated fmt and clippy pins.
fn read_gate_pins(ci_text: &str) -> Result<[(&'static str, DatedChannel); 2], String> {
    let pin = |key, channel| {
        DatedChannel::env_pin(ci_text, key, channel)
            .map(|p| (key, p))
            .map_err(|e| format!("{CI_YML}: {e}"))
    };

    Ok([
        pin("FMT_TOOLCHAIN", "nightly")?,
        pin("CLIPPY_TOOLCHAIN", "beta")?,
    ])
}

/// Offline invariants: release pin == every ci.yml pin, MSRV <= pin, and
/// mutants.yml's nightly == ci.yml's fmt nightly.
fn consistency() -> Result<(), String> {
    let release = read_release_pin()?;
    let ci_text = fs::read_to_string(CI_YML).map_err(|e| format!("{CI_YML}: {e}"))?;
    let mutants_text =
        fs::read_to_string(MUTANTS_YML).map_err(|e| format!("{MUTANTS_YML}: {e}"))?;
    let cargo_text = fs::read_to_string(CARGO_TOML).map_err(|e| format!("{CARGO_TOML}: {e}"))?;

    let [(_, fmt_pin), _] = read_gate_pins(&ci_text)?;
    let mutants_pin = DatedChannel::env_pin(&mutants_text, "MUTANTS_TOOLCHAIN", "nightly")
        .map_err(|e| format!("{MUTANTS_YML}: {e}"))?;
    if mutants_pin != fmt_pin {
        return Err(format!(
            "{MUTANTS_YML} pins {mutants_pin} but {CI_YML} pins {fmt_pin}; bump both in one PR"
        ));
    }
    println!("ok: mutants nightly {mutants_pin} matches the fmt pin");

    let ci_pins = Version::workflow_pins(&ci_text);
    if ci_pins.is_empty() {
        return Err(format!(
            "no pinned toolchain leg in {CI_YML}; the release toolchain {release} is untested by CI"
        ));
    }
    for pin in &ci_pins {
        if *pin != release {
            return Err(format!(
                "{CI_YML} pins {pin} but {RELEASE_YML} pins {release}; bump both in one PR"
            ));
        }
    }

    if let Some(msrv) = Version::msrv(&cargo_text) {
        let msrv = msrv.map_err(|e| format!("{CARGO_TOML} rust-version: {e}"))?;
        if msrv > release {
            return Err(format!(
                "rust-version {msrv} exceeds RELEASE_TOOLCHAIN {release}"
            ));
        }
        println!("ok: MSRV {msrv} <= release toolchain {release}");
    }

    println!(
        "ok: release toolchain {release} matches {} CI pin(s)",
        ci_pins.len()
    );
    Ok(())
}

/// Staleness check against the stable channel manifest on stdin: the release
/// pin must not lag by two or more minors, and the dated gate pins must not
/// predate the current stable release.
fn staleness() -> Result<(), String> {
    let release = read_release_pin()?;
    let ci_text = fs::read_to_string(CI_YML).map_err(|e| format!("{CI_YML}: {e}"))?;
    let gate_pins = read_gate_pins(&ci_text)?;

    let mut manifest = String::new();
    std::io::stdin()
        .read_to_string(&mut manifest)
        .map_err(|e| format!("stdin: {e}"))?;
    let stable = Version::stable_channel(&manifest)
        .ok_or("no [pkg.rust] version in the channel manifest on stdin")?;
    let released = manifest_date(&manifest).ok_or("no date in the channel manifest on stdin")?;

    if release > stable {
        return Err(format!(
            "RELEASE_TOOLCHAIN {release} is ahead of stable {stable}; not a released toolchain"
        ));
    }
    if release.lags(&stable) {
        return Err(format!(
            "RELEASE_TOOLCHAIN {release} lags stable {stable} by two or more minors; bump the pin"
        ));
    }

    println!("ok: release toolchain {release} is current against stable {stable}");

    for (key, pin) in &gate_pins {
        if pin.date.as_str() < released {
            return Err(format!(
                "{key} {pin} predates stable {stable} ({released}); bump it once the fmt and clippy canaries pass"
            ));
        }
        println!("ok: {key} {pin} is not older than stable {stable} ({released})");
    }
    Ok(())
}

fn main() -> ExitCode {
    let mode = std::env::args().nth(1).unwrap_or_default();

    let result = match mode.as_str() {
        "consistency" => consistency(),
        "staleness" => staleness(),
        _ => Err("usage: toolchain-drift <consistency|staleness>".to_string()),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("toolchain-drift: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_manifest_versions() {
        let plain: Version = "1.97.1".parse().unwrap();
        assert_eq!(plain.to_string(), "1.97.1");

        let manifest = "[pkg.rust]\nversion = \"1.97.1 (8bab26f4f 2026-07-14)\"\n";
        assert_eq!(Version::stable_channel(manifest), Some(plain));
    }

    #[test]
    fn rejects_malformed_versions() {
        assert!("1.97".parse::<Version>().is_err());
        assert!("stable".parse::<Version>().is_err());
        assert!("1.97.x".parse::<Version>().is_err());
    }

    #[test]
    fn extracts_pins_and_ignores_floating_channels() {
        let release = "env:\n  RELEASE_TOOLCHAIN: \"1.97.1\"\n";
        assert_eq!(Version::release_pin(release), "1.97.1".parse().ok());

        let ci = concat!(
            "          toolchain: stable\n",
            "          toolchain: \"1.97.1\"\n",
            "          toolchain: nightly\n",
            "          toolchain: \"1.97.1\"\n",
        );
        assert_eq!(Version::workflow_pins(ci).len(), 2);
    }

    #[test]
    fn parses_dated_channels() {
        let pin: DatedChannel = "nightly-2026-10-01".parse().unwrap();
        assert_eq!(pin.channel, "nightly");
        assert_eq!(pin.date, "2026-10-01");
        assert_eq!(pin.to_string(), "nightly-2026-10-01");

        assert!("nightly".parse::<DatedChannel>().is_err());
        assert!("nightly-2026-10".parse::<DatedChannel>().is_err());
        assert!("nightly-2026/10/01".parse::<DatedChannel>().is_err());
        assert!("1.99.0".parse::<DatedChannel>().is_err());
    }

    #[test]
    fn extracts_env_pins() {
        let ci = concat!(
            "env:\n",
            "  # Bump FMT_TOOLCHAIN: with care.\n",
            "  FMT_TOOLCHAIN: nightly-2026-10-01\n",
            "  CLIPPY_TOOLCHAIN: \"beta-2026-10-02\"\n",
            "          toolchain: ${{ env.FMT_TOOLCHAIN }}\n",
        );
        let fmt = DatedChannel::env_pin(ci, "FMT_TOOLCHAIN", "nightly").unwrap();
        assert_eq!(fmt.to_string(), "nightly-2026-10-01");
        let clippy = DatedChannel::env_pin(ci, "CLIPPY_TOOLCHAIN", "beta").unwrap();
        assert_eq!(clippy.to_string(), "beta-2026-10-02");

        assert!(DatedChannel::env_pin(ci, "CLIPPY_TOOLCHAIN", "nightly").is_err());
        assert!(DatedChannel::env_pin(ci, "MUTANTS_TOOLCHAIN", "nightly").is_err());
        let floating = "  FMT_TOOLCHAIN: nightly\n";
        assert!(DatedChannel::env_pin(floating, "FMT_TOOLCHAIN", "nightly").is_err());
    }

    #[test]
    fn reads_manifest_date() {
        let manifest = "manifest-version = \"2\"\ndate = \"2026-10-01\"\n[pkg.rust]\n";
        assert_eq!(manifest_date(manifest), Some("2026-10-01"));
        assert_eq!(manifest_date("[pkg.rust]\n"), None);
    }

    #[test]
    fn ignores_dated_channels() {
        let ci = concat!(
            "  FMT_TOOLCHAIN: nightly-2026-10-01\n",
            "          toolchain: ${{ env.FMT_TOOLCHAIN }}\n",
            "          toolchain: nightly-2026-10-01\n",
            "          toolchain: \"beta-2026-10-02\"\n",
            "          toolchain: \"1.97.1\"\n",
        );
        assert_eq!(Version::workflow_pins(ci), vec!["1.97.1".parse().unwrap()]);
    }

    #[test]
    fn extracts_msrv() {
        let cargo = "edition = \"2021\"\nrust-version = \"1.91.0\"\n";
        assert_eq!(Version::msrv(cargo), Some("1.91.0".parse()));
        let short = "rust-version = \"1.89\"\n";
        assert_eq!(Version::msrv(short), Some("1.89.0".parse()));
        assert!(matches!(
            Version::msrv("rust-version = \"1.x\"\n"),
            Some(Err(_))
        ));
        assert_eq!(Version::msrv("edition = \"2021\"\n"), None);
    }

    #[test]
    fn lag_boundaries() {
        let pin: Version = "1.97.1".parse().unwrap();

        let same: Version = "1.97.5".parse().unwrap();
        let one_minor: Version = "1.98.0".parse().unwrap();
        let two_minors: Version = "1.99.0".parse().unwrap();
        let next_major: Version = "2.0.0".parse().unwrap();

        assert!(!pin.lags(&same));
        assert!(!pin.lags(&one_minor));
        assert!(pin.lags(&two_minors));
        assert!(pin.lags(&next_major));
    }
}
