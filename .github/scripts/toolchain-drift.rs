//! Guards the release-toolchain pin invariant chain: `rust-version`
//! (Cargo.toml, once declared) <= `RELEASE_TOOLCHAIN` (release.yml)
//! == every pinned `toolchain: "X.Y.Z"` leg in ci.yml <= current stable.
//!
//! Modes:
//! - `toolchain-drift consistency` — offline; checks the checked-out tree. Run
//!   on pull requests (devops.yml).
//! - `curl -s .../channel-rust-stable.toml | toolchain-drift staleness` — reads
//!   the channel manifest on stdin and fails when the pin lags stable by two or
//!   more minor releases. Run on the weekly cron (audit.yml).

use std::{fmt, fs, io::Read, process::ExitCode, str::FromStr};

/// Path to the release workflow holding the `RELEASE_TOOLCHAIN` pin.
const RELEASE_YML: &str = ".github/workflows/release.yml";
/// Path to the CI workflow holding the pinned test-matrix leg.
const CI_YML: &str = ".github/workflows/ci.yml";
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
    /// from workflow text. Floating channels (`stable`, `nightly`) are
    /// unquoted in our workflows and are not pins.
    fn workflow_pins(text: &str) -> Vec<Version> {
        text.lines()
            .filter(|l| l.trim_start().starts_with("toolchain: \""))
            .filter_map(quoted)
            .filter_map(|v| v.parse().ok())
            .collect()
    }

    /// Extracts `rust-version = "X.Y.Z"` from Cargo.toml text.
    fn msrv(text: &str) -> Option<Version> {
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with("rust-version"))?;

        quoted(line)?.parse().ok()
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

/// Offline invariants: release pin == every ci.yml pin, and MSRV <= pin.
fn consistency() -> Result<(), String> {
    let release = read_release_pin()?;
    let ci_text = fs::read_to_string(CI_YML).map_err(|e| format!("{CI_YML}: {e}"))?;
    let cargo_text = fs::read_to_string(CARGO_TOML).map_err(|e| format!("{CARGO_TOML}: {e}"))?;

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

/// Staleness check: the release pin must not lag the stable channel
/// manifest supplied on stdin by two or more minors.
fn staleness() -> Result<(), String> {
    let release = read_release_pin()?;

    let mut manifest = String::new();
    std::io::stdin()
        .read_to_string(&mut manifest)
        .map_err(|e| format!("stdin: {e}"))?;
    let stable = Version::stable_channel(&manifest)
        .ok_or("no [pkg.rust] version in the channel manifest on stdin")?;

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
    fn extracts_msrv() {
        let cargo = "edition = \"2021\"\nrust-version = \"1.91.0\"\n";
        assert_eq!(Version::msrv(cargo), "1.91.0".parse().ok());
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
