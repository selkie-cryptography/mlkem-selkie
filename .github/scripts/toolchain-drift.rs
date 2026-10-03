//! Guards the release-toolchain pin invariant chain: `rust-version`
//! (Cargo.toml, once declared) <= `RELEASE_TOOLCHAIN` (release.yml)
//! == every pinned `toolchain: "X.Y.Z"` leg in ci.yml <= current stable.
//! Also flags ci.yml's `STABLE_TOOLCHAIN` once a newer stable minor ships.
//!
//! Modes:
//! - `toolchain-drift consistency` — offline; checks the checked-out tree. Run
//!   on pull requests (devops.yml).
//! - `curl -s .../channel-rust-stable.toml | toolchain-drift staleness` — reads
//!   the channel manifest on stdin and fails when `RELEASE_TOOLCHAIN` lags
//!   stable by two or more minor releases, or `STABLE_TOOLCHAIN` by one or
//!   more. Run weekly (toolchain-canary.yml).

use std::{fmt, fs, io::Read, process::ExitCode, str::FromStr};

/// Path to the release workflow holding the `RELEASE_TOOLCHAIN` pin.
const RELEASE_YML: &str = ".github/workflows/release.yml";
/// Path to the CI workflow holding the pinned test-matrix leg and
/// `STABLE_TOOLCHAIN`.
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
    /// Extracts the `KEY: "X.Y.Z"` env pin from workflow text.
    fn env_pin(text: &str, key: &str) -> Option<Version> {
        let prefix = format!("{key}:");
        let line = text.lines().find(|l| l.trim_start().starts_with(&prefix))?;

        quoted(line)?.parse().ok()
    }

    /// Reads the `KEY: "X.Y.Z"` env pin from the workflow at `path`.
    fn read_env_pin(path: &str, key: &str) -> Result<Version, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;

        Version::env_pin(&text, key).ok_or_else(|| format!("no {key} pin in {path}"))
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

    /// Whether `self` lags `stable` by `minors` or more minor releases (or
    /// any major release).
    fn lags(&self, stable: &Version, minors: u32) -> bool {
        stable.major > self.major || stable.minor >= self.minor + minors
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

/// Offline invariants: release pin == every ci.yml pin, MSRV <= pin, and
/// ci.yml's `STABLE_TOOLCHAIN` parses.
fn consistency() -> Result<(), String> {
    let release = Version::read_env_pin(RELEASE_YML, "RELEASE_TOOLCHAIN")?;
    let ci_stable = Version::read_env_pin(CI_YML, "STABLE_TOOLCHAIN")?;
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
    println!("ok: STABLE_TOOLCHAIN {ci_stable} parses");
    Ok(())
}

/// Staleness check against the stable channel manifest on stdin. Fails when
/// `RELEASE_TOOLCHAIN` lags it by two or more minors, `STABLE_TOOLCHAIN` by
/// one or more, or either pin is ahead of it.
fn staleness() -> Result<(), String> {
    let mut manifest = String::new();
    std::io::stdin()
        .read_to_string(&mut manifest)
        .map_err(|e| format!("stdin: {e}"))?;
    let stable = Version::stable_channel(&manifest)
        .ok_or("no [pkg.rust] version in the channel manifest on stdin")?;

    // Checks every pin, so one run names all the pins to bump.
    let mut stale = Vec::new();
    for (key, path, minors) in [
        ("RELEASE_TOOLCHAIN", RELEASE_YML, 2),
        ("STABLE_TOOLCHAIN", CI_YML, 1),
    ] {
        let pin = Version::read_env_pin(path, key)?;
        if pin > stable {
            stale.push(format!(
                "{key} {pin} is ahead of stable {stable}; not a released toolchain"
            ));
        } else if pin.lags(&stable, minors) {
            stale.push(format!(
                "{key} {pin} lags stable {stable} by {minors} or more minors; bump {key} in {path}"
            ));
        } else {
            println!("ok: {key} {pin} is current against stable {stable}");
        }
    }

    if stale.is_empty() {
        Ok(())
    } else {
        Err(stale.join("\n"))
    }
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
            for line in message.lines() {
                eprintln!("toolchain-drift: {line}");
            }
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
        assert_eq!(
            Version::env_pin(release, "RELEASE_TOOLCHAIN"),
            "1.97.1".parse().ok()
        );

        let ci = concat!(
            "          toolchain: stable\n",
            "          toolchain: \"1.97.1\"\n",
            "          toolchain: nightly\n",
            "          toolchain: \"1.97.1\"\n",
        );
        assert_eq!(Version::workflow_pins(ci).len(), 2);
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

        assert!(!pin.lags(&same, 2));
        assert!(!pin.lags(&one_minor, 2));
        assert!(pin.lags(&two_minors, 2));
        assert!(pin.lags(&next_major, 2));

        // STABLE_TOOLCHAIN's window: a newer minor lags, a newer patch doesn't.
        assert!(!pin.lags(&same, 1));
        assert!(pin.lags(&one_minor, 1));
        assert!(pin.lags(&next_major, 1));
    }

    #[test]
    fn extracts_stable_pin_and_skips_references() {
        let ci = concat!(
            "  # Bump STABLE_TOOLCHAIN: see toolchain-canary.yml.\n",
            "  STABLE_TOOLCHAIN: \"1.99.0\"\n",
            "          toolchain: ${{ env.STABLE_TOOLCHAIN }}\n",
        );
        assert_eq!(
            Version::env_pin(ci, "STABLE_TOOLCHAIN"),
            "1.99.0".parse().ok()
        );
        assert_eq!(Version::env_pin(ci, "RELEASE_TOOLCHAIN"), None);

        let unquoted = "  STABLE_TOOLCHAIN: 1.99.0\n";
        assert_eq!(Version::env_pin(unquoted, "STABLE_TOOLCHAIN"), None);
    }
}
