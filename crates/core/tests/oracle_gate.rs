//! The `naive` oracle and the route-forcing hook are absent from a default release build and
//! present with `oracle`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Builds the library in release mode into its own target dir and returns its rlib bytes.
fn rlib_bytes(features: &[&str], target: &Path) -> Vec<u8> {
    // timing-ok: a compile-time path, fixed per build
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["build", "-p", "ladde", "--lib", "--release"])
        .arg("--target-dir")
        .arg(target)
        // timing-ok: a compile-time path, fixed per build
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    if !features.is_empty() {
        cmd.args(["--features", &features.join(",")]);
    }
    assert!(cmd.status().expect("cargo runs").success());
    let rlib: PathBuf = target.join("release/libladde.rlib");
    std::fs::read(rlib).expect("rlib exists")
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn default_release_build_has_no_oracle() {
    // timing-ok: names a scratch directory only; no outcome depends on it
    let dir = std::env::temp_dir().join(format!("mp-oracle-gate-{}", std::process::id()));
    let with = rlib_bytes(&["oracle"], &dir.join("with"));
    // Positive control: the probe string is really there when the feature is on.
    assert!(contains(&with, b"naive_two_sided"));
    assert!(contains(&with, b"REANCHOR_EVERY"));
    let without = rlib_bytes(&[], &dir.join("without"));
    assert!(!contains(&without, b"naive_two_sided"));
    assert!(!contains(&without, b"naive_left_topk"));
    // The route-forcing hook of the route harness is part of the same gate.
    assert!(!contains(&without, b"REANCHOR_EVERY"));
    assert!(!contains(&without, b"ANCHOR_SEED"));
    let _ = std::fs::remove_dir_all(dir);
}
