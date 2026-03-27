//! Smoke tests for the `rvl` CLI binary.
//!
//! These tests verify help text and version output without requiring a Redis
//! connection.

use std::process::Command;

fn rvl_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rvl"))
}

#[test]
fn version_prints_semver() {
    let output = rvl_bin()
        .arg("version")
        .output()
        .expect("failed to run rvl");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should be a semver-like version string
    assert!(
        stdout.trim().contains('.'),
        "expected semver output, got: {stdout}"
    );
}

#[test]
fn help_shows_usage() {
    let output = rvl_bin().arg("--help").output().expect("failed to run rvl");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Redis Vector Library CLI"));
    assert!(stdout.contains("index"));
    assert!(stdout.contains("version"));
    assert!(stdout.contains("stats"));
}

#[test]
fn index_help_shows_subcommands() {
    let output = rvl_bin()
        .args(["index", "--help"])
        .output()
        .expect("failed to run rvl");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("create"));
    assert!(stdout.contains("delete"));
    assert!(stdout.contains("info"));
    assert!(stdout.contains("listall"));
}

#[test]
fn missing_subcommand_exits_with_error() {
    let output = rvl_bin().output().expect("failed to run rvl");
    assert!(!output.status.success());
}

#[test]
fn index_create_missing_schema_exits_with_error() {
    let output = rvl_bin()
        .args(["index", "create"])
        .output()
        .expect("failed to run rvl");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--schema"));
}
