//! Smoke tests for the `rvl` CLI binary.
//!
//! These tests verify help text, version output, and argument validation
//! without requiring a Redis connection.

use std::process::Command;

fn rvl_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rvl"))
}

// ── Version command ─────────────────────────────────────────────────────────

#[test]
fn version_long_format_includes_prefix() {
    let output = rvl_bin()
        .arg("version")
        .output()
        .expect("failed to run rvl");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("RedisVL version"),
        "expected 'RedisVL version' prefix, got: {stdout}"
    );
    assert!(
        stdout.trim().contains('.'),
        "expected semver in output, got: {stdout}"
    );
}

#[test]
fn version_short_prints_only_semver() {
    let output = rvl_bin()
        .args(["version", "--short"])
        .output()
        .expect("failed to run rvl");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    assert!(
        !trimmed.contains("RedisVL"),
        "short version should not include prefix, got: {trimmed}"
    );
    assert!(
        trimmed.contains('.'),
        "expected semver output, got: {trimmed}"
    );
}

#[test]
fn version_short_flag_alias() {
    let output = rvl_bin()
        .args(["version", "-s"])
        .output()
        .expect("failed to run rvl");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("RedisVL"), "-s should behave like --short");
}

// ── Top-level help ──────────────────────────────────────────────────────────

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
fn missing_subcommand_exits_with_error() {
    let output = rvl_bin().output().expect("failed to run rvl");
    assert!(!output.status.success());
}

// ── Index subcommand help ───────────────────────────────────────────────────

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
    assert!(stdout.contains("destroy"));
    assert!(stdout.contains("info"));
    assert!(stdout.contains("listall"));
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

// ── --index / --schema argument validation ──────────────────────────────────

#[test]
fn index_info_no_args_exits_with_error() {
    let output = rvl_bin()
        .args(["index", "info"])
        .output()
        .expect("failed to run rvl");
    assert!(
        !output.status.success(),
        "info without --index or --schema should fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Index name or schema must be provided"),
        "expected helpful error message, got: {stderr}"
    );
}

#[test]
fn index_delete_no_args_exits_with_error() {
    let output = rvl_bin()
        .args(["index", "delete"])
        .output()
        .expect("failed to run rvl");
    assert!(
        !output.status.success(),
        "delete without --index or --schema should fail"
    );
}

#[test]
fn stats_no_args_exits_with_error() {
    let output = rvl_bin()
        .args(["stats"])
        .output()
        .expect("failed to run rvl");
    assert!(
        !output.status.success(),
        "stats without --index or --schema should fail"
    );
}

#[test]
fn index_info_with_bad_schema_file_exits_with_error() {
    let output = rvl_bin()
        .args(["index", "info", "--schema", "/nonexistent/file.yaml"])
        .output()
        .expect("failed to run rvl");
    assert!(!output.status.success());
}

// ── listall does not require --schema or --index ────────────────────────────

#[test]
fn index_listall_help_shows_no_schema_required() {
    let output = rvl_bin()
        .args(["index", "listall", "--help"])
        .output()
        .expect("failed to run rvl");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("redis-url"),
        "listall should show --redis-url option"
    );
}

// ── destroy help shows it deletes documents ─────────────────────────────────

#[test]
fn index_destroy_help_mentions_documents() {
    let output = rvl_bin()
        .args(["index", "destroy", "--help"])
        .output()
        .expect("failed to run rvl");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("documents"),
        "destroy help should mention documents, got: {stdout}"
    );
}
