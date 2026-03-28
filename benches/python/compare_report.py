#!/usr/bin/env python3
"""Generate a Markdown comparison table from Rust and Python benchmark JSON.

Usage::

    python benches/python/compare_report.py target/bench-compare/

Reads ``rust_core.json``, ``python_core.json`` (and optionally
``rust_redis.json``, ``python_redis.json``) from the given directory and
prints a Markdown table to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt_time(ns: float) -> str:
    """Format nanoseconds into a human-readable string."""
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.3f} s"


def _load(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _build_lookup(data: dict | None) -> dict[str, float]:
    if data is None:
        return {}
    return {r["name"]: r["mean_ns"] for r in data.get("results", [])}


def _ratio_str(rust_ns: float, python_ns: float) -> str:
    if rust_ns <= 0 or python_ns <= 0:
        return "—"
    ratio = python_ns / rust_ns
    return f"{ratio:.1f}×"


def _print_table(title: str, rust: dict[str, float], python: dict[str, float]) -> None:
    all_names = sorted(set(rust.keys()) | set(python.keys()))
    if not all_names:
        return

    print(f"\n### {title}\n")
    print("| Benchmark | Rust | Python | Ratio (Py/Rust) |")
    print("|-----------|------|--------|-----------------|")

    for name in all_names:
        r = rust.get(name)
        p = python.get(name)
        r_str = _fmt_time(r) if r is not None else "—"
        p_str = _fmt_time(p) if p is not None else "—"
        ratio = _ratio_str(r, p) if (r and p) else "—"
        print(f"| {name} | {r_str} | {p_str} | {ratio} |")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: compare_report.py <results-dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    print("# Rust vs Python Benchmark Comparison\n")
    print(f"Results directory: `{results_dir}`\n")

    # Core benchmarks
    rust_core = _load(results_dir / "rust_core.json")
    python_core = _load(results_dir / "python_core.json")
    _print_table(
        "Core Benchmarks (no Redis)",
        _build_lookup(rust_core),
        _build_lookup(python_core),
    )

    # Redis benchmarks
    rust_redis = _load(results_dir / "rust_redis.json")
    python_redis = _load(results_dir / "python_redis.json")
    if rust_redis or python_redis:
        _print_table(
            "Redis-Backed Benchmarks",
            _build_lookup(rust_redis),
            _build_lookup(python_redis),
        )

    print("\n---")
    print("*Ratio > 1× means Rust is faster. "
          "Lower absolute time is better.*")
    print()


if __name__ == "__main__":
    main()
