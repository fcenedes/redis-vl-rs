"""Benchmark timing harness shared by Python benchmark scripts.

Provides a micro-benchmark runner that mirrors Criterion's methodology:
warmup, then multiple sample iterations, reporting mean/median/stddev in
nanoseconds.  Results are emitted as JSON for machine consumption.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional


def _run_bench(
    name: str,
    fn: Callable[[], Any],
    *,
    warmup_iters: int = 10,
    sample_iters: int = 100,
    setup: Optional[Callable[[], None]] = None,
    teardown: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Run *fn* repeatedly, returning timing statistics in nanoseconds."""
    if setup:
        setup()

    # Warmup
    for _ in range(warmup_iters):
        fn()

    # Sample
    times_ns: List[float] = []
    for _ in range(sample_iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times_ns.append(float(t1 - t0))

    if teardown:
        teardown()

    mean = statistics.mean(times_ns)
    median = statistics.median(times_ns)
    stddev = statistics.stdev(times_ns) if len(times_ns) > 1 else 0.0

    return {
        "name": name,
        "samples": len(times_ns),
        "mean_ns": mean,
        "median_ns": median,
        "stddev_ns": stddev,
        "min_ns": min(times_ns),
        "max_ns": max(times_ns),
    }


class BenchSuite:
    """Collects and runs benchmarks, emitting JSON results."""

    def __init__(self, suite_name: str) -> None:
        self.suite_name = suite_name
        self._results: List[Dict[str, Any]] = []
        self._benches: List[tuple] = []

    def add(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        warmup_iters: int = 10,
        sample_iters: int = 100,
        setup: Optional[Callable[[], None]] = None,
        teardown: Optional[Callable[[], None]] = None,
    ) -> None:
        self._benches.append(
            (name, fn, warmup_iters, sample_iters, setup, teardown)
        )

    def run(self, *, verbose: bool = False) -> List[Dict[str, Any]]:
        for name, fn, warmup, samples, setup, teardown in self._benches:
            if verbose:
                print(f"  running: {name} ...", file=sys.stderr, flush=True)
            result = _run_bench(
                name,
                fn,
                warmup_iters=warmup,
                sample_iters=samples,
                setup=setup,
                teardown=teardown,
            )
            self._results.append(result)
        return self._results

    def emit_json(self, *, file=None) -> None:
        out = {
            "suite": self.suite_name,
            "language": "python",
            "results": self._results,
        }
        json.dump(out, file or sys.stdout, indent=2)
        print(file=file or sys.stdout)
