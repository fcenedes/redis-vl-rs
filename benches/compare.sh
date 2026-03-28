#!/usr/bin/env bash
# Rust-vs-Python benchmark comparison runner.
#
# Usage:
#   benches/compare.sh              # core (pure-computation) benchmarks only
#   benches/compare.sh --redis      # include Redis-backed benchmarks
#   benches/compare.sh --verbose    # show progress on stderr
#
# Environment:
#   REDIS_URL          Redis connection URL (default: redis://127.0.0.1:6379)
#   PYTHON             Python interpreter   (default: python3)
#   CARGO_BENCH_ARGS   Extra args for cargo bench (e.g. --features sql)
#
# Prerequisites:
#   - Rust toolchain (cargo, criterion)
#   - Python 3.10+ with redisvl installed:
#       pip install -r benches/python/requirements.txt
#   - For --redis: a running Redis 8+ / Redis Stack instance
#
# Output:
#   Prints a Markdown comparison table to stdout.
#   Raw JSON results are saved under target/bench-compare/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$ROOT_DIR/target/bench-compare"
PYTHON="${PYTHON:-python3}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379}"
VERBOSE=""
RUN_REDIS=""

for arg in "$@"; do
    case "$arg" in
        --redis)   RUN_REDIS=1 ;;
        --verbose) VERBOSE="--verbose" ;;
        *)         echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

# -------------------------------------------------------------------------
# 1. Run Rust Criterion benchmarks (core)
# -------------------------------------------------------------------------
echo ">>> Running Rust core benchmarks …" >&2
cargo bench -p redis-vl --bench core_benchmarks ${CARGO_BENCH_ARGS:-} \
    -- --save-baseline rust-vs-python 2>&1 | tail -1 >&2 || true

# Parse Criterion estimates into JSON
"$PYTHON" - "$ROOT_DIR" > "$OUT_DIR/rust_core.json" <<'PYEOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
criterion_dir = root / "target" / "criterion"
results = []
for est in sorted(criterion_dir.rglob("new/estimates.json")):
    parts = est.relative_to(criterion_dir).parts
    name = parts[0]
    with open(est) as f:
        data = json.load(f)
    mean_ns = data.get("mean", {}).get("point_estimate", 0)
    stddev_ns = data.get("std_dev", {}).get("point_estimate", 0)
    median_ns = data.get("median", {}).get("point_estimate", 0)
    results.append({
        "name": name,
        "mean_ns": mean_ns,
        "median_ns": median_ns,
        "stddev_ns": stddev_ns,
    })
json.dump({"suite": "core", "language": "rust", "results": results}, sys.stdout, indent=2)
print()
PYEOF

# -------------------------------------------------------------------------
# 2. Run Python core benchmarks
# -------------------------------------------------------------------------
echo ">>> Running Python core benchmarks …" >&2
(cd "$SCRIPT_DIR/python" && "$PYTHON" bench_core.py $VERBOSE) > "$OUT_DIR/python_core.json"

# -------------------------------------------------------------------------
# 3. Redis-backed benchmarks (optional)
# -------------------------------------------------------------------------
if [[ -n "$RUN_REDIS" ]]; then
    echo ">>> Running Rust Redis benchmarks …" >&2
    REDISVL_RUN_BENCHMARKS=1 REDIS_URL="$REDIS_URL" \
        cargo bench -p redis-vl --bench redis_benchmarks ${CARGO_BENCH_ARGS:-} \
            -- --save-baseline rust-vs-python 2>&1 | tail -1 >&2 || true

    "$PYTHON" - "$ROOT_DIR" > "$OUT_DIR/rust_redis.json" <<'PYEOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
criterion_dir = root / "target" / "criterion"
results = []
for est in sorted(criterion_dir.rglob("new/estimates.json")):
    parts = est.relative_to(criterion_dir).parts
    name = parts[0]
    with open(est) as f:
        data = json.load(f)
    mean_ns = data.get("mean", {}).get("point_estimate", 0)
    stddev_ns = data.get("std_dev", {}).get("point_estimate", 0)
    median_ns = data.get("median", {}).get("point_estimate", 0)
    results.append({
        "name": name,
        "mean_ns": mean_ns,
        "median_ns": median_ns,
        "stddev_ns": stddev_ns,
    })
json.dump({"suite": "redis", "language": "rust", "results": results}, sys.stdout, indent=2)
print()
PYEOF

    echo ">>> Running Python Redis benchmarks …" >&2
    (cd "$SCRIPT_DIR/python" && REDIS_URL="$REDIS_URL" "$PYTHON" bench_redis.py $VERBOSE) \
        > "$OUT_DIR/python_redis.json"
fi

# -------------------------------------------------------------------------
# 4. Generate comparison report
# -------------------------------------------------------------------------
echo ">>> Generating comparison report …" >&2
"$PYTHON" "$SCRIPT_DIR/python/compare_report.py" "$OUT_DIR"
