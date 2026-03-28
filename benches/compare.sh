#!/usr/bin/env bash
# Rust-vs-Python benchmark comparison runner.
#
# Usage:
#   benches/compare.sh              # core (pure-computation) benchmarks only
#   benches/compare.sh --redis      # include Redis-backed benchmarks
#   benches/compare.sh --verbose    # show progress on stderr
#   benches/compare.sh --keep-env   # keep the uv virtualenv after the run
#
# The script creates an isolated virtualenv via `uv` under
# target/bench-compare/.venv, installs benchmark-only Python dependencies,
# and removes the virtualenv when done (unless --keep-env is passed).
#
# Environment:
#   REDIS_URL          Redis connection URL (default: redis://127.0.0.1:6379)
#   CARGO_BENCH_ARGS   Extra args for cargo bench (e.g. --features sql)
#
# Prerequisites:
#   - Rust toolchain (cargo, criterion)
#   - uv (https://docs.astral.sh/uv/)
#   - For --redis: a running Redis 8+ / Redis Stack instance
#
# Output:
#   Prints a Markdown comparison table to stdout.
#   Raw JSON results are saved under target/bench-compare/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$ROOT_DIR/target/bench-compare"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379}"
VERBOSE=""
RUN_REDIS=""
KEEP_ENV=""

for arg in "$@"; do
    case "$arg" in
        --redis)    RUN_REDIS=1 ;;
        --verbose)  VERBOSE="--verbose" ;;
        --keep-env) KEEP_ENV=1 ;;
        *)          echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

# -------------------------------------------------------------------------
# 0. Create isolated uv virtualenv with benchmark dependencies
# -------------------------------------------------------------------------
VENV_DIR="$OUT_DIR/.venv"

echo ">>> Setting up Python virtualenv via uv …" >&2
uv venv "$VENV_DIR" --quiet 2>&1 >&2
uv pip install --quiet -r "$SCRIPT_DIR/python/requirements.txt" -p "$VENV_DIR/bin/python" 2>&1 >&2

PYTHON="$VENV_DIR/bin/python"

# Clean up the virtualenv on exit unless --keep-env was passed
if [[ -z "$KEEP_ENV" ]]; then
    trap 'rm -rf "$VENV_DIR" 2>/dev/null || true' EXIT
fi

# -------------------------------------------------------------------------
# 1. Run Rust Criterion benchmarks (core)
# -------------------------------------------------------------------------
echo ">>> Running Rust core benchmarks …" >&2
cargo bench -p redis-vl --bench core_benchmarks ${CARGO_BENCH_ARGS:-} \
    -- --save-baseline rust-vs-python >&2

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
    # Wait for Redis to be ready (Docker may be slow to accept connections)
    echo ">>> Checking Redis readiness at $REDIS_URL …" >&2
    _redis_host="${REDIS_URL#redis://}"
    _redis_host="${_redis_host%%/*}"
    _redis_host_only="${_redis_host%%:*}"
    _redis_port="${_redis_host##*:}"
    _redis_port="${_redis_port:-6379}"
    for _i in $(seq 1 30); do
        if "$PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('$_redis_host_only', $_redis_port))
    s.sendall(b'PING\r\n')
    data = s.recv(64)
    s.close()
    sys.exit(0 if b'PONG' in data else 1)
except Exception:
    s.close()
    sys.exit(1)
" 2>/dev/null; then
            echo "  Redis is ready." >&2
            break
        fi
        echo "  Waiting for Redis (attempt $_i/30)…" >&2
        sleep 2
    done

    echo ">>> Running Rust Redis benchmarks …" >&2
    REDISVL_RUN_BENCHMARKS=1 REDIS_URL="$REDIS_URL" \
        cargo bench -p redis-vl --bench redis_benchmarks ${CARGO_BENCH_ARGS:-} \
            -- --save-baseline rust-vs-python >&2

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
