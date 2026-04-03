#!/usr/bin/env bash
# smoke_test.sh – build and run the fast test suite for one or more Python versions.
#
# By default all versions run in PARALLEL.  Set PARALLEL=0 to run sequentially.
#
# Usage:
#   devtools/scripts/smoke_test.sh [PYTHON_VERSION ...]
#
# Examples:
#   devtools/scripts/smoke_test.sh                     # defaults: 3.12 3.13
#   devtools/scripts/smoke_test.sh 3.10 3.11 3.12 3.13 3.14
#
# Environment variables:
#   TEST_FLAGS   – pytest flags (default: --skip-slow --skip-stochastic --skip-network)
#   LOG_DIR      – directory for per-version log files (default: devtools/logs)
#   PARALLEL     – set to 0 to run sequentially (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKERFILE="$REPO_ROOT/devtools/docker/Dockerfile"
TEST_FLAGS="${TEST_FLAGS:---skip-slow --skip-stochastic --skip-network}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/devtools/logs}"
PARALLEL="${PARALLEL:-1}"

# Default to testing Python 3.7 through 3.14, but allow overriding via command-line args
if [ $# -gt 0 ]; then
    VERSIONS=("$@")
else
    VERSIONS=(3.7 3.8 3.9 3.10 3.11 3.12 3.13 3.14)
fi

mkdir -p "$LOG_DIR"

# ---------- worker function (one per Python version) ----------
run_one() {
    local PY="$1"
    local LOG="$LOG_DIR/smoke_py${PY}.log"
    local IMAGE="frustratometer-test:py${PY}"

    {
        echo "════════════════════════════════════════"
        echo " Python ${PY} – started $(date '+%H:%M:%S')"
        echo "════════════════════════════════════════"

        echo "--- docker build ---"
        if ! docker build \
            --build-arg PYTHON_VERSION="${PY}" \
            -f "$DOCKERFILE" \
            -t "$IMAGE" \
            "$REPO_ROOT" ; then
            echo "BUILD FAILED for Python ${PY}"
            return 1
        fi

        echo ""
        echo "--- pytest ---"
        if docker run --rm "$IMAGE" \
            conda run -n test --no-capture-output \
            python -m pytest tests/ $TEST_FLAGS -v --tb=short --color=no ; then
            echo ""
            echo "RESULT: PASS (Python ${PY})"
            return 0
        else
            echo ""
            echo "RESULT: FAIL (Python ${PY})"
            return 1
        fi
    } > "$LOG" 2>&1
}

# ---------- launch ----------
PIDS=()
for PY in "${VERSIONS[@]}"; do
    if [ "$PARALLEL" = "1" ]; then
        run_one "$PY" &
        PIDS+=("$!:$PY")
    else
        run_one "$PY" || true
    fi
done

# ---------- wait & collect ----------
PASS=()
FAIL=()

if [ "$PARALLEL" = "1" ]; then
    for entry in "${PIDS[@]}"; do
        PID="${entry%%:*}"
        PY="${entry##*:}"
        if wait "$PID"; then
            PASS+=("$PY")
        else
            FAIL+=("$PY")
        fi
    done
else
    for PY in "${VERSIONS[@]}"; do
        if grep -q "^RESULT: PASS" "$LOG_DIR/smoke_py${PY}.log" 2>/dev/null; then
            PASS+=("$PY")
        else
            FAIL+=("$PY")
        fi
    done
fi

# ---------- summary ----------
echo ""
echo "════════════════════════════════════════"
echo " Summary"
echo "════════════════════════════════════════"
[ ${#PASS[@]} -gt 0 ] && echo "  PASSED: ${PASS[*]}"
[ ${#FAIL[@]} -gt 0 ] && echo "  FAILED: ${FAIL[*]}"
echo "  Logs:   $LOG_DIR/"
echo ""

# Print failures inline for convenience
for PY in "${FAIL[@]+"${FAIL[@]}"}"; do
    echo ""
    echo "──── FAILURES for Python ${PY} (last 50 lines) ────"
    tail -50 "$LOG_DIR/smoke_py${PY}.log"
done

[ ${#FAIL[@]} -eq 0 ]   # exit 0 only if nothing failed
