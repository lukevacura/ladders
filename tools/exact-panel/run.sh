#!/usr/bin/env bash
# Usage: PY_OURS=... PY_STUMPY=... [PY_SCAMP=...] [PY_TRUTH=...] [PANEL_OUT=dir] [PANEL_NS=512,1024] [PROCS=8] ./run.sh
# Each PY_* is a Python interpreter: PY_OURS has ladders, PY_STUMPY has stumpy, PY_SCAMP has pyscamp
# (left unset, SCAMP is skipped). PY_TRUTH needs only numpy (default: PY_OURS).
set -euo pipefail
cd "$(dirname "$0")"
# The exact reference lives beside this directory.
export PYTHONPATH="$PWD/../exact-ref${PYTHONPATH:+:$PYTHONPATH}"
: "${PY_OURS:?}" "${PY_STUMPY:?}"
PY_TRUTH="${PY_TRUTH:-$PY_OURS}"; PROCS="${PROCS:-8}"
export PANEL_OUT="${PANEL_OUT:-$PWD/out}"
rm -rf "$PANEL_OUT"; mkdir -p "$PANEL_OUT"
"$PY_TRUTH" make_data.py
"$PY_OURS" dump_ours.py
"$PY_STUMPY" dump_stumpy.py 2>&1 | grep -v -e RuntimeWarning -e Σ_T_inverse || true
if [ -n "${PY_SCAMP:-}" ]; then "$PY_SCAMP" dump_scamp.py; else echo "PY_SCAMP is not set: SCAMP skipped"; fi
"$PY_TRUTH" truth.py --selftest > "$PANEL_OUT/selftest.txt"
"$PY_TRUTH" truth.py "$PROCS"
"$PY_TRUTH" classify.py "$PROCS"
"$PY_TRUTH" breakdown.py | tee "$PANEL_OUT/breakdown.md"
echo "tables: $PANEL_OUT/classify.md and breakdown.md  json: $PANEL_OUT/classify.json"
