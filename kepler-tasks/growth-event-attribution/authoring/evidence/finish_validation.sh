#!/usr/bin/env bash
# Run the sealed verifier on the oracle (three times), on an empty submission
# and on every baseline, four at a time, and record the outcome in runs.txt.
#   finish_validation.sh <oracle_dir> <baseline_root> <python>
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ORACLE="$1"; BASE="$2"; PY="${3:-python}"
OUT="$HERE/verifier_runs"
rm -rf "$OUT"; mkdir -p "$OUT" /tmp/nop_empty
run_one() {
    name="$1"; sub="$2"
    ( cd "$ROOT/tests" && GEA_SUBMISSION="$sub" "$PY" -m pytest test_science.py \
        -p no:cacheprovider -q --tb=line > "$OUT/$name.log" 2>&1 )
    tail -1 "$OUT/$name.log" | sed "s/^/$name: /"
}
export -f run_one; export ROOT PY OUT
{
  echo "oracle_1 $ORACLE"; echo "oracle_2 $ORACLE"; echo "oracle_3 $ORACLE"
  echo "nop /tmp/nop_empty"
  for d in "$BASE"/*/; do echo "$(basename "$d") $d"; done
} | xargs -P 4 -L 1 bash -c 'run_one "$0" "$1"' | tee "$HERE/runs.txt"
