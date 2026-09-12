#!/usr/bin/env bash
# Run the sealed verifier on a submission directory outside Docker.
#   run_gates.sh <submission_dir> [pytest args]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB="$1"; shift
cd "$ROOT/tests"
GEA_SUBMISSION="$SUB" python -m pytest test_science.py -p no:cacheprovider -q --tb=line "$@"
