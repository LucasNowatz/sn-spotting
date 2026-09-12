#!/usr/bin/env bash
# Run the sealed verifier on one submission directory outside Docker.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../../tests"
EA_SUBMISSION="$1" python -m pytest test_verify.py -p no:cacheprovider -q --tb=line
