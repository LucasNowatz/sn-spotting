#!/usr/bin/env bash
# Offline verifier entry point.  Always leaves a binary reward behind, including
# on a crash, a missing artifact or a pytest collection error.
set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

cd "$(dirname "$0")"
python3 -m pytest test_ozone_budget.py -v \
    --ctrf /logs/verifier/ctrf-report.json \
    > /logs/verifier/pytest.log 2>&1
status=$?

if [ "$status" -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi

cat /logs/verifier/pytest.log
exit 0
