#!/usr/bin/env bash
# Sealed verifier entry point.  A reward is written on every path: it starts
# as 0 and becomes 1 only when pytest reports a clean pass.
set -u
mkdir -p /logs/verifier
printf '0' > /logs/verifier/reward.txt
cd /tests
python -m pytest test_verify.py -p no:cacheprovider -q --tb=short --ctrf /logs/verifier/ctrf.json
rc=$?
if [ "$rc" -eq 0 ]; then printf '1' > /logs/verifier/reward.txt; else printf '0' > /logs/verifier/reward.txt; fi
if [ ! -s /logs/verifier/ctrf.json ]; then
  printf '{"results":{"tool":{"name":"pytest"},"summary":{"tests":0,"passed":0,"failed":1,"pending":0,"skipped":0,"other":0,"start":0,"stop":0},"tests":[{"name":"verifier","status":"failed","duration":0,"message":"pytest produced no report"}]}}' > /logs/verifier/ctrf.json
fi
echo "reward: $(cat /logs/verifier/reward.txt)"
exit 0
