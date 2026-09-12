#!/usr/bin/env bash
# Reference solution: staged Bayesian inversion, counterfactual attribution,
# restricted sensitivity analysis, withheld predictions and lineage.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /app/output /app/work
cp -r "$HERE/src/." /app/work/
chmod +x /app/work/run.py

python /app/work/run.py --input-dir /app/data --output-dir /app/output
