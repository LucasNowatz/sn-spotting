#!/usr/bin/env bash
# Reference solution: Bayesian inversion, counterfactual attribution,
# restricted sensitivity analysis, withheld predictions and region shares.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /app/output /app/work
cp -r "$HERE/src/." /app/work/
python /app/work/pipeline.py --data /app/data --out /app/output
