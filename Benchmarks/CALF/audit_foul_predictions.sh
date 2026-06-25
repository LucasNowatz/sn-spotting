#!/usr/bin/env bash
# Audit foul predictions on valid + test; whole-clip analysis for 006.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="${DATASET_ROOT:-${REPO}/dataset_root/turbo_result}"
SPLITS="${SPLITS:-${DATASET}/splits_foul_ball.json}"
MODEL="${MODEL:-${REPO}/Benchmarks/CALF/models/CALF_turbo_foul_ball/model.pth.tar}"
OUT="${OUT:-${REPO}/Benchmarks/CALF/outputs/audit}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CALF-pytorch

python "${REPO}/Benchmarks/CALF/audit_foul_predictions.py" \
  --dataset_root "${DATASET}" \
  --splits_path "${SPLITS}" \
  --weights "${MODEL}" \
  --output_dir "${OUT}" \
  --chunk_size 20 \
  --receptive_field 6 \
  --GPU 0 \
  "$@"
