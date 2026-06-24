#!/usr/bin/env bash
# Run Foul-only CALF inference on turbo_result clips.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="${DATASET_ROOT:-${REPO}/dataset_root/turbo_result}"
MODEL="${MODEL:-${REPO}/Benchmarks/CALF/models/CALF_turbo_foul/model.pth.tar}"

python "${REPO}/Benchmarks/CALF/predict_turbo_batch.py" \
  --dataset_root "${DATASET}" \
  --weights "${MODEL}" \
  --class_set foul_1 \
  --num_classes 1 \
  --chunk_size 10 \
  --receptive_field 4 \
  "$@"
