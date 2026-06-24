#!/usr/bin/env bash
# Run Foul + Ball out of play CALF inference on turbo_result clips.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="${DATASET_ROOT:-${REPO}/dataset_root/turbo_result}"
MODEL="${MODEL:-${REPO}/Benchmarks/CALF/models/CALF_turbo_foul_ball/model.pth.tar}"

python "${REPO}/Benchmarks/CALF/predict_turbo_batch.py" \
  --dataset_root "${DATASET}" \
  --weights "${MODEL}" \
  --class_set foul_ball_2 \
  --num_classes 2 \
  --chunk_size 20 \
  --receptive_field 6 \
  "$@"
