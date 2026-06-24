#!/usr/bin/env bash
# Batch 8-class CALF predictions for turbo_result clips.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO}/Benchmarks/CALF"

DATASET_ROOT="${DATASET_ROOT:-${REPO}/dataset_root/turbo_result}"
MODEL="${MODEL:-${REPO}/Benchmarks/CALF/models/CALF_turbo_8class/model.pth.tar}"

conda activate CALF-pytorch

python predict_turbo_batch.py \
  --dataset_root "${DATASET_ROOT}" \
  --weights "${MODEL}" \
  --class_set turbo_8 \
  --num_classes 8 \
  --GPU 0 \
  --chunk_size 10 \
  --receptive_field 4 \
  "$@"
