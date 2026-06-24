#!/usr/bin/env bash
# Batch 17-class CALF predictions for turbo_result clips.
# Feature extraction uses SoccerNet-FeatureExtraction (TF2 + opencv).
# CALF inference uses CALF-pytorch (PyTorch + CUDA).
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO}/Benchmarks/CALF"

DATASET_ROOT="${DATASET_ROOT:-${REPO}/dataset_root/turbo_result}"
if [[ ! -d "${DATASET_ROOT}" && -d "/root/workspace/spider/sn-spotting/dataset_root/turbo_result" ]]; then
  DATASET_ROOT="/root/workspace/spider/sn-spotting/dataset_root/turbo_result"
fi

EXTRACT_ONLY=false
INFER_ONLY=false
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --features_only) EXTRACT_ONLY=true ;;
    --no-extract) INFER_ONLY=true ;;
    *) EXTRA_ARGS+=("${arg}") ;;
  esac
done

if [[ "${INFER_ONLY}" == false ]]; then
  conda activate SoccerNet-FeatureExtraction
  echo "Extracting features under ${DATASET_ROOT} (CPU, ~1 min/clip)..."
  python "${REPO}/Features/batch_extract_dataset.py" \
    --dataset_root "${DATASET_ROOT}" \
    --gpu -1 \
    "${EXTRA_ARGS[@]}"
elif ! find "${DATASET_ROOT}" -name "1_ResNET_TF2_PCA512.npy" -print -quit | grep -q .; then
  echo "ERROR: --no-extract set but no feature files found under ${DATASET_ROOT}" >&2
  echo "Run ./predict_turbo.sh without --no-extract first." >&2
  exit 1
fi

if [[ "${EXTRACT_ONLY}" == true ]]; then
  exit 0
fi

conda activate CALF-pytorch
python predict_turbo_batch.py \
  --dataset_root "${DATASET_ROOT}" \
  --GPU 0 \
  --chunk_size 120 \
  --receptive_field 40 \
  "${EXTRA_ARGS[@]}"
