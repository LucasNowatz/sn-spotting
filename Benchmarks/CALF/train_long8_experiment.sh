#!/usr/bin/env bash
# Long-8 foul_ball experiment: C (120/40 scratch) vs D (120/40 + CALF backbone_seg).
# Z = zero-shot full CALF_benchmark (v2_17, test_only).
# Split: 6 train / 1 valid / 1 test, max_epochs=31.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="${DATASET_ROOT:-${REPO}/dataset_root/turbo_result}"
SPLITS="${SPLITS:-${REPO}/dataset_root/splits_long8_foul_ball.json}"
CALF_DIR="${REPO}/Benchmarks/CALF"
BENCHMARK_WEIGHTS="${BENCHMARK_WEIGHTS:-${CALF_DIR}/models/CALF_benchmark/model.pth.tar}"

RUN="${1:-both}"
EXTRA_ARGS=("${@:2}")

if [[ "${RUN}" != "C" && "${RUN}" != "D" && "${RUN}" != "Z" && "${RUN}" != "both" ]]; then
  echo "Usage: $0 [C|D|Z|both] [extra main.py args...]" >&2
  exit 1
fi

if [[ ! -f "${SPLITS}" ]]; then
  echo "ERROR: Missing ${SPLITS}" >&2
  exit 1
fi

cd "${CALF_DIR}"
conda activate CALF-pytorch

run_train() {
  local model_name="$1"
  shift
  python src/main.py \
    --SoccerNet_path="${DATASET}" \
    --custom_dataset \
    --class_set=foul_ball_2 \
    --splits_path="${SPLITS}" \
    --num_features=512 \
    --model_name="${model_name}" \
    --chunk_size=120 \
    --receptive_field=40 \
    --chunks_per_epoch=2000 \
    --background_weight=5 \
    --event_class_weights=4,1 \
    --batch_size=4 \
    --LR=1e-4 \
    --max_epochs=31 \
    --evaluation_frequency=10 \
    --GPU=0 \
    "$@" \
    "${EXTRA_ARGS[@]}"
}

if [[ "${RUN}" == "C" || "${RUN}" == "both" ]]; then
  echo "=== Run C: 120/40 scratch (long-8) ==="
  run_train CALF_long8_C
fi

if [[ "${RUN}" == "D" || "${RUN}" == "both" ]]; then
  if [[ ! -f "${BENCHMARK_WEIGHTS}" ]]; then
    echo "ERROR: Missing ${BENCHMARK_WEIGHTS}" >&2
    exit 1
  fi
  echo "=== Run D: 120/40 + backbone_seg (long-8) ==="
  run_train CALF_long8_D \
    --load_weights="${BENCHMARK_WEIGHTS}" \
    --load_mode=backbone_seg
fi

if [[ "${RUN}" == "Z" ]]; then
  if [[ ! -f "${BENCHMARK_WEIGHTS}" ]]; then
    echo "ERROR: Missing ${BENCHMARK_WEIGHTS}" >&2
    exit 1
  fi
  echo "=== Run Z: zero-shot CALF_benchmark (v2_17, test clip only) ==="
  python src/main.py \
    --SoccerNet_path="${DATASET}" \
    --custom_dataset \
    --class_set=v2_17 \
    --splits_path="${SPLITS}" \
    --num_features=512 \
    --model_name=CALF_benchmark \
    --test_only \
    --chunk_size=120 \
    --receptive_field=40 \
    --GPU=0 \
    "${EXTRA_ARGS[@]}"
fi
