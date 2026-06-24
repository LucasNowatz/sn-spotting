#!/usr/bin/env bash
# Train Foul-only CALF (1 event class + background) on turbo_result.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="${DATASET_ROOT:-${REPO}/dataset_root/turbo_result}"
SPLITS="${SPLITS:-${DATASET}/splits_foul.json}"
CALF_DIR="${REPO}/Benchmarks/CALF"

cd "${CALF_DIR}"

python "${REPO}/dataset_root/create_splits.py" \
  --dataset_root "${DATASET}" \
  --class_set foul_1 \
  --require-features \
  --no-require-annotations \
  --output "${SPLITS}"

conda activate CALF-pytorch

python src/main.py \
  --SoccerNet_path="${DATASET}" \
  --custom_dataset \
  --class_set=foul_1 \
  --splits_path="${SPLITS}" \
  --num_features=512 \
  --model_name=CALF_turbo_foul \
  --chunk_size=10 \
  --receptive_field=4 \
  --chunks_per_epoch=2000 \
  --background_weight=5 \
  --batch_size=16 \
  --LR=1e-3 \
  --max_epochs=200 \
  --evaluation_frequency=10 \
  --GPU=0 \
  "$@"
