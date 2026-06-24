#!/usr/bin/env bash
# Train Foul + Ball out of play CALF (2 event classes + background) on turbo_result.
# Trains from scratch (no CALF_benchmark checkpoint).
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="${DATASET_ROOT:-${REPO}/dataset_root/turbo_result}"
SPLITS="${SPLITS:-${DATASET}/splits_foul_ball.json}"
CALF_DIR="${REPO}/Benchmarks/CALF"

cd "${CALF_DIR}"

python "${REPO}/dataset_root/create_splits.py" \
  --dataset_root "${DATASET}" \
  --class_set foul_ball_2 \
  --require-features \
  --no-require-annotations \
  --output "${SPLITS}"

conda activate CALF-pytorch

python src/main.py \
  --SoccerNet_path="${DATASET}" \
  --custom_dataset \
  --class_set=foul_ball_2 \
  --splits_path="${SPLITS}" \
  --num_features=512 \
  --model_name=CALF_turbo_foul_ball \
  --chunk_size=20 \
  --receptive_field=6 \
  --chunks_per_epoch=2000 \
  --background_weight=5 \
  --event_class_weights=4,1 \
  --batch_size=4 \
  --LR=5e-4 \
  --max_epochs=31 \
  --evaluation_frequency=10 \
  --GPU=0 \
  "$@"
