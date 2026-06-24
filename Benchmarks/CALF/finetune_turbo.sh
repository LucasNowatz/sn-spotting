#!/usr/bin/env bash
# Fine-tune 17-class CALF_benchmark on turbo_result clips (Option B).
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="${DATASET_ROOT:-${REPO}/dataset_root/turbo_result}"
SPLITS="${SPLITS:-${DATASET}/splits.json}"
CALF_DIR="${REPO}/Benchmarks/CALF"
BENCHMARK_WEIGHTS="${CALF_DIR}/models/CALF_benchmark/model.pth.tar"

cd "${CALF_DIR}"

if [[ ! -f "${SPLITS}" ]]; then
  python "${REPO}/dataset_root/create_splits.py" --dataset_root "${DATASET}"
fi

if [[ ! -f "${BENCHMARK_WEIGHTS}" ]]; then
  echo "ERROR: Missing ${BENCHMARK_WEIGHTS}" >&2
  echo "Copy CALF_benchmark/model.pth.tar into Benchmarks/CALF/models/CALF_benchmark/" >&2
  exit 1
fi

conda activate CALF-pytorch

python src/main.py \
  --SoccerNet_path="${DATASET}" \
  --custom_dataset \
  --class_set=v2_17 \
  --splits_path="${SPLITS}" \
  --load_weights="${BENCHMARK_WEIGHTS}" \
  --num_features=512 \
  --model_name=CALF_turbo_finetune \
  --chunk_size=120 \
  --receptive_field=40 \
  --chunks_per_epoch=2000 \
  --batch_size=8 \
  --LR=1e-4 \
  --max_epochs=50 \
  --evaluation_frequency=5 \
  --GPU=0 \
  "$@"
