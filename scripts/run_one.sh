#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <A|B|C|D> <seed> [gpu-id]" >&2
  exit 2
fi

VARIANT="$1"
SEED="$2"
GPU_ID="${3:-4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${VARIANT}_seed${SEED}"
RUN_DIR="runs/${RUN_NAME}"
mkdir -p runs/.locks
exec 9>"runs/.locks/${RUN_NAME}.lock"
flock 9
if [[ -f "${RUN_DIR}/metrics.json" ]]; then
  echo "[scheduler] ${RUN_NAME} is already complete; skipping duplicate run"
  exit 0
fi
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$REPO_ROOT/src"

conda run --no-capture-output -n py310 env PYTHONPATH="$REPO_ROOT/src" python -m clarity_rrt.train \
  --config configs/experiment.yaml \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --device cuda:0 \
  2>&1 | tee "${RUN_DIR}/train.log"

conda run --no-capture-output -n py310 env PYTHONPATH="$REPO_ROOT/src" python -m clarity_rrt.evaluate run \
  --config configs/experiment.yaml \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --device cuda:0 \
  2>&1 | tee "${RUN_DIR}/evaluate.log"
