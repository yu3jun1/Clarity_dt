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

mkdir -p "runs/${VARIANT}_seed${SEED}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$REPO_ROOT/src"

conda run --no-capture-output -n py310 env PYTHONPATH="$REPO_ROOT/src" python -m clarity_rrt.train \
  --config configs/experiment.yaml \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --device cuda:0 \
  2>&1 | tee "runs/${VARIANT}_seed${SEED}/train.log"

conda run --no-capture-output -n py310 env PYTHONPATH="$REPO_ROOT/src" python -m clarity_rrt.evaluate run \
  --config configs/experiment.yaml \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --device cuda:0 \
  2>&1 | tee "runs/${VARIANT}_seed${SEED}/evaluate.log"
