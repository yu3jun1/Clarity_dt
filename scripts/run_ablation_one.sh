#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <B|D> <rrt-weight> <seed> [gpu-id]" >&2
  exit 2
fi

VARIANT="$1"
WEIGHT="$2"
SEED="$3"
GPU_ID="${4:-4}"
LABEL="${WEIGHT/./p}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_DIR="$ROOT/outputs/stagewise_recursive/ablation/${VARIANT}_lambda${LABEL}_seed${SEED}"

case "$GPU_ID" in
  4|5|6|7) ;;
  *) echo "GPU ID must be one of 4, 5, 6, 7" >&2; exit 2 ;;
esac

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" python -m clarity_rrt.train \
  --config "$ROOT/configs/stagewise_recursive.yaml" \
  --variant "$VARIANT" --seed "$SEED" --device cuda:0 \
  --rrt-weight "$WEIGHT" 2>&1 | tee "$RUN_DIR/train.log"

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" python -m clarity_rrt.evaluate run \
  --config "$ROOT/configs/stagewise_recursive.yaml" \
  --variant "$VARIANT" --seed "$SEED" --device cuda:0 \
  --rrt-weight "$WEIGHT" 2>&1 | tee "$RUN_DIR/evaluate.log"
