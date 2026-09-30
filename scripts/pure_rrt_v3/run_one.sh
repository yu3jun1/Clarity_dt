#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <A|B|C|D> <seed> <gpu-id>" >&2
  exit 2
fi

VARIANT="$1"
SEED="$2"
GPU_ID="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$ROOT/outputs/pure_rrt_v3_step2400/primary/${VARIANT}_seed${SEED}"

case "$VARIANT" in
  A|B|C|D) ;;
  *) echo "Variant must be A, B, C, or D" >&2; exit 2 ;;
esac
case "$GPU_ID" in
  4|5|6|7) ;;
  *) echo "GPU ID must be one of 4, 5, 6, 7" >&2; exit 2 ;;
esac

cd "$ROOT"
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" python -m clarity_rrt_v3.train \
  --config "$ROOT/configs/pure_rrt_v3.yaml" \
  --variant "$VARIANT" --seed "$SEED" --device cuda:0 \
  2>&1 | tee "$RUN_DIR/train.log"

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" python -m clarity_rrt_v3.evaluate run \
  --config "$ROOT/configs/pure_rrt_v3.yaml" \
  --variant "$VARIANT" --seed "$SEED" --device cuda:0 \
  2>&1 | tee "$RUN_DIR/evaluate.log"
