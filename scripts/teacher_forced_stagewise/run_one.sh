#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <seed> <gpu-id>" >&2
  exit 2
fi

SEED="$1"
GPU_ID="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/configs/ablations/teacher_forced_stagewise.yaml"
RUN_DIR="$ROOT/outputs/ablations/teacher_forced_stagewise/primary/E_seed${SEED}"

case "$GPU_ID" in
  0|1|2|3|4|5|6|7) ;;
  *) echo "GPU ID must be between 0 and 7" >&2; exit 2 ;;
esac

cd "$ROOT"
mkdir -p "$RUN_DIR"
LOCK_FILE="$ROOT/outputs/ablations/teacher_forced_stagewise/E_seed${SEED}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[tf-stagewise] E seed=${SEED} is already running; skipping duplicate"
  exit 0
fi
if [[ -f "$RUN_DIR/status.log" ]] && grep -q 'complete=' "$RUN_DIR/status.log"; then
  echo "[tf-stagewise] E seed=${SEED} is already complete; skipping duplicate"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
printf '[tf-stagewise] variant=E seed=%s gpu=%s train_start=%s\n' \
  "$SEED" "$GPU_ID" "$(date -u +%FT%TZ)" > "$RUN_DIR/status.log"

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt_v3.train --config "$CONFIG" \
  --variant E --seed "$SEED" --device cuda:0 2>&1 | tee "$RUN_DIR/train.log"

printf '[tf-stagewise] variant=E seed=%s evaluate_start=%s\n' \
  "$SEED" "$(date -u +%FT%TZ)" >> "$RUN_DIR/status.log"
conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt_v3.evaluate run --config "$CONFIG" \
  --variant E --seed "$SEED" --device cuda:0 2>&1 | tee "$RUN_DIR/evaluate.log"
printf '[tf-stagewise] variant=E seed=%s complete=%s\n' \
  "$SEED" "$(date -u +%FT%TZ)" >> "$RUN_DIR/status.log"
