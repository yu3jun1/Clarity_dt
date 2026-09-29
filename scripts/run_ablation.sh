#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEED="${1:-42}"

echo "[ablation] seed=$SEED start=$(date -u +%FT%TZ)"

run_worker() {
  local gpu="$1"
  local weight="$2"
  bash "$ROOT/scripts/run_ablation_one.sh" B "$weight" "$SEED" "$gpu"
  bash "$ROOT/scripts/run_ablation_one.sh" D "$weight" "$SEED" "$gpu"
}

pids=()
run_worker 4 0 & pids+=("$!")
run_worker 5 0.01 & pids+=("$!")
run_worker 6 0.05 & pids+=("$!")
run_worker 7 0.1 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[ablation] failed" >&2
  exit 1
fi

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt.evaluate aggregate \
  --config "$ROOT/configs/stagewise_recursive.yaml"
echo "[ablation] seed=$SEED complete=$(date -u +%FT%TZ)"
