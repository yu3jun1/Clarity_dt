#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEED="${1:-42}"

echo "[single-seed] seed=$SEED start=$(date -u +%FT%TZ)"
pids=()
bash "$ROOT/scripts/run_one.sh" A "$SEED" 7 & pids+=("$!")
bash "$ROOT/scripts/run_one.sh" B "$SEED" 5 & pids+=("$!")
bash "$ROOT/scripts/run_one.sh" C "$SEED" 6 & pids+=("$!")
bash "$ROOT/scripts/run_one.sh" D "$SEED" 4 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[single-seed] failed" >&2
  exit 1
fi

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt.evaluate aggregate \
  --config "$ROOT/configs/stagewise_recursive.yaml"
echo "[single-seed] seed=$SEED complete=$(date -u +%FT%TZ)"
