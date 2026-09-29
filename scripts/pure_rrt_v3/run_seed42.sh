#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "[pure-rrt-v3] seed=42 start=$(date -u +%FT%TZ)"
pids=()
bash "$ROOT/scripts/pure_rrt_v3/run_one.sh" D 42 4 & pids+=("$!")
bash "$ROOT/scripts/pure_rrt_v3/run_one.sh" B 42 5 & pids+=("$!")
bash "$ROOT/scripts/pure_rrt_v3/run_one.sh" C 42 6 & pids+=("$!")
bash "$ROOT/scripts/pure_rrt_v3/run_one.sh" A 42 7 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[pure-rrt-v3] seed=42 failed" >&2
  exit 1
fi

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt_v3.evaluate aggregate \
  --config "$ROOT/configs/pure_rrt_v3.yaml"

echo "[pure-rrt-v3] seed=42 complete=$(date -u +%FT%TZ)"
