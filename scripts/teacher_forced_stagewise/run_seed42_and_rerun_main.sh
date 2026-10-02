#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

run_a_then_e() {
  bash "$ROOT/scripts/pure_rrt_v3/run_one.sh" A 42 3
  bash "$ROOT/scripts/teacher_forced_stagewise/run_one.sh" 42 3
}

echo "[seed42-rerun+tf] start=$(date -u +%FT%TZ)"
pids=()
bash "$ROOT/scripts/pure_rrt_v3/run_one.sh" D 42 0 & pids+=("$!")
bash "$ROOT/scripts/pure_rrt_v3/run_one.sh" B 42 1 & pids+=("$!")
bash "$ROOT/scripts/pure_rrt_v3/run_one.sh" C 42 2 & pids+=("$!")
run_a_then_e & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[seed42-rerun+tf] failed" >&2
  exit 1
fi

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt_v3.evaluate aggregate \
  --config "$ROOT/configs/pure_rrt_v3.yaml"
conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt_v3.evaluate aggregate \
  --config "$ROOT/configs/ablations/teacher_forced_stagewise.yaml"

echo "[seed42-rerun+tf] complete=$(date -u +%FT%TZ)"
