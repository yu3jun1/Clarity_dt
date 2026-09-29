#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

wait_for_workers() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

run_primary_worker() {
  local gpu="$1"
  shift
  local task variant seed
  for task in "$@"; do
    variant="${task%%:*}"
    seed="${task##*:}"
    bash "$ROOT/scripts/run_one.sh" "$variant" "$seed" "$gpu"
  done
}

echo "[pipeline] phase=primary start=$(date -u +%FT%TZ)"
primary_pids=()
run_primary_worker 4 A:43 C:43 & primary_pids+=("$!")
run_primary_worker 5 B:43 D:43 & primary_pids+=("$!")
run_primary_worker 6 A:44 C:44 & primary_pids+=("$!")
run_primary_worker 7 B:44 D:44 & primary_pids+=("$!")

if ! wait_for_workers "${primary_pids[@]}"; then
  echo "[pipeline] primary phase failed; ablation will not start" >&2
  exit 1
fi

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" python -m clarity_rrt.evaluate aggregate \
  --config "$ROOT/configs/stagewise_recursive.yaml"
echo "[pipeline] phase=primary complete=$(date -u +%FT%TZ)"

run_ablation_worker() {
  local gpu="$1"
  local variant="$2"
  local worker="$3"
  local index=0
  local weight seed
  for weight in 0 0.01 0.05 0.1; do
    for seed in 42 43 44; do
      if (( index % 4 == worker )); then
        bash "$ROOT/scripts/run_ablation_one.sh" \
          "$variant" "$weight" "$seed" "$gpu"
      fi
      index=$((index + 1))
    done
  done
}

echo "[pipeline] phase=ablation start=$(date -u +%FT%TZ)"
ablation_pids=()
for worker in 0 1 2 3; do
  gpu=$((worker + 4))
  run_ablation_worker "$gpu" B "$worker" &
  ablation_pids+=("$!")
done
for worker in 0 1 2 3; do
  gpu=$((worker + 4))
  run_ablation_worker "$gpu" D "$worker" &
  ablation_pids+=("$!")
done

if ! wait_for_workers "${ablation_pids[@]}"; then
  echo "[pipeline] ablation phase failed" >&2
  exit 1
fi

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" python -m clarity_rrt.evaluate aggregate \
  --config "$ROOT/configs/stagewise_recursive.yaml"
echo "[pipeline] complete=$(date -u +%FT%TZ)"
