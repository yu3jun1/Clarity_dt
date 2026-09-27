#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the high-memory GPU pool requested for this experiment. Override with
# GPU_IDS="..." only when deliberately changing the allocation.
read -r -a GPU_LIST <<< "${GPU_IDS:-4 5 6 7}"
GPU_COUNT="${#GPU_LIST[@]}"
WORKER_COUNT="${MAX_JOBS:-$GPU_COUNT}"
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
if (( WORKER_COUNT > GPU_COUNT )); then
  WORKER_COUNT="$GPU_COUNT"
fi
if (( WORKER_COUNT < 1 )); then
  echo "MAX_JOBS must be at least one" >&2
  exit 2
fi

TASKS=()
for variant in A B C D; do
  for seed in 42 43 44; do
    TASKS+=("${variant}:${seed}")
  done
done

wait_for_gpu_memory() {
  local gpu="$1"
  local free_mib
  while true; do
    free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      echo "[scheduler] GPU $gpu ready: ${free_mib} MiB free"
      return 0
    fi
    echo "[scheduler] GPU $gpu has ${free_mib:-unknown} MiB free; waiting for ${MIN_FREE_MIB} MiB"
    sleep 60
  done
}

run_worker() {
  local worker_index="$1"
  local gpu="${GPU_LIST[$worker_index]}"
  local task_index spec variant seed
  for ((task_index=worker_index; task_index<${#TASKS[@]}; task_index+=WORKER_COUNT)); do
    spec="${TASKS[$task_index]}"
    variant="${spec%%:*}"
    seed="${spec##*:}"
    wait_for_gpu_memory "$gpu"
    echo "[scheduler] starting ${variant}_seed${seed} on physical GPU $gpu"
    bash scripts/run_one.sh "$variant" "$seed" "$gpu"
  done
}

worker_pids=()
for ((worker_index=0; worker_index<WORKER_COUNT; worker_index++)); do
  run_worker "$worker_index" &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one experiment worker failed; skipping aggregate." >&2
  exit "$status"
fi

export PYTHONPATH="$REPO_ROOT/src"
conda run --no-capture-output -n py310 env PYTHONPATH="$REPO_ROOT/src" python -m clarity_rrt.evaluate aggregate \
  --config configs/experiment.yaml
