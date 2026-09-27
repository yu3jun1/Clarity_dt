#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPU_LIST <<< "${GPU_IDS:-0}"
GPU_COUNT="${#GPU_LIST[@]}"
MAX_JOBS="${MAX_JOBS:-$GPU_COUNT}"
job_index=0

for variant in A B C D; do
  for seed in 42 43 44; do
    while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
      wait -n
    done
    gpu="${GPU_LIST[$((job_index % GPU_COUNT))]}"
    bash scripts/run_one.sh "$variant" "$seed" "$gpu" &
    job_index=$((job_index + 1))
  done
done
wait

export PYTHONPATH="$REPO_ROOT/src"
conda run -n py310 env PYTHONPATH="$REPO_ROOT/src" python -m clarity_rrt.evaluate aggregate \
  --config configs/experiment.yaml
