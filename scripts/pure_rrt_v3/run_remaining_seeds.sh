#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

run_worker() {
  local gpu="$1"
  shift
  local task variant seed
  for task in "$@"; do
    variant="${task%%:*}"
    seed="${task##*:}"
    bash "$ROOT/scripts/pure_rrt_v3/run_one.sh" "$variant" "$seed" "$gpu"
  done
}

echo "[pure-rrt-v3] remaining-seeds start=$(date -u +%FT%TZ)"
pids=()
run_worker 4 D:43 D:44 & pids+=("$!")
run_worker 5 B:43 B:44 & pids+=("$!")
run_worker 6 C:43 C:44 & pids+=("$!")
run_worker 7 A:43 A:44 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[pure-rrt-v3] remaining-seeds failed" >&2
  exit 1
fi

conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt_v3.evaluate aggregate \
  --config "$ROOT/configs/pure_rrt_v3.yaml"

echo "[pure-rrt-v3] remaining-seeds complete=$(date -u +%FT%TZ)"
