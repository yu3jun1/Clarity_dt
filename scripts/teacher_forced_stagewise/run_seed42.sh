#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/configs/ablations/teacher_forced_stagewise.yaml"
cd "$ROOT"

echo "[tf-stagewise] seed=42 start=$(date -u +%FT%TZ)"
bash "$ROOT/scripts/teacher_forced_stagewise/run_one.sh" 42 "$GPU_ID"
conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt_v3.evaluate aggregate --config "$CONFIG"
echo "[tf-stagewise] seed=42 complete=$(date -u +%FT%TZ)"
