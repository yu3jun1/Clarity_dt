#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-7]$ ]]; then
  echo "Usage: $0 <idle-gpu-id 0-7>" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AUDIT_ROOT="$ROOT/outputs/reproducibility/seed42_same_gpu"
mkdir -p "$AUDIT_ROOT"
exec conda run --no-capture-output -n py310 env PYTHONPATH="$ROOT/src" \
  python -m clarity_rrt_v3.replicate_audit pipeline --gpu "$1" --output-root "$AUDIT_ROOT" \
  >> "$AUDIT_ROOT/pipeline.log" 2>&1
