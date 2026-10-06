#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$1"
shift
if (( $# == 0 )); then
    set -- campaign
fi
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tanyuejun/miniconda3/envs/py310/bin/python -m clarity_ensemble_study "$@"
