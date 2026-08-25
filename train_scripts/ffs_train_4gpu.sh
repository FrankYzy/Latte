#!/usr/bin/env bash
set -euo pipefail

if ! command -v torchrun >/dev/null 2>&1; then
    echo "Error: torchrun was not found. Run 'uv sync', then invoke this script with 'uv run bash'." >&2
    exit 127
fi

LATTE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LATTE_REPO_ROOT="$(cd -- "${LATTE_SCRIPT_DIR}/.." && pwd)"

# Override this when the desired four GPUs are not 0,1,2,3, for example:
# LATTE_CUDA_DEVICES=4,5,6,7 uv run bash train_scripts/ffs_train_4gpu.sh
export CUDA_VISIBLE_DEVICES="${LATTE_CUDA_DEVICES:-0,1,2,3}"

cd "${LATTE_REPO_ROOT}"

exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
    --config ./configs/ffs/ffs_train.yaml
