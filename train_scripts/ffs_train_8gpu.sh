#!/usr/bin/env bash
set -euo pipefail

if ! command -v torchrun >/dev/null 2>&1; then
    echo "Error: torchrun was not found. Run 'uv sync', then invoke this script with 'uv run bash'." >&2
    exit 127
fi

LATTE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LATTE_REPO_ROOT="$(cd -- "${LATTE_SCRIPT_DIR}/.." && pwd)"

# Matches the official single-node 8-GPU setup in slurm_scripts/ffs.slurm.
# Override this when the desired eight GPUs are not 0-7, for example:
# LATTE_CUDA_DEVICES=8,9,10,11,12,13,14,15 uv run bash train_scripts/ffs_train_8gpu.sh
export CUDA_VISIBLE_DEVICES="${LATTE_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"

# Override the training config with LATTE_CONFIG, for example LATTE_CONFIG=./configs/ffs/ffs_train_10k.yaml.
cd "${LATTE_REPO_ROOT}"

exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
    --config "${LATTE_CONFIG:-./configs/ffs/ffs_train.yaml}"
