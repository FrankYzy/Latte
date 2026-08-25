#!/usr/bin/env bash
set -euo pipefail

.venv/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
    --config configs/ffs/ffs_train.yaml
