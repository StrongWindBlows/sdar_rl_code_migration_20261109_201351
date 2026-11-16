#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
export PATH="/ssd_data/liuyang/miniconda3/envs/dirl/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export HF_HOME="${SDAR_CACHE_DIR:-$ROOT/../../.cache/huggingface}"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_MODULES_CACHE"
exec torchrun --standalone --nproc_per_node=4 "$HERE/verify_parallel_long_gpu.py" \
    --model "${MODEL:-/ssd_data/liuyang/hug/model/SDAR-4B-Chat-b32}" \
    --output "${OUTPUT:-$ROOT/verification/long_tp2_dp2_3000_20261109}" \
    --length "${LONG_TOKENS:-3000}" --block-size 32 "$@"
