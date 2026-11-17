#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# 模型与训练数据。
export MODEL="${MODEL:-$ROOT/../hug/model/SDAR-4B-Chat-b32}"
export DATA="${DATA:-$ROOT/../hug/data/JustGRPO-MATH-500/train.jsonl}"

# 训练配置：每轮将全部回答作为一个 minibatch，默认每轮更新一次。
export OBJECTIVE="${OBJECTIVE:-grpo}"  # gmpo、grpo 或 coupled
export COUPLING_BETA="${COUPLING_BETA:-1e-4}"  # coupled 的完整回答 KL 系数
export LR="${LR:-5e-6}"
export STEPS="${STEPS:-5}"
export BATCH_PROMPTS="${BATCH_PROMPTS:-128}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((BATCH_PROMPTS * GROUP_SIZE))}"
export PPO_EPOCHS="${PPO_EPOCHS:-1}"
# 每次前向合并的 phase 数，共享一份 clean；1 为逐 phase。
export TRAIN_PHASE_BATCH_SIZE="${TRAIN_PHASE_BATCH_SIZE:-4}"
export TRAIN_LOGIT_CHUNK_SIZE="${TRAIN_LOGIT_CHUNK_SIZE:-128}"
export CLIP="${CLIP:-0.4}"  # log-ratio clipping
export SEED="${SEED:-1234}"

# 生成配置：每个 TP 推理副本的并发为 ROLLOUT_BATCH_SIZE。
export MAX_TOKENS="${MAX_TOKENS:-3000}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export BLOCK_SIZE="${BLOCK_SIZE:-32}"
export TEMPERATURE="${TEMPERATURE:-1.0}"

# 训练采样与评估共用的推理设置。
export INFERENCE_CACHE_FRACTION="${INFERENCE_CACHE_FRACTION:-0.8}"
export INFERENCE_CUDA_GRAPH="${INFERENCE_CUDA_GRAPH:-true}"
# 当前 SDAR 的前缀缓存复用未通过输出一致性检查。
export INFERENCE_PREFIX_CACHING="${INFERENCE_PREFIX_CACHING:-false}"
export INFERENCE_MAX_PREFILL_TOKENS="${INFERENCE_MAX_PREFILL_TOKENS:-4096}"
export INFERENCE_PROGRESS_TIMEOUT="${INFERENCE_PROGRESS_TIMEOUT:-600}"
export INFERENCE_RETRIES="${INFERENCE_RETRIES:-1}"
export TRAIN_TF32="${TRAIN_TF32:-true}"

# 输出与恢复：RESUME 留空时，新目录开始训练，已有实验自动恢复。
export RUN_TAG="${RUN_TAG:-sdar4b-just${OBJECTIVE}-math-rbs${ROLLOUT_BATCH_SIZE}}"
export OUTPUT="${OUTPUT:-$ROOT/sdar_justgmpo/runs/$RUN_TAG}"
export SAVE_EVERY="${SAVE_EVERY:-5}"
export RESUME="${RESUME:-}"

# 评估时保留该轮 BF16 权重；0 表示跟随 SAVE_EVERY。
export EVAL_EVERY="${EVAL_EVERY:-1}"
export EVAL_AT_START="${EVAL_AT_START:-true}"
export EVAL_DATA="${EVAL_DATA:-$ROOT/../hug/data/JustGRPO-MATH-500/test.jsonl}"
export EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-3000}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export EVAL_SEED="${EVAL_SEED:-113}"
# 固定基座回答前缀的熵探针，SIZE 为每个 DP 副本的题数；0 表示关闭。
export ENTROPY_PROBE_SIZE="${ENTROPY_PROBE_SIZE:-4}"
export ENTROPY_PROBE_TOKENS="${ENTROPY_PROBE_TOKENS:-128}"

# 四卡并行与运行环境。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#DEVICES[@]} -ne 4 ]]; then
    printf '%s\n' 'TP=2 / DP=2 requires exactly four visible GPUs.' >&2
    exit 1
fi
export TP_SIZE=2 NPROC_PER_NODE=4
export HF_HOME="${SDAR_CACHE_DIR:-$ROOT/../.cache/huggingface}"
export HF_MODULES_CACHE="$HF_HOME/modules"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
# The current PCIe host passed NCCL through shared memory; direct P2P stalled.
# Set NCCL_P2P_DISABLE=0 only after validating that transport on the target host.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
mkdir -p "$HF_MODULES_CACHE"
exec bash "$ROOT/run_train_portable.sh" "$@" --dp-size 2
