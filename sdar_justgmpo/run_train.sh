#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$SCRIPT_DIR"

# 激活含 PyTorch 的环境，或显式设置 TORCHRUN_BIN。
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"
if [[ -z "$TORCHRUN_BIN" ]]; then
    printf '%s\n' '请先激活包含 PyTorch 的环境，或设置 TORCHRUN_BIN。' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-1}"
MODEL="${MODEL:-$SCRIPT_DIR/../../hug/model/SDAR-4B-Chat-b32}"
DATA="${DATA:-$SCRIPT_DIR/../../hug/data/JustGRPO-MATH-500/train.jsonl}"

# GMPO 论文配方：128题 × 8回答 = 1024条；每128条更新一次，共8次。
# PPO_EPOCHS=1 表示遍历本轮轨迹一次，并非只更新一次参数。
BATCH_PROMPTS="${BATCH_PROMPTS:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-128}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
# 正常 EOS 停止；保留 EOS 的 log-prob。达到长度上限的未结束回答奖励为0。
MAX_TOKENS="${MAX_TOKENS:-3000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LR="${LR:-1e-6}"
# LOG-ratio clipping，0.4 对应 ratio [exp(-0.4), exp(0.4)]。
CLIP="${CLIP:-0.4}"
OBJECTIVE="${OBJECTIVE:-gmpo}"  # grpo 提供相同批内更新与clip范围的对照。
COUPLING_BETA="${COUPLING_BETA:-1e-4}"
SEED="${SEED:-1234}"
# STEPS 是 rollout 轮数。默认50轮 = 400次 optimizer 更新。
STEPS="${STEPS:-50}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-0}"
EVAL_AT_START="${EVAL_AT_START:-false}"
EVAL_DATA="${EVAL_DATA:-$SCRIPT_DIR/../../hug/data/JustGRPO-MATH-500/test.jsonl}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-3000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_SEED="${EVAL_SEED:-113}"
INFERENCE_CACHE_FRACTION="${INFERENCE_CACHE_FRACTION:-0.8}"
INFERENCE_CUDA_GRAPH="${INFERENCE_CUDA_GRAPH:-true}"
INFERENCE_PREFIX_CACHING="${INFERENCE_PREFIX_CACHING:-false}"
INFERENCE_MAX_PREFILL_TOKENS="${INFERENCE_MAX_PREFILL_TOKENS:-4096}"
INFERENCE_PROGRESS_TIMEOUT="${INFERENCE_PROGRESS_TIMEOUT:-600}"
INFERENCE_RETRIES="${INFERENCE_RETRIES:-1}"
ENTROPY_PROBE_SIZE="${ENTROPY_PROBE_SIZE:-0}"
ENTROPY_PROBE_TOKENS="${ENTROPY_PROBE_TOKENS:-128}"
TRAIN_TF32="${TRAIN_TF32:-true}"
TRAIN_PHASE_BATCH_SIZE="${TRAIN_PHASE_BATCH_SIZE:-1}"
TRAIN_LOGIT_CHUNK_SIZE="${TRAIN_LOGIT_CHUNK_SIZE:-128}"
RUN_TAG="${RUN_TAG:-sdar4b-just${OBJECTIVE}-math-rbs${ROLLOUT_BATCH_SIZE}}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/runs/$RUN_TAG}"
RESUME="${RESUME:-}"

DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
    DRY_RUN=true
    shift
fi
# CLI 指定恢复/输出时，也必须走恢复分支，避免默认训练参数覆盖检查点。
CLI_ARGS=("$@")
for ((i=0; i<${#CLI_ARGS[@]}; i++)); do
    case "${CLI_ARGS[i]}" in
        --resume) RESUME="${CLI_ARGS[i+1]:?--resume needs a checkpoint or auto}" ;;
        --resume=*) RESUME="${CLI_ARGS[i]#--resume=}" ;;
        --output) OUTPUT="${CLI_ARGS[i+1]:?--output needs a directory}" ;;
        --output=*) OUTPUT="${CLI_ARGS[i]#--output=}" ;;
    esac
done
# 同一实验再次启动时恢复最近完整检查点；新实验请指定新的 OUTPUT。
# 恢复时检查 checkpoint 完整性和并行布局，允许修改源码。
if [[ -z "$RESUME" && -f "$OUTPUT/config.json" ]]; then
    RESUME=auto
    printf '发现已有实验，自动恢复最近完整 checkpoint: %s\n' "$OUTPUT" >&2
fi
COMMAND=("$TORCHRUN_BIN" --standalone "--nproc_per_node=$NPROC_PER_NODE" "$SCRIPT_DIR/train.py")
if [[ -n "$RESUME" ]]; then
    # 算法与采样配置继承 checkpoint；下方推理性能参数可调整。
    COMMAND+=(--resume "$RESUME" --output "$OUTPUT" --steps "$STEPS" --save-every "$SAVE_EVERY")
else
    COMMAND+=(--model "$MODEL" --data "$DATA" --output "$OUTPUT"
        --batch-prompts "$BATCH_PROMPTS" --group-size "$GROUP_SIZE" --minibatch-size "$MINIBATCH_SIZE"
        --ppo-epochs "$PPO_EPOCHS" --max-tokens "$MAX_TOKENS"
        --temperature "$TEMPERATURE" --lr "$LR" --clip "$CLIP" --objective "$OBJECTIVE" --tp-size "$TP_SIZE"
        --coupling-beta "$COUPLING_BETA"
        --seed "$SEED" --steps "$STEPS" --save-every "$SAVE_EVERY")
    if [[ -n "$BLOCK_SIZE" ]]; then COMMAND+=(--block-size "$BLOCK_SIZE"); fi
fi
COMMAND+=(--rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --train-phase-batch-size "$TRAIN_PHASE_BATCH_SIZE"
    --train-logit-chunk-size "$TRAIN_LOGIT_CHUNK_SIZE"
    --inference-cache-fraction "$INFERENCE_CACHE_FRACTION"
    --inference-max-prefill-tokens "$INFERENCE_MAX_PREFILL_TOKENS"
    --inference-progress-timeout "$INFERENCE_PROGRESS_TIMEOUT" --inference-retries "$INFERENCE_RETRIES"
    --entropy-probe-size "$ENTROPY_PROBE_SIZE" --entropy-probe-tokens "$ENTROPY_PROBE_TOKENS"
    --eval-every "$EVAL_EVERY"
    --eval-data "$EVAL_DATA" --eval-max-tokens "$EVAL_MAX_TOKENS"
    --eval-batch-size "$EVAL_BATCH_SIZE" --eval-seed "$EVAL_SEED")
if [[ "$EVAL_AT_START" == true ]]; then COMMAND+=(--eval-at-start); else COMMAND+=(--no-eval-at-start); fi
if [[ "$INFERENCE_CUDA_GRAPH" == true ]]; then
    COMMAND+=(--inference-cuda-graph)
else
    COMMAND+=(--no-inference-cuda-graph)
fi
if [[ "$INFERENCE_PREFIX_CACHING" == true ]]; then
    COMMAND+=(--inference-prefix-caching)
else
    COMMAND+=(--no-inference-prefix-caching)
fi
if [[ "$TRAIN_TF32" == true ]]; then
    COMMAND+=(--train-tf32)
else
    COMMAND+=(--no-train-tf32)
fi
COMMAND+=("$@")
printf 'CUDA_VISIBLE_DEVICES=%s WANDB_MODE=%s\n' "$CUDA_VISIBLE_DEVICES" "$WANDB_MODE"
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" == true ]]; then exit 0; fi
exec "${COMMAND[@]}"
