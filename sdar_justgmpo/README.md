# SDAR JustGMPO

用 SDAR 的精确 LTR 条件概率实现 GRPO 和 GMPO。训练与采样使用一致的 `block_size`，每次采样提交一个 token；训练端将多个 noisy phase 与一份共享 clean 序列拼接计算。未显式指定时读取模型配置，旧模型配置缺少该字段时沿用 4；对于 b32 权重须显式设置 `BLOCK_SIZE=32` 或 `--block-size 32`。

核心文件：

- `algorithm.py`：单次更新 GRPO、通用 GRPO/GMPO 目标、旧概率缓存和跨 phase 反向传播。
- `train.py`：采样一轮 → 选择计算路径 → 打乱回答 → 分 minibatch 更新 → 再采样。
- `data.py`：训练和评测共用的 MATH 数据加载、答案提取。
- `eval.py`：基线模型和训练检查点的独立贪心评测。
- `likelihood.py`：从审计项目复用的 SDAR 概率计算；entropy 改为逐 token 返回。
- `rollout.py`：复用 LMDeploy 的采样布局修正、全词表采样、原始 token 流和权重同步，改为正常 EOS 停止。
- `runtime.py`、`checkpointing.py`、`run_state.py`：复用梯度归约、模型导出及完整断点恢复。
- `vendor/gmpo/math_grader.py`：原样复制 GMPO 的数学判分器。

## 与论文和源码的对应

对回答 i 的有效 token，令 `d = new_logp - old_logp`：

```python
sign = +1 if advantage >= 0 else -1
c = sign * minimum(sign * d, clamp(sign * d, -epsilon, epsilon))
ratio = exp(mean(c))
loss = -advantage * ratio
```

对应本地 `../GMPO/train_zero_math_gmpo.py:675–701` 和论文 Algorithm 1。默认 `epsilon=0.4` 是 **log-ratio** 阈值，对应概率比 `[exp(-0.4), exp(0.4)]`。每个回答先独立聚合，再对 optimizer minibatch 内的所有回答取平均。零 advantage 回答可跳过计算，但仍计入回答数分母。

通用入口 `sdar_justgmpo/run_train.sh` 沿用 GMPO 的主要训练设置。四卡入口 `run_train_tp2_dp2.sh` 当前默认 GRPO、全局 minibatch 1024、每轮更新一次、训练5轮、学习率5e-6；其余配置直接在该脚本中查看和调整。

| 参数 | 默认值 |
|---|---|
| 每轮题数 / 每题回答数 | 128 / 8，共1024条回答 |
| optimizer minibatch | 全局128条回答 |
| 每轮更新次数 / PPO epoch | 8 / 1 |
| learner microbatch | 每卡1条回答 |
| 每次前向合并的 phase 数 | 通用入口1，四卡入口4 |
| 最大回答长度 | 3000 |
| 每个推理副本并发 | 32 |
| 采样 | sequential、温度1、top-k=0、top-p=1、完整模型词表 |
| advantage | 组均值、样本标准差、epsilon=1e-8 |
| 奖励 | 正确1，错误/格式错误0；未正常结束的截断回答为0 |
| EOS | 包含首个 EOS 的概率；其后不判分、不训练 |
| 优化器 | AdamW，lr=1e-6，betas=(0.9,0.999)，eps=1e-8，wd=0 |
| 梯度裁剪 / KL | 1.0 / 0 |
| 初次实验长度 | 50个 rollout round，即400次 optimizer 更新 |

启动脚本默认使用 `/ssd_data/liuyang/hug/model/SDAR-4B-Chat-b32` 和 `/ssd_data/liuyang/hug/data/JustGRPO-MATH-500/train.jsonl`，路径相对于工作区定位。原始训练集 7,500 题，按 JustGRPO 的 `0 < len(problem) < 1500` 规则筛出 7,461 题，再报告并排除原始第 309、4791 行的空标准答案，最终使用 **7,459 题**；测试集完整保留 500 题。这是 **GMPO 算法在 SDAR/ankner 数据上的移植**，模型、数据和实验时长与原文 Qwen/MATH level3–5 实验不同。prompt 使用 SDAR Chat 模板和 boxed 指令。

`data.py` 直接读取原始 JSONL：`problem` 映射为 `question`，从 `solution` 的最后一个 `\boxed{}` 提取 `ground_truth_answer`，兼容数据中的 `\boxed 9` 写法。原有包含 `question`、`ground_truth_answer` 的 JSON 数组也能使用，保留其既有样本选择。空标准答案在原始训练集加载时明确警告并排除；其他提取失败及测试集答案缺失会报告文件和行号。不需要另存转换后的数据。

`OBJECTIVE=grpo` 使用算术平均目标，保持相同的采样、reward、回答长度归一化和 clip 区间。每轮更新次数为 `BATCH_PROMPTS * GROUP_SIZE / MINIBATCH_SIZE * PPO_EPOCHS`。

当目标为 GRPO 且每轮只更新一次时，自动使用单遍路径：跳过旧概率缓存，每个 phase 分组只进行一次带梯度前向和反传。更新前策略参数始终相同，`exp(logp - logp.detach())` 的值为1，梯度等于 log-prob 的梯度，保持原 GRPO 在该更新点的损失与梯度。entropy 从同一次前向统计；零 advantage 回答跳过计算，仍计入 batch 分母。多次更新 GRPO 和 GMPO 使用原有通用路径，旧概率在整轮内固定。

`TRAIN_PHASE_BATCH_SIZE` / `--train-phase-batch-size` 控制每组 phase 数，四卡入口默认4。每组布局为 `[noisy_0, ..., noisy_k, clean]`：noisy 分支之间不可见，各自仅看本分支的当前 block 和 clean 的历史 block；clean 使用原有 block-causal 注意力。位置编码保持原始位置，clean 保留完整梯度，组内统一反传、组间累积梯度；每个目标 token 只计入一次。设置为1即逐 phase 布局。分组用于两种训练路径及熵探针，增加分组大小会增加单次前向的激活显存。

## SDAR phase 如何保持 GMPO 梯度

GMPO 的几何平均必须覆盖完整回答，不能给每个 phase 分别取 exp。实现采用两遍计算以保持显存可控：

1. no-grad 遍历所有 phase，得到当前回答的 `q = exp(mean(c))`。
2. 保持模型参数不变，再逐 phase 反传 `-A * stopgrad(q) * sum(c_phase) / T / local_minibatch_size`。

这个 surrogate 与完整 GMPO 的**梯度相同**；日志记录真实 `-A*q`，不把 surrogate 数值当作 GMPO loss。SDAR attention dropout 为0；模型的所有 phase 完成后才执行 optimizer 更新。

通用路径在每轮最开始保存所有回答的 `old_logps`，这一缓存跨 minibatch 和 PPO epoch 保持固定。之后各 minibatch 使用已经更新的当前模型，因此 ratio 会偏离1。`ppo_epochs=1` 表示每条轨迹使用一次；例如1024条回答按128条划分，本轮仍有8次参数更新。

## 运行

本机四卡 `TP=2、DP=2` 请使用包根目录的 `run_train_tp2_dp2.sh`，详见 [四卡启动和验证说明](../TP2_DP2_README.md)。它同时接入 learner 参数切分、两个 TP 推理副本和分片检查点；下述原始入口默认仍为 `TP=1`。

使用已有 `dirl` 环境；脚本从当前环境寻找 `torchrun`，也支持显式设置 `TORCHRUN_BIN`。

```bash
cd /ssd_data/liuyang/sdar_rl_code_migration_20261109_201351
export PATH=/ssd_data/liuyang/miniconda3/envs/dirl/bin:$PATH
bash run_train_tp2_dp2.sh --dry-run
bash run_train_tp2_dp2.sh
```

参数优先级：末尾 CLI 参数 > 环境变量 > 脚本默认值。例如：

```bash
# 每轮一次 GRPO 更新，每次合并4个 phase，使用独立目录。
OBJECTIVE=grpo TRAIN_PHASE_BATCH_SIZE=4 RUN_TAG=sdar4b-grpo-phase4 bash run_train_tp2_dp2.sh

# b32 权重：采样 block length、denoising steps、训练 attention 与 phase 均使用32。
MODEL=/path/to/SDAR-4B-Chat-b32 BLOCK_SIZE=32 bash run_train_tp2_dp2.sh

# 同一目录从最近完整 checkpoint 恢复到总计50轮。
OUTPUT=/path/to/run RESUME=auto STEPS=50 bash run_train_tp2_dp2.sh
```

`STEPS`、`--steps` 和 checkpoint 名称里的数字均为 **rollout round**。`metrics.jsonl` 另记 `optimizer_steps` 和每次更新的 `train/updates`，不会把两种 step 混用。`SAVE_EVERY` 控制完整断点间隔，`EVAL_EVERY` 控制评估间隔（默认跟随保存间隔），每次完整保存及最后的 `final/` 也会评估。独立评估轮次保留 `inference-step-N/` BF16 权重及 `eval/` 结果，这种权重不包含优化器状态。`EVAL_AT_START=true` 还会将基座评估写入 `eval-step0/`。每份完整 checkpoint 包含 BF16 推理导出、FP32 主权重、Adam 状态、各 rank RNG 及本地日志恢复位置；4B 的 TP=1 检查点约57.5 GiB，TP=2 约66.23 GiB。恢复仅在完整轮次边界进行。

例如，5 轮诊断实验可用 `STEPS=5 EVAL_EVERY=1 EVAL_AT_START=true SAVE_EVERY=5 ENTROPY_PROBE_SIZE=4 RUN_TAG=sdar4b-gmpo-diagnostic` 启动。熵探针在首次采样时固定每个 DP 副本的 4 道训练题及各题首个回答的前 128 tokens，记录 `entropy-probe-dp*.json`；`entropy-probe-step0.json` 和各轮 `metrics.jsonl` 的 `probe/entropy` 使用相同的上下文计算完整词表熵，不按 advantage 筛选。

TP 采样和自动评估以实际生成的 tokens 更新进度。`INFERENCE_PROGRESS_TIMEOUT=600` 表示连续 600 秒没有新 tokens 就终止整个推理进程组；`INFERENCE_RETRIES=1` 表示用相同问题、权重和种子重试一次，重试时关闭 CUDA graph。再次失败时各训练 rank 保存最后完成轮次的完整状态并退出，避免长期占卡。启动及引擎关闭阶段也受此超时约束。

不再采集或校验源码、数据和环境指纹，也不需要源码白名单。修改代码后可以继续恢复；恢复默认继承保存的训练配置，并检查文件完整性、TP/DP 布局和进度。已有检查点里的 `fingerprint` 字段不参与恢复校验。

phase 分组大小属于运行参数，可以在恢复时调整；已有检查点缺少该字段时使用当前入口的配置。单遍 GRPO 由恢复后的目标和每轮更新次数自动判断，不改变检查点格式。

W&B 默认 offline。本地恢复会还原权重、优化器和日志；W&B offline 生成相同 ID 的新本地记录片段，不声称已在线恢复。新实验请用不同 `RUN_TAG`，避免覆盖旧记录。

## 独立评测

自动评估复用训练的 TP/DP 布局，每个 TP 副本处理一个连续题目分片。四卡 TP2/DP2 使用两组 GPU 各评估250题，输出位于 `eval/shard-0/` 和 `eval/shard-1/`；全部成功后，rank 0 按原题号合并到 `eval/predictions.jsonl`，并按题数汇总 `eval/metrics.json`。汇总耗时取各分片推理耗时的最大值。各分片独立超时、重试，恢复时复用已完成的评估。独立入口默认单副本，也可通过 `--num-shards` 和 `--shard-index` 指定分片。

从包根目录运行下列命令。`--model` 可以是基线模型，也可以是训练输出的 `checkpoint-N` 或 `final`；评测不加载优化器状态。

```bash
CUDA_VISIBLE_DEVICES=0,1 python sdar_justgmpo/eval.py \
  --model /ssd_data/liuyang/hug/model/SDAR-4B-Chat-b32 \
  --data /ssd_data/liuyang/hug/data/JustGRPO-MATH-500/test.jsonl \
  --tp-size 2 --block-size 32 --output sdar_justgmpo/runs/math-baseline-eval

CUDA_VISIBLE_DEVICES=0,1 python sdar_justgmpo/eval.py \
  --model sdar_justgmpo/runs/sdar4b-justgmpo-math/final \
  --data /ssd_data/liuyang/hug/data/JustGRPO-MATH-500/test.jsonl \
  --tp-size 2 --output sdar_justgmpo/runs/sdar4b-justgmpo-math/eval-final
```

默认每题贪心生成一次、上限 3,000 tokens、推理并发 32，block size 读取模型配置；本地 b32 基线模型没有此字段，需要显式传 `--block-size 32`，训练导出的检查点已包含该字段。评测完整测试集，不做题长过滤。Chat 模板、EOS 处理和判分器与训练一致，未正常结束的截断回答计错。用 `--limit 4 --max-tokens 128` 做短测试，用 `--batch-size` 调整并发。输出目录中的 `config.json` 记录配置，`predictions.jsonl` 保存逐题回答，`metrics.json` 记录准确率（0–1）、平均 token 数、截断率和判分超时数。已有结果的目录不会被覆盖。

这里评测的是 JustGRPO 官方代码使用的 `ankner/math-500` 测试划分，与相邻目录 `HuggingFaceH4/MATH-500` 的测试集不同。

## 验证

```bash
cd sdar_justgmpo
python tests/run_cpu_tests.py
CUDA_VISIBLE_DEVICES=0 python verify_gpu.py
```

CPU 测试直接执行 GMPO 上游 loss 语句作为 oracle；验证正负/零 advantage、padding、组标准差、实际 SDAR 串行与分 phase 梯度、冻结旧概率后的多次更新、采样 metadata 布局、EOS、截断奖励以及已知错误判分反例。

逐轮评估接入还通过了 49 项 CPU 测试、TP=2/DP=2 数值测试、真实四卡两轮训练与逐轮评估，以及单个 DP 副本失败后的完整保存和退出演练。TP=2 采样与训练 log-prob 抽检、固定熵探针及清理记录见 `runs/diagnostic-check-20261114/validation.json`；验证用的大型权重已清理，保留日志和评估结果。

`verify_gpu.py` 加载真实4B模型，核对引擎提交 token 的 log-prob、不同 prompt 的批量采样、保留 EOS、完整 phase 反向传播、固定 old 的更新，以及3000-token长轨迹的显存路径。验证中的 SGD 更新只作用于内存，不写入原模型。

本次数据接入验收记录位于 `runs/math-integration-20261112/validation.json`，历史日志位于 `verification/`。36 项 CPU 测试及四进程 TP/DP 数值测试通过；真实模型完成短训练、完整保存和新进程恢复。基线模型在完整 500 题上贪心评测得到 346/500（69.2%），上限 3,000 tokens、TP=2、batch size=32，13 条回答截断，判分超时 0 次。

训练效果应通过独立评测判断；训练正确率本身不代表泛化收益。`active_token_entropy`、ratio 和 clip 诊断只统计非零 advantage 回答；`grad_norm` 为裁剪前范数。

四卡 TP=2、DP=2 的真实 3000-token 压力测试也已通过，覆盖两次 AdamW 更新及优化器状态驻留时的再次长生成，详见 [四卡长序列验证](../TP2_DP2_README.md#3000-token-四卡压力测试)。复现入口为 `tests/run_parallel_long_gpu.sh`；测试显式指定 advantage 并屏蔽 EOS，仅验证执行与显存路径。
