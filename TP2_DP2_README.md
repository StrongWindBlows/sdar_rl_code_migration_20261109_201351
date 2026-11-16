# SDAR 四卡 TP=2、DP=2

入口为 `run_train_tp2_dp2.sh`，使用已有的 PyTorch / LMDeploy 和本地 SDAR 模型，不需要安装 Megatron 或 verl。当前支持单机四卡：

| GPU / rank | 工作 |
|---|---|
| 0、1 | 第一个 TP=2 模型副本，处理 DP 分片 0 |
| 2、3 | 第二个 TP=2 模型副本，处理 DP 分片 1 |
| 0↔2、1↔3 | 对应参数分片的梯度平均 |

GPU 编号按 `CUDA_VISIBLE_DEVICES` 中的顺序映射。推理和训练都采用 TP=2；两个 DP 副本处理不同题目，同一 TP 组处理相同轨迹。attention 和 MLP 的线性层切分，embedding、词表输出层、norm 保留副本，因此单卡显存不会恰好降为原来的一半。

## 当前机器运行

```bash
cd /ssd_data/liuyang/sdar_rl_code_migration_20261109_201351
export PATH=/ssd_data/liuyang/miniconda3/envs/dirl/bin:$PATH

# 复现短测试：合成题目，48-token 上限，1 轮、2 次 AdamW 更新。
# 用独立目录，保留 verification 下的原始验证记录。
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT="$PWD/sdar_justgmpo/runs/tp2-dp2-check" \
bash sdar_justgmpo/verification/tp2_dp2_20261109/run_smoke.sh

# 同一目录从已保存的完整轮次恢复，继续到总计 2 轮。
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT="$PWD/sdar_justgmpo/runs/tp2-dp2-check" \
bash sdar_justgmpo/verification/tp2_dp2_20261109/run_smoke.sh --resume auto --steps 2
```

正式训练默认使用 `/ssd_data/liuyang/hug/model/SDAR-4B-Chat-b32` 与 `/ssd_data/liuyang/hug/data/JustGRPO-MATH-500/train.jsonl`。原始 JSONL 的 `problem` 和 `solution` 由共享加载器转换；按 JustGRPO 题长规则筛出 7,461 题，再明确报告并排除 2 条空标准答案，实际使用 **7,459 道训练题**。也支持原有包含 `question` 和 `ground_truth_answer` 的 JSON 数组。

```bash
RUN_TAG=sdar4b-justgmpo-math-rbs32 bash run_train_tp2_dp2.sh --dry-run
RUN_TAG=sdar4b-justgmpo-math-rbs32 bash run_train_tp2_dp2.sh

# 恢复到总计 10 轮；恢复时继承检查点的算法和采样参数。
RUN_TAG=sdar4b-justgmpo-math-rbs32 RESUME=auto STEPS=60 bash run_train_tp2_dp2.sh
```

`BATCH_PROMPTS` 是全局题数，`MINIBATCH_SIZE` 是全局回答数，两者都须能被 DP=2 整除。`BATCH_PROMPTS * GROUP_SIZE` 须能被 `MINIBATCH_SIZE` 整除。`ROLLOUT_BATCH_SIZE` 是每个 TP 推理副本的并发回答数。`STEPS` 是 rollout 轮数。

当前四卡入口默认每轮 128 题、每题 8 个回答、30 轮，学习率 5e-6，每轮 8 次优化器更新；回答上限 3,000 tokens。`MODEL`、`DATA`、`OUTPUT` 可通过环境变量覆盖。短测试可使用 `BATCH_PROMPTS=2 GROUP_SIZE=4 MINIBATCH_SIZE=4 MAX_TOKENS=128 STEPS=1`，并指定独立的 `RUN_TAG`。

## 集中配置推理

生成和评估的推理设置均在根目录 `run_train_tp2_dp2.sh` 中显式列出：

| 配置 | 默认值 | 含义 |
|---|---|---|
| `ROLLOUT_BATCH_SIZE` | 32 | 每个 DP 副本的并发回答数，两个副本合计最多 64 条 |
| `EVAL_BATCH_SIZE` | 32 | 单个 TP 评估副本的并发回答数 |
| `INFERENCE_CACHE_FRACTION` | 0.8 | 推理引擎将扣除运行开销后的空闲显存按此比例分配给 KV 缓存 |
| `INFERENCE_CUDA_GRAPH` | true | 开启解码阶段 CUDA Graph |
| `INFERENCE_PREFIX_CACHING` | false | 跨请求复用相同前缀的 KV 缓存 |
| `INFERENCE_MAX_PREFILL_TOKENS` | 4096 | 每次预填充的 token 上限 |
| `TRAIN_TF32` | true | 允许支持 TF32 的 GPU 加速 FP32 矩阵乘法；主要训练计算仍使用 BF16 |

布尔配置使用 `true` 或 `false`。CLI 参数优先于环境变量，例如 `--inference-cache-fraction 0.25`、`--no-inference-cuda-graph`。这些选项也可直接传给 `eval.py`；评估并发使用 `--batch-size`。

当前 SDAR 的前缀缓存检查能命中缓存，但重复批次的贪心输出与关闭缓存时不一致，因此默认保持关闭。

2026-11-12 在 A800 80GB、TP=2/DP=2 上验证了每副本 32 条、每条 3000 tokens 的完整生成，期间 FP32 主权重、Adam 状态和梯度均驻留显存。KV 比例 0.3 和 0.8 均通过，整卡峰值分别约 62.0 和 76.9 GiB（包含其他常驻进程，按 5 秒间隔采样）；两个副本合计吞吐分别约 846 和 854 tokens/s。最终默认采用 0.8。另通过 42 项 CPU 测试，以及真实采样、参数更新、保存、33 题评估和恢复时修改运行参数的检查。详见 [验证记录](sdar_justgmpo/runs/inference32-check-20261112-194124/validation.json)。

并发、缓存和上述引擎选项属于运行参数，恢复时可以调整；未显式传入则继承 checkpoint 配置，旧 checkpoint 没有的选项使用当前默认值。训练的 batch、group、学习率等仍按原规则继承。默认运行目录名带 `-rbs32`，与此前并发为 1 且尚无 checkpoint 的已停止运行分开。

## 保存时自动评估

`run_train_tp2_dp2.sh` 将保存和评估配置集中在一起：

| 配置 | 默认值 | 含义 |
|---|---|---|
| `SAVE_EVERY` | 10 | 每隔多少个 rollout round 保存并评估 |
| `EVAL_DATA` | `../hug/data/JustGRPO-MATH-500/test.jsonl` | 完整 500 题测试集 |
| `EVAL_MAX_TOKENS` | 3000 | 每题评估生成长度上限 |
| `EVAL_BATCH_SIZE` | 32 | 评估推理并发 |
| `EVAL_SEED` | 113 | 评估随机种子 |

每次完整 checkpoint 保存成功后，训练暂停，rank 0 调用现有 `eval.py` 读取该 checkpoint 的 BF16 权重，使用第一组 TP 卡评估。其他 rank 通过 Gloo 在 CPU 上等待，评估完成后继续训练。评估沿用训练的 TP 和 block size，每题贪心生成一次；评估长度和并发独立配置。最后一轮保存 `final` 时同样评估一次。

```bash
SAVE_EVERY=5 EVAL_MAX_TOKENS=3000 EVAL_BATCH_SIZE=32 bash run_train_tp2_dp2.sh
```

每份 checkpoint 的 `eval/` 子目录保存 `config.json`、`predictions.jsonl`、`metrics.json` 和 `eval.log`，控制台输出准确率、正确题数、平均生成长度、截断率及判分超时数。评估完成后继续训练，不会修改该 checkpoint 的权重或恢复状态。

评估失败会停止训练并保留完整 checkpoint。恢复时先检查所恢复 checkpoint 的评估结果：已完成的直接复用；未完成的清理其部分评估结果后重跑，再继续训练。即使已经达到目标轮数，也会补完评估并写入 `completed.json`。启动脚本在恢复时仍传入当前评估配置；后续 checkpoint 使用该配置，已完成的历史评估保留原结果。

## 独立评测基线和训练检查点

评测使用独立进程，不占用训练的 optimizer 状态。每题贪心生成一次，与训练共用 Chat 模板和判分器。

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

测试集完整使用 500 题；`--limit 4 --max-tokens 128` 可做短测试。本地 b32 基线的配置缺少 block size，必须指定 `--block-size 32`；训练导出的检查点会自动读取该值。配置、逐题结果和汇总分别保存在 `config.json`、`predictions.jsonl`、`metrics.json`，重复运行使用新输出目录。这是 `ankner/math-500` 划分，不能与 `HuggingFaceH4/MATH-500` 的分数直接混用。

启动脚本默认将 HF 缓存放到 `/ssd_data/liuyang/.cache/huggingface`，避免当前环境原有的不可写缓存目录；可用 `SDAR_CACHE_DIR` 修改。W&B 默认 offline。

## 通信、权重同步和恢复

当前这台 PCIe A800 机器的四卡 NCCL 最小 all-reduce 测试，在直接 P2P 开启时超时，`NCCL_P2P_DISABLE=1` 时通过。入口默认采用后者。其他机器只有在验证直接 P2P 可用后，才应覆盖为 `NCCL_P2P_DISABLE=0`。

每轮训练开始，DP 分片 0 将 TP 权重拼回完整 BF16 模型，写入输出目录内的临时目录。两个 TP 推理进程组分别读取它，完成生成后退出，再进入 learner 更新。这使每轮推理使用当前权重，但存在磁盘导出和引擎启动开销。当前没有跨 TP 的 GPU IPC 热更新。

采样结束后，各 rank 通过 Gloo 的 CPU barrier 等待所有推理副本退出，然后才进行 NCCL 广播和 learner 计算，避免较早结束的副本占用另一副本仍在使用的 GPU 通信资源。

TP 检查点采用 format 2，包括两个 `training_state-tp*.pt`（FP32 主权重和 Adam 状态）、各 rank RNG、日志恢复位置，以及可独立加载的完整 BF16 推理导出。恢复保留文件完整性、TP/DP 布局和进度检查；源码、数据和环境指纹不再采集或校验，已有指纹字段也不会阻止恢复。旧的单卡检查点不能直接恢复成 TP=2。

每份真实 4B TP=2 检查点实测为 66.23 GiB，因词表参数在 TP 分片中保留副本，比单卡完整状态更大。保存目录通过临时目录原子发布；失败留下的隐藏 `*-incomplete-*` 目录不作为完整检查点。

## 验证范围

2026-11-12 数据接入验证：36 项 CPU 测试和四进程数值验证通过；真实数学题短训练完成 2 次更新，新进程恢复后达到累计 4 次更新，并验证两个 TP 分片的参数继续变化。导出的模型通过独立评测入口加载。完整 500 题基线评测为 346/500（69.2%，greedy、3,000-token 上限、TP=2、batch size=32），13 条回答截断，无判分超时。记录位于 `sdar_justgmpo/runs/math-integration-20261112/validation.json`。

以下为 2026-11-09 的历史验证。日志和结果位于 `sdar_justgmpo/verification/tp2_dp2_20261109/`，最终状态见其中的 `results.json`。

- 原有及 block32 CPU 测试：29/29 通过。
- 四进程 CPU TP/DP 数值验证：梯度相对单模型参考最大绝对误差 `2.220446049250313e-16`；梯度裁剪、Adam 更新、检查点恢复后的下一次更新均通过。数值基准使用小型 SDAR 和 FP64，不代表真实 4B BF16 内核达到同样误差。
- 真实 4B-b32 四卡首轮：16 条回答，2 次非零梯度 AdamW 更新，完成完整检查点保存；rank 0 learner 的 PyTorch 峰值分配为 43.02 GiB。该数字不包含独立推理进程、其他用户进程和非 PyTorch 分配。
- 真实 4B-b32 新进程恢复：从首轮检查点继续生成 16 条回答，新增 2 次非零梯度更新，保存 `final-step-2/`，累计 2 轮、4 次更新，正常退出。恢复轮次 rank 0 learner 的峰值分配为 43.06 GiB。真实 GPU 测试验证恢复可继续执行；逐元素相同的恢复数值验证来自上述 CPU 测试。
- 合成测试通过随机选择题产生正负 advantage，用于验证训练链路，正确率没有模型效果评估意义。

上述完整保存/恢复测试的回答上限为 48 tokens；追加的 3000-token 压力测试见下节。8B 模型、多机训练和长期训练稳定性尚未验证。采样固定为 sequential，每个 block 的 32 个 phase 均保留；low-confidence 策略尚未接入对应的训练轨迹概率计算。

之前在单卡引擎与 learner 间观察到的 log-prob 数值差异仍待定位；TP/DP 接入没有解决或重新验证该差异，不能将运行通过解释为训推数值完全一致。

实现前的代码备份是 `verification/tp2_dp2_20261109/before_tp.tar.gz`（相对 `sdar_justgmpo/`），变更补丁为同目录的 `tp2_dp2_changes.patch`。

## 3000-token 四卡压力测试

2026-11-09 使用真实 SDAR-4B-Chat-b32、TP=2、DP=2、block size=32 通过以下测试，进程正常退出，总耗时 571.46 秒：

- 两个 TP 推理副本分别实际生成 3000 tokens，prompt 长度分别为 45 和 48 tokens。测试显式屏蔽 EOS，以确保覆盖完整长度。
- 使用生成的完整轨迹及测试指定的正负 advantage，完成全部 32 个 phase 的反向传播和两次 AdamW 更新。第二次更新在 FP32 Adam 状态已驻留显存时执行；两次全局梯度范数为 1.155697、1.130428，所有 rank 一致，抽查的参数分片均发生更新。
- 旧概率缓存保持不变，第二次更新时新旧概率出现非零差异，数值均有限。
- 保留 Adam 状态与梯度，重新导出更新后的权重；两个 TP 推理副本再次各生成 3000 tokens。共生成 4 条完整长回答。

四个 learner 的 PyTorch 峰值分配均为 44.32 GiB，最大峰值预留为 48.67 GiB。这不包含独立推理进程和其他任务。每 5 秒采样的整卡统计中，GPU 3 总占用最高为 69.57 GiB，包含该卡原有约 17.67 GiB 的其他任务。

该历史测试中每个 TP 副本的推理并发为 **1**，learner microbatch 为 **1**。当前入口的并发配置见上文。此次历史测试不重复保存大型检查点，保存/恢复覆盖来自前面的短测试。

这是执行与显存压力测试，强制长度和指定 advantage 不用于证明正常 EOS、真实奖励训练效果或训推概率一致性。

```bash
cd /ssd_data/liuyang/sdar_rl_code_migration_20261109_201351
# OUTPUT 必须是尚不存在的目录；保留原始验证结果。
OUTPUT="$PWD/sdar_justgmpo/verification/long3000-rerun" \
bash sdar_justgmpo/tests/run_parallel_long_gpu.sh
```

结果和逐 rank 记录在 [long_tp2_dp2_3000_20261109/results.json](sdar_justgmpo/verification/long_tp2_dp2_3000_20261109/results.json)，完整日志在同级 `long_tp2_dp2_3000_20261109.log`，整卡显存采样在 `long_tp2_dp2_3000_20261109_memory.csv`。`test_manifest.json` 保存测试脚本与生产源码的哈希。
