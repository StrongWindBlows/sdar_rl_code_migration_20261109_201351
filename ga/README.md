# SDAR 重组实验

首版代码及真实开发集 pilot 已完成。统一入口支持短 GPU 检查、Experiment 0、恢复验收、Experiment 1、报告及配置冻结。首轮结果见 [报告](runs/pilot8192-20261110/REPORT.md)：309 个唯一候选续跑未显示跨轮重组收益，当时计划的廉价评分器与 GA 未启动。

补充的 [Experiment 1.5](runs/pilot8192-20261110/exp1_5/REPORT.md) 已完成已有轨迹分析及 12 条小规模控制/撤回续跑；其中一题的早期单 token 重开恢复正确答案。该结果独立于历史候选重组，仍仅使用原 pilot。

[Experiment 1.6](runs/pilot8192-20261110/action_value/REPORT.md) 已完成 T8/T4 的 100 条同预算早期动作续跑：20 个恢复控制与全部动作检查通过，存在局部新增正确，但当前换位置、延迟和撤回规则均无净正确数提升。整题 forward 上限包含重放成本，未增加块内去噪轮数。

[新 Experiment 2-Joint](runs/joint-oracle-20261111/ANALYSIS.md) 已完成 261 个唯一动作：T8 的单因素并集 oracle 为 7/10、联合为 8/10，出现一个经复现的联合独有成功状态；T4 没有联合额外覆盖。所有 20 个 Current 及旧 Delay1/Swap1 对照通过，独立审计与 23 个 CPU 校验通过。成绩为事后 oracle，尚未实现在线评分或搜索。

[50 题扩量](runs/joint50-20261111/ANALYSIS.md) 已完成：全池 T8 为 35/50、T4 为 18/50，单因素并集分别为 33/50、16/50。新增 40 题中联合额外覆盖 T8 一题、T4 两题，但等候选数比较没有联合池整体优势。当前开发/评估用途已调整，见 [DATA_USAGE.md](DATA_USAGE.md)；旧 20/80 划分仅保留为历史归档。

[长程退化与预警](runs/degeneration-20261112/REPORT.md) 已完成同一批 50 题的 T8 / Dynamic 自然轨迹及按题预测检验：新采集使用 GPU 0–4。T8 精确重复 14/50、Dynamic 2/50；轻量预警未通过门槛，本轮未执行报警后的干预。总览见 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)。

[Threshold Landscape Study](runs/threshold-landscape-20261112/REPORT.md) 已完成 900 条开发生成与冻结后的 250 条评估生成。固定 0.95 在确认题上为 33/50、188.5 次平均 forward；factor=1 为 30/50、151.0 次。factor 相对固定 0.85 同为 30/50、forward 点估计降低 4.9%，区间仍跨过零收益。本轮结束于 E2，不进入 E3；400 题最终验证保留。

实验定义以 [PLAN.md](../experiments/crossover_b32/PLAN.md) 为准。这里规定文件组织、引擎接入和首轮执行范围；候选组合、Oracle、forward 预算及 GA 研究路线沿用该计划。

## 文件范围

主实验由 3 个业务模块、配置、文档及 CPU 校验组成；补充诊断使用独立入口：

| 文件 | 职责 |
| --- | --- |
| `README.md` | 文件职责、运行命令、执行限制及结果入口 |
| `config.json` | 模型、数据清单、GPU 4–7、五组基线、greedy、长度及记录参数；默认仅开发集 pilot |
| `run.py` | 主实验入口；阶段调度、每卡 worker、任务分片、结果落盘和报告生成 |
| `engine.py` | 本地 LMDeploy 适配：采样配置、请求 ID 映射、逐轮记录、状态恢复、动作注入、forward 计数 |
| `protocol.py` | 数据隔离校验、源状态选择、候选去重、分歧统计、复用数学判分器、配对结果汇总和恢复轨迹比较 |
| `test_protocol.py` | 小规模 CPU 校验：数据隔离、退化候选池、固定位置、预算和轨迹差异定位 |
| `early.py` | CPU 分析现有轨迹：人工错误标注对齐、重复起点、提交清单、分支路径与答案统计 |
| `early_rollback.py` | 从已标注早期状态执行 0/1/2 token 重开，核对恢复、下一轮输入和共同预算 |
| `action_value.py` | 以提交前信息选择早期状态，比较换位置、延迟和撤回；整题 512 forward 上限、块内轮数不增加 |
| `test_action_value.py` | 检查源状态选择的因果性、撤回风险的信息口径，以及预算耗尽时不输出含洞块 |
| `joint_protocol.py` | 联合位置/词值配对矩阵、动作去重、实际分组、分支验收与联合独有收益分析 |
| `joint.py` | 联合 Oracle 的 prepare/check/run/report；复用已有恢复接口与判分，按原块内轮数补交 |
| `test_joint.py` | 检查原动作保留、完整配对、退化去重、合法词值、联合归因和预算约束 |
| `joint50.py` | 固定扩为 50 题；采集基线/源状态、核对并复用 pilot、运行新分支，分开报告原 10 与新增 40 |
| `test_joint50.py` | 校验数据用途变更、固定抽样和缺少源状态时仍保留完整题目分母 |
| `degeneration.py` | T8 / Dynamic 自然轨迹采集、重复时间对齐、因果特征、按题分组预测与图表 |
| `test_degeneration.py` | 检查乱序提交时间、删失、未来信息隔离、可见值与题级折隔离 |
| `threshold_study.py` | 完整生成、单卡常驻副本、阶段调度、轻量记录、审计和质量—成本图表 |
| `threshold_rules.py` | 固定阈值、step 调度、OSDT/CadLLM 阈值组件和已有 factor 规则 |
| `threshold_study.json` | 冻结六组固定阈值、调度参数、factor 与低阈值支持对照 |
| `test_thresholds.py` | 校验调度因果性、校准范围、状态回退与去噪/预算口径 |

GPU 验收由 `run.py` 的独立阶段执行，不另建一套测试入口。首版不增加新的依赖环境、启动脚本或通用插件框架。

**修改既有源码：0 个文件。** `DiRL/lmdeploy/`、`sdar_justgmpo/`、checkpoint 和数据文件均直接复用。实验计划仅更新执行状态与实测配置。所有新增实现、运行输出及新增缓存位于 `ga/` 下，运行结果写入 `ga/runs/<run_id>/`。

## 引擎接入

使用已有 `dirl` 环境和项目内 LMDeploy，显式指定 block length 32。每卡一个 TP=1 worker；同配置下复用常驻引擎，候选分批提交，不为每个动作重载模型。模型配置变化时按组执行，避免不同调度策略混入同一批。

`engine.py` 集中管理少量运行时适配，优先绑定到本 worker 的引擎实例；必要的替换仅作用于该 worker 进程，关闭时恢复。保留现有采样参数展平修正及 EOS 处理语义，不调用固定 sequential 的训练 Rollout 入口，不复制整份训练器或推理引擎。

接入点分三类：

1. **请求与 forward 输入**：传递稳定 request ID，将轨迹和历史绑定到请求；对每次实际模型调用计数，区分 prefill、去噪、缓存写入与恢复重放。
2. **提交前后**：在 `post_sampling` 附近保存 mask、当前/历史 top-k、提交集合及其分歧；动作注入也从此处进入。记录所需的原始概率要在 logits 被采样处理修改前取得。
3. **状态恢复**：从原 prompt 开始，按已保存的提交轨迹重放至源状态，由引擎重建完全相同计算路径下的 KV，再注入指定动作。保留 mask 空位，各分支拥有独立的可变状态。重放开销与干预后的 forward 预算分开计数。

必须以 Current 恢复路径与不中断路径一致为通过条件。入口要求先通过检查，再记录基线；每个选中的源状态均通过恢复验收后，才允许进入 Experiment 1。

实现时发现两项数值限制：直接将整个已生成前缀重新 prefill 会改变 BF16 计算路径，后续提交排序可能不同，因此采用提交轨迹重放；batch=4 与 batch=1 也观察到排序差异，首版默认 `batch_size=1`，由四张卡并行处理请求。改变并发配置需重新验收。适配器按单个 forward 调度并关闭预取，确保不会越过停止或预算边界继续计算。

## 首轮执行

| 顺序 | 工作 | 完成标准 / 产物 |
| --- | --- | --- |
| 1 | 实现配置、协议逻辑与引擎封装，运行 CPU 校验和短 GPU 检查 | 数据仅取开发集；静态提交数、EOS、尾块、批请求映射正确；记录开关不改变输出；短样本核对原生与引擎语义 |
| 2 | Experiment 0：开发集固定 10 题 × 五组配置 | 正确率、截断、forward、耗时、逐轮轨迹，以及 T8 的 `d=0…4` 和 T4 的对应统计；吞吐与详细记录开销分别测量 |
| 3 | 依据开发集结果确定工作点并验证恢复 | 保存 → 恢复 → Current → 续跑；完整 token IDs、逐轮提交、终止和未来预算一致；失败时输出首个分歧并修复 |
| 4 | Experiment 1：最多 10 个开发状态 | 固定 S、去重后最多约 310 条续跑，生成父代与两候选池的配对结果；恢复未通过不得执行 |
| 5 | 根据任务奖励证据决定是否扩展 | 正面信号才扩展开发集；工作点与方法冻结后再执行 80 题评估，400 题留作最终验证 |

先完成这条路径。Experiment 2/3 的实现依据开发集证据再增加，首版不预建 GA 模块。

首轮使用 GPU 4–7，每卡单请求。候选去重后按卡分片；只保存 top-k 和必要诊断，避免逐轮写出完整词表 logits。前缀重放增加运行成本，但保证恢复使用相同的计算路径；安全共享 KV 的优化留待机制验证后再做。

## 运行与产物约定

在项目根目录使用已有 `dirl` 环境，所有阶段指定同一个 run ID：

```bash
export PATH=/ssd_data/liuyang/miniconda3/envs/dirl/bin:$PATH
python ga/run.py --run-id b32-pilot-01 --stage check
python ga/run.py --run-id b32-pilot-01 --stage exp0
python ga/run.py --run-id b32-pilot-01 --stage report
# 查看开发集的质量与分歧，再确定 source；以下以 T8 为例。
python ga/run.py --run-id b32-pilot-01 --stage verify-resume --source T8
python ga/run.py --run-id b32-pilot-01 --stage exp1 --source T8
python ga/run.py --run-id b32-pilot-01 --stage report --source T8
```

默认读取 `ga/config.json`，回答上限为 4096，数据范围为开发集内 10 题 pilot。真实 pilot 出现明显截断后，已统一用 `--max-new-tokens 8192` 重跑并用于分支实验。选择 T4 时，对恢复、干预和报告阶段传入 `--source T4`；高分歧位置诊断使用 `--positions disagreement`，产物位于独立分支目录。T4 仍采用完整八位置提交、其中四坐标可变的既定方案；干预后统一使用配置中的 `continuation=T8`。

`--scope dev` 扩大到 20 题开发集。开发完成后执行 `--stage freeze`，再用相同 source、positions 和 run ID 配合 `--scope eval` 运行 80 题评估；未冻结或配置不一致时入口拒绝评估。`--no-trace --stage exp0` 单独记录吞吐，避免与详细记录运行的耗时混淆。

运行清单记录代码/模型/数据指纹及环境。模型文件采用完整 SHA256，首次检查会读取权重文件。共享配置或代码发生变化时使用新 run ID。工作点和位置规则在开发阶段可调整，其产物按 `T8-confidence` 等目录区分；冻结后不再允许这些修改。已有阶段结果不会被覆盖。

每次运行保留逐题结果、压缩轨迹、选中状态、恢复验收结果及 `summary.json` / `summary.md`。评估入口检查冻结配置和数据划分；首轮默认范围仅为开发集 pilot。新运行使用独立目录，不覆盖已完成结果。

检查阶段仅运行短合成轨迹，不作为数学正确率结论。恢复比较完整 token IDs、逐轮提交、终止原因和未来 forward 次数；同时检查两种干预轮次、替代 token 注入、严格预算、记录开关、请求顺序和短原生模型对照。数学判分器复用现有实现；超时单列，含判分超时的状态不参与 oracle 上界比较。

## 已完成验证

- 当前版本：10 个 CPU 校验全部通过；[短 GPU 验收](runs/pilot8192-20261110/pilot/check-gpu4.json)通过五组调度、记录开关、请求顺序、两种干预轮次的恢复、替代动作和严格预算检查。短原生对照的提交 top-1 一致，最大 top-1 log-prob 差约 0.088。
- 四卡端到端冒烟：开发集 10 题、五组配置共 50 条基线；10/10 自然状态恢复一致；179 条去重候选续跑完成，判分无超时。产物在 [implementation-smoke-128](runs/implementation-smoke-128/)。该轮上限为 **128 tokens**，仅验证代码路径，不用于选择工作点或判断质量收益；之后仅补充分轮次统计和显式源状态种子，推理与干预路径保持相同。
- 真实 pilot：4096/8192 上限各完成 10 题 × 五组基线；8192 下 50 对记录开关输出一致，10/10 源状态严格恢复一致；自然位置 163 个、高分歧位置诊断 146 个唯一候选续跑完成。分歧、质量和诊断结果见 [报告](runs/pilot8192-20261110/REPORT.md)。
- 原项目 454 个 Python 源文件的 SHA256 保持不变，模型与数据哈希也保持不变，见 [指纹核查](runs/pilot8192-20261110/source_integrity.json)。所有实现、验证记录和缓存均位于 `ga/`。

`implementation-final` 保留首次 4096-token 基线；之后为完整长度候选队列将执行超时移到每个请求，当前代码和 8192-token 结果对应 `pilot8192-20261110`。查看当前自然位置报告：

```bash
python ga/run.py --run-id pilot8192-20261110 --max-new-tokens 8192 --stage report
```

已完成阶段不会重复运行。当前只使用固定 pilot 的 10 道开发题，未冻结配置、未运行 80 题评估或剩余 400 题。

Experiment 1.5 的离线分析可单独重算：

```bash
python ga/early.py --run ga/runs/pilot8192-20261110
```

人工标注、两份代码指纹、12 条验证的原始输出及总结保存在该 run 的 `exp1_5/`。已提交位置的原始 logits 不作为自身稳定性的估计；重开后的 masked 分布单独记录。反事实结果属于回溯诊断，尚不是在线纠错策略。

Experiment 1.6 在相同的 10 题上执行早期动作试验，使用独立的规则清单及目录：

```bash
python ga/action_value.py --run ga/runs/pilot8192-20261110 --stage prepare
python ga/action_value.py --run ga/runs/pilot8192-20261110 --stage run
python ga/action_value.py --run ga/runs/pilot8192-20261110 --stage report
```

`prepare` 在 GPU 运行前保存源状态、五种动作、预算与指纹；`run` 不覆盖已有结果。每个状态先验证 Current 续跑，再执行其他动作。T8/T4 各 10 题，单次干预，最多 100 条续跑；源块的延迟或撤回在原定最后一轮补交，之后返回对应的 T8/T4 策略。512 次整题上限包含重放、prefill 和缓存写入；EOS 可提前结束。结果与压缩轨迹保存在 `action_value/`，各动作单独报告，不将事后最佳动作当成可部署策略。

## Experiment 2-Joint

联合 Oracle 按 [执行计划](JOINT_ORACLE_PLAN.md) 复用上述 20 个早期状态，改变本轮的提交集合和词值。最多 5 个位置方案 × 4 个词值方案，冻结的 376 个矩阵单元去重为 261 个唯一动作。原 Current、Delay1、Swap1 均作为配对控制保留。每条路径仍以 512 次整题 forward 为上限，源块的去噪轮数不增加。

```bash
python ga/joint.py --stage prepare
python ga/joint.py --stage check
python ga/joint.py --stage run
python ga/joint.py --stage report
```

默认产物在 `ga/runs/joint-oracle-20261111/`，包含运行前清单、候选矩阵、逐动作结果、轨迹和报告。`prepare` 不覆盖已冻结的清单；`run` 根据已落盘的动作继续，代码指纹必须相同。GPU 小检查包括非 top-1 注入、两位置延迟收尾，以及不同动作后再次恢复 Current；正式运行逐状态核对原路径与旧 Delay1/Swap1。各池保留 Current，报告联合独有成功、完整四格配对和 32 个等候选数对照；oracle 结果不等于在线策略成绩。

`report` 重建 `summary.json` 与简要 `REPORT.md`；人工数学复核、成功配对和解释保存在独立的 [ANALYSIS.md](runs/joint-oracle-20261111/ANALYSIS.md)。本轮汇总曾修复 T8/T4 的旧状态 ID 重名问题，原执行清单与代码、修正记录均保留在运行目录；候选和推理实现没有改变。

## 50 题扩量

```bash
python ga/joint50.py --stage prepare
python ga/joint50.py --stage baseline
python ga/joint50.py --stage run
python ga/joint50.py --stage report
```

默认目录为 `ga/runs/joint50-20261111/`。抽样保留原 dev20，再从原 eval80 中固定分层抽取 30 题，组成 dev50；剩余 eval50 与最终 400 题保持未用。`selection.json` 是当前完整用途清单，原数据文件不修改。`prepare` 不覆盖已有清单；基线和分支阶段分别按已落盘结果继续。

原 pilot 的 20 条基线、快照与候选池重新核对一致后，复用旧 261 条分支及其审计证书；新增 40 题采集新状态并执行全部唯一动作。旧分支成本、新基线、新分支和额外复现分开记账。无合法状态时保留 Current，结果仍按 50 题统计。该入口不修改旧运行代码和候选协议。

## Threshold Landscape Study

配置为 `ga/threshold_study.json`。新运行指定独立 `--out`，按 `prepare → check → E0 → calibrate → E1 → support → E2 → report → timing → audit` 执行。每个 GPU 一个 TP=1 常驻副本，按同题同卡、规则顺序打乱的方式处理任务；已落盘的 `(question_id, rule, repeat)` 自动跳过。每条路径从原始 prompt 开始生成。

默认目录为 `ga/runs/threshold-landscape-20261112/`。`report` 生成开发阶段的 `DEVELOPMENT.md`、指标和图表；最终研究报告为 [REPORT.md](runs/threshold-landscape-20261112/REPORT.md)。本次条件触发的确认配置由运行目录内 `confirm.py` 冻结到 `heldout50/freeze.json`，五组配置完成同一批 50 道原保留题；没有按确认结果继续调参。

仅重新审计和分析已有结果：

```bash
python ga/threshold_study.py audit
python ga/runs/threshold-landscape-20261112/analyze.py
```

`analyze.py` 生成独立状态审计、确认配对区间和图表，不调用模型。完整计算量、计时范围及代码修订来源见最终报告。当轮累计 37 个 CPU 校验通过，400 道最终验证题仍未使用。

## Experiment T3：Marginal Commit Utility

`marginal_commit.py` 复用 Dynamic .90 的已保存自然轨迹和引擎恢复接口，在三个固定生成位置区间各抽一个真实状态。固定当前 top-1，比较 k0−2/k0/k0+2/k0+4，以及 .85/.95/factor1 建议的提交数量；合法裁剪后去重。只干预一次，然后继续 Dynamic .90；每条路径最多512次整题forward，包含重放、源forward、prefill和缓存写入。块内完成步数是测量结果，不强制末轮补交。

```bash
python ga/marginal_commit.py prepare
python ga/marginal_commit.py check
python ga/marginal_commit.py pilot
python ga/marginal_commit.py run
python ga/marginal_commit.py report
python ga/marginal_commit.py audit
```

默认输出为 `ga/runs/marginal-commit-20261112/`。`prepare` 冻结137个状态和499个唯一动作，阈值按原生BF16分数精度比较；GPU0–7、TP=1、batch=1。先验收，再原10题pilot，通过后扩到原开发50题。`run` 复用已完成的pilot，不重复计数；同状态的Current必须先复现原路径，其余动作才可执行。

配置为 `marginal_commit.json`，CPU语义校验为 `test_marginal_commit.py`。主报告使用每题最早区间的一个状态，其他区间单列；合并指标先在题内平均。浅树按题五折比较置信度、熵、进度与分布结构，预测与训练规则均落盘。Oracle只是反事实上界，已确认50题和最终400题不进入T3。

本轮已完成：499条正式与31条短路径通过审计，累计46个CPU校验通过。主比较Current为29/50、全池oracle为30/50；简单/结构浅树均26/50，未得到事前选择增益。结果解释见 [ANALYSIS.md](runs/marginal-commit-20261112/ANALYSIS.md)，自动完整指标见 [REPORT.md](runs/marginal-commit-20261112/REPORT.md)。运行目录内的 `analyze.py` 重算配对比较、成本拆解、分桶与最终图表，不调用模型。首轮0.95比较边界的精度修正、原始清单及7条被替代对照保留在 `rounding_revision/`；主要结果不变。

## Experiment T4：Confidence Sufficiency

`confidence_audit.py` 只读取T3最终版本，复算标签、原生提交数量和历史来源。比较C0置信度摘要、C1完整masked置信度排序、C2额外margin/熵、历史和位置特征；沿用按题五折，固定浅树与正则化线性模型，另作不依赖标签的相近置信度配对。

```bash
python ga/confidence_audit.py prepare
python ga/confidence_audit.py analyze
python ga/runs/confidence-sufficiency-20261112/verify.py
```

默认目录为 `ga/runs/confidence-sufficiency-20261112/`，`prepare`冻结本轮规则与输入指纹，已有运行不再次prepare。独立复算脚本也重绘最终图表。详细判断见[ANALYSIS.md](runs/confidence-sufficiency-20261112/ANALYSIS.md)。本次新增GPU forward为0，额外特征未改善折外选择；52个CPU校验通过。50题仍属开发数据，不将本轮折外结果称为独立确认。

## R1：高置信度重复恢复

入口`repetition_recovery.py`，配置`repetition_recovery.json`，CPU校验`test_repetition_recovery.py`。单次在线确认、单位置扰动，支持无回退采样/定向换词、统一块边界回退、Dynamic切换和greedy恢复控制。每条策略512总forward，丢弃计算和前缀重建均计入；采样使用完整模型词表及请求独立RNG。

新运行按`prepare → pilot → run → report`执行，使用新的`--out`。若旧Current不能复现，先`collect → refresh`采集当前权重自然状态，再做pilot；未触发的原题记录缺失，不补题。`collect`和`run`默认GPU4–7，pilot默认GPU6。不要对已完成目录重新prepare/refresh。

本轮遵照用户选择使用原目录，但第二分片与历史/官方哈希不同，并定位到四段56 KiB全零内容。原14题重新生成后只有7题触发，完成119条正式恢复路径，另有当前权重下42条直接Dynamic对照与3条确认。结果、限制和校验副本位置见[ANALYSIS.md](runs/repetition-recovery-20261112/ANALYSIS.md)。后续官方权重复验见下节。

只重算本轮审计、指标和完整性检查，不调用模型：

```bash
python ga/runs/repetition-recovery-20261112/analyze.py
```

执行时源码保留在运行目录`source/`，最终业务文件仅修正自动报告的范围措辞；指纹差异见`source_integrity.json`。R1有单题奖励恢复，但未达到多题稳定恢复或超过从头Dynamic的门槛，本轮不扩量。

### R1 官方权重复验

配置`repetition_recovery_official.json`使用完整校验的官方副本，数据和动作规则不变。新运行可用`prepare --config ga/repetition_recovery_official.json --out <新目录>`，随后同目录执行`pilot --gpu 3 → run → report`。本轮输出`ga/runs/repetition-recovery-official-20261112/`，GPU0–3；无需对旧运行做collect/refresh。

原14条Current及回退greedy全部复现；238条正式分支、42条新Dynamic对照已完成。12条路径退出重复，但正确恢复为0，未启动后续集成。最终报告为[ANALYSIS.md](runs/repetition-recovery-official-20261112/ANALYSIS.md)。只重算审计和指标：

```bash
python ga/runs/repetition-recovery-official-20261112/analyze.py
```

最后一条路径因会话中断补跑，前237条直接复用；未完成尝试单独归档，额外成本范围见`interruption.json`。正式结果无重复计数。
