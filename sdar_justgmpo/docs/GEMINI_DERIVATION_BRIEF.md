# SDAR 顺序与并行解码协同强化学习的推导任务

我正在使用 SDAR 扩散语言模型做数学推理强化学习，希望让同一个模型的顺序解码能力和并行解码能力相互促进。请从明确的概率模型、优化目标和梯度估计出发，独立推导一种可实现的方法。优先考虑沿用一次顺序 rollout，并复用其中已经计算的信息；如果必须增加并行采样，请说明它提供了哪些无法从现有样本获得的信息。

我关心的是数理依据与实际生成过程是否一致，而不是先拼接几个 loss 再解释。请先独立分析正文，最后再审查附录中的已有候选推导；可以否定它，也可以提出更合适的目标。不必沿用 CORY 的两个模型、合作奖励或角色交换。

代码与实验背景截至 2026 年 11 月 17 日。以下实现摘录和实验数据足以独立阅读；本机路径供能够访问项目的研究者进一步核对。

## 研究愿景与已经对齐的定义

我有三个相互关联的想法。

第一，顺序解码每次只提交一个 token，但模型仍会预测当前块中的其他 masked 位置。这些尚未提交的预测，是否能在 RL 中提供有数学依据的学习信号？例如提前预测能力、条件分布随提交动作的变化、对后续推理的信息贡献。

第二，能否使用 full-mask 重建辅助学习或信用分配？**这里的 full mask 已明确限定为：保留题目和已经完成的历史块，只把当前 32-token 块的待生成位置全部 mask。不是遮住整段回答。** 对首个包含 prompt 的混合块，prompt token 保留；最后一块只对首个 EOS 及其之前的有效生成 token 计损失。

第三，同一组参数在顺序解码和并行解码下会访问不同的状态、形成不同的输出分布。能否使顺序模式找到的高质量轨迹改善并行模式，同时由并行模式的能力、失败状态或独特解法反过来改善顺序模式？仅仅让顺序模式当固定教师，未必覆盖我希望研究的双向促进。

“更自由的解码方式拥有更高上限”“正确轨迹很稀疏”都是待检验的假设。请区分策略类的表达能力、固定采样算法的可达分布、有限预算下的探索能力与最终准确率，不预设它们之间的大小关系。

## 当前模型与训练配置

项目根目录是 `/ssd_data/liuyang/sdar_rl_code_migration_20261109_201351`，主要训练代码在 `sdar_justgmpo/`。虽然目录名保留了 GMPO，最近这轮实验使用的是 **GRPO**。

| 项目 | 本轮实际设置 |
| --- | --- |
| 基座 | `SDAR-4B-Chat-b32`，块长度 32 |
| 模型路径 | `/ssd_data/liuyang/hug/model/SDAR-4B-Chat-b32` |
| 数据 | JustGRPO 官方代码所用 `ankner/math-500` 的本地 JSONL |
| 训练集 | 原始 7500 题，按题长过滤并排除 2 条空答案后使用 7459 题 |
| 测试集 | 完整 500 题；与相邻目录的 `HuggingFaceH4/MATH-500` 不同 |
| 每轮采样 | 128 道题，每题 8 条回答，共 1024 条 |
| 更新 | 全局 minibatch 1024，PPO epoch 1，每轮恰好一次 optimizer update |
| 实验长度 | 共 10 轮，即 10 次参数更新 |
| 采样规则 | 顺序提交，每次一个 token，temperature 1，top-k 0，top-p 1 |
| 最大生成长度 | 3000 tokens |
| 奖励 | 最终数学答案正确为 1；错误、格式错误、截断为 0 |
| 终止 | 保留首个 EOS 及其概率，首个 EOS 后的内容不参与判分和训练 |
| 优化器 | AdamW，学习率 5e-6，betas 为 0.9 和 0.999，weight decay 0 |
| KL 系数 | 当前为 0 |
| 训练并行 | TP2、DP2，共 4 张卡；每个推理副本并发 32 条请求 |
| 精度 | FP32 主权重，BF16 计算和推理 |
| phase 分组 | 每组计算 4 个 noisy phase，共享一份 clean 分支 |

phase 分组为 4 只是训练计算的合并方式，**不表示每步并行生成 4 个 token**。当前 rollout 始终顺序提交一个 token。

当前 advantage 是同题 8 条回答的组内归一化：减组均值，再除以样本标准差加 1e-8。损失先对每个回答的有效 token 取平均，再对回答取平均。零 advantage 回答跳过反传，但仍计入 batch 分母。

## 解码与概率重算的真实实现

### 顺序 rollout

`sdar_justgmpo/rollout.py` 用本地 LMDeploy 引擎，关键配置如下：

```python
dllm_block_length = 32
dllm_denoising_steps = 32
dllm_unmasking_strategy = 'sequential'
```

推理端会为当前块的 32 个位置计算隐藏状态和词表预测，然后只提交最左侧尚未确定的一个位置。其他未提交位置的候选预测不会作为已确定 token 保留到下一轮，它们仍然是 mask。不是每一步都还有 32 个 mask，而是一个块内的 mask 数随顺序提交逐渐减少。

训练轨迹目前只保存 prompt、最终 token 序列、是否正常结束、最终 reward 和回答级 advantage，没有保存每次前向中全部未提交位置的完整词表分布。

因此，不能把未提交位置视为已经执行并获得最终奖励的动作。若要使用这些预测，需要说明它们在概率模型中是监督目标、边缘化的候选、辅助估计量，还是新增的动作。

### 顺序 likelihood 与共享 clean

`sdar_justgmpo/likelihood.py` 按块内位置构造 phase。对于 phase t，每个块内已经生成的位置保留真实轨迹 token，从第 t 个待生成位置起都使用 mask；每个块仅以本 phase 中实际提交的那个 token 为 likelihood target。

多个 phase 的布局是：

```text
[noisy_phase_0, noisy_phase_1, ..., noisy_phase_k, shared_clean]
```

注意力关系是：

- noisy 分支之间不可见；noisy query 可以看本分支的当前块。
- noisy query 可以看 clean 分支的严格历史块，不能看 clean 的当前块与未来块。
- clean 分支使用 block-causal attention，同一块内可见，历史块可见。
- position ID 保持原始位置；clean 分支参与梯度计算。

这用于重算顺序 rollout 的条件概率。所有 phase 完成梯度累积以后，才进行一次参数更新。

当前 `SDARPolicy.forward` 的核心顺序如下，先选 target，再做词表投影：

```python
hidden = self.model(
    inputs_embeds=embeds,
    position_ids=position_ids,
    attention_mask=mask,
    use_cache=False,
).last_hidden_state
hidden = hidden[:, target_positions] if labels.ndim == 2 else hidden[0, target_positions]
logits = F.linear(hidden, self.lm_head.weight[:active_vocab]).float() / temperature
log_distribution = logits.log_softmax(-1)
logp = log_distribution.gather(-1, labels[..., None]).squeeze(-1)
```

所以，训练骨干已经计算其他位置的 hidden，但没有为它们全部计算词表概率。rollout 端计算了全部位置的预测，也不意味着 learner 可以直接利用那些推理结果反传；rollout 在 inference mode 下运行。

### 单次更新 GRPO

`sdar_justgmpo/algorithm.py` 中的单次更新路径省略统计后如下：

```python
if response.advantage == 0:
    return
for _, logps, _ in phase_logps(model, response, temperature, phase_batch_size):
    ratio = (logps - logps.detach()).exp()
    loss = -response.advantage * ratio.sum() / len(response.tokens) / minibatch_size
    loss.backward()
```

仅当每轮恰好一次 GRPO 更新时使用这条路径。参数在整个采样批次的梯度累积期间不变；更新点上的 ratio 为 1，其梯度保留为 log-prob 的梯度，因此无需缓存一份旧概率或再做一遍预计算。DP 梯度在更新前归约。

多次更新的 GRPO 和 GMPO 使用另一条通用路径，整轮冻结旧概率。本次数学推导可以先考虑一次更新，但迁移时应明确适用范围。不要把当前实现的组内标准差归一化、随机回答长度归一化、梯度裁剪和原始期望奖励的无偏梯度默认视为同一个对象。通用路径的 clip 参数 0.4 是 log-ratio 区间阈值；本轮更新点上尚不发生 clipping。

### full mask phase 的复用与成本

上述块内 full-mask 输入状态已经存在于 phase 0。对同一个进入 likelihood 计算的回答，只要沿用当前注意力关系，就可以在已有 phase 0 中增加其他有效 masked 位置的重建目标，共享骨干前向，并在计算图释放前把损失合并反传。

新增开销主要来自更多位置的 `lm_head`、全词表归一化、loss 与相应梯度，不必再独立跑一遍 Transformer。一个完整生成块中，原有 32 个顺序 phase 合计投影 32 个目标位置；若在 phase 0 补齐所有 32 个目标，会额外投影 31 个位置。这个计数说明词表投影总量接近翻倍，**不能据此说整个训练的计算量翻倍**。独立教师、额外重建前向或多个新 mask 状态会另增成本。

当前零 advantage 回答完全跳过 likelihood。如果新目标也要使用这些回答，就会新增它们原本没有执行的骨干计算，必须单独计入成本。只在 rollout 侧缓存概率可能节省某些统计计算，但不能替代带梯度重算。

### 当前动态并行解码

动态评测使用 `low_confidence_dynamic`，规则在 `DiRL/lmdeploy/pytorch/strategies/dllm/unmasking.py`：

1. 为各 masked 位置得到候选 token 及其概率。
2. 同时提交置信度达到 0.9 的位置。
3. 至少提交其中置信度最高的一个位置，保证进展。

现有评测是 greedy，所以候选为各位置 argmax。若将其改成随机采样用于训练，候选 token 和位置选择的依赖也要进入概率分析。固定连续分组、固定数量按置信度选位置、动态阈值选位置，是三种不同的策略，不能相互替代。

## 已有实验观察

同一轮训练的基座和 Step 1 至 Step 10 均完成了全部 500 题评测：

| 权重 | 顺序准确率 | 动态准确率 | 动态 TPF |
| --- | --- | --- | --- |
| Base | 68.8% | 62.2% | 4.320 |
| Step 1 | 69.6% | 63.2% | 4.344 |
| Step 2 | 70.2% | 63.4% | 4.382 |
| Step 3 | 70.8% | 62.8% | 4.385 |
| Step 4 | 70.4% | 62.4% | 4.603 |
| Step 5 | 71.6% | 61.0% | 4.641 |
| Step 6 | 70.2% | 63.6% | 4.598 |
| Step 7 | 69.6% | 61.8% | 4.945 |
| Step 8 | 71.0% | 58.2% | 5.054 |
| Step 9 | 70.0% | 56.0% | 5.149 |
| Step 10 | 70.6% | 53.6% | 5.500 |

表中的 TPF 是全数据集返回的有效生成 token 总数，除以实际包含待生成 mask 的 denoising forward 总数；包含首个 EOS，排除 EOS 后尾部 token，排除纯 prefill 和缓存写入前向。这是汇总比值，不是逐题 TPF 的简单平均，也不等于实际 wall-clock 加速比。

顺序评测使用 TP2、batch 32；动态评测使用 TP1、batch 1、BF16 eager、阈值 0.9。两者均为 greedy，块长度 32，最大生成长度 3000，使用相同题目、模板和答案判分。每一列内部的 checkpoint 比较保持该列设置一致；跨列差异不能全部归因于解码规则。

固定阈值下，训练改变置信度时实际并行程度也会变化。因此，Step 10 的动态准确率下降与 TPF 上升同时发生，不足以单独证明并行能力全面退化。需要比较相同 TPF、相同 forward 预算或相近准确率下的效率。

Step 6 的逐题交叉结果为：两种方式都正确 299 题，只有顺序正确 52 题，只有动态正确 19 题，两种都错误 130 题。两者正确集合的并集是 370/500，即 74.0%；这是知道标准答案后的事后选择结果，不是现成可实现的推理准确率。单次 greedy 的集合互补也不能直接等同于总体策略分布或 pass@k 的互补。

固定训练上下文上的完整词表熵探针，从基座约 0.905 降到 Step 5 的 0.658、Step 8 的 0.419、Step 10 的 0.341。它覆盖固定的 8 条回答、合计 1024 个 token。该现象提示需要检查熵与置信度变化，但目前没有建立它与动态错误之间的因果关系。

本轮每步耗时从约 56 分钟降到约 36 分钟，其中 rollout 约 13 至 16 分钟；这些耗时不含评测，且回答长度、编译与运行状态也在变化，不能当作新方法的固定成本。概率重算与反传是重要开销，双倍 rollout 和额外骨干遍历都需要有充分收益依据。

JustGRPO 原论文也展示过顺序训练后并行推理的准确率与效率收益，不能由本次实验推出“顺序训练无法改善并行解码”。其并行实验使用 EB-Sampler，和这里的固定置信度阈值策略有差异。[JustGRPO 论文第 5.2 节](https://arxiv.org/html/2601.15165v2#S5.SS2)

## 希望独立推导的问题

请从下面的问题出发选择一条主线，不必把所有机制同时加入。

1. **生成过程应怎样定义。** 明确 state、action、提交位置集合、token 采样、状态转移、EOS 与长度上限，以及真实 reward。区分去噪条件分布、生成轨迹概率和最终回答的边缘概率。顺序与并行是否需要统一为一个带解码规则参数的策略族？
2. **一次顺序 rollout 足以估计什么。** 尚未提交的 31 个位置，对顺序策略的原始期望奖励是否提供额外有效梯度？它们若不影响实际转移，相关 score 项的条件期望是否为零？若要估计并行策略收益或梯度，重要性采样是否满足支持条件，权重是否可计算、方差是否可控？不要把“一次采样加一次概率重算”误写成“两次 rollout”。
3. **full-mask 重建是什么数学对象。** 是合法生成策略的 likelihood、条件监督目标、似然下界、价值函数估计，还是关于某种一致性的正则项？当前块 full-mask 下对全部位置取 log-prob 之和，何时能对应实际并行策略概率，何时只能对应一个不同的生成过程？
4. **信用分配是否成立。** 揭示一个 token 后，其他位置对最终成功轨迹的重建概率变化，能否估计信息贡献或因果贡献？用已经受当前动作影响的最终轨迹作为标签或 baseline，会引入什么偏差？若使用 potential-based shaping，请给出状态势函数、终止边界、return-to-go 与策略梯度的推导；仅有望远镜求和还不足以保证更好的学习信号。
5. **如何定义相互促进。** 是提高两个模式的加权期望奖励、提高受约束的并行收益下界、减少解码分布差异，还是改善准确率与计算量的 Pareto 前沿？单向轨迹学习、双向联合优化和利用另一模式发现的新解法，应分别说明。若方法没有并行独立探索，哪些意义上的反向促进仍然成立？
6. **怎样防止表面收益。** 避免把提高置信度、减少熵、缩短回答或减少多样性误当成并行推理能力提升。请分析同一步位置独立采样不能表示的联合依赖，是否构成不可消除的误差，强行一致是否可能削弱顺序模式。
7. **怎样落到当前代码。** 标明 rollout、概率重算、反向传播、优化器更新各发生几次；哪些 tensor 可以共享，哪些量必须 detach，额外显存和计算主要在哪里。若只在一次 on-policy 更新点等价，请说明多次更新时需要哪些修正。严谨推导可以先使用固定长度、固定提交计划，但要明确这些简化如何恢复到当前真实设置。

## 期望得到的研究输出

请给出一条最有依据的主方案，并把以下内容连起来说明：

- 精确定义的目标函数、概率空间和必要假设。
- 从目标到梯度估计的逐步推导，说明哪些项是无偏估计，哪些是近似、代理目标或正则项。
- 单次顺序 rollout 是否足够；如不够，指出缺失的信息和最少需要增加的采样。
- 一个可枚举的二元 token 小例子或反例，用来验证梯度、支持条件及两种策略的区别。
- 贴近当前 phase 实现的训练伪代码，以及相对现有 GRPO 增加的实际计算。
- 最小消融设计：以相同采样预算和训练计算预算分别比较，评测完整 500 题，并画出准确率与 TPF 或实际 forward 数的关系。应有独立验证集用于调参，避免反复依据这 500 道测试题选择方案和超参数；后续还需重复实验或统计区间。

我希望方法简洁，能解释每一个新增项为何存在。优先复用现有参数和 full-mask phase；若需要 critic、独立教师、额外 head 或第二类 rollout，说明它解决的具体不可辨识性或估计问题。没有成立的结论时，请直接指出缺口，不要预设必须得到 1+1 大于 2。

## 附录 已有候选推导供审查

以下是此前讨论中的一个候选方向，**不是要求采用的方案**。请在完成独立建模后检查它的成立条件、价值与局限。

设 Sθ 和 Pθ 分别是同一模型在顺序与某种并行规则下定义的、规范化的完整回答分布。候选目标是：

```math
F(\theta)
=\mathbb E_{y\sim S_\theta(\cdot\mid x)}[R(x,y)]
-\beta D_{\mathrm{KL}}\!\left(S_\theta(\cdot\mid x)\|P_\theta(\cdot\mid x)\right).
```

当奖励在 0 到 1 之间时，对同一个问题可以写出：

```math
J_P(x)\ge J_S(x)-\sqrt{\tfrac12D_{\mathrm{KL}}(S_\theta\|P_\theta)}.
```

对数据分布平均后应正确处理条件 KL 与平方根，不要把平均 token KL 直接代入完整回答分布的界。该界可能很松；增大前面的拉格朗日目标不自动保证右侧这个非线性下界单调提高。

令 dθ(y) 为两种完整回答的 log-prob 差，在可交换求导与期望、支持与可微条件满足时，候选梯度为：

```math
\begin{aligned}
d_\theta(y)&=\log S_\theta(y\mid x)-\log P_\theta(y\mid x),\\
\nabla_\theta F&=\mathbb E_{S_\theta}\!\Big[
\left(R-\beta d_\theta(y)\right)\nabla_\theta\log S_\theta(y\mid x)\\
&\hspace{4em}+\beta\nabla_\theta\log P_\theta(y\mid x)\Big].
\end{aligned}
```

两个分布共享参数，候选推导没有将任何一边当作固定教师。该梯度曾用一个完全枚举的小模型对照直接求导核对；这不证明把当前动态解码或任意重建分数代入 Pθ 就成立。

如果并行规则是事先确定的连续位置分组，每组从同一个状态独立采样，固定长度下可写出：

```math
P_\theta(y\mid x)=\prod_b\prod_{j\in C_b}p_\theta(y_j\mid s_b).
```

每组 32 个位置时，条件分布可由当前 full-mask phase 提供；但实际 EOS 停止、最后一组长度、prompt 混合块与 EOS 后采样变量的边缘化仍须严格说明。用已知最终长度构造的重建分数，不会自动成为包含长度生成机制的规范化概率。

对于固定连续分组，还曾提出把 dθ 分解为逐 token 条件 log-prob 差 cj，用 `R − β × 从当前位置起的 cj 之和` 为顺序动作构造 return-to-go，并保留并行 log-prob 的直接梯度项。请验证该因果分解、哪些项可作为 baseline，以及非连续动态提交下是否仍然成立。组内归一化、长度归一化和多次更新的处理也需要单独推导。

还有一个重要限制。先忽略 EOS，对给定前缀的固定长度块，设顺序模式的块联合分布为 S_B，一次并行生成的独立位置分布为各 qj 的乘积，则：

```math
D_{\mathrm{KL}}\!\left(S_B\middle\|\prod_jq_j\right)
=D_{\mathrm{KL}}\!\left(S_B\middle\|\prod_jS_j\right)
+\sum_jD_{\mathrm{KL}}(S_j\|q_j).
```

右侧第一项是块内总相关性。即使边缘分布拟合完美，独立位置采样也可能仍有误差；联合训练还可能通过压低顺序分布的多样性来减小差异。请判断这个候选是否合理、应如何约束，或者它是否从目标选择上就不合适。

## 本机代码与结果定位

以下均相对于项目根目录。主要函数名可用于搜索，路径本身不能代替上面的实现背景。

| 文件 | 相关入口或内容 |
| --- | --- |
| `run_train_tp2_dp2.sh` | 四卡启动与可调参数 |
| `sdar_justgmpo/train.py` | `make_responses`、`single_update_grpo`、采样与更新循环 |
| `sdar_justgmpo/algorithm.py` | `group_advantages`、`phase_logps`、`backward_on_policy_grpo`、通用更新 |
| `sdar_justgmpo/likelihood.py` | `phase_inputs`、`packed_attention`、`SDARPolicy.forward` |
| `sdar_justgmpo/rollout.py` | `Rollout`、顺序引擎配置、首个 EOS 处理 |
| `sdar_justgmpo/reward.py` | 二值数学判分与截断处理 |
| `DiRL/lmdeploy/pytorch/strategies/dllm/model_agent.py` | `slice_outputs`、`post_sampling`，当前块预测与提交 |
| `DiRL/lmdeploy/pytorch/strategies/dllm/unmasking.py` | `sequential`、`low_confidence_dynamic` |
| `ga/engine.py` | 现有动态推理 Backend，前向计数和事件追踪 |
| `ga/protocol.py` | `Grader` |

训练记录在 `sdar_justgmpo/runs/sdar4b-grpo-phase4-20261116-155816/`，实际参数在 `config.json`，训练与熵指标在 `metrics.jsonl`。Step 5 权重目录是 `final/`，Step 6 为 `inference-step-6/`，Step 10 为 `final-step-10/`；各自的 `eval/` 保存顺序评测。

动态汇总在 `sdar_justgmpo/runs/dynamic-allsteps-20261117-161412/`，包括 `comparison.csv`、`summary.json`、`measure_dynamic.py`，以及逐权重的 `predictions.jsonl`、`metrics.json` 和事件记录。部分基座与 Step 5、Step 8 结果通过符号链接复用此前同设置的完整 500 题评测。

当前代码尚未加入本文设想的 full-mask 重建或协同 RL 目标；现有共享 phase 和单次更新 GRPO 是可复用的实现基础。

## 相关原始资料

- [JustGRPO 论文](https://arxiv.org/html/2601.15165v2)与[官方代码](https://github.com/LeapLabTHU/JustGRPO)：顺序 RL 及其并行推理结果；本地模型、生成长度与解码器并非原文设置。
- [Coevolving with the Other You 论文](https://papers.neurips.cc/paper_files/paper/2024/file/1c2b1c8f7d317719a9ce32dd7386ba35-Paper-Conference.pdf)：提供协同学习启发，不要求复刻其双模型结构。
- [Gradient Estimation Using Stochastic Computation Graphs](https://papers.nips.cc/paper_files/paper/2015/file/de03beffeed9da5f3639a621bcab5dd4-Paper.pdf)：用于区分随机节点、确定性依赖和参数相关目标的梯度。
- [Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)：如果考虑用重建构造奖励整形，需要核对其条件。
- [Mask-Aware Policy Gradients for Diffusion Language Models](https://arxiv.org/html/2607.15200v1)：与 token 及位置选择联合建模有关。
- [Denoising Token Credit Assignment](https://arxiv.org/html/2610.35201v1)：与提交 token 后剩余位置预测变化的信用信号有关，可用于检查已有工作边界。

请直接引用原始资料核对相关结论，区分已有方法、对本问题的推论和新的假设，不预设方案的新颖性。
