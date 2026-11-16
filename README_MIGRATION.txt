SDAR RL 代码迁移包

本包仅包含代码、依赖快照和迁移说明。不含模型权重、tokenizer 资源、训练数据、
Conda 环境、历史 checkpoint、训练日志或评测缓存。

目录：
  sdar_justgmpo/                 训练循环、算法、判分、概率计算、断点恢复、测试和 vendor
  DiRL/lmdeploy/                rollout 实际依赖的本地 LMDeploy 源码
  DiRL/LICENSE                  DiRL 许可证
  GMPO/train_zero_math_gmpo.py  CPU 测试解析的上游 loss 参考源码，不参与训练运行
  GMPO/readme.md                上游项目说明
  requirements-dirl.txt        本机 dirl 环境中的包版本快照
  run_train_portable.sh        使用当前激活环境 torchrun 的便携启动入口
  MANIFEST.sha256              包内文件的 SHA256 清单（清单自身除外）
  PACKAGE_INFO.json            来源、打包范围和环境说明
  VALIDATION.txt               本次本地验证结果

一、迁移

将整个压缩包复制到目标服务器后解压，保持上述相对目录关系。
rollout.py 会加载旁边的 DiRL/lmdeploy，不需要改成本机 pip 上游 LMDeploy。
不需要一并复制完整 DiRL 仓库。

二、准备环境

原环境：Python 3.12.14，PyTorch 2.8.0+cu128，FlashAttention 2.8.3，Linux x86_64。
requirements-dirl.txt 是现有环境快照，不是已经解决所有依赖冲突的锁文件。
源环境存在已知冲突：latex2sympy2 1.9.1 要求 antlr4-python3-runtime 4.7.2，
但实际安装的是 4.9.3。普通 pip install -r 会拒绝这组精确版本。
若按快照复刻，文件头给出了 --no-deps 安装方式；它保留源环境的冲突。
FlashAttention 的来源记录指向 Python 3.12 / Torch 2.8 / ABI FALSE wheel；
本机实际 PyTorch ABI 检查为 TRUE，因此不能仅凭这个历史 URL 选择新机 wheel。
本机 import flash_attn 成功；新机仍需按实际 PyTorch ABI 选择兼容 wheel。
GPU 驱动和系统库需单独准备。
本包未下载依赖 wheel，也没有执行任何环境安装或替换。

三、准备外部模型和数据

模型目录需包含完整权重、配置、tokenizer 和 SDAR 自定义 Python 文件。
原默认模型为原生 SDAR-4B-Chat；当前实现使用原生 block_size=4 和严格 LTR 采样。
原默认数据为 JustGRPO-MATH/train.sdar.json，但本包不包含该文件。
数据格式是 JSON 数组，每条至少包含 question 和 ground_truth_answer。
迁移其他 SDAR 模型时，应核对模型结构、词表、特殊 token 和 block 配置。

四、启动

先激活目标 Python 环境，在解压后的包根目录运行：

export MODEL=/绝对路径/SDAR-4B-Chat
export DATA=/绝对路径/train.sdar.json
export OUTPUT=/绝对路径/新实验输出目录
bash run_train_portable.sh --dry-run

确认打印的路径和参数正确后：

bash run_train_portable.sh

默认使用 8 张 GPU，每个训练进程加载一份完整模型，不做模型张量切分。
需要其他卡数时，同时设置 CUDA_VISIBLE_DEVICES 和 NPROC_PER_NODE，
并确认 batch-prompts、minibatch-size 等参数满足 train.py 中的整除约束。
输出目录请使用新目录；已有实验目录会触发原代码的自动恢复逻辑。
当前代码包不包含历史训练状态。

五、自定义算法

algorithm.py：优势计算、old log-prob、策略损失和分 phase 反向传播。
train.py：采样、固定 old、minibatch 更新的组织方式。
reward.py：奖励与正确性判分。
likelihood.py、vendor/sdar：SDAR 训练概率路径。
rollout.py、DiRL/lmdeploy：采样引擎、采样语义与权重同步。
runtime.py、checkpointing.py、run_state.py：梯度归约、导出和完整状态恢复。

六、验证

在包根目录运行 CPU 测试：

python sdar_justgmpo/tests/run_cpu_tests.py

GPU 验证需要另外准备好模型，并在具备足够显存的 GPU 节点运行：

CUDA_VISIBLE_DEVICES=0 python sdar_justgmpo/verify_gpu.py --model "$MODEL"

verify_gpu.py 包含真实 4B 模型、3000-token 反向等检查，会使用较多显存。
它的更新只作用于内存，不写入基础模型；验证结果写入 verification/。
本次迁移打包没有运行 GPU 训练。

校验文件内容：

sha256sum -c MANIFEST.sha256
