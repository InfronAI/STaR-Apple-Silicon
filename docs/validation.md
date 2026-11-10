# 实机验证

后续按默认 32 道训练题、16 道验证题、两轮完整 epoch 运行的录制实验，见 [最新端到端报告](experiments/2026-11-10-e2e/report.zh-CN.md)（[English](experiments/2026-11-10-e2e/report.md)）。本页保留最初平台适配时的小规模检查记录。

日期：2026-11-10。机器：Apple M1 Pro，16 GiB 统一内存，macOS 26.6.1，原生 arm64 Python 3.11.16。运行环境为 PyTorch 2.8.0 / Transformers 4.51.3 / NumPy 2.2.6。完整可读指标保存在 [validation.json](validation.json)。

## 已执行的检查

| 检查 | 结果 |
| --- | --- |
| `bash scripts/setup_mac.sh` | 安装成功，依赖一致性通过，MPS 实际前向/反向通过 |
| `pytest -q` | 33 项通过，包含真实 CPU 和 MPS 训练测试，MPS 测试未跳过 |
| `ruff check src tests`、`ruff format --check src tests` | 通过 |
| `pip check`、wheel 构建、`git diff --check` | 通过 |
| 离线小型 GPT-J，MPS / CPU 各两轮 | 全流程通过，每次首轮包含 1 个无提示成功样本和 1 个提示后恢复样本 |
| checkpoint 重载、已完成实验 `--resume` | 通过；恢复不重复训练已完成轮次 |
| 真实 GPT-2 / 原始 CommonsenseQA，两轮 MPS | 两轮实际执行全参数训练，checkpoint 保存并重载成功 |
| 真实 GPT-2 / 原始 GSM、生成的加法数据 | 每个任务实际评估 2 个问题，均完整完成 |
| 数据迁移完整性 | 12 个原始数据与提示文件同上游 Git blob 逐字节一致 |

离线 smoke 使用刻意拟合的微型 GPT-J 和相互重叠的测试题，专门验证生成、筛选、rationalization 和训练的连接是否正确。它不是泛化评测，不能用其中的正确率说明 STaR 的研究效果。单元测试另外验证未完成轮次重放、只对错题重推理、无提示评估、基座重置、尾批、损失掩码和文件篡改检测。

## 整理后目录上的真实模型验证

模型：`openai-community/gpt2`，固定 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`。使用原始训练集按 seed=42 打乱后的前 6 题、原始验证集前 2 题，完整运行 2 轮。生成最多 96 tokens，序列长度 512，每轮为控制验收耗时仅更新 1 个 optimizer step。参数、梯度和优化器均在 MPS 上执行，使用 float32。

| 轮次 | 无提示生成正确 | 提示重试 | 提示后正确 | 保留样本 | 实际训练步数 | 验证正确 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 / 6 | 5 | 0 | 1 | 1 | 0 / 2 |
| 2 | 2 / 6 | 4 | 0 | 2 | 1 | 0 / 2 |

第 1、2 轮的训练 loss 分别为 3.20695 和 3.66150。两轮都从固定的 `base_model/` 初始化；不同训练样本的 loss 不宜作下降曲线比较。第 2 轮有两条可训练记录，但验收配置只更新一步、batch=1，所以实际训练只读取一条；常规任务配置默认按 epoch 训练。

真实 GSM 和 arithmetic 的两题评估均为 **0 / 2**。这些结果说明小模型能完成计算流程，不说明它获得了可靠的数学或常识推理能力。没有为提高验收指标而把参考推理填入训练记录。

整理前还执行过 16 道训练题、4 道验证题、每轮两次更新的 GPT-2 流程；两轮分别保留 5 和 6 条样本。该重复验证的模型已删除以节省约 1.4 GiB，最终保留整理后目录的实验。

## 本机保留的可用产物

- `runs/final-validation-config.json`：整理后真实模型验收配置。
- `runs/final-validation/summary.json`：模型路径与两轮详细指标。
- `runs/final-validation/iteration_0002/model/`：可加载的真实微调 GPT-2。
- `runs/smoke-mps-verified/`、`runs/smoke-cpu/`：离线集成验证的配置、记录与微型模型。
- `runs/gsm-evaluation/`、`runs/arithmetic-evaluation/`：实际任务评估输出。

这些本机产物被 Git 忽略；可用 README 中的命令在其他机器重建。机器可读报告使用相对路径，未包含模型权重或机器专有的绝对目录。

本次仅在这台 16 GiB M1 Pro 实机验收。其他 Apple Silicon 型号、8 GiB 内存、Python 3.12 及 Linux CPU 的运行表现需要在各自环境验证；仓库 CI 定义了 Python 3.11/3.12 的 CPU 检查，但本次没有声称已经执行远程 CI。未运行论文规模实验，也未验证原 GPT-J-6B 权重的全参数训练。
