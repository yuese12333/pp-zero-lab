# CLAUDE.md

> 本文件供项目中的 AI agent（Claude Code / Cowork 等）阅读，用于快速理解项目目标、技术栈与工作约定。人类开发者也可参考。

## 1. 项目目标

构建一个**分布式训练显存与吞吐优化实验平台**，在小型 Transformer（nanoGPT / GPT-2 small）上实测并可视化对比：

- **流水线并行调度**：GPipe（同步）vs 1F1B（PipeDream 风格异步）的气泡率、吞吐
- **ZeRO 显存优化**：Baseline vs ZeRO-1 / ZeRO-2 / ZeRO-3 三阶段的每卡显存占用
- **可视化前端**：气泡时间线、显存对比柱状图、吞吐曲线

最终交付：课题报告、系统手册、可视化系统、演示视频。本仓库主要承载**代码 + 实验数据 + 可视化**。

## 2. 当前 agent 的核心任务（A 的工作线）

你（agent）服务的开发者负责 **实验与代码（角色 A）**，是项目关键路径。核心目标按优先级：

1. **跑通 baseline**：单机/单卡 nanoGPT 训练循环能正常前向反向。
2. **接入 DeepSpeed**，依次跑通 ZeRO-1 / ZeRO-2 / ZeRO-3，采集每卡显存。
3. **实现/配置流水线调度**（GPipe 与 1F1B），采集气泡率与吞吐。
4. **导出标准化 CSV**（见 §5），这是交付给下游（可视化 B、报告 C）的唯一接口。

> ⏰ 硬截止：**6/19 前产出完整 CSV**。在此之前一切优先服务于"把数据跑出来"，而非代码优雅度。

## 3. 技术栈与环境

- **语言/框架**：Python 3.10+，PyTorch 2.x，DeepSpeed
- **模型**：nanoGPT（karpathy/nanoGPT）或 HuggingFace GPT-2 small，参数量控制在能单机跑通的规模
- **可视化（B 负责，agent 了解即可）**：前端 + 图表库（Plotly / Recharts / D3 任选）
- **显存测量**：`torch.cuda.max_memory_allocated()` / DeepSpeed 内存报告
- **资源**：优先单机多卡；若多卡不可用，用单卡 + DeepSpeed 单进程测 ZeRO 显存，气泡率可用公式 `(K-1)/(M+K-1)` 推算 + 调度仿真补充

## 4. 目录结构（约定，agent 应遵守）

```
pp-zero-lab/
├── CLAUDE.md                # 本文件
├── README.md                # 人类用快速说明
├── requirements.txt
├── configs/                 # DeepSpeed JSON 配置
│   ├── zero1.json
│   ├── zero2.json
│   └── zero3.json
├── src/
│   ├── model.py             # 模型定义（nanoGPT / GPT-2 封装）
│   ├── train.py             # 训练入口，--config 切换实验
│   ├── pipeline.py          # GPipe / 1F1B 调度封装
│   ├── profiler.py          # 显存/吞吐/气泡率采集
│   └── export.py            # 汇总实验结果 → CSV
├── data/                    # 训练用小数据集（如 tinyshakespeare）
├── results/
│   └── metrics.csv          # ★ 交付给下游的唯一数据接口
├── viz/                     # B 的可视化前端（agent 不主动改）
└── scripts/
    └── run_all.sh           # 一键跑全部实验
```

## 5. 数据接口契约（最重要 — 不要擅自改列名）

`results/metrics.csv` 是与下游（可视化、报告）约定好的**唯一接口**，列名锁定如下：

| 列名 | 含义 | 示例 |
|------|------|------|
| `config` | 实验配置名 | `baseline` / `zero1` / `zero2` / `zero3` / `gpipe` / `1f1b` |
| `category` | 实验类别 | `zero` / `pipeline` |
| `stage` | ZeRO 阶段（流水线实验留空） | `0`/`1`/`2`/`3` |
| `num_gpus` | 卡数 | `1` / `4` |
| `micro_batches` | 微批次数 M（流水线实验用） | `8` |
| `mem_per_gpu_gb` | 每卡峰值显存(GB) | `16.6` |
| `throughput_samples_s` | 吞吐(samples/s) | `1240` |
| `bubble_ratio` | 气泡率(0–1，流水线实验用) | `0.08` |
| `comm_volume` | 通信量(以 Ψ 为单位的倍数) | `2` / `3` |

**约束**：
- 列名、单位**一旦定下不要改**。下游已按此开发，改名会连锁炸掉可视化和报告。
- 缺失值用空字符串，不要用 `NaN`/`null` 字符串。
- 每跑完一组实验**追加**写入，不要覆盖历史行。

## 6. agent 工作约定

- **先跑通，再优化**：6/19 前，可读性 < 出数据。先要 baseline 能动。
- **小步验证**：每加一个 DeepSpeed stage，先用 1~2 个 step 跑通确认不 OOM，再全量跑。
- **OOM 是常态**：显存炸了优先调小 batch / 模型层数 / seq_len，而不是改测量逻辑。记录每次配置，别让"调参"污染对比的公平性（其它超参保持一致）。
- **数据可信度**：显存测量在 `torch.cuda.empty_cache()` 后、稳态若干 step 取峰值，避免冷启动噪声。
- **不要碰 `viz/`**：那是 B 的目录，除非被明确要求。
- **不要联网拉大模型权重**：用随机初始化或小数据集即可，目标是测系统行为不是测精度。

## 7. 关键背景知识（实验设计依据）

- ZeRO 三阶段显存（7.5B×64 卡论文参考值）：Baseline 120GB → ZeRO-1 ~31GB → ZeRO-2 ~16.6GB → ZeRO-3 ~1.9GB；通信量 ZeRO-1/2 为 2Ψ，ZeRO-3 为 3Ψ。
- 气泡率公式：`bubble = (K-1)/(M+K-1)`，K=流水线阶段数，M=微批次数；M 越大气泡越小。
- GPipe = 同步、每 mini-batch flush、有固定启停气泡；1F1B = 异步连续、稳态满载、气泡更低但有权重陈旧。
- 我们的小模型实测值**不会**等于论文的大模型数字，重点是**趋势一致**（stage 越高显存越省、M 越大气泡越小）。

## 8. 协作边界

| 角色 | 负责 | 与本 agent 的接口 |
|------|------|--------------------|
| **A（本 agent 服务对象）** | 代码 + 实验 + 数据 | 产出 `results/metrics.csv` |
| B | 可视化前端 + 系统手册 | 消费 `metrics.csv` |
| C | 课题报告 | 消费 `metrics.csv` + 实验日志 |
| D | 视频 + 整合 | 需要可运行的 demo |

下游全部通过 `metrics.csv` 解耦，因此**保证 CSV 正确、按时**比任何其它事都重要。