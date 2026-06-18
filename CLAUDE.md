# CLAUDE.md

> 本文件供项目中的 AI agent（Claude Code / Cowork 等）阅读，用于快速理解项目目标、技术栈与工作约定。人类开发者也可参考。

## 1. 项目目标

构建一个**分布式训练显存与吞吐优化实验平台**，在小型 Transformer（nanoGPT 风格 GPT）上实测并可视化对比：

- **流水线并行调度**：GPipe / 1F1B 的气泡率、吞吐（单卡分段前向仿真 + 公式）
- **ZeRO 显存优化**：Baseline vs ZeRO-1 / ZeRO-2 / ZeRO-3 的每卡显存占用
- **扩展分析**：多卡 ZeRO 理论值、参数扫描、PP×ZeRO 组合建模（见 `theory.py` / `sweep.py` / `combo.py`）
- **可视化前端**：气泡时间线、显存对比柱状图、吞吐曲线（B 负责）

最终交付：课题报告、系统手册、可视化系统、演示视频。本仓库主要承载**代码 + 实验数据 + 可视化**。

## 2. 当前 agent 的核心任务（A 的工作线）

你（agent）服务的开发者负责 **实验与代码（角色 A）**，是项目关键路径。核心目标按优先级：

1. **跑通 baseline**：单机/单卡训练循环能正常前向反向。
2. **接入 DeepSpeed**，依次跑通 ZeRO-1 / ZeRO-2 / ZeRO-3，采集每卡显存。
3. **实现/配置流水线调度**（GPipe 与 1F1B），采集气泡率与吞吐。
4. **导出标准化 CSV**（见 §5），`metrics.csv` 是交付给下游（可视化 B、报告 C）的**主接口**。
5. **扩展实验**（可选）：`theory.py` 多卡理论值、`sweep.py` 扫描曲线、`combo.py` 组合建模。

> ⏰ 硬截止：**6/19 前产出完整 CSV**。在此之前一切优先服务于"把数据跑出来"，而非代码优雅度。

> **运行环境**：ZeRO 实验需在 **WSL2 / Linux + CUDA + DeepSpeed** 下跑；Windows 原生 DeepSpeed 不可靠。详见 [系统手册.md](系统手册.md)。

## 3. 技术栈与环境

- **语言/框架**：Python 3.10+，PyTorch 2.x，DeepSpeed
- **模型**：自包含 nanoGPT 风格 GPT（`src/model.py`），随机初始化，~30M 参数默认配置
- **可视化（B 负责，agent 了解即可）**：前端 + 图表库（Plotly / Recharts / D3 任选）
- **显存测量**：`torch.cuda.max_memory_allocated()`，稳态 step 取峰值
- **ZeRO 优化器**：PyTorch `AdamW` + json 中 `"torch_adam": true`（避免 JIT 编译 fused_adam）
- **资源**：优先 WSL2 单卡；多卡不可用时可单卡测 ZeRO + 公式/理论脚本补充多卡趋势

## 4. 目录结构（约定，agent 应遵守）

```
pp-zero-lab/
├── CLAUDE.md                # 本文件
├── README.md                # 人类用快速说明
├── 系统手册.md              # 部署指南 + 完整接口说明（B 也可维护）
├── requirements.txt
├── configs/                 # DeepSpeed JSON 配置（train_batch_size=4）
│   ├── zero1.json
│   ├── zero2.json
│   └── zero3.json
├── src/
│   ├── model.py             # 模型定义
│   ├── train.py             # 训练入口，--config 切换实验
│   ├── pipeline.py          # 流水线分段前向仿真
│   ├── profiler.py          # 显存/吞吐/气泡率采集
│   ├── export.py            # 汇总实验结果 → metrics.csv
│   ├── theory.py            # 多卡 ZeRO 理论显存值
│   ├── sweep.py             # 参数扫描（bubble / scale）→ sweep.csv
│   └── combo.py             # PP×ZeRO 组合 + 权衡前沿 → combo.csv / tradeoff.csv
├── data/                    # 训练数据占位（当前用合成 token）
├── results/
│   ├── metrics.csv          # ★ 下游主接口（实测 + zero_theory）
│   ├── sweep.csv            # 扫描曲线（扩展，不改 metrics 列名）
│   ├── combo.csv            # PP×ZeRO 组合显存
│   └── tradeoff.csv         # 气泡-吞吐权衡网格
├── viz/                     # B 的可视化前端（agent 不主动改）
└── scripts/
    └── run_all.sh           # 一键跑核心实测（覆盖 metrics.csv）
```

## 5. 数据接口契约（最重要 — 不要擅自改列名）

### 5.1 `results/metrics.csv`（下游主接口）

列名锁定如下：

| 列名 | 含义 | 示例 |
|------|------|------|
| `config` | 实验配置名 | `baseline` / `zero1` / `zero2` / `zero3` / `gpipe` / `1f1b` / `*_theory` |
| `category` | 实验类别 | `zero` / `pipeline` / `zero_theory` |
| `stage` | ZeRO 阶段（流水线实验留空） | `0`/`1`/`2`/`3` |
| `num_gpus` | 卡数 | `1` / `4` / `8` / `16` / `64` |
| `micro_batches` | 微批次数 M（流水线实验用） | `8` |
| `mem_per_gpu_gb` | 每卡峰值显存(GB) | `2.68` |
| `throughput_samples_s` | 吞吐(samples/s) | `73.33` |
| `bubble_ratio` | 气泡率(0–1，流水线实验用) | `0.27` |
| `comm_volume` | 通信量(以 Ψ 为单位的倍数) | `2` / `3` |

**约束**：
- 列名、单位**一旦定下不要改**。下游已按此开发，改名会连锁炸掉可视化和报告。
- 缺失值用空字符串，不要用 `NaN`/`null` 字符串。
- **写入策略**：
  - `scripts/run_all.sh` → **覆盖** `metrics.csv`（先清空表头再写实测）。
  - `train.py` / `theory.py` 单次运行 → **追加**一行。
- `zero_theory` 行由 `theory.py` 生成，与单卡实测 `zero` 区分。

### 5.2 扩展 CSV（不改 metrics 列名，供报告/深入可视化）

| 文件 | 产出脚本 | 用途 |
|------|---------|------|
| `sweep.csv` | `sweep.py` | 气泡率/规模扫描曲线 |
| `combo.csv` | `combo.py` | PP×ZeRO 组合显存网格 |
| `tradeoff.csv` | `combo.py` | K×M 气泡-有效算力权衡 |

列定义见 [系统手册.md](系统手册.md) §5.2。扩展 CSV 均为**追加**写入。

## 6. agent 工作约定

- **先跑通，再优化**：可读性 < 出数据。先要 baseline 能动。
- **小步验证**：每加一个 DeepSpeed stage，先用少量 step 跑通确认不 OOM，再全量跑。
- **OOM 是常态**：显存炸了优先调小 batch / 模型层数 / seq_len，而不是改测量逻辑。记录每次配置，别让"调参"污染对比的公平性（其它超参保持一致）。
- **数据可信度**：显存测量在 `torch.cuda.empty_cache()` 后、稳态若干 step 取峰值，避免冷启动噪声。
- **DeepSpeed 命令**：`deepspeed ... --deepspeed`（无值开关）；json 路径由 `--config zeroN` 自动映射，**不要**写 `--deepspeed configs/zero1.json`。
- **batch 对齐**：`configs/zeroN.json` 的 `train_batch_size` 须等于 `batch_size × grad_accum × num_gpus`（单卡默认 4）。
- **不要碰 `viz/`**：那是 B 的目录，除非被明确要求。
- **不要联网拉大模型权重**：随机初始化 + 合成 token 即可，目标是测系统行为不是测精度。
- **Windows ↔ WSL**：在 Windows 侧改代码后，WSL 跑实验前需同步（`rsync` 或 `git pull`）。

## 7. 关键背景知识（实验设计依据）

- ZeRO 三阶段显存（7.5B×64 卡论文参考值）：Baseline 120GB → ZeRO-1 ~31GB → ZeRO-2 ~16.6GB → ZeRO-3 ~1.9GB；通信量 ZeRO-1/2 为 2Ψ，ZeRO-3 为 3Ψ。
- 气泡率公式：`bubble = (K-1)/(M+K-1)`，K=流水线阶段数，M=微批次数；M 越大气泡越小。
- 当前 `gpipe` / `1f1b` 为单卡**分段前向仿真**，气泡率用公式；尚未做严格 GPipe/1F1B 时间线调度区分。
- 单卡 ZeRO 显存可能不降反升（无法分片 + DeepSpeed 开销），多卡趋势见 `theory.py`。
- 小模型实测值**不会**等于论文大模型数字，重点是**趋势一致**。

## 8. 协作边界

| 角色 | 负责 | 与本 agent 的接口 |
|------|------|--------------------|
| **A（本 agent 服务对象）** | 代码 + 实验 + 数据 | 产出 `results/metrics.csv`（+ 扩展 CSV） |
| B | 可视化前端 + 系统手册 | 消费 `metrics.csv` 等 |
| C | 课题报告 | 消费 `metrics.csv` + 实验日志 |
| D | 视频 + 整合 | 需要可运行的 demo |

下游以 `metrics.csv` 为主接口解耦；扩展 CSV 供深入分析。**保证主 CSV 正确、按时**比任何其它事都重要。

## 9. 常用命令速查

```bash
# 核心实测（覆盖 metrics.csv）+ 理论值（追加）
bash scripts/run_all.sh && python src/theory.py

# 扩展实验（追加各自 CSV）
python src/sweep.py
python src/combo.py

# 单实验
deepspeed --num_gpus=1 src/train.py --config zero1 --steps 20 --batch-size 4 --deepspeed
python src/train.py --config gpipe --steps 20 --num-stages 4 --micro-batches 8
```
