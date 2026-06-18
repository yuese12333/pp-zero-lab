# pp-zero-lab

分布式训练显存与吞吐优化实验平台。在小型 GPT 上对比 ZeRO 各阶段显存占用，以及 GPipe / 1F1B 流水线调度的气泡率与吞吐。

仓库：https://github.com/yuese12333/pp-zero-lab

## 快速开始

**环境**：WSL2 + conda（`ppzero`）+ PyTorch（CUDA）+ DeepSpeed。完整部署见 [系统手册.md](系统手册.md)。

```bash
git clone https://github.com/yuese12333/pp-zero-lab.git
cd pp-zero-lab
conda activate ppzero

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 核心实验：覆盖 metrics.csv + 追加理论值
bash scripts/run_all.sh && python src/theory.py

# 扩展实验（追加 sweep.csv / combo.csv / tradeoff.csv）
python src/sweep.py
python src/combo.py
```

### 单个实验

```bash
# Baseline（单卡全量）
python src/train.py --config baseline --steps 20 --batch-size 4

# ZeRO（需 DeepSpeed；--deepspeed 为开关，json 由 --config 自动映射）
deepspeed --num_gpus=1 src/train.py --config zero1 --steps 20 --batch-size 4 --deepspeed
deepspeed --num_gpus=1 src/train.py --config zero2 --steps 20 --batch-size 4 --deepspeed
deepspeed --num_gpus=1 src/train.py --config zero3 --steps 20 --batch-size 4 --deepspeed

# 流水线仿真
python src/train.py --config gpipe --steps 20 --batch-size 4 --num-stages 4 --micro-batches 8
python src/train.py --config 1f1b  --steps 20 --batch-size 4 --num-stages 4 --micro-batches 8
```

> ZeRO 的 `configs/zeroN.json` 中 `train_batch_size` 须与 `--batch-size` 一致（默认均为 4）。

## 目录

| 路径 | 说明 |
|------|------|
| `src/train.py` | 训练入口，`--config` 切换实验 |
| `src/theory.py` | 多卡 ZeRO 理论显存（追加到 metrics.csv） |
| `src/sweep.py` | 气泡率 / 规模扫描 → `sweep.csv` |
| `src/combo.py` | PP×ZeRO 组合建模 → `combo.csv` / `tradeoff.csv` |
| `configs/` | DeepSpeed ZeRO JSON 配置 |
| `results/metrics.csv` | **下游主接口**（实测 + 理论） |
| `scripts/run_all.sh` | 一键跑核心实测（覆盖 metrics.csv） |
| `viz/` | 可视化前端（B 负责） |
| `系统手册.md` | 部署指南 + 完整接口说明 |

## 实验配置

| config | category | 说明 |
|--------|----------|------|
| `baseline` | `zero` | 单卡全量，stage=0 |
| `zero1` / `zero2` / `zero3` | `zero` | DeepSpeed ZeRO 各阶段 |
| `gpipe` / `1f1b` | `pipeline` | 流水线分段仿真 |
| `*_theory` | `zero_theory` | 多卡公式推算（`theory.py`） |

## 数据接口

- **主接口**：`results/metrics.csv`（列名固定，见 [CLAUDE.md](CLAUDE.md) §5）
- **扩展**：`sweep.csv`、`combo.csv`、`tradeoff.csv`（列定义见 [系统手册.md](系统手册.md)）

Agent 工作约定见 [CLAUDE.md](CLAUDE.md)。
