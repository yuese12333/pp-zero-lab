# pp-zero-lab

分布式训练显存与吞吐优化实验平台。在小型 Transformer 上对比 ZeRO 各阶段显存占用，以及 GPipe / 1F1B 流水线调度的气泡率与吞吐。

## 快速开始

```bash
pip install -r requirements.txt

# 单卡 baseline（CPU 也可跑，显存指标为 0）
python src/train.py --config baseline --steps 10

# ZeRO 实验（需 CUDA + DeepSpeed）
deepspeed --num_gpus=1 src/train.py --config zero1 --steps 10 --batch-size 4 --deepspeed

# 流水线仿真（无需多卡）
python src/train.py --config gpipe --steps 10 --num-stages 4 --micro-batches 8
python src/train.py --config 1f1b --steps 10 --num-stages 4 --micro-batches 8

# 一键跑全部实验（Linux/macOS）
bash scripts/run_all.sh
```

## 目录

| 路径 | 说明 |
|------|------|
| `src/train.py` | 训练入口，`--config` 切换实验 |
| `configs/` | DeepSpeed ZeRO JSON 配置 |
| `results/metrics.csv` | 实验结果（下游唯一接口） |
| `viz/` | 可视化前端（B 负责） |

## 实验配置

| config | 类别 | 说明 |
|--------|------|------|
| `baseline` | zero | 无 ZeRO，stage=0 |
| `zero1` / `zero2` / `zero3` | zero | DeepSpeed ZeRO 各阶段 |
| `gpipe` / `1f1b` | pipeline | 流水线调度（仿真 + 可选实测） |

详细约定见 [CLAUDE.md](CLAUDE.md)。
