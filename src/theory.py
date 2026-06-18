"""多卡 ZeRO 理论显存值生成（依据 ZeRO 论文 SC 2020 显存公式）。

为单卡实测补充多卡趋势：展示 stage 越高、卡数 N 越大，模型状态显存越接近 1/N。
理论行 category='zero_theory'，与单卡实测 category='zero' 区分。

显存公式（每参数，Adam fp16 训练 K=12）：
  Baseline      (2+2+K)*Psi
  ZeRO-1        2*Psi + 2*Psi + K*Psi/N
  ZeRO-2        2*Psi + (2+K)*Psi/N
  ZeRO-3        (2+2+K)*Psi/N
"""
from __future__ import annotations
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from export import append_row
from model import build_model

K = 12
BYTES_PER_GB = 1024 ** 3
GPU_COUNTS = [1, 4, 8, 16, 64]


def model_state_gb(psi: int, stage: int, n: int) -> float:
    if stage == 0:
        per_param = (2 + 2 + K)
    elif stage == 1:
        per_param = 2 + 2 + K / n
    elif stage == 2:
        per_param = 2 + (2 + K) / n
    elif stage == 3:
        per_param = (2 + 2 + K) / n
    else:
        raise ValueError(stage)
    return psi * per_param / BYTES_PER_GB


def comm_volume(stage: int) -> str:
    if stage in (1, 2):
        return "2"
    if stage == 3:
        return "3"
    return ""


def main() -> None:
    model = build_model(n_layer=6, n_embd=384, block_size=128)
    psi = model.get_num_params()
    print(f"模型参数量 Psi = {psi:,}")
    rows = []
    for stage in (0, 1, 2, 3):
        for n in GPU_COUNTS:
            if stage == 0 and n != 1:
                continue
            mem = model_state_gb(psi, stage, n)
            config = "baseline" if stage == 0 else f"zero{stage}"
            rows.append({
                "config": f"{config}_theory",
                "category": "zero_theory",
                "stage": stage,
                "num_gpus": n,
                "micro_batches": "",
                "mem_per_gpu_gb": round(mem, 4),
                "throughput_samples_s": "",
                "bubble_ratio": "",
                "comm_volume": comm_volume(stage),
            })
    for r in rows:
        append_row(r)
        print(f"appended: {r['config']:18s} N={r['num_gpus']:<3} mem={r['mem_per_gpu_gb']} GB")
    print(f"\n共追加 {len(rows)} 行理论值到 results/metrics.csv")


if __name__ == "__main__":
    main()
