"""创新实验：流水线并行 × ZeRO 组合显存建模 + 气泡-吞吐权衡分析。

实验 A — PP×ZeRO 正交组合显存模型
  流水线把模型切成 P 段，每段约 1/P 参数；段内再用 N_dp 路数据并行 + ZeRO 分片。
  对应汇报 PPT 第20页 Megatron-LM 的 3D 并行（节点内/跨节点/最外层），
  此处给出每卡模型状态显存的量化估算，回答"组合后到底省多少"。

  每卡显存(模型状态) = ψ_stage × per_param(zero_stage, N_dp)
  其中 ψ_stage = Ψ / P （流水线切分后单段参数量）

实验 B — 气泡率 × 吞吐 权衡前沿
  扫描 (K, M) 网格，输出气泡率与相对有效算力，刻画"增大 M / 减少 K"如何逼近零气泡。
  对应汇报 PPT 第21页 从 GPipe 到 Zero Bubble 的演进。

结果写入 results/combo.csv。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import build_model
from profiler import bubble_ratio_formula

COMBO_CSV = SRC_DIR.parent / "results" / "combo.csv"
K_OPT = 12          # Adam 优化器状态系数
BYTES_PER_GB = 1024 ** 3


def per_param_bytes(zero_stage: int, n_dp: int) -> float:
    """单参数模型状态字节数（参数2 + 梯度2 + 优化器状态K）。"""
    if zero_stage == 0:
        return 2 + 2 + K_OPT
    if zero_stage == 1:
        return 2 + 2 + K_OPT / n_dp
    if zero_stage == 2:
        return 2 + (2 + K_OPT) / n_dp
    if zero_stage == 3:
        return (2 + 2 + K_OPT) / n_dp
    raise ValueError(zero_stage)


def ensure_csv(path: Path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def write_rows(rows, header, path=COMBO_CSV):
    ensure_csv(path, header)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


HEADER_A = [
    "exp", "pipeline_stages_P", "zero_stage", "dp_degree_N",
    "total_gpus", "params_total", "params_per_stage",
    "mem_per_gpu_gb", "vs_baseline_x",
]
HEADER_B = [
    "exp", "num_stages_K", "micro_batches_M",
    "bubble_ratio", "effective_compute_pct",
]


def exp_a_combo(psi: int):
    """PP×ZeRO 组合显存网格。"""
    print("\n=== 实验A: PP × ZeRO 正交组合显存（每卡模型状态）===")
    baseline = psi * per_param_bytes(0, 1) / BYTES_PER_GB
    rows = []
    P_list = [1, 2, 4, 8]          # 流水线段数
    zero_list = [0, 1, 2, 3]       # ZeRO 阶段
    N_list = [1, 4, 8]             # 段内数据并行度
    for P in P_list:
        psi_stage = psi / P
        for zs in zero_list:
            for n in N_list:
                if zs == 0 and n != 1:
                    continue           # baseline 不分片，N 无意义
                mem = psi_stage * per_param_bytes(zs, n) / BYTES_PER_GB
                rows.append({
                    "exp": "combo",
                    "pipeline_stages_P": P,
                    "zero_stage": zs,
                    "dp_degree_N": n,
                    "total_gpus": P * n,
                    "params_total": psi,
                    "params_per_stage": int(psi_stage),
                    "mem_per_gpu_gb": round(mem, 4),
                    "vs_baseline_x": round(baseline / mem, 2) if mem > 0 else "",
                })
    write_rows(rows, HEADER_A)
    # 打印几个代表点
    for r in rows:
        if r["dp_degree_N"] in (1, 8) and r["pipeline_stages_P"] in (1, 4):
            print(f"  P={r['pipeline_stages_P']} ZeRO-{r['zero_stage']} "
                  f"N={r['dp_degree_N']} totGPU={r['total_gpus']:<3} "
                  f"mem={r['mem_per_gpu_gb']}GB  {r['vs_baseline_x']}x")
    print(f"  共 {len(rows)} 组组合写入")


def exp_b_tradeoff():
    """气泡-吞吐权衡前沿。"""
    print("\n=== 实验B: 气泡率 × 有效算力 权衡（K×M 网格）===")
    rows = []
    for K in (2, 4, 8, 16):
        for M in (1, 2, 4, 8, 16, 32, 64):
            b = bubble_ratio_formula(K, M)
            rows.append({
                "exp": "tradeoff",
                "num_stages_K": K,
                "micro_batches_M": M,
                "bubble_ratio": round(b, 4),
                "effective_compute_pct": round((1 - b) * 100, 2),
            })
    write_rows(rows, HEADER_B, COMBO_CSV.parent / "tradeoff.csv")
    for r in rows:
        if r["micro_batches_M"] in (1, 8, 32):
            print(f"  K={r['num_stages_K']:<2} M={r['micro_batches_M']:<3} "
                  f"bubble={r['bubble_ratio']:.4f} "
                  f"eff={r['effective_compute_pct']}%")
    print(f"  共 {len(rows)} 组写入 tradeoff.csv")


def main():
    model = build_model(n_layer=6, n_embd=384, block_size=128)
    psi = model.get_num_params()
    print(f"模型参数量 Ψ = {psi:,}")
    exp_a_combo(psi)
    exp_b_tradeoff()
    print(f"\n完成。结果见 {COMBO_CSV} 和 tradeoff.csv")


if __name__ == "__main__":
    main()
