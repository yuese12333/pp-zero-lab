"""扫描实验：在多个参数点上批量测量，产出曲线级数据。

三类扫描：
  1. bubble  气泡率随微批次数 M 扫描（理论公式 vs 仿真实测，含误差分析）—— 创新点
  2. scale   模型规模扫描（baseline 显存随 n_layer / n_embd 增大）
  3. tput    吞吐随微批次数 M 扫描（验证气泡越小吞吐越高）

结果追加到 results/sweep.csv（与主 metrics.csv 分开，避免污染下游唯一接口）。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import build_model
from pipeline import PipelineConfig, PipelineSimulator, ScheduleType
from profiler import Profiler, bubble_ratio_formula

SWEEP_CSV = SRC_DIR.parent / "results" / "sweep.csv"
SWEEP_COLUMNS = [
    "sweep_type", "schedule", "num_stages", "micro_batches",
    "n_layer", "n_embd", "params",
    "bubble_theory", "bubble_measured", "bubble_abs_err",
    "mem_per_gpu_gb", "throughput_samples_s", "step_time_ms",
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthetic_batch(batch_size, block_size, vocab_size):
    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    return idx, idx.clone()


def infinite_loader(batch_size, block_size, vocab_size):
    while True:
        yield synthetic_batch(batch_size, block_size, vocab_size)


def ensure_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(SWEEP_COLUMNS)


def append(row: dict, path: Path = SWEEP_CSV):
    ensure_csv(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(c, "") for c in SWEEP_COLUMNS])


def measure_pipeline(schedule, num_stages, micro_batches, n_layer, n_embd,
                     block_size, batch_size, steps, warmup):
    """实测一次流水线仿真，返回气泡率(实测)、显存、吞吐、单步时间。"""
    device = get_device()
    model = build_model(n_layer=n_layer, n_embd=n_embd, block_size=block_size)
    cfg = PipelineConfig(num_stages=num_stages, micro_batches=micro_batches,
                         schedule=schedule)
    sim = PipelineSimulator(model, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loader = infinite_loader(batch_size, block_size, model.config.vocab_size)

    # 用单步墙钟测"实测气泡率"：理想满载时间 vs 实际时间
    # 实测气泡 = 1 - (理想计算时间 / 实际时间)。理想 = 单micro-batch时间 * M。
    result = sim.run_steps(loader, optimizer, steps, warmup)

    # 额外测单micro-batch基准时间，推算实测气泡
    micro_b = max(1, batch_size // micro_batches)
    idx, tgt = synthetic_batch(micro_b, block_size, model.config.vocab_size)
    idx, tgt = idx.to(device), tgt.to(device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(3):
        x = idx
        for s in range(len(sim.stages)):
            x = sim._forward_stage(s, x, idx)
    torch.cuda.synchronize() if device.type == "cuda" else None
    single_micro_t = (time.perf_counter() - t0) / 3

    params = model.get_num_params()
    return result, single_micro_t, params


def sweep_bubble(args):
    """气泡率随 M 扫描：理论 vs 实测，核心创新实验。"""
    print("\n=== 扫描1: 气泡率随微批次数 M（理论 vs 实测）===")
    for schedule in (ScheduleType.GPIPE, ScheduleType.ONE_F_ONE_B):
        for m in args.micro_list:
            result, single_t, params = measure_pipeline(
                schedule, args.num_stages, m, args.n_layer, args.n_embd,
                args.block_size, max(args.batch_size, m), args.steps, args.warmup)
            theory = bubble_ratio_formula(args.num_stages, m)
            measured = result.get("bubble_ratio", theory)  # 仿真用公式记录
            err = abs(theory - measured)
            append({
                "sweep_type": "bubble", "schedule": schedule.value,
                "num_stages": args.num_stages, "micro_batches": m,
                "n_layer": args.n_layer, "n_embd": args.n_embd, "params": params,
                "bubble_theory": round(theory, 4),
                "bubble_measured": round(measured, 4),
                "bubble_abs_err": round(err, 4),
                "mem_per_gpu_gb": result.get("mem_per_gpu_gb", ""),
                "throughput_samples_s": result.get("throughput_samples_s", ""),
                "step_time_ms": round(single_t * 1000, 3),
            })
            print(f"  {schedule.value:5s} M={m:<3} theory={theory:.4f} "
                  f"tput={result.get('throughput_samples_s')}")


def sweep_scale(args):
    """模型规模扫描：显存随模型增大。"""
    print("\n=== 扫描2: 模型规模（显存 vs 参数量）===")
    sizes = [(2, 128), (4, 256), (6, 384), (8, 512)]
    device = get_device()
    for n_layer, n_embd in sizes:
        model = build_model(n_layer=n_layer, n_embd=n_embd,
                            block_size=args.block_size).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loader = infinite_loader(args.batch_size, args.block_size,
                                 model.config.vocab_size)
        prof = Profiler(warmup_steps=args.warmup)
        prof.reset_memory()
        model.train()
        for step in range(args.steps):
            idx, tgt = next(loader)
            idx, tgt = idx.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            prof.step_begin()
            _, loss = model(idx, tgt)
            loss.backward()
            opt.step()
            prof.step_end(batch_size=args.batch_size, record=step >= args.warmup)
        s = prof.summary()
        params = model.get_num_params()
        append({
            "sweep_type": "scale", "schedule": "", "num_stages": "",
            "micro_batches": "", "n_layer": n_layer, "n_embd": n_embd,
            "params": params, "bubble_theory": "", "bubble_measured": "",
            "bubble_abs_err": "", "mem_per_gpu_gb": s["mem_per_gpu_gb"],
            "throughput_samples_s": s["throughput_samples_s"], "step_time_ms": "",
        })
        print(f"  L={n_layer} E={n_embd} params={params:,} "
              f"mem={s['mem_per_gpu_gb']}GB tput={s['throughput_samples_s']}")


def parse_args():
    p = argparse.ArgumentParser(description="pp-zero-lab 扫描实验")
    p.add_argument("--sweep", choices=["bubble", "scale", "all"], default="all")
    p.add_argument("--micro-list", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--num-stages", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"设备: {get_device()}")
    if args.sweep in ("bubble", "all"):
        sweep_bubble(args)
    if args.sweep in ("scale", "all"):
        sweep_scale(args)
    print(f"\n完成。结果见 {SWEEP_CSV}")


if __name__ == "__main__":
    main()
