"""训练入口：--config 切换 baseline / ZeRO / 流水线实验。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# 允许从项目根或 src 目录运行
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from export import append_row
from model import build_model
from pipeline import PipelineConfig, PipelineSimulator, ScheduleType
from profiler import Profiler, comm_volume_for_zero

ZERO_CONFIGS = {
    "zero1": ("zero", 1, "configs/zero1.json"),
    "zero2": ("zero", 2, "configs/zero2.json"),
    "zero3": ("zero", 3, "configs/zero3.json"),
}

PIPELINE_CONFIGS = {"gpipe": ScheduleType.GPIPE, "1f1b": ScheduleType.ONE_F_ONE_B}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthetic_batch(batch_size: int, block_size: int, vocab_size: int):
    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = idx.clone()
    return idx, targets


def infinite_loader(batch_size: int, block_size: int, vocab_size: int):
    while True:
        yield synthetic_batch(batch_size, block_size, vocab_size)


def run_baseline(args: argparse.Namespace) -> dict:
    device = get_device()
    model = build_model(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        block_size=args.block_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = infinite_loader(args.batch_size, args.block_size, model.config.vocab_size)
    profiler = Profiler(warmup_steps=args.warmup_steps)
    profiler.reset_memory()
    model.train()

    for step in range(args.steps):
        idx, targets = next(loader)
        idx, targets = idx.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        profiler.step_begin()
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        profiler.step_end(batch_size=args.batch_size, record=step >= args.warmup_steps)

    print(f"[baseline] params={model.get_num_params():,} device={device}")
    return profiler.summary()


def run_zero(args: argparse.Namespace, stage: int, ds_config_path: str) -> dict:
    import deepspeed

    device = get_device()
    if not torch.cuda.is_available():
        print("[warn] ZeRO 实验建议 CUDA；当前无 GPU，回退 baseline 逻辑。")
        return run_baseline(args)

    model = build_model(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        block_size=args.block_size,
    )
    root = SRC_DIR.parent
    ds_path = root / ds_config_path
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        config=str(ds_path),
    )
    loader = infinite_loader(
        args.batch_size, args.block_size, model.config.vocab_size
    )
    profiler = Profiler(warmup_steps=args.warmup_steps)
    profiler.reset_memory()
    model_engine.train()

    for step in range(args.steps):
        idx, targets = next(loader)
        idx, targets = idx.to(device), targets.to(device)
        profiler.step_begin()
        _, loss = model_engine(idx, targets)
        model_engine.backward(loss)
        model_engine.step()
        profiler.step_end(batch_size=args.batch_size, record=step >= args.warmup_steps)

    print(f"[zero{stage}] params={model.get_num_params():,} device={device}")
    return profiler.summary()


def run_pipeline(args: argparse.Namespace, schedule: ScheduleType) -> dict:
    device = get_device()
    model = build_model(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        block_size=args.block_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = infinite_loader(args.batch_size, args.block_size, model.config.vocab_size)

    pipe_config = PipelineConfig(
        num_stages=args.num_stages,
        micro_batches=args.micro_batches,
        schedule=schedule,
    )
    sim = PipelineSimulator(model, pipe_config, device)
    result = sim.run_steps(loader, optimizer, args.steps, args.warmup_steps)
    print(
        f"[{schedule.value}] stages={args.num_stages} "
        f"micro_batches={args.micro_batches} bubble={result.get('bubble_ratio')}"
    )
    return result


def build_result_row(args: argparse.Namespace, metrics: dict) -> dict:
    config = args.config
    if config == "baseline":
        category, stage = "zero", "0"
        comm = ""
    elif config in ZERO_CONFIGS:
        category, stage, _ = ZERO_CONFIGS[config]
        comm = comm_volume_for_zero(int(stage))
    else:
        category, stage, comm = "pipeline", "", ""

    num_gpus = args.num_gpus
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        num_gpus = max(num_gpus, torch.cuda.device_count())

    return {
        "config": config,
        "category": category,
        "stage": stage,
        "num_gpus": num_gpus,
        "micro_batches": args.micro_batches if category == "pipeline" else "",
        "mem_per_gpu_gb": metrics.get("mem_per_gpu_gb", ""),
        "throughput_samples_s": metrics.get("throughput_samples_s", ""),
        "bubble_ratio": metrics.get("bubble_ratio", ""),
        "comm_volume": comm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pp-zero-lab training")
    parser.add_argument(
        "--config",
        choices=["baseline", "zero1", "zero2", "zero3", "gpipe", "1f1b"],
        default="baseline",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--micro-batches", type=int, default=8)
    parser.add_argument("--no-export", action="store_true", help="不写入 metrics.csv")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed_argparser(parser)
    return parser.parse_args()


def deepspeed_argparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    try:
        import deepspeed

        parser = deepspeed.add_config_arguments(parser)
    except ImportError:
        pass
    return parser


def main() -> None:
    args = parse_args()
    print(f"Running config={args.config} steps={args.steps} batch_size={args.batch_size}")

    if args.config == "baseline":
        metrics = run_baseline(args)
    elif args.config in ZERO_CONFIGS:
        _, stage, ds_path = ZERO_CONFIGS[args.config]
        metrics = run_zero(args, stage, ds_path)
    elif args.config in PIPELINE_CONFIGS:
        metrics = run_pipeline(args, PIPELINE_CONFIGS[args.config])
    else:
        raise ValueError(f"Unknown config: {args.config}")

    print(f"Metrics: {metrics}")
    if not args.no_export:
        row = build_result_row(args, metrics)
        append_row(row)
        print(f"Appended to results/metrics.csv: {row}")


if __name__ == "__main__":
    main()
