"""汇总实验结果 → results/metrics.csv（下游唯一接口）。"""

from __future__ import annotations

import csv
from pathlib import Path

CSV_COLUMNS = [
    "config",
    "category",
    "stage",
    "num_gpus",
    "micro_batches",
    "mem_per_gpu_gb",
    "throughput_samples_s",
    "bubble_ratio",
    "comm_volume",
]

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "results" / "metrics.csv"


def ensure_csv(path: Path | str = DEFAULT_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
    return path


def append_row(row: dict, path: Path | str = DEFAULT_CSV) -> None:
    path = ensure_csv(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([str(row.get(col, "")) for col in CSV_COLUMNS])
