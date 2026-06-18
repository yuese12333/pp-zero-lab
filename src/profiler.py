"""显存、吞吐、气泡率采集。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


def bubble_ratio_formula(num_stages: int, micro_batches: int) -> float:
    """GPipe / 1F1B 稳态气泡率近似: (K-1)/(M+K-1)。"""
    k, m = num_stages, micro_batches
    if k <= 1 or m <= 0:
        return 0.0
    return (k - 1) / (m + k - 1)


def comm_volume_for_zero(stage: int) -> str:
    if stage == 0:
        return ""
    if stage in (1, 2):
        return "2"
    if stage == 3:
        return "3"
    return ""


@dataclass
class Profiler:
    warmup_steps: int = 2
    _step_times: list[float] = field(default_factory=list)
    _samples: int = 0
    _peak_mem_bytes: int = 0

    def reset_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def step_begin(self) -> None:
        self._t0 = time.perf_counter()

    def step_end(self, batch_size: int, record: bool = True) -> None:
        elapsed = time.perf_counter() - self._t0
        if record:
            self._step_times.append(elapsed)
            self._samples += batch_size
        if torch.cuda.is_available():
            self._peak_mem_bytes = max(
                self._peak_mem_bytes, torch.cuda.max_memory_allocated()
            )

    @property
    def peak_mem_gb(self) -> float:
        return self._peak_mem_bytes / (1024**3)

    @property
    def throughput_samples_s(self) -> float:
        if not self._step_times:
            return 0.0
        total_time = sum(self._step_times)
        if total_time <= 0:
            return 0.0
        return self._samples / total_time

    def summary(self) -> dict:
        return {
            "mem_per_gpu_gb": round(self.peak_mem_gb, 4),
            "throughput_samples_s": round(self.throughput_samples_s, 2),
        }
