"""GPipe / 1F1B 流水线调度封装（仿真 + 可选分段前向）。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from profiler import Profiler, bubble_ratio_formula


class ScheduleType(str, Enum):
    GPIPE = "gpipe"
    ONE_F_ONE_B = "1f1b"


@dataclass
class PipelineConfig:
    num_stages: int = 4
    micro_batches: int = 8
    schedule: ScheduleType = ScheduleType.GPIPE


def split_model_stages(model: nn.Module, num_stages: int) -> list[nn.Module]:
    """将 GPT 的 transformer block 均分到 num_stages 段。"""
    blocks = list(model.transformer.h)
    n = len(blocks)
    if num_stages <= 0 or num_stages > n:
        raise ValueError(f"num_stages must be in [1, {n}], got {num_stages}")

    chunk_size = n // num_stages
    remainder = n % num_stages
    stages: list[nn.Module] = []
    start = 0
    for i in range(num_stages):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        stage_blocks = blocks[start:end]
        stages.append(nn.Sequential(*stage_blocks))
        start = end
    return stages


class PipelineSimulator:
    """在无多卡环境下用分段前向模拟流水线步进，并记录吞吐。"""

    def __init__(
        self,
        model: nn.Module,
        pipe_config: PipelineConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = pipe_config
        self.device = device
        self.stages = split_model_stages(model, pipe_config.num_stages)
        for s in self.stages:
            s.to(device)
        self.profiler = Profiler()

    def _forward_stage(
        self, stage_idx: int, x: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        if stage_idx == 0:
            b, t = idx.size()
            pos = torch.arange(0, t, dtype=torch.long, device=self.device).unsqueeze(0)
            tok_emb = self.model.transformer.wte(idx)
            pos_emb = self.model.transformer.wpe(pos)
            x = self.model.transformer.drop(tok_emb + pos_emb)

        x = self.stages[stage_idx](x)

        if stage_idx == len(self.stages) - 1:
            x = self.model.transformer.ln_f(x)
        return x

    def train_step(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        m = self.config.micro_batches
        b = idx.size(0)
        micro_b = max(1, b // m)
        total_loss = torch.tensor(0.0, device=self.device)

        for micro in range(m):
            start = micro * micro_b
            end = start + micro_b if micro < m - 1 else b
            if start >= b:
                break
            micro_idx = idx[start:end]
            micro_targets = targets[start:end]

            x = micro_idx  # type: ignore[assignment]
            for s in range(len(self.stages)):
                x = self._forward_stage(s, x, micro_idx)  # type: ignore[arg-type]

            logits = self.model.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), micro_targets.view(-1)
            )
            total_loss = total_loss + loss
            loss.backward()

        return total_loss / max(1, min(m, b))

    def run_steps(
        self, data_loader, optimizer, num_steps: int, warmup_steps: int = 2
    ) -> dict:
        self.profiler.reset_memory()
        self.model.train()

        for step in range(num_steps):
            idx, targets = next(data_loader)
            idx = idx.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            self.profiler.step_begin()
            self.train_step(idx, targets)
            optimizer.step()
            self.profiler.step_end(
                batch_size=idx.size(0), record=step >= warmup_steps
            )

        bubble = bubble_ratio_formula(
            self.config.num_stages, self.config.micro_batches
        )
        result = self.profiler.summary()
        result["bubble_ratio"] = round(bubble, 4)
        return result
