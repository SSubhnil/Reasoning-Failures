from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

@dataclass(frozen=True, slots=True)
class GenerationOutput:
    traces: list[str]
    extracted_steps: list[list[str]]
    hidden_states: torch.Tensor
    transcoder_z: dict[str, torch.Tensor]
    target_states: Optional[torch.Tensor]
    step_spans: list[list[tuple[int, int]]]
    finish_reasons: list[str]
    rescue_counts: dict[str, int] | None = None

    token_counts: list[int] | None = None
