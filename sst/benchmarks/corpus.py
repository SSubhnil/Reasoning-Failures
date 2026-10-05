from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch

@dataclass(frozen=True, slots=True)
class BenchmarkTrace:
    trace_id: str
    problem_id: str
    question: str
    trace: str
    ground_truth: str
    extracted_steps: list[str]

    step_spans: list[tuple[int, int]]
    is_correct: bool
    hidden_states: torch.Tensor
    canonical_length: int
    dataset: str
    finish_reason: str

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkTrace:
        return cls(**{f.name: d[f.name] for f in fields(cls)})

def load_corpus(corpus_dir: str | Path) -> list[BenchmarkTrace]:
    corpus_dir = Path(corpus_dir)
    shard_paths = sorted(corpus_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {corpus_dir}")

    traces: list[BenchmarkTrace] = []
    for sp in shard_paths:
        shard_dicts: list[dict] = torch.load(sp, map_location="cpu", weights_only=False)
        for d in shard_dicts:
            traces.append(BenchmarkTrace.from_dict(d))
    return traces
