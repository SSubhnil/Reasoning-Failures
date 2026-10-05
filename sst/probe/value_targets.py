from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch

@dataclass(frozen=True, slots=True)
class McTargets:
    md5: str
    trace_id: str
    is_correct: bool
    T: int
    steps: tuple[int, ...]
    p: tuple[float, ...]
    k: tuple[int, ...]

def _iter_steps_files(paths: Path | str | Iterable[Path | str]) -> Iterator[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in paths:
        p = Path(p)
        if p.is_dir():
            direct = p / "steps.jsonl"
            if direct.exists():
                yield direct
            yield from sorted(p.glob("slice_*/steps.jsonl"))
        else:
            yield p

def load_mc_targets(paths: Path | str | Iterable[Path | str]) -> dict[str, McTargets]:
    out: dict[str, McTargets] = {}
    seen: set[str] = set()
    for sf in _iter_steps_files(paths):
        with open(sf) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                tid = r["trace_id"]
                if tid in seen:
                    continue
                seen.add(tid)
                items = sorted((int(s), v) for s, v in r["steps"].items())
                out[r["md5"]] = McTargets(
                    md5=r["md5"],
                    trace_id=tid,
                    is_correct=bool(r["is_correct"]),
                    T=int(r["T"]),
                    steps=tuple(s for s, _ in items),
                    p=tuple(float(v["p_correct_mc"]) for _, v in items),
                    k=tuple(int(v["n_rollouts"]) for _, v in items),
                )
    return out

def load_mc_step_targets(paths: Path | str) -> dict[str, "McTargets"]:
    campaign_dir = Path(paths)
    by_md5 = load_mc_targets(campaign_dir)
    if by_md5:
        return by_md5
    gt_path = campaign_dir / "mc_ground_truth.pt"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"No slice_*/steps.jsonl and no mc_ground_truth.pt under {campaign_dir}"
        )
    gt = torch.load(gt_path, weights_only=False)
    out: dict[str, McTargets] = {}
    for tid, steps in gt.items():
        items = sorted((int(s), v) for s, v in steps.items())
        out[tid] = McTargets(
            md5="", trace_id=tid, is_correct=False,
            T=(items[-1][0] + 1 if items else 0),
            steps=tuple(s for s, _ in items),
            p=tuple(float(v["p_correct_mc"]) for _, v in items),
            k=tuple(int(v["n_rollouts"]) for _, v in items),
        )
    return out

def reachability_ceiling(t: McTargets) -> float:
    return max(t.p) if t.p else 0.0

def is_recoverable(t: McTargets, theta: float = 0.1) -> bool:
    return reachability_ceiling(t) > theta

def build_target_row(
    t: McTargets, Lmax: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    target = torch.zeros(Lmax)
    k = torch.zeros(Lmax)
    mask = torch.zeros(Lmax, dtype=torch.bool)
    for s, pi, ki in zip(t.steps, t.p, t.k):
        if 0 <= s < Lmax:
            target[s] = float(pi)
            k[s] = float(ki)
            mask[s] = True
    return target, k, mask
