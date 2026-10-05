"""Fraction of pit traces whose failure step (crossing rule) is the first step attaining the minimum step success."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CAMP = ROOT / "runs/hf_pull/mc_rollouts_locus_k16/qwen3-{scale}/campaign"


def same_step_fraction(scale: str) -> tuple[float, int]:
    """Fraction of pit traces whose failure step is the first argmin of step success, and their number."""
    camp = Path(str(CAMP).format(scale=scale))
    labels = json.loads((camp / "gold_pits.json").read_text())
    steps = torch.load(camp / "mc_ground_truth.pt", weights_only=False)
    agree = []
    for trace_id, row in labels.items():
        if row["gold_pit"] is None:
            continue
        visited = sorted((int(t), v["p_correct_mc"]) for t, v in steps[trace_id].items())
        argmin_step = visited[int(np.argmin([p for _, p in visited]))][0]
        agree.append(argmin_step == row["gold_pit"])
    return float(np.mean(agree)), len(agree)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", nargs="+", default=["4b", "8b"])
    args = ap.parse_args()
    for scale in args.scales:
        frac, n = same_step_fraction(scale)
        print(f"{scale}: crossing rule versus argmin_t V, same step {frac:.3f} ({n} pit traces)")


if __name__ == "__main__":
    main()
