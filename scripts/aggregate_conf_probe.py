from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ("ece", "ece_temp", "brier", "auroc", "aupr", "acc_at_80")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/conf_probe"))
    ap.add_argument("--out", type=Path, default=Path("runs/conf_probe/summary.csv"))
    ap.add_argument("--regime", default="as_written", choices=["as_written", "strict", "pre_answer"])
    args = ap.parse_args()

    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seeds: dict[tuple[str, str], list[int]] = defaultdict(list)
    for mf in sorted(glob.glob(str(args.root / "qwen3-*/seed*/*/metrics.json"))):
        m = json.loads(Path(mf).read_text())
        key = (m["model"], m["loss"])
        reg = m["regimes"][args.regime]
        for k in KEYS:
            if k in reg and reg[k] == reg[k]:
                acc[key][k].append(float(reg[k]))
        seeds[key].append(int(m["seed"]))

    rows = []
    for (model, loss), md in sorted(acc.items()):
        row = {"model": model, "loss": loss, "regime": args.regime,
               "n_seeds": len(seeds[(model, loss)])}
        for k in KEYS:
            v = np.array(md.get(k, []))
            row[f"{k}_mean"] = float(v.mean()) if len(v) else float("nan")
            row[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)

    if not rows:
        print(f"[aggregate] no metrics.json under {args.root}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
