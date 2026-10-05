#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
from scripts import deconfound_conf_probe as _dc

SCORES = Path(os.environ.get("SST_DECONF_SCORES",
                             ROOT / "runs" / "conf_probe" / "blindspot_scores"))
MODELS, LOSSES, SEEDS = _dc.MODELS, _dc.LOSSES, _dc.SEEDS
DYN = _dc.DYN_PREDICTORS
ANSWER_COLS = ["dens_finalization", "finalization_timing"]
KEEP = [c for c in DYN if c not in ANSWER_COLS]

MARGIN = 0.05

def verdict(point: float, lo: float) -> str:
    if lo <= 0.0:
        return "matches"
    return "beats" if point >= MARGIN else "exceeds_without_margin"

def run_cell(model: str, loss: str, dyn: pd.DataFrame, n_boot: int,
             rng_seed: int) -> dict:
    raw, surf = [], []
    boot_raw, boot_surf = [], []
    for seed in SEEDS:
        f = SCORES / f"{model}_{loss}_s{seed}.parquet"
        d = pd.read_parquet(f, columns=["trace_id", "problem_id", "Y", "c_T", "split"])
        d = d.merge(dyn, on="trace_id", how="left", validate="one_to_one")
        if d[KEEP].isna().any().any():
            raise ValueError(f"{f.name}: traces missing dynamics features")
        val = (d["split"] == "val").to_numpy()
        trn = ~val
        y = d["Y"].to_numpy()
        c_T = d["c_T"].to_numpy(dtype=float)
        pcode = d["problem_id"].astype("category").cat.codes.to_numpy()

        X = d[KEEP].to_numpy(dtype=float)
        mu, sd = X[val].mean(axis=0), X[val].std(axis=0)
        if (sd == 0).any():
            raise ValueError(f"{f.name}: zero-variance dynamics column on val rows")
        Xs = (X - mu) / sd

        lg = LogisticRegression(max_iter=2000).fit(Xs[trn], y[trn])
        s = lg.decision_function(Xs[val])
        raw.append(roc_auc_score(y[val], c_T[val]))
        surf.append(roc_auc_score(y[val], s))
        boot_raw.append((pcode[val], y[val], c_T[val]))
        boot_surf.append((pcode[val], y[val], s))

    diff = _dc.cluster_ci_diff(boot_raw, boot_surf, n_boot, rng_seed)
    return {
        "columns": KEEP,
        "dropped_answer_columns": ANSWER_COLS,
        "raw_auroc": float(np.mean(raw)),
        "surface_null_holdout_noanswer": float(np.mean(surf)),
        "surface_null_holdout_noanswer_std": float(np.std(surf, ddof=1)),
        "surface_null_holdout_noanswer_per_seed": [float(x) for x in surf],
        "probe_minus_surface": diff,
        "verdict": verdict(diff["point"], diff["ci_lo"]),
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    out = {}
    for model in MODELS:
        dyn = pd.read_parquet(a.dynamics_dir / f"cot_dynamics_{model}.parquet",
                              columns=["trace_id"] + DYN)
        for loss in LOSSES:
            key = f"{model}_{loss}"
            r = run_cell(model, loss, dyn, a.n_boot, a.rng_seed)
            out[key] = r

    payload = {
        "description": ("E-1 leakage variant: held-out CoT-dynamics surface null "
                        "with the two answer-cue columns dropped, and the paired "
                        "problem-clustered probe-minus-surface difference."),
        "scores_dir": str(SCORES), "n_boot": a.n_boot, "rng_seed": a.rng_seed,
        "margin": MARGIN, "per_cell": out,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
