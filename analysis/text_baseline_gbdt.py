#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
from scripts import deconfound_conf_probe as _dc

SCORES = Path(os.environ.get("SST_DECONF_SCORES",
                             ROOT / "runs" / "conf_probe" / "blindspot_scores"))
MODELS, LOSSES, SEEDS = _dc.MODELS, _dc.LOSSES, _dc.SEEDS
DYN = _dc.DYN_PREDICTORS
ANSWER_COLS = ["dens_finalization", "finalization_timing"]
KEEP9 = [c for c in DYN if c not in ANSWER_COLS]
MARGIN = 0.05

GBDT_KW = dict(random_state=0, early_stopping=True, validation_fraction=0.1,
               n_iter_no_change=10)

def verdict(point: float, lo: float) -> str:
    if lo <= 0.0:
        return "matches"
    return "beats" if point >= MARGIN else "exceeds_without_margin"

def run_cell(model: str, loss: str, dyn: pd.DataFrame, cols: list[str],
             n_boot: int, rng_seed: int) -> dict:
    raw, surf = [], []
    boot_raw, boot_surf = [], []
    split_sig = None
    for seed in SEEDS:
        f = SCORES / f"{model}_{loss}_s{seed}.parquet"
        d = pd.read_parquet(f, columns=["trace_id", "problem_id", "Y", "c_T", "split"])
        d = d.merge(dyn, on="trace_id", how="left", validate="one_to_one")
        if d[cols].isna().any().any():
            raise ValueError(f"{f.name}: traces missing dynamics features")
        val = (d["split"] == "val").to_numpy()
        trn = ~val

        sig = (int(val.sum()), int(np.flatnonzero(val)[:64].sum()))
        if split_sig is None:
            split_sig = {}
        split_sig[seed] = sig

        y = d["Y"].to_numpy()
        c_T = d["c_T"].to_numpy(dtype=float)
        pcode = d["problem_id"].astype("category").cat.codes.to_numpy()
        X = d[cols].to_numpy(dtype=float)

        clf = HistGradientBoostingClassifier(**GBDT_KW).fit(X[trn], y[trn])
        s = clf.predict_proba(X[val])[:, 1]
        raw.append(roc_auc_score(y[val], c_T[val]))
        surf.append(roc_auc_score(y[val], s))
        boot_raw.append((pcode[val], y[val], c_T[val]))
        boot_surf.append((pcode[val], y[val], s))

    diff = _dc.cluster_ci_diff(boot_raw, boot_surf, n_boot, rng_seed)
    return {
        "n_columns": len(cols), "columns": cols,
        "raw_auroc": float(np.mean(raw)),
        "gbdt_surface_holdout": float(np.mean(surf)),
        "gbdt_surface_holdout_std": float(np.std(surf, ddof=1)),
        "gbdt_surface_holdout_per_seed": [float(x) for x in surf],
        "probe_minus_surface": diff,
        "verdict": verdict(diff["point"], diff["ci_lo"]),
        "split_signature": {str(k): list(v) for k, v in split_sig.items()},
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics-dir", type=Path, required=True)
    ap.add_argument("--logistic-ref", type=Path, required=True,
                    help="deconfounded_auroc_e1holdout.json for the same split")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    ref = json.load(open(a.logistic_ref))["per_cell"]
    out = {}
    for model in MODELS:
        dyn = pd.read_parquet(a.dynamics_dir / f"cot_dynamics_{model}.parquet",
                              columns=["trace_id"] + DYN)
        for loss in LOSSES:
            key = f"{model}_{loss}"
            out[key] = {}
            for tag, cols in (("d11", DYN), ("d9", KEEP9)):
                r = run_cell(model, loss, dyn, cols, a.n_boot, a.rng_seed)
                lin = ref[key]["surface_null_holdout"]
                r["logistic_holdout_11col"] = float(lin)
                r["gain_over_logistic"] = float(r["gbdt_surface_holdout"] - lin)
                out[key][tag] = r

            if loss == LOSSES[-1]:
                a_sig = out[f"{model}_{LOSSES[0]}"]["d11"]["split_signature"]
                b_sig = out[key]["d11"]["split_signature"]
                if a_sig != b_sig:
                    raise ValueError(f"{model}: split labels differ across losses")

    payload = {
        "description": ("E-1 nonlinear surface null: HistGradientBoosting on the "
                        "CoT-dynamics columns, fit on train rows, scored on val, "
                        "with the paired problem-clustered probe-minus-surface Δ."),
        "estimator": {"class": "HistGradientBoostingClassifier", **GBDT_KW},
        "scores_dir": str(SCORES), "n_boot": a.n_boot, "rng_seed": a.rng_seed,
        "margin": MARGIN, "per_cell": out,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
