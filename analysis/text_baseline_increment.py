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
logit = _dc.logit

GBDT_KW = dict(random_state=0, early_stopping=True, validation_fraction=0.1,
               n_iter_no_change=10)

def run_cell(model: str, loss: str, dyn: pd.DataFrame, n_boot: int,
             rng_seed: int) -> dict:
    A, B, C = [], [], []
    probe_train_auc = []
    bA, bB, bC = [], [], []
    for seed in SEEDS:
        f = SCORES / f"{model}_{loss}_s{seed}.parquet"
        d = pd.read_parquet(f, columns=["trace_id", "problem_id", "Y", "c_T", "split"])
        d = d.merge(dyn, on="trace_id", how="left", validate="one_to_one")
        if d[DYN].isna().any().any():
            raise ValueError(f"{f.name}: traces missing dynamics features")
        val = (d["split"] == "val").to_numpy()
        trn = ~val
        y = d["Y"].to_numpy()
        c_T = d["c_T"].to_numpy(dtype=float)
        z = logit(c_T)
        pcode = d["problem_id"].astype("category").cat.codes.to_numpy()

        Xs = d[DYN].to_numpy(dtype=float)
        Xz = np.column_stack([Xs, z])

        clf_a = HistGradientBoostingClassifier(**GBDT_KW).fit(Xs[trn], y[trn])
        clf_b = HistGradientBoostingClassifier(**GBDT_KW).fit(Xz[trn], y[trn])
        sa = clf_a.predict_proba(Xs[val])[:, 1]
        sb = clf_b.predict_proba(Xz[val])[:, 1]

        A.append(roc_auc_score(y[val], sa))
        B.append(roc_auc_score(y[val], sb))
        C.append(roc_auc_score(y[val], c_T[val]))
        probe_train_auc.append(roc_auc_score(y[trn], c_T[trn]))
        bA.append((pcode[val], y[val], sa))
        bB.append((pcode[val], y[val], sb))
        bC.append((pcode[val], y[val], c_T[val]))

    d_f = _dc.cluster_ci_diff(bB, bA, n_boot, rng_seed)
    d_r = _dc.cluster_ci_diff(bB, bC, n_boot, rng_seed)
    return {
        "A_surface_only": float(np.mean(A)),
        "B_surface_plus_probe": float(np.mean(B)),
        "C_probe_only": float(np.mean(C)),
        "A_per_seed": [float(x) for x in A],
        "B_per_seed": [float(x) for x in B],
        "forward_increment_B_minus_A": d_f,
        "reverse_increment_B_minus_C": d_r,
        "probe_train_auroc": float(np.mean(probe_train_auc)),
        "probe_train_minus_val": float(np.mean(probe_train_auc) - np.mean(C)),
        "prediction_met": bool(d_f["point"] >= 0.03 and d_f["ci_lo"] > 0.0),
        "kill_redundant": bool(d_f["ci_lo"] <= 0.0 <= d_f["ci_hi"]),
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics-dir", type=Path, required=True)
    ap.add_argument("--gbdt-ref", type=Path, required=True,
                    help="e1_surface_null_gbdt.json for the same split (A must reproduce)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    ref = json.load(open(a.gbdt_ref))["per_cell"]
    out = {}
    for model in MODELS:
        dyn = pd.read_parquet(a.dynamics_dir / f"cot_dynamics_{model}.parquet",
                              columns=["trace_id"] + DYN)
        for loss in LOSSES:
            key = f"{model}_{loss}"
            r = run_cell(model, loss, dyn, a.n_boot, a.rng_seed)
            banked = ref[key]["d11"]["gbdt_surface_holdout"]
            if abs(r["A_surface_only"] - banked) > 1e-9:
                raise ValueError(
                    f"{key}: surface-only AUROC {r['A_surface_only']} does not "
                    f"reproduce the banked GBDT-11 {banked}")
            if r["B_surface_plus_probe"] < r["A_surface_only"] - 0.01:
                raise ValueError(
                    f"{key}: B {r['B_surface_plus_probe']} far below A "
                    f"{r['A_surface_only']}; check the fitted model")
            out[key] = r

    payload = {
        "description": ("E-1 incremental value: GBDT-11 surface alone (A) vs the "
                        "same GBDT with logit(c_T) stacked in (B) vs the probe "
                        "alone (C), held out, with paired problem-clustered CIs."),
        "estimator": {"class": "HistGradientBoostingClassifier", **GBDT_KW},
        "scores_dir": str(SCORES), "n_boot": a.n_boot, "rng_seed": a.rng_seed,
        "prediction": "forward increment >= +0.03 with CI excluding 0 at both scales",
        "kill_line": "forward increment CI covers 0 at either scale => redundant",
        "per_cell": out,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
