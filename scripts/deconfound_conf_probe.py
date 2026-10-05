#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

SCORES = Path(os.environ.get("SST_DECONF_SCORES",
                             ROOT / "runs" / "conf_probe" / "blindspot_scores"))
LOOKUP = ROOT / "runs" / "conf_probe" / "confound_lookup.parquet"
OUT = ROOT / "runs" / "conf_probe" / "deconfounded_auroc.json"

MODELS = ["4b", "8b"]
LOSSES = ["bce", "brier"]
SEEDS = [0, 1, 2, 3, 4]

PREDICTORS = ["length", "difficulty_z", "corpus_deepmath", "log_ans_len", "is_numeric"]

DYN_PREDICTORS = [
    "dens_planning", "dens_derivation", "dens_verification",
    "dens_backtracking", "dens_finalization",
    "cycle_rate", "weighted_self_transition", "transition_entropy",
    "finalization_timing", "log_n_sentences", "log_mean_sent_chars",
]

def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def trace_table(model: str, loss: str, seed: int, look: pd.DataFrame,
                dyn: pd.DataFrame | None = None) -> pd.DataFrame:

    f = SCORES / f"{model}_{loss}_s{seed}.parquet"
    d = pd.read_parquet(
        f, columns=["trace_id", "problem_id", "Y", "c_t", "c_T", "split"])
    d["length"] = d["c_t"].apply(len).astype(float)
    d = d.drop(columns=["c_t"])
    m = d.merge(look, on="problem_id", how="left", validate="many_to_one")
    if m["difficulty_raw"].isna().any():
        bad = m[m["difficulty_raw"].isna()]["problem_id"].head(5).tolist()
        raise ValueError(f"{f.name}: unmatched problem_ids {bad}")
    if dyn is not None:
        m = m.merge(dyn, on="trace_id", how="left", validate="one_to_one")
        miss = m[DYN_PREDICTORS].isna().any(axis=1)
        if miss.any():
            bad = m[miss]["trace_id"].head(5).tolist()
            raise ValueError(f"{f.name}: {int(miss.sum())} traces missing dynamics "
                             f"features, e.g. {bad}")
    return m

def standardize(col: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu = float(ref.mean())
    sd = float(ref.std())
    if sd == 0.0:
        raise ValueError("zero-variance predictor on val fit rows")
    return (col - mu) / sd

def build_design(df: pd.DataFrame, val_mask: np.ndarray,
                 use_dyn: bool = False) -> np.ndarray:

    diff_z = np.zeros(len(df), dtype=float)
    for corp in ("deepmath", "math"):
        cm = (df["corpus"] == corp).to_numpy()
        ref = df.loc[cm & val_mask, "difficulty_raw"].to_numpy()
        sd = float(ref.std())
        mu = float(ref.mean())
        diff_z[cm] = 0.0 if sd == 0.0 else (df.loc[cm, "difficulty_raw"].to_numpy() - mu) / sd

    length = df["length"].to_numpy(dtype=float)
    log_ans = np.log1p(df["ans_char_len"].to_numpy(dtype=float))
    is_num = df["is_numeric"].to_numpy(dtype=float)
    corpus_dm = (df["corpus"] == "deepmath").to_numpy(dtype=float)

    v = val_mask
    cols = {
        "length": standardize(length, length[v]),
        "difficulty_z": standardize(diff_z, diff_z[v]),
        "corpus_deepmath": corpus_dm,
        "log_ans_len": standardize(log_ans, log_ans[v]),
        "is_numeric": is_num,
    }
    X = np.column_stack([cols[c] for c in PREDICTORS])
    if not use_dyn:
        return X
    D = np.column_stack([standardize(df[c].to_numpy(dtype=float),
                                     df.loc[val_mask, c].to_numpy(dtype=float))
                         for c in DYN_PREDICTORS])
    return np.column_stack([X, D])

def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("degenerate bootstrap block: one class absent")
    r = rankdata(score)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def cluster_ci(per_seed: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
               n_boot: int, rng_seed: int) -> dict:

    rng = np.random.default_rng(rng_seed)
    codes = np.unique(np.concatenate([p for p, _, _ in per_seed]))
    n_g = codes.size

    groups = []
    for pcode, _, _ in per_seed:
        order = np.argsort(pcode, kind="stable")
        sortp = pcode[order]
        starts = np.searchsorted(sortp, codes, side="left")
        ends = np.searchsorted(sortp, codes, side="right")
        groups.append((order, starts, ends))
    point = float(np.mean([fast_auc(y, sc) for _, y, sc in per_seed]))
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        vals = []
        for (order, starts, ends), (_, y, sc) in zip(groups, per_seed):
            idx = np.concatenate([order[starts[g]:ends[g]] for g in pick])
            vals.append(fast_auc(y[idx], sc[idx]))
        draws[b] = float(np.mean(vals))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot": n_boot, "n_problems": int(n_g)}

def cluster_ci_diff(per_seed_a: list, per_seed_b: list, n_boot: int,
                    rng_seed: int) -> dict:

    rng = np.random.default_rng(rng_seed)
    codes = np.unique(np.concatenate([p for p, _, _ in per_seed_a]))
    n_g = codes.size
    groups = []
    for pcode, _, _ in per_seed_a:
        order = np.argsort(pcode, kind="stable")
        sortp = pcode[order]
        groups.append((order, np.searchsorted(sortp, codes, side="left"),
                       np.searchsorted(sortp, codes, side="right")))
    point = float(np.mean([fast_auc(y, sa) - fast_auc(y, sb)
                           for (_, y, sa), (_, _, sb) in zip(per_seed_a, per_seed_b)]))
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        vals = []
        for (order, st, en), (_, y, sa), (_, _, sb) in zip(groups, per_seed_a, per_seed_b):
            idx = np.concatenate([order[st[g]:en[g]] for g in pick])
            vals.append(fast_auc(y[idx], sa[idx]) - fast_auc(y[idx], sb[idx]))
        draws[b] = float(np.mean(vals))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot": n_boot, "n_problems": int(n_g)}

def run_cell(model: str, loss: str, look: pd.DataFrame,
             dyn: pd.DataFrame | None = None, n_boot: int = 0,
             rng_seed: int = 0) -> dict:
    raw_aurocs, deconf_aurocs, beta_rows = [], [], []
    dyn_aurocs, null_d_aurocs, null_all_aurocs, dyn_beta_rows = [], [], [], []
    null_d_hold, null_all_hold = [], []
    boot_raw, boot_d5, boot_d16, boot_null = [], [], [], []
    boot_null_h, boot_probe_raw = [], []
    for seed in SEEDS:
        df = trace_table(model, loss, seed, look, dyn)
        val = (df["split"] == "val").to_numpy()
        y = df["Y"].to_numpy()
        c_T = df["c_T"].to_numpy(dtype=float)

        pcode = df["problem_id"].astype("category").cat.codes.to_numpy()

        raw_aurocs.append(roc_auc_score(y[val], c_T[val]))

        z = logit(c_T)
        X = build_design(df, val)
        reg = LinearRegression().fit(X[val], z[val])
        resid = z - reg.predict(X)
        deconf_aurocs.append(roc_auc_score(y[val], resid[val]))
        beta_rows.append(reg.coef_)
        boot_raw.append((pcode[val], y[val], c_T[val]))
        boot_d5.append((pcode[val], y[val], resid[val]))

        if dyn is None:
            continue

        Xd = build_design(df, val, use_dyn=True)
        regd = LinearRegression().fit(Xd[val], z[val])
        residd = z - regd.predict(Xd)
        dyn_aurocs.append(roc_auc_score(y[val], residd[val]))
        dyn_beta_rows.append(regd.coef_)
        boot_d16.append((pcode[val], y[val], residd[val]))

        D = Xd[:, len(PREDICTORS):]
        lg = LogisticRegression(max_iter=2000).fit(D[val], y[val])
        s_null = lg.decision_function(D[val])
        null_d_aurocs.append(roc_auc_score(y[val], s_null))
        boot_null.append((pcode[val], y[val], s_null))
        lg_all = LogisticRegression(max_iter=2000).fit(Xd[val], y[val])
        null_all_aurocs.append(
            roc_auc_score(y[val], lg_all.decision_function(Xd[val])))

        trn = ~val
        lg_h = LogisticRegression(max_iter=2000).fit(D[trn], y[trn])
        s_h = lg_h.decision_function(D[val])
        null_d_hold.append(roc_auc_score(y[val], s_h))
        boot_null_h.append((pcode[val], y[val], s_h))
        lg_ah = LogisticRegression(max_iter=2000).fit(Xd[trn], y[trn])
        null_all_hold.append(
            roc_auc_score(y[val], lg_ah.decision_function(Xd[val])))
        boot_probe_raw.append((pcode[val], y[val], c_T[val]))

    betas = np.mean(np.vstack(beta_rows), axis=0)
    out = {
        "raw_auroc": float(np.mean(raw_aurocs)),
        "raw_auroc_std": float(np.std(raw_aurocs)),
        "deconf_auroc": float(np.mean(deconf_aurocs)),
        "deconf_auroc_std": float(np.std(deconf_aurocs)),
        "auroc_drop": float(np.mean(raw_aurocs) - np.mean(deconf_aurocs)),
        "raw_auroc_per_seed": [float(x) for x in raw_aurocs],
        "deconf_auroc_per_seed": [float(x) for x in deconf_aurocs],
        "confound_betas": {p: float(b) for p, b in zip(PREDICTORS, betas)},
    }
    if dyn is not None:
        dbetas = np.mean(np.vstack(dyn_beta_rows), axis=0)
        names = PREDICTORS + DYN_PREDICTORS
        out.update({
            "deconf_auroc_dyn": float(np.mean(dyn_aurocs)),
            "deconf_auroc_dyn_std": float(np.std(dyn_aurocs, ddof=1)),
            "deconf_auroc_dyn_per_seed": [float(x) for x in dyn_aurocs],
            "surface_null_auroc": float(np.mean(null_d_aurocs)),
            "surface_null_auroc_std": float(np.std(null_d_aurocs, ddof=1)),
            "surface_null_auroc_per_seed": [float(x) for x in null_d_aurocs],
            "surface_null_auroc_all": float(np.mean(null_all_aurocs)),
            "surface_null_auroc_all_per_seed": [float(x) for x in null_all_aurocs],
            "surface_null_holdout": float(np.mean(null_d_hold)),
            "surface_null_holdout_std": float(np.std(null_d_hold, ddof=1)),
            "surface_null_holdout_per_seed": [float(x) for x in null_d_hold],
            "surface_null_all_holdout": float(np.mean(null_all_hold)),
            "surface_null_all_holdout_per_seed": [float(x) for x in null_all_hold],
            "deconf_auroc_std_ddof1": float(np.std(deconf_aurocs, ddof=1)),
            "raw_auroc_std_ddof1": float(np.std(raw_aurocs, ddof=1)),
            "betas_with_dynamics": {n: float(b) for n, b in zip(names, dbetas)},
        })
    if n_boot > 0:
        out["ci"] = {
            "raw": cluster_ci(boot_raw, n_boot, rng_seed),
            "deconf_5": cluster_ci(boot_d5, n_boot, rng_seed),
        }
        if dyn is not None:
            out["ci"]["deconf_16"] = cluster_ci(boot_d16, n_boot, rng_seed)
            out["ci"]["surface_null"] = cluster_ci(boot_null, n_boot, rng_seed)
            out["ci"]["surface_null_holdout"] = cluster_ci(boot_null_h, n_boot, rng_seed)

            out["ci"]["probe_minus_holdout_null"] = cluster_ci_diff(
                boot_probe_raw, boot_null_h, n_boot, rng_seed)
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dynamics-dir", type=Path, default=None,
                    help="dir holding cot_dynamics_{4b,8b}.parquet (E-1)")
    ap.add_argument("--n-boot", type=int, default=0,
                    help="problem-clustered bootstrap replicates (0 = none)")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args(argv)

    look = pd.read_parquet(LOOKUP)
    dyn_by_model = {}
    if args.dynamics_dir is not None:
        for model in MODELS:
            f = args.dynamics_dir / f"cot_dynamics_{model}.parquet"
            d = pd.read_parquet(f, columns=["trace_id"] + DYN_PREDICTORS)
            dyn_by_model[model] = d

    results = {}
    for model in MODELS:
        for loss in LOSSES:
            key = f"{model}_{loss}"
            results[key] = run_cell(model, loss, look,
                                    dyn_by_model.get(model), args.n_boot,
                                    args.rng_seed)

    payload = {
        "description": ("Slot 5: confidence-detector (c_T vs Y) AUROC on val, "
                        "raw vs after residualizing logit(c_T) on "
                        "[trace length, within-corpus difficulty, corpus, "
                        "log answer-length, numeric-answer]. Fit on val only, "
                        "seed-averaged over 5 seeds."),
        "predictors": PREDICTORS,
        "dynamics_predictors": DYN_PREDICTORS if dyn_by_model else None,
        "n_boot": args.n_boot,
        "rng_seed": args.rng_seed,
        "per_cell": results,
        "residualized_peak_layer": None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
