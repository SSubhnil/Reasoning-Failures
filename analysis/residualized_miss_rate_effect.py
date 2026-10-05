#!/usr/bin/env python

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .blindness_common import (DELTA_RESID, MODELS, N_BOOT, PRIMARY_LOSS, VSTAR_EDGES,
                            build_master, vstar_bin, write_json)

COLS = ["pit", "capability"] + ["vstar0", "length", "difficulty_z", "corpus_deepmath",
                                "log_ans_len", "is_numeric", "answer_position_norm"]
PIT_IDX, CAP_IDX = 0, 1

def difficulty_within_corpus_z(m: pd.DataFrame) -> np.ndarray:
    dz = np.zeros(len(m), dtype=float)
    for corp in ("deepmath", "math"):
        cm = (m["corpus"] == corp).to_numpy()
        ref = m.loc[cm, "difficulty_raw"].to_numpy(dtype=float)
        sd = float(ref.std())
        if sd == 0.0:
            raise ValueError(f"zero-variance difficulty within corpus={corp}")
        dz[cm] = (m.loc[cm, "difficulty_raw"].to_numpy(dtype=float) - float(ref.mean())) / sd
    return dz

def make_design(m: pd.DataFrame, scaler: StandardScaler | None):
    dz = difficulty_within_corpus_z(m)
    cont = np.column_stack([
        m["vstar0"].to_numpy(dtype=float),
        m["length"].to_numpy(dtype=float),
        dz,
        np.log1p(m["ans_char_len"].to_numpy(dtype=float)),
        m["answer_position_norm"].to_numpy(dtype=float),
    ])
    if scaler is None:
        scaler = StandardScaler().fit(cont)
    contz = scaler.transform(cont)
    pit = (m["regime"] == "pit").to_numpy(dtype=float)
    cap = (m["regime"] == "capability").to_numpy(dtype=float)
    corp = (m["corpus"] == "deepmath").to_numpy(dtype=float)
    num = m["is_numeric"].to_numpy(dtype=float)
    X = np.column_stack([pit, cap, contz[:, 0], contz[:, 1], contz[:, 2],
                         corp, contz[:, 3], num, contz[:, 4]])
    return X, scaler

def fit_logit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000).fit(X, y)
    return clf

def ame(clf: LogisticRegression, X: np.ndarray, col: int) -> float:
    X1 = X.copy(); X1[:, col] = 1.0
    X0 = X.copy(); X0[:, col] = 0.0
    return float((clf.predict_proba(X1)[:, 1] - clf.predict_proba(X0)[:, 1]).mean())

def fit_stats(m: pd.DataFrame, scaler: StandardScaler) -> dict:
    X, _ = make_design(m, scaler)
    y = m["conf"].to_numpy(dtype=float)
    clf = fit_logit(X, y)
    coef = clf.coef_[0]
    return {
        "pit_logodds": float(coef[PIT_IDX]),
        "cap_logodds": float(coef[CAP_IDX]),
        "pit_ame": ame(clf, X, PIT_IDX),
        "cap_ame": ame(clf, X, CAP_IDX),
    }

def cluster_bootstrap_fit(m: pd.DataFrame, scaler: StandardScaler,
                          n_boot: int, seed: int = 20260627) -> dict:
    rng = np.random.default_rng(seed)
    idx_lists = list(m.groupby("problem_id").indices.values())
    ng = len(idx_lists)
    point = fit_stats(m, scaler)
    keys = list(point.keys())
    draws = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        pick = rng.integers(0, ng, size=ng)
        idx = np.concatenate([idx_lists[p] for p in pick])
        sv = fit_stats(m.iloc[idx], scaler)
        for k in keys:
            draws[k][b] = sv[k]
    out = {}
    for k in keys:
        d = draws[k]
        out[k] = {"point": float(point[k]),
                  "ci_lo": float(np.percentile(d, 2.5)),
                  "ci_hi": float(np.percentile(d, 97.5))}
    return out

def cell_stratified_gap(m: pd.DataFrame, pos_regime: str, min_n: int = 30) -> dict:
    pr = m[m["regime"].isin([pos_regime, "reaches"])].copy()
    pr["vbin"] = vstar_bin(pr["vstar0"].to_numpy())
    q = np.quantile(m.loc[m["regime"].isin(["pit", "reaches"]), "length"].to_numpy(float),
                    [1 / 3, 2 / 3])
    pr["ltert"] = np.digitize(pr["length"].to_numpy(dtype=float), q)
    cells = []
    signs = []
    for vb in range(len(VSTAR_EDGES) - 1):
        for lt in range(3):
            sel = (pr["vbin"] == vb) & (pr["ltert"] == lt)
            pos = pr[sel & (pr["regime"] == pos_regime)]
            rch = pr[sel & (pr["regime"] == "reaches")]
            populated = len(pos) >= min_n and len(rch) >= min_n
            gap = (float(pos["conf"].mean()) - float(rch["conf"].mean())
                   ) if (len(pos) and len(rch)) else None
            cells.append({"vbin": vb, "ltert": lt, "n_pos": int(len(pos)),
                          "n_reaches": int(len(rch)), "gap": gap,
                          "populated": bool(populated)})
            if populated and gap is not None:
                signs.append(gap)
    same_sign = bool(signs) and (all(g > 0 for g in signs) or all(g < 0 for g in signs))
    return {"pos_regime": pos_regime, "cells": cells, "n_populated": len(signs),
            "populated_gaps": signs, "all_same_sign": same_sign,
            "all_positive": bool(signs) and all(g > 0 for g in signs),
            "mean_populated_gap": (float(np.mean(signs)) if signs else None)}

def run_model(model: str, loss: str, n_boot: int) -> dict:
    m = build_master(model, loss, with_answer_pos=True)
    _, scaler = make_design(m, None)
    stats = cluster_bootstrap_fit(m, scaler, n_boot)
    cells_pit = cell_stratified_gap(m, "pit")
    cells_cap = cell_stratified_gap(m, "capability")

    pit_lo = stats["pit_logodds"]
    pit_ci_excl0 = (pit_lo["ci_lo"] is not None
                    and (pit_lo["ci_lo"] > 0 or pit_lo["ci_hi"] < 0))
    pit_ame_big = stats["pit_ame"]["point"] >= DELTA_RESID
    gate_pass = bool(pit_ci_excl0 and pit_ame_big and cells_pit["all_same_sign"]
                     and cells_pit["all_positive"])
    return {
        "model": model, "loss": loss, "n_traces": int(len(m)),
        "n_dropped_no_locus": m.attrs["n_dropped_no_locus"],
        "n_dropped_answerless": m.attrs["n_dropped_answerless"],
        "coefficients": stats,
        "pit_logodds_ci_excludes_0": bool(pit_ci_excl0),
        "pit_ame_ge_delta": bool(pit_ame_big),
        "cell_stratified_pit_vs_reaches": cells_pit,
        "cell_stratified_cap_vs_reaches": cells_cap,
        "gate_pass": gate_pass,
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--skip-brier", action="store_true",
                    help="primary BCE only (the psplit rerun trained no Brier arm).")
    args = ap.parse_args(argv)

    primary = {m: run_model(m, PRIMARY_LOSS, args.n_boot) for m in MODELS}
    robustness_brier = (None if args.skip_brier else
                        {m: run_model(m, "brier", args.n_boot) for m in MODELS})

    gate_both = all(primary[m]["gate_pass"] for m in MODELS)
    payload = {
        "task": "T2_residualize_nuisances_gate_inferential",
        "covariates": COLS,
        "answer_position": "real normalized first-\\boxed step index (NOT 0.9 heuristic)",
        "delta_resid": DELTA_RESID,
        "note": ("'coefficient' reported on two scales: log-odds (literal logistic "
                 "coef) and AME (probability gap, comparable to delta_resid). "
                 "GATE on pit = AME>=delta_resid AND log-odds CI excludes 0 AND "
                 "cell-stratified pit-reaches gap same-sign-positive, both scales. "
                 "capability: regression coef is modest-positive; the length-MATCHED "
                 "cell gap is where it collapses (see cell_stratified)."),
        "gate_is_on_primary_bce_only": True,
        "paper_number": {
            "spec": "primary BCE held-out-union logistic; report AME (point) + log-odds + cluster-bootstrap CI",
            "pit_ame_4b": primary["4b"]["coefficients"]["pit_ame"],
            "pit_ame_8b": primary["8b"]["coefficients"]["pit_ame"],
            "honest_caveat": ('Interpret the marginal-effect interval relative to delta_resid using the selected split. A point estimate above the threshold does not imply an interval above it.'),
        },
        "primary_bce": primary,
        "robustness_brier": robustness_brier,
        "brier_note": ('Secondary Brier-loss sensitivity. Report its marginal effect and interval separately from the primary BCE gate.'),
        "GATE_pit_survives_both_scales": gate_both,
    }
    write_json(payload, "t2_residualized_logistic.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
