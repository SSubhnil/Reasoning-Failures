#!/usr/bin/env python

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCORES = Path(os.environ.get("SST_CONCEAL_SCORES",
                             ROOT / "runs" / "conf_probe" / "blindspot_scores"))
LOOKUP = ROOT / "runs" / "conf_probe" / "confound_lookup.parquet"
OUT = Path(os.environ.get("SST_CONCEAL_OUT",
                          ROOT / "runs" / "conf_probe" / "conceal"))

SEEDS = [0, 1, 2, 3, 4]
PRIMARY_LOSS = "bce"
MODELS = ["4b", "8b"]

KAPPA = 0.5
VSTAR_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0]
CLASSES = ["capability", "pit", "reaches"]

N_BOOT = 1000
BOOT_SEED = 12345

DELTA_RAW = 0.15
DELTA_RESID = 0.08

def locus_path(model: str) -> Path:
    return ROOT / "runs" / "resp" / f"locus_{model}_k16" / "campaign" / "gold_pits.json"

def steps_cache_path(model: str) -> Path:
    return OUT / f"steps_{model}.parquet"

def load_confidence_union(model: str, loss: str, incorrect_only: bool = True) -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        f = SCORES / f"{model}_{loss}_s{s}.parquet"
        d = pd.read_parquet(
            f,
            columns=["trace_id", "problem_id", "Y", "c_t", "c_T",
                     "c_pre_answer_mean", "regime", "split"],
        )
        d = d[d["split"] == "val"].copy()
        if incorrect_only:
            d = d[d["Y"] == 0].copy()
        d["length"] = d["c_t"].apply(len).astype(int)
        frames.append(d.drop(columns=["c_t"]))
    allv = pd.concat(frames, ignore_index=True)
    if allv.empty:
        raise ValueError(f"no val rows for {model}_{loss}")
    g = (
        allv.groupby("trace_id")
        .agg(
            problem_id=("problem_id", "first"),
            Y=("Y", "first"),
            regime=("regime", "first"),
            length=("length", "first"),
            c_T=("c_T", "mean"),
            c_pre_answer_mean=("c_pre_answer_mean", "mean"),
            n_val_seeds=("c_T", "size"),
        )
        .reset_index()
    )
    return g

def load_vstar(model: str) -> pd.DataFrame:
    g = json.loads(locus_path(model).read_text())
    rows = []
    for tid, rec in g.items():
        curve = rec["curve"]
        if not curve:
            raise ValueError(f"{model}: empty curve for {tid}")
        rows.append({
            "trace_id": tid,
            "vstar0": float(max(curve)),
            "gold_pit": rec["gold_pit"],
            "regime_locus": rec["regime"],
        })
    return pd.DataFrame(rows)

def load_nuisances() -> pd.DataFrame:
    look = pd.read_parquet(LOOKUP)
    need = {"problem_id", "corpus", "difficulty_raw", "ans_char_len", "is_numeric"}
    missing = need - set(look.columns)
    if missing:
        raise ValueError(f"confound_lookup missing columns {missing}")
    return look

def build_master(
    model: str,
    loss: str = PRIMARY_LOSS,
    with_answer_pos: bool = False,
) -> pd.DataFrame:

    conf = load_confidence_union(model, loss)
    vs = load_vstar(model)
    nui = load_nuisances()

    m = conf.merge(vs, on="trace_id", how="left", validate="one_to_one")
    n_pre = len(m)
    n_no_locus = int(m["vstar0"].isna().sum())
    m = m.dropna(subset=["vstar0", "regime"]).copy()

    m = m.merge(nui, on="problem_id", how="left", validate="many_to_one")
    if m["difficulty_raw"].isna().any():
        bad = m.loc[m["difficulty_raw"].isna(), "problem_id"].head(5).tolist()
        raise ValueError(f"{model}_{loss}: unmatched problem_ids in nuisances {bad}")

    if with_answer_pos:
        cache = steps_cache_path(model)
        if not cache.exists():
            raise FileNotFoundError(
                f"missing step cache {cache}; run answer_position_cache.py --model {model}")
        steps = pd.read_parquet(cache, columns=["trace_id", "answer_position_norm"])
        m = m.merge(steps, on="trace_id", how="left", validate="one_to_one")

        if m["answer_position_norm"].isna().any() and (
                ~m["trace_id"].isin(set(steps["trace_id"]))).any():
            bad = m.loc[~m["trace_id"].isin(set(steps["trace_id"])), "trace_id"].head(5).tolist()
            raise ValueError(f"{model}_{loss}: trace_ids absent from step cache {bad}")

        n_answerless = int(m["answer_position_norm"].isna().sum())
        m = m.dropna(subset=["answer_position_norm"]).copy()
        m.attrs["n_dropped_answerless"] = n_answerless

    if m["regime"].nunique() < 3 or not set(CLASSES).issubset(set(m["regime"])):
        raise ValueError(f"{model}_{loss}: regimes present = {sorted(m['regime'].unique())}")

    m["conf"] = (m["c_T"] >= KAPPA).astype(float)
    m.attrs["n_pre_dropna"] = n_pre
    m.attrs["n_dropped_no_locus"] = n_no_locus
    m.attrs["model"] = model
    m.attrs["loss"] = loss
    return m

def vstar_bin(vstar0: np.ndarray) -> np.ndarray:
    edges = np.asarray(VSTAR_EDGES, dtype=float)
    b = np.digitize(vstar0, edges[1:-1], right=False)
    return np.clip(b, 0, len(edges) - 2)

def cluster_bootstrap(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], dict],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:

    rng = np.random.default_rng(seed)
    groups = df.groupby("problem_id").indices
    idx_lists = list(groups.values())
    n_groups = len(idx_lists)

    point = stat_fn(df)
    keys = list(point.keys())
    draws = {k: np.empty(n_boot, dtype=float) for k in keys}
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([idx_lists[p] for p in pick])
        sub = df.iloc[idx]
        sv = stat_fn(sub)
        for k in keys:
            draws[k][b] = sv[k]

    out = {}
    for k in keys:
        d = draws[k]
        finite = d[np.isfinite(d)]
        out[k] = {
            "point": float(point[k]) if np.isfinite(point[k]) else None,
            "ci_lo": float(np.percentile(finite, 2.5)) if finite.size else None,
            "ci_hi": float(np.percentile(finite, 97.5)) if finite.size else None,
            "n_boot_valid": int(finite.size),
        }
    return out

def write_json(payload: dict, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(payload, indent=2))
    return p
