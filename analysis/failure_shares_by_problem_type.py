#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/type_shares.md"
LOOKUP = ROOT / "data/problem_lookup.parquet"
CLASSES = ["capability", "pit", "reaches"]
DM_RE = re.compile(r"deepmath_shard(\d)_ge7/(\d+)")
MATH_RE = re.compile(r"math_train_l4_5/hmath_train_(.+)_(\d+)")
N_BOOT, BOOT_SEED = 1000, 12345
N_PERM, PERM_SEED = 1000, 20260930

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def problem_types(problems: list[str]) -> dict[str, str]:
    topics = pd.read_parquet(LOOKUP).set_index("problem_id")["topic"]
    out = {}
    for p in problems:
        if DM_RE.fullmatch(p) is not None:
            if p not in topics.index or pd.isna(topics[p]):
                raise ValueError(f"{p}: no level-two topic in {LOOKUP.name}")
            out[p] = topics[p]
        elif (m := MATH_RE.fullmatch(p)) is not None:
            out[p] = f"MATH:{m.group(1)}"
        else:
            raise ValueError(f"{p}: matches no id rule")
    return out

def flips(model: str) -> set[str]:
    pc = pd.read_parquet(ROOT / f"runs/paper/labels/packed_classes_{model}.parquet")
    lg = pd.read_parquet(ROOT / f"runs/paper/labels/letter_gold/letter_gold_traces_{model}.parquet")
    a = set(pc.loc[(pc["cls_pre"] != "correct") & (pc["cls_post"] == "correct"), "trace_key"])
    c = set(lg.loc[lg["flip_to_correct"], "trace_id"])
    if not a or not c:
        raise ValueError(f"{model}: empty flip set")
    return a | c

def cramers_v(cls: np.ndarray, typ: np.ndarray) -> float:
    tab = pd.crosstab(cls, typ).to_numpy(dtype=float)
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    chi2 = float(((tab - exp) ** 2 / exp).sum())
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))

def shares_with_ci(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    out = {}
    for t, sub in df.groupby("type"):
        counts = sub.groupby("problem_id")["regime"].value_counts().unstack(fill_value=0)
        counts = counts.reindex(columns=CLASSES, fill_value=0).to_numpy(dtype=float)
        point = counts.sum(0) / counts.sum()
        draws = np.empty((N_BOOT, len(CLASSES)))
        for b in range(N_BOOT):
            pick = counts[rng.integers(0, len(counts), len(counts))].sum(0)
            draws[b] = pick / pick.sum()
        out[t] = {"n_traces": int(len(sub)), "n_problems": int(sub["problem_id"].nunique()),
                  **{c: {"point": float(point[i]),
                         "ci_lo": float(np.percentile(draws[:, i], 2.5)),
                         "ci_hi": float(np.percentile(draws[:, i], 97.5))}
                     for i, c in enumerate(CLASSES)}}
    return out

def permutation_v(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    real = cramers_v(df["regime"].to_numpy(), df["type"].to_numpy())
    prob_type = df.groupby("problem_id")["type"].first()
    codes = df["problem_id"].map({p: i for i, p in enumerate(prob_type.index)}).to_numpy()
    types = prob_type.to_numpy()
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = cramers_v(df["regime"].to_numpy(), rng.permutation(types)[codes])
    return {"V": real, "perm_p": float((1 + (null >= real).sum()) / (1 + N_PERM)),
            "null_p95": float(np.percentile(null, 95)), "null_mean": float(null.mean())}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    payload = {"prereg": str(PROTOCOL.relative_to(ROOT)), "prereg_sha256": sha256(PROTOCOL),
               "driver_sha256": sha256(Path(__file__)), "per_model": {}}
    for model in ["4b", "8b"]:
        g = json.loads((ROOT / f"runs/mc_gold/locus_{model}_k16/gold_pits.json").read_text())
        df = pd.DataFrame({"trace_id": list(g), "regime": [r["regime"] for r in g.values()]})
        df["problem_id"] = df["trace_id"].str.split("__").str[0]
        tmap = problem_types(sorted(df["problem_id"].unique()))
        df["type"] = df["problem_id"].map(tmap)
        fl = flips(model)
        n_fl = int(df["trace_id"].isin(fl).sum())
        if n_fl == 0:
            raise ValueError(f"{model}: flips match 0 broad-collection traces")
        res = {"n_types": int(df["type"].nunique()), "n_problems": int(df["problem_id"].nunique()),
               "flips_in_population": n_fl}
        for arm, sub in [("S", df), ("A", df[~df["trace_id"].isin(fl)])]:
            res[arm] = {"n_traces": int(len(sub)),
                        "shares": shares_with_ci(sub, np.random.default_rng(BOOT_SEED)),
                        "cramers_v": permutation_v(sub, np.random.default_rng(PERM_SEED))}
            cap = [v["capability"]["point"] for v in res[arm]["shares"].values()]
            print(f"[{model}/{arm}] n={len(sub)} V={res[arm]['cramers_v']['V']:.4f} "
                  f"p={res[arm]['cramers_v']['perm_p']:.4f} cap share range "
                  f"{min(cap):.3f}-{max(cap):.3f}", flush=True)
        payload["per_model"][model] = res
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
