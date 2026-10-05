#!/usr/bin/env python
'Run paired localization test.'
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import length_rule_baseline as _pn
TOL, fit_argmax, predict = _pn.TOL, _pn.fit_argmax, _pn.predict

COLUMNS = ["md5", "problem_id", "n_steps", "gold_pit", "pred_step", "hit", "seed"]

def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}; expected {COLUMNS}")
    recomputed = (np.abs(df["pred_step"] - df["gold_pit"]) <= TOL).astype(int)
    bad = int((recomputed != df["hit"].astype(int)).sum())
    if bad:
        raise ValueError(f"{path}: {bad} rows where hit != (|pred-gold| <= {TOL})")
    return df

def paired_stats(
    hit_a: np.ndarray, hit_b: np.ndarray, problem: np.ndarray,
    n_boot: int, seed: int,
) -> dict:

    a = int(np.sum(hit_a & hit_b))
    b = int(np.sum(hit_a & ~hit_b))
    c = int(np.sum(~hit_a & hit_b))
    d = int(np.sum(~hit_a & ~hit_b))
    n = a + b + c + d

    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)

    diff = hit_a.astype(float) - hit_b.astype(float)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(problem, return_inverse=True)
    by_problem = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(by_problem), len(by_problem))
        boot[i] = diff[np.concatenate([by_problem[j] for j in pick])].mean()
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    return {
        "n": n, "n_problems": len(uniq),
        "hit_a": float(hit_a.mean()), "hit_b": float(hit_b.mean()),
        "both_hit": a, "b_only_a": b, "c_only_b": c, "both_miss": d,
        "n_discordant": b + c,
        "diff": float(diff.mean()), "mcnemar_p": p,
        "boot_lo": lo, "boot_hi": hi,
        "boot_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }

def report(tag: str, s: dict) -> None:
    print(f"\n=== {tag} ===")
    print(f"  rows={s['n']} over {s['n_problems']} problems "
          f"({s['n']/s['n_problems']:.2f} pits/problem)")
    print(f"  Hit@±{TOL}: A={s['hit_a']:.4f}  B={s['hit_b']:.4f}  "
          f"paired diff={s['diff']:+.4f}")
    print(f"  cells: both hit={s['both_hit']}  A-only b={s['b_only_a']}  "
          f"B-only c={s['c_only_b']}  both miss={s['both_miss']}")
    print(f"  discordant b+c={s['n_discordant']}"
          + ("   <- both localizers hit and miss the same rows"
             if s['n_discordant'] == 0 else ""))
    print(f"  McNemar exact p={s['mcnemar_p']:.4g}"
          + ("  (no information: b+c=0)" if s['n_discordant'] == 0 else ""))
    print(f"  clustered paired bootstrap 95% [{s['boot_lo']:+.4f}, {s['boot_hi']:+.4f}]"
          f"  excludes 0: {s['boot_excludes_zero']}")
    sig_mc = s["mcnemar_p"] < 0.05
    if s["n_discordant"] and sig_mc != s["boot_excludes_zero"]:
        print("  DISAGREEMENT: McNemar and the clustered bootstrap differ at the 5% "
              "level. Pits cluster within problems, which breaks McNemar's "
              "independence assumption - read the bootstrap.")

def null_predictions(labels: Path, val_problems: Path) -> pd.DataFrame:
    gold, L, prob, md5 = _pn.load(labels)
    val = set(json.loads(val_problems.read_text()))
    te = np.array([p in val for p in prob])
    f = fit_argmax(gold[~te], L[~te])
    pred = predict(f, L[te])
    print(f"[paired] positional null refit on train split: f={f:.3f}, "
          f"applied to {int(te.sum())} eval pits")
    return pd.DataFrame({
        "md5": md5[te], "problem_id": prob[te], "n_steps": L[te].astype(int),
        "gold_pit": gold[te].astype(int), "pred_step": pred.astype(int),
        "hit": (np.abs(pred - gold[te]) <= TOL).astype(int),
    })

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-a", type=Path, default=None, help="predictions CSV for A")
    ap.add_argument("--pred-b", type=Path, default=None, help="predictions CSV for B")
    ap.add_argument("--vs-null", action="store_true",
                    help="use the refit positional null as predictor B")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--val-problems", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.pred_a is None:
        raise ValueError("--pred-a is required")
    if (args.pred_b is None) == (not args.vs_null):
        raise ValueError("supply exactly one of --pred-b or --vs-null")

    a = load_predictions(args.pred_a)
    b = (null_predictions(args.labels, args.val_problems) if args.vs_null
         else load_predictions(args.pred_b))

    if "md5" not in b.columns:
        raise ValueError("predictor B has no md5 column; the paired join needs it")

    for sd in sorted(a["seed"].unique()):
        ra = a[a["seed"] == sd]
        rb = b[b["seed"] == sd] if "seed" in b.columns else b
        if ra["md5"].duplicated().any() or rb["md5"].duplicated().any():
            raise ValueError(f"seed {sd}: duplicate md5 in A or B - rows are not unique")
        m = ra.merge(rb, on="md5", suffixes=("_a", "_b"), how="inner")
        if len(m) != len(ra) or len(m) != len(rb):
            raise ValueError(
                f"seed {sd}: A has {len(ra)} rows, B has {len(rb)}, {len(m)} shared - "
                f"the paired test requires identical eval rows"
            )
        if not np.array_equal(m["gold_pit_a"].to_numpy(), m["gold_pit_b"].to_numpy()):
            raise ValueError(f"seed {sd}: gold_pit differs between A and B on shared rows")
        report(f"seed {sd}: A={args.pred_a.name} vs "
               f"B={'positional null' if args.vs_null else args.pred_b.name}",
               paired_stats(m["hit_a"].to_numpy().astype(bool),
                            m["hit_b"].to_numpy().astype(bool),
                            m["problem_id_a"].to_numpy(), args.n_boot, args.seed))

if __name__ == "__main__":
    main()
