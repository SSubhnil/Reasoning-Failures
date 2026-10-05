#!/usr/bin/env python
'Run positional null.'
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

TOL = 2
GRID = np.arange(0.0, 1.0005, 0.005)

def hit(gold: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gold) <= TOL))

def predict(f: float, L: np.ndarray) -> np.ndarray:
    return np.rint(f * (L - 1.0))

def random_floor(gold: np.ndarray, L: np.ndarray) -> float:
    width = np.minimum(gold + TOL, L - 1.0) - np.maximum(gold - TOL, 0.0) + 1.0
    return float(np.mean(width / L))

def fit_argmax(gold: np.ndarray, L: np.ndarray) -> float:
    return float(GRID[int(np.argmax([hit(gold, predict(f, L)) for f in GRID]))])

def fit_meandepth(gold: np.ndarray, L: np.ndarray) -> float:
    ok = L > 1
    return float(np.mean(gold[ok] / (L[ok] - 1.0)))

def fit_constant(gold: np.ndarray, L: np.ndarray) -> float:
    cand = np.arange(0, int(L.max()))
    return float(cand[int(np.argmax([hit(gold, np.full_like(gold, c)) for c in cand]))])

def clustered_ci(
    gold: np.ndarray, pred: np.ndarray, prob: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:

    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(prob, return_inverse=True)
    by_problem = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    correct = (np.abs(pred - gold) <= TOL).astype(float)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_problem), len(by_problem))
        idx = np.concatenate([by_problem[i] for i in pick])
        stats[b] = correct[idx].mean()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def load(labels: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    items = [(k, r) for k, r in torch.load(labels, weights_only=False,
                                           map_location="cpu").items()
             if r["gold_pit"] >= 0]
    if not items:
        raise ValueError(f"{labels} carries no rows with gold_pit >= 0")
    return (np.array([r["gold_pit"] for _, r in items], float),
            np.array([r["n_steps"] for _, r in items], float),
            np.array([r["problem_id"] for _, r in items], object),
            np.array([k for k, _ in items], object))

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--fit-labels", type=Path, default=None,
                    help="fit the length rule on this label file and evaluate it on --labels")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gold, L, prob, _ = load(args.labels)
    print(f"[null] {args.labels}: {len(gold)} pits over {len(set(prob))} problems")
    print(f"[null] relative depth gold/(L-1): mean={np.mean(gold[L>1]/(L[L>1]-1)):.4f} "
          f"median={np.median(gold[L>1]/(L[L>1]-1)):.4f}")
    print(f"[null] {'randomfloor':10s} exact={random_floor(gold, L):.4f}  "
          f"unclipped (2*TOL+1)/mean(L)={(2*TOL+1)/np.mean(L):.4f}  mean(L)={np.mean(L):.3f}")

    if args.fit_labels is not None:
        gold_fit, L_fit, prob_fit, _ = load(args.fit_labels)
        f = fit_argmax(gold_fit, L_fit)
        pred = predict(f, L)
        lo, hi = clustered_ci(gold, pred, prob, args.n_boot, args.seed)
        print(f"[null] argmax fitted on {args.fit_labels} ({len(gold_fit)} pits over {len(set(prob_fit))} problems): "
              f"f={f:.3f}  Hit@±2={hit(gold, pred):.4f} [{lo:.4f}, {hi:.4f}]  (clustered on problem)")
        return

    fitters = {"argmax": fit_argmax, "meandepth": fit_meandepth}
    uniq = np.array(sorted(set(prob)), object)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[order[i]]: i % args.folds for i in range(len(uniq))}
    fold = np.array([fold_of[p] for p in prob])
    print(f"[null] {args.folds}-fold CV over problems (no held-out split supplied)")
    for name, fn in fitters.items():
        pred = np.empty_like(gold)
        fs = []
        for k in range(args.folds):
            tr, te = fold != k, fold == k
            f = fn(gold[tr], L[tr])
            fs.append(f)
            pred[te] = predict(f, L[te])
        lo, hi = clustered_ci(gold, pred, prob, args.n_boot, args.seed)
        print(f"[null] {name:10s} f={np.mean(fs):.3f}±{np.std(fs):.3f}  "
              f"Hit@±2={hit(gold, pred):.4f} [{lo:.4f}, {hi:.4f}]  (clustered on problem)")
    pred = np.empty_like(gold)
    for k in range(args.folds):
        tr, te = fold != k, fold == k
        pred[te] = fit_constant(gold[tr], L[tr])
    lo, hi = clustered_ci(gold, pred, prob, args.n_boot, args.seed)
    print(f"[null] {'constant':10s}         Hit@±2={hit(gold, pred):.4f} "
          f"[{lo:.4f}, {hi:.4f}]  (length-free control)")

if __name__ == "__main__":
    main()
