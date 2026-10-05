from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

PUBLISHED_BCE = {
    "4b": {"auroc": 0.817, "ece": 0.027, "ece_temp": 0.015, "brier": 0.125, "acc_at_80": 0.882},
    "8b": {"auroc": 0.817, "ece": 0.019, "ece_temp": 0.012, "brier": 0.113, "acc_at_80": 0.901},
}

ROOT = Path(__file__).resolve().parents[1]

SCORES = Path(os.environ.get("SST_ECE_SCORES", ROOT / "runs/conf_probe/blindspot_scores"))
REEVAL = ROOT / "runs/conf_probe/grouped_split_reeval.csv"
REEVAL_GATE = os.environ.get("SST_ECE_REEVAL", "1") == "1"
OUT = Path(os.environ.get("SST_ECE_OUT", ROOT / "runs/conf_probe/ece_by_objective.json"))

def ece_equal_width(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    n = len(p)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        err += (mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return float(err)

def temperature_scale(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    eps = 1e-7
    logit = np.log(np.clip(p, eps, 1 - eps)) - np.log(np.clip(1 - p, eps, 1 - eps))

    def nll(t: float) -> float:
        q = 1.0 / (1.0 + np.exp(-logit / t))
        q = np.clip(q, eps, 1 - eps)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    t_opt = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x
    return 1.0 / (1.0 + np.exp(-logit / t_opt))

def acc_at_coverage(y: np.ndarray, p: np.ndarray, coverage: float = 0.8) -> float:
    k = int(round(coverage * len(p)))
    kept = np.argsort(-p)[:k]
    return float(y[kept].mean())

def main() -> None:
    reeval = {}
    with open(REEVAL) as f:
        for row in csv.DictReader(f):
            if row["slice"] == "all" and row["seed"] != "POOLED":
                reeval[(row["model"], int(row["seed"]))] = float(row["ece"])

    result: dict[str, dict] = {
        "description": (
            "Held-out (split=='val') metrics on c_T per retrain seed, from "
            "blindspot_scores parquets: raw AUROC, ECE (10 equal-width bins), ECE "
            "after a single-temperature NLL fit on the same held-out logits, Brier "
            "score, and accuracy at 80% coverage accepting by c_T. BCE per-seed "
            "ECEs verified against grouped_split_reeval.csv; BCE column means "
            "checked against the published tab:conf as-written rows."
        ),
        "per_cell": {},
        "bce_reproduction": {},
    }
    metrics = ("auroc", "ece", "ece_temp", "brier", "acc_at_80")
    for model in ("4b", "8b"):
        for obj in ("bce", "brier"):
            per_seed: dict[str, list[float]] = {m: [] for m in metrics}
            for seed in range(5):
                df = pd.read_parquet(SCORES / f"{model}_{obj}_s{seed}.parquet")
                val = df[df["split"] == "val"]
                y = val["Y"].to_numpy().astype(float)
                p = val["c_T"].to_numpy().astype(float)
                e = ece_equal_width(y, p)
                if obj == "bce" and REEVAL_GATE:
                    ref = reeval[(model, seed)]
                    if abs(e - ref) > 1e-6:
                        raise SystemExit(
                            f"ECE mismatch {model} bce s{seed}: {e:.8f} vs reeval {ref:.8f}"
                        )
                vals = {
                    "auroc": float(roc_auc_score(y, p)),
                    "ece": e,
                    "ece_temp": ece_equal_width(y, temperature_scale(y, p)),
                    "brier": float(((p - y) ** 2).mean()),
                    "acc_at_80": acc_at_coverage(y, p),
                }
                for m in metrics:
                    per_seed[m].append(vals[m])
            cell = {"n_val": int(len(val))}
            for m in metrics:
                cell[f"{m}_per_seed"] = per_seed[m]
                cell[f"{m}_mean"] = float(np.mean(per_seed[m]))
                cell[f"{m}_std"] = float(np.std(per_seed[m], ddof=1))
            result["per_cell"][f"{model}_{obj}"] = cell
            if obj == "bce":
                repro = {}
                for m in metrics:
                    diff = abs(cell[f"{m}_mean"] - PUBLISHED_BCE[model][m])
                    repro[m] = {"published": PUBLISHED_BCE[model][m],
                                "recomputed": cell[f"{m}_mean"],
                                "abs_diff": diff,
                                "matched": bool(diff <= 0.002)}
                result["bce_reproduction"][model] = repro
    OUT.write_text(json.dumps(result, indent=1))

if __name__ == "__main__":
    main()
