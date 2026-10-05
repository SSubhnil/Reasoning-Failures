"""Replay the transcoder failure-type classifier table from the stored per-partition scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

TEXT_ARMS = ["surface_null", "count_len_null", "surface_null_count", "tracetext_null", "surface_null_strong",
             "surface_null_full"]
BASELINES = TEXT_ARMS + [f"{arm}__gbdt" for arm in TEXT_ARMS]
MARGIN = 0.05
PARTITIONS = 6
MIN_INTERVALS_EXCLUDING_ZERO = 5
OOF_ARRAYS = ["y", "groups", "oof__z_presence_union", "oof__h_presence_single"] + [f"oof__{arm}" for arm in BASELINES]


def paired_interval(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_boot: int = 1000,
                    seed: int = 0) -> tuple[float, float]:
    """Problem-resampled 95% interval of AUROC(a) - AUROC(b) over fixed out-of-fold scores."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rows = {g: np.where(groups == g)[0] for g in unique}
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def load_oof(path: Path) -> dict[str, np.ndarray]:
    """The numeric out-of-fold arrays of one partition."""
    with np.load(path) as stored:
        return {name: stored[name] for name in OOF_ARRAYS}


def partition_row(score: dict, oof: dict[str, np.ndarray] | None) -> dict:
    """One fold partition: AUROCs, the strongest baseline and the two margins with their intervals."""
    arms, read = score["arms"], score["registered_read"]
    strongest = max(BASELINES, key=lambda arm: arms[arm]["auroc"])
    row = {"transcoder_auroc": arms["z_presence_union"]["auroc"], "hidden_auroc": arms["h_presence_single"]["auroc"],
           "strongest_baseline": strongest, "strongest_baseline_auroc": arms[strongest]["auroc"],
           "transcoder_minus_strongest": read["z_minus_strongest"],
           "transcoder_minus_strongest_ci95": read["z_minus_strongest_ci95"],
           "transcoder_minus_hidden": read["z_minus_h_single"],
           "transcoder_minus_hidden_ci95": read["z_minus_h_single_ci95"]}
    if oof is not None:
        y, groups, transcoder = oof["y"], oof["groups"], oof["oof__z_presence_union"]
        row["recomputed"] = {
            "transcoder_auroc": float(roc_auc_score(y, transcoder)),
            "transcoder_minus_strongest_ci95": list(paired_interval(y, transcoder, oof[f"oof__{strongest}"], groups)),
            "transcoder_minus_hidden_ci95": list(paired_interval(y, transcoder, oof["oof__h_presence_single"], groups))}
    return row


def recompute_gap(rows: list[dict]) -> float:
    """Largest absolute difference between stored and recomputed transcoder AUROC and interval ends."""
    gaps = []
    for r in rows:
        stored, again = r, r["recomputed"]
        gaps.append(abs(again["transcoder_auroc"] - stored["transcoder_auroc"]))
        for key in ("transcoder_minus_strongest_ci95", "transcoder_minus_hidden_ci95"):
            gaps.extend(abs(a - b) for a, b in zip(again[key], stored[key]))
    return float(max(gaps))


def summarize(rows: list[dict]) -> dict:
    """Means and sample standard deviations over fold partitions, interval counts and the margin rule."""
    margin = [r["transcoder_minus_strongest"] for r in rows]
    intervals = [r["transcoder_minus_strongest_ci95"] for r in rows]
    excluding = sum(lo > 0 for lo, _ in intervals)
    gap = [r["transcoder_minus_hidden"] for r in rows]
    return {"n_partitions": len(rows),
            "transcoder_auroc_mean": float(np.mean([r["transcoder_auroc"] for r in rows])),
            "hidden_auroc_mean": float(np.mean([r["hidden_auroc"] for r in rows])),
            "strongest_baseline_auroc_mean": float(np.mean([r["strongest_baseline_auroc"] for r in rows])),
            "transcoder_minus_strongest_mean": float(np.mean(margin)),
            "transcoder_minus_strongest_sd": float(np.std(margin, ddof=1)),
            "intervals_excluding_zero": excluding,
            "intervals_containing_margin": sum(lo <= MARGIN <= hi for lo, hi in intervals),
            "partitions_at_or_above_margin": sum(m >= MARGIN for m in margin),
            "meets_margin": float(np.mean(margin)) >= MARGIN and excluding >= MIN_INTERVALS_EXCLUDING_ZERO,
            "transcoder_minus_hidden_mean": float(np.mean(gap)),
            "transcoder_minus_hidden_sd": float(np.std(gap, ddof=1)),
            "transcoder_minus_hidden_intervals_excluding_zero": sum(r["transcoder_minus_hidden_ci95"][0] > 0 for r in rows),
            "partitions": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="gd_{scale}_{arm}_seed{seed}")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    table = {}
    for scale in ("4b", "8b"):
        for arm, label in (("s", "stored"), ("d", "defects_removed")):
            rows = []
            for seed in range(PARTITIONS):
                stem = args.scores_dir / args.pattern.format(scale=scale, arm=arm, seed=seed)
                oof = load_oof(Path(f"{stem}.oof.npz")) if args.recompute else None
                rows.append(partition_row(json.loads(Path(f"{stem}.json").read_text()), oof))
            cell = summarize(rows)
            if args.recompute:
                cell["recompute_max_abs_difference"] = recompute_gap(rows)
            table[f"{scale}_{label}"] = cell
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
