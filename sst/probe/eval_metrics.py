from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

def compute_binary_detection_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:

    nan_result = {"auroc": float("nan"), "aupr": float("nan"), "fpr_at_95": float("nan")}
    if len(scores) == 0 or len(labels) == 0:
        return nan_result
    if len(np.unique(labels)) < 2:
        return nan_result

    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr_at_95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0

    return {"auroc": auroc, "aupr": aupr, "fpr_at_95": fpr_at_95}

def compute_calibration_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:

    if len(probs) == 0:
        return float("nan")
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(probs[mask].mean())
        avg_acc = float(labels[mask].mean())
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / max(len(probs), 1))
