from __future__ import annotations

import numpy as np
import torch

from sst.probe.eval_metrics import (
    compute_binary_detection_metrics,
    compute_calibration_ece,
)

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(probs) == 0:
        return float("nan")
    return float(np.mean((np.asarray(probs, dtype=np.float64) - np.asarray(labels, dtype=np.float64)) ** 2))

def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    if len(logits) == 0 or len(np.unique(labels)) < 2:
        return 1.0
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())

def probs_to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def selective_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
    coverages: tuple[float, ...] = (0.5, 0.8, 1.0),
) -> dict:

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return {"acc_at": {c: float("nan") for c in coverages}, "coverage": [], "accuracy": []}
    pred = (probs > 0.5).astype(np.float64)
    correct = (pred == labels).astype(np.float64)
    dec_conf = np.maximum(probs, 1.0 - probs)
    order = np.argsort(-dec_conf)
    correct_sorted = correct[order]
    cum_acc = np.cumsum(correct_sorted) / np.arange(1, n + 1)
    cov_grid = np.arange(1, n + 1) / n
    acc_at = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        acc_at[c] = float(cum_acc[k - 1])
    return {"acc_at": acc_at, "coverage": cov_grid.tolist(), "accuracy": cum_acc.tolist()}

def summarize_trace_level(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    det = compute_binary_detection_metrics(probs, labels)
    ece = compute_calibration_ece(probs, labels, n_bins=n_bins)
    T = fit_temperature(probs_to_logits(probs), labels)
    probs_temp = 1.0 / (1.0 + np.exp(-probs_to_logits(probs) / max(T, 1e-6)))
    ece_temp = compute_calibration_ece(probs_temp, labels, n_bins=n_bins)
    sel = selective_accuracy(probs, labels)
    return {
        "n": int(len(probs)),
        "base_rate": float(labels.mean()) if len(labels) else float("nan"),
        "ece": ece,
        "ece_temp": ece_temp,
        "temperature": T,
        "brier": brier_score(probs, labels),
        "auroc": det["auroc"],
        "aupr": det["aupr"],
        "fpr_at_95": det["fpr_at_95"],
        "acc_at_50": sel["acc_at"][0.5],
        "acc_at_80": sel["acc_at"][0.8],
    }

def per_step_calibration(
    step_probs: list[list[float]],
    labels: list[float],
    n_pos_bins: int = 10,
) -> list[dict]:

    pos_p: list[list[float]] = [[] for _ in range(n_pos_bins)]
    pos_y: list[list[float]] = [[] for _ in range(n_pos_bins)]
    for sp, y in zip(step_probs, labels):
        L = len(sp)
        if L == 0 or y < 0:
            continue
        for t, p in enumerate(sp):
            frac = 0.0 if L == 1 else t / (L - 1)
            b = min(n_pos_bins - 1, int(frac * n_pos_bins))
            pos_p[b].append(float(p))
            pos_y[b].append(float(y))
    out = []
    for b in range(n_pos_bins):
        if not pos_p[b]:
            continue
        p = np.asarray(pos_p[b]); yv = np.asarray(pos_y[b])
        out.append({
            "pos_bin": (b + 0.5) / n_pos_bins,
            "n": int(len(p)),
            "mean_conf": float(p.mean()),
            "mean_acc": float(yv.mean()),
            "ece": compute_calibration_ece(p, yv),
            "brier": brier_score(p, yv),
        })
    return out
