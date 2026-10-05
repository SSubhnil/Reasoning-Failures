from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from sst.probe import conf_eval, q_utils
from sst.probe.heads import TemporalProbe

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "train_conf_probe", ROOT / "scripts" / "train_conf_probe.py")
tcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcp)

def test_confidence_head_forward_shape_and_range():
    probe = TemporalProbe(d_input=32, d_proj=16, n_heads=2, n_transformer_layers=2,
                          max_steps=8, value_head="confidence")
    x = torch.randn(4, 5, 32)
    out = probe(x)
    assert out.shape == (4, 5)
    assert out.min() >= 0.0 and out.max() <= 1.0

    assert not hasattr(probe, "q_head")
    assert not hasattr(probe, "hazard_head")

def test_unknown_value_head_raises():
    try:
        TemporalProbe(d_input=8, value_head="nope")
    except ValueError as e:
        assert "confidence" in str(e)
    else:
        raise AssertionError("expected ValueError")

def _toy_probe():
    torch.manual_seed(0)
    return TemporalProbe(d_input=16, d_proj=16, n_heads=2, n_transformer_layers=1,
                         max_steps=8, value_head="confidence")

def test_loss_bce_and_brier_finite_and_masked():
    probe = _toy_probe()
    x = torch.randn(3, 4, 16)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    y = torch.tensor([1.0, 0.0, 1.0])
    for loss in ("bce", "brier"):
        l, stats = q_utils._probe_loss_confidence(probe, "cpu", x, y, mask, loss=loss)
        assert torch.isfinite(l) and l.item() >= 0.0
        assert "loss_conf" in stats

def test_loss_excludes_unknown_label():
    probe = _toy_probe()
    x = torch.randn(2, 3, 16)
    mask = torch.ones(2, 3, dtype=torch.bool)

    y = torch.tensor([-1.0, -1.0])
    l, _ = q_utils._probe_loss_confidence(probe, "cpu", x, y, mask, loss="bce")
    assert l.item() == 0.0

def test_loss_label_smoothing_changes_value():
    probe = _toy_probe()
    x = torch.randn(2, 3, 16)
    mask = torch.ones(2, 3, dtype=torch.bool)
    y = torch.tensor([1.0, 0.0])
    a, _ = q_utils._probe_loss_confidence(probe, "cpu", x, y, mask, loss="bce", label_smooth=0.0)
    b, _ = q_utils._probe_loss_confidence(probe, "cpu", x, y, mask, loss="bce", label_smooth=0.2)
    assert abs(a.item() - b.item()) > 1e-6

def test_bad_loss_name_raises():
    probe = _toy_probe()
    x = torch.randn(1, 2, 16)
    mask = torch.ones(1, 2, dtype=torch.bool)
    try:
        q_utils._probe_loss_confidence(probe, "cpu", x, torch.tensor([1.0]), mask, loss="hinge")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad loss")

def test_stratified_split_preserves_base_rate_and_disjoint():
    ic = [1.0] * 770 + [0.0] * 230
    train, val = tcp.stratified_split(ic, val_frac=0.3, seed=1)
    assert set(train).isdisjoint(set(val))
    assert len(train) + len(val) == 1000
    base = np.mean(ic)
    tr_rate = np.mean([ic[i] for i in train])
    va_rate = np.mean([ic[i] for i in val])
    assert abs(tr_rate - base) < 0.02 and abs(va_rate - base) < 0.02
    assert abs(len(val) / 1000 - 0.3) < 0.01

def test_stratified_split_deterministic():
    ic = [1.0, 0.0] * 100
    assert tcp.stratified_split(ic, 0.3, 7) == tcp.stratified_split(ic, 0.3, 7)

def test_brier_perfect_is_zero():
    assert conf_eval.brier_score(np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])) == 0.0

def test_temperature_positive_and_improves_overconfident():
    rng = np.random.default_rng(0)
    n = 4000
    pred = rng.integers(0, 2, n)
    flip = rng.random(n) > 0.7
    labels = np.where(flip, 1 - pred, pred).astype(float)
    logits = 5.0 * (2 * pred - 1)
    T = conf_eval.fit_temperature(logits, labels)
    assert T > 1.0
    p0 = 1 / (1 + np.exp(-logits))
    pT = 1 / (1 + np.exp(-logits / T))
    assert conf_eval.compute_calibration_ece(pT, labels) <= conf_eval.compute_calibration_ece(p0, labels) + 1e-9

def test_selective_full_coverage_equals_overall_accuracy():
    probs = np.array([0.9, 0.1, 0.8, 0.4])
    labels = np.array([1.0, 0.0, 0.0, 0.0])
    sel = conf_eval.selective_accuracy(probs, labels, coverages=(1.0,))
    overall = np.mean((probs > 0.5).astype(float) == labels)
    assert abs(sel["acc_at"][1.0] - overall) < 1e-9

def test_selective_high_coverage_not_better_than_low():
    rng = np.random.default_rng(1)
    probs = rng.random(500)
    labels = (rng.random(500) < probs).astype(float)
    sel = conf_eval.selective_accuracy(probs, labels, coverages=(0.2, 1.0))
    assert sel["acc_at"][0.2] >= sel["acc_at"][1.0] - 0.05

def test_per_step_calibration_bins():
    step_probs = [[0.5, 0.6, 0.7], [0.4, 0.8]]
    labels = [1.0, 0.0]
    rec = conf_eval.per_step_calibration(step_probs, labels, n_pos_bins=4)
    assert all("ece" in r and "mean_conf" in r and r["n"] > 0 for r in rec)

def test_summarize_trace_level_keys():
    rng = np.random.default_rng(2)
    probs = rng.random(300)
    labels = (rng.random(300) < probs).astype(float)
    s = conf_eval.summarize_trace_level(probs, labels)
    for k in ("ece", "ece_temp", "brier", "auroc", "temperature", "acc_at_80"):
        assert k in s
