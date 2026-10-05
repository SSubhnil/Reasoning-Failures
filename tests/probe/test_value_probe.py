from __future__ import annotations

import math

import pytest
import torch

from sst.probe.heads import TemporalProbe
from sst.probe.q_utils import binomial_nll, _probe_loss_value
from sst.probe.value_targets import (
    McTargets,
    build_target_row,
    is_recoverable,
    load_mc_targets,
    reachability_ceiling,
)

def _value_probe(d_input=16, with_conf_head=False, causal_mask=False, **kw):
    torch.manual_seed(0)
    return TemporalProbe(
        d_input=d_input, d_proj=32, n_heads=4, n_transformer_layers=2,
        max_steps=16, value_head="value", causal_mask=causal_mask,
        with_conf_head=with_conf_head, **kw,
    )

def test_binomial_nll_perfect_prediction_equals_entropy():
    v = torch.tensor([0.5])
    p = torch.tensor([0.5])
    k = torch.tensor([8.0])
    loss = binomial_nll(v, p, k)
    assert abs(float(loss) - math.log(2.0)) < 1e-5

def test_binomial_nll_minimized_at_truth():
    p = torch.tensor([0.7])
    k = torch.tensor([10.0])
    at_truth = binomial_nll(torch.tensor([0.7]), p, k)
    below = binomial_nll(torch.tensor([0.5]), p, k)
    above = binomial_nll(torch.tensor([0.9]), p, k)
    assert float(at_truth) < float(below)
    assert float(at_truth) < float(above)

def test_binomial_nll_k_weights_by_precision():
    v = torch.tensor([1.0, 1.0])
    p = torch.tensor([0.0, 1.0])
    bad_dominant = binomial_nll(v, p, torch.tensor([1000.0, 1.0]))
    good_dominant = binomial_nll(v, p, torch.tensor([1.0, 1000.0]))
    assert float(good_dominant) < float(bad_dominant)

def test_binomial_nll_is_weighted_mean_bce():
    v = torch.tensor([0.3, 0.8])
    p = torch.tensor([0.4, 0.6])
    k = torch.tensor([3.0, 9.0])
    bce0 = -(0.4 * math.log(0.3) + 0.6 * math.log(0.7))
    bce1 = -(0.6 * math.log(0.8) + 0.4 * math.log(0.2))
    expected = (3.0 * bce0 + 9.0 * bce1) / 12.0
    assert abs(float(binomial_nll(v, p, k)) - expected) < 1e-5

def test_binomial_nll_clamps_extremes():
    loss = binomial_nll(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]),
                        torch.tensor([5.0, 5.0]))
    assert torch.isfinite(loss) and float(loss) >= 0.0

def test_value_head_forward_is_bare_value_in_unit_range():
    p = _value_probe().eval()
    x = torch.randn(3, 7, 16)
    with torch.no_grad():
        out = p(x)
    assert isinstance(out, torch.Tensor) and out.shape == (3, 7)
    assert float(out.min()) > 0.0 and float(out.max()) < 1.0

def test_value_head_has_no_conf_head_when_separated():
    p = _value_probe(with_conf_head=False)
    assert not hasattr(p, "conf_head")

def test_value_head_with_conf_head_returns_tuple():
    p = _value_probe(with_conf_head=True)
    out = p(torch.randn(2, 5, 16))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (2, 5) and out[1].shape == (2, 5)

def test_dropping_conf_head_only_allowed_for_value_head():
    with pytest.raises(ValueError):
        TemporalProbe(d_input=8, value_head="td_q", with_conf_head=False)

def test_value_head_bidirectional_uses_future_steps():
    p = _value_probe(causal_mask=False).eval()
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        base = p(x).clone()
        x2 = x.clone()
        x2[0, 5] += 5.0
        moved = p(x2)
    assert float((moved[0, 0] - base[0, 0]).abs()) > 1e-6

def test_value_head_causal_ignores_future_steps():
    p = _value_probe(causal_mask=True).eval()
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        base = p(x).clone()
        x2 = x.clone()
        x2[0, 5] += 5.0
        moved = p(x2)
    assert float((moved[0, 0] - base[0, 0]).abs()) < 1e-6

def _batch(B=4, L=6, d=16):
    x = torch.randn(B, L, d)
    length_mask = torch.ones(B, L, dtype=torch.bool)
    length_mask[0, 4:] = False
    targets = torch.rand(B, L)
    k = torch.full((B, L), 8.0)
    visited = torch.zeros(B, L, dtype=torch.bool)
    visited[:, 1] = True
    visited[:, 3] = True
    return x, length_mask, targets, k, visited

def test_probe_loss_value_runs_and_grads_into_value_head():
    p = _value_probe()
    x, lm, t, k, vis = _batch()
    loss, stats = _probe_loss_value(p, "cpu", x, t, vis, k, lm)
    assert torch.isfinite(loss) and float(loss) > 0.0
    assert stats["n_anchored"] == int((vis & lm).sum())
    loss.backward()
    g = p.value_mlp[-1].weight.grad
    assert g is not None and bool(torch.isfinite(g).all())

def test_probe_loss_value_ignores_unvisited_steps():
    p = _value_probe().eval()
    x, lm, t, k, vis = _batch()
    loss_a, _ = _probe_loss_value(p, "cpu", x, t, vis, k, lm)
    t2 = t.clone()
    t2[~vis] = 1.0 - t2[~vis]
    loss_b, _ = _probe_loss_value(p, "cpu", x, t2, vis, k, lm)
    assert abs(float(loss_a) - float(loss_b)) < 1e-6

def test_probe_loss_value_empty_mask_is_zero():
    p = _value_probe()
    x, lm, t, k, _ = _batch()
    none_visited = torch.zeros_like(lm)
    loss, stats = _probe_loss_value(p, "cpu", x, t, none_visited, k, lm)
    assert float(loss) == 0.0 and stats["n_anchored"] == 0

def test_probe_loss_value_raises_on_nonfinite_input():
    p = _value_probe()
    x, lm, t, k, vis = _batch()
    x[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        _probe_loss_value(p, "cpu", x, t, vis, k, lm)

def test_reachability_ceiling_and_recoverable_filter():
    doomed = McTargets(md5="a", trace_id="t0", is_correct=False, T=5,
                       steps=(0, 2, 4), p=(0.0, 0.0, 0.0), k=(8, 8, 16))
    recov = McTargets(md5="b", trace_id="t1", is_correct=False, T=5,
                      steps=(0, 2, 4), p=(0.9, 0.4, 0.02), k=(8, 8, 16))
    assert reachability_ceiling(doomed) == 0.0
    assert reachability_ceiling(recov) == pytest.approx(0.9)
    assert not is_recoverable(doomed, theta=0.1)
    assert is_recoverable(recov, theta=0.1)

def test_build_target_row_places_visited_steps_only():
    rec = McTargets(md5="b", trace_id="t1", is_correct=False, T=6,
                    steps=(1, 3), p=(0.8, 0.2), k=(8, 16))
    target, k, mask = build_target_row(rec, Lmax=6)
    assert target.shape == (6,) and k.shape == (6,) and mask.shape == (6,)
    assert bool(mask[1]) and bool(mask[3])
    assert int(mask.sum()) == 2
    assert float(target[1]) == pytest.approx(0.8) and float(k[3]) == pytest.approx(16.0)
    assert float(target[0]) == 0.0 and not bool(mask[0])

def test_build_target_row_drops_out_of_range_steps():
    rec = McTargets(md5="c", trace_id="t2", is_correct=False, T=10,
                    steps=(1, 8), p=(0.5, 0.1), k=(8, 8))
    target, k, mask = build_target_row(rec, Lmax=4)
    assert int(mask.sum()) == 1 and bool(mask[1])

def test_load_mc_targets_accepts_dir_with_direct_steps_file(tmp_path):
    import json
    (tmp_path / "steps.jsonl").write_text(json.dumps(
        {"trace_id": "t", "md5": "abc", "is_correct": False, "T": 3,
         "steps": {"0": {"p_correct_mc": 0.5, "n_rollouts": 8}}}))
    tg = load_mc_targets(tmp_path)
    assert "abc" in tg and tg["abc"].p == (0.5,)
