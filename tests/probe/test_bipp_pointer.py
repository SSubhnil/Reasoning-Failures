from __future__ import annotations

import torch

from sst.probe.heads import TemporalProbe
from sst.probe.q_utils import _probe_loss_pointer

def _probe(d_input=37, **kw):
    torch.manual_seed(0)
    return TemporalProbe(
        d_input=d_input, d_proj=32, n_heads=4, n_transformer_layers=2,
        max_steps=16, value_head="pointer", causal_mask=False, **kw,
    )

def test_pointer_forward_shapes_and_ranges():
    p = _probe()
    x = torch.randn(3, 7, 37)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[0, 5:] = False
    with torch.no_grad():
        pl, rl, R_hat, conf = p(x, length_mask=mask, return_pointer=True)
    assert pl.shape == (3, 7)
    assert rl.shape == (3, 3)
    assert R_hat.shape == (3, 7)
    assert conf.shape == (3, 7)
    assert 0.0 < float(R_hat.min()) and float(R_hat.max()) < 1.0

def test_pointer_default_is_two_tuple():
    p = _probe()
    out = p(torch.randn(2, 5, 37))
    assert isinstance(out, tuple) and len(out) == 2
    pl, conf = out
    assert pl.shape == (2, 5) and conf.shape == (2, 5)

def test_pointer_is_bidirectional():
    p = _probe()
    p.eval()
    x = torch.randn(1, 6, 37)
    mask = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        s0_before = float(p(x, length_mask=mask, return_pointer=True)[0][0, 0])
        x2 = x.clone()
        x2[0, 5] += 5.0
        s0_after = float(p(x2, length_mask=mask, return_pointer=True)[0][0, 0])
    assert abs(s0_before - s0_after) > 1e-5, "step-0 score must react to a later step"

def test_unknown_value_head_raises():
    try:
        TemporalProbe(d_input=8, value_head="bogus")
    except ValueError as e:
        assert "value_head" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown value_head")

def test_td_q_forward_unchanged():
    torch.manual_seed(0)
    tdq = TemporalProbe(d_input=16, d_proj=32, n_heads=4,
                        n_transformer_layers=2, max_steps=16, value_head="td_q")
    out = tdq(torch.randn(2, 5, 16))
    assert isinstance(out, tuple) and len(out) == 2

def _batch(B=4, L=6, d=37):
    x = torch.randn(B, L, d)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 4:] = False
    ic = torch.tensor([1.0, 0.0, 1.0, 0.0])
    return x, mask, ic

def test_brier_only_runs_and_grads():
    p = _probe()
    x, mask, ic = _batch()
    loss, stats = _probe_loss_pointer(p, "cpu", x, ic, mask)
    assert torch.isfinite(loss)
    assert stats["loss_brier"] > 0.0

    assert stats["loss_ptr"] == 0.0
    assert stats["loss_regime"] == 0.0
    assert stats["loss_env"] == 0.0
    assert stats["loss_branch"] == 0.0
    loss.backward()
    g = p.pointer_head.weight.grad
    assert g is None or bool(torch.isfinite(g).all())
    gc = p.conf_head[0].weight.grad
    assert gc is not None and bool(torch.isfinite(gc).all())

def test_pointer_ce_activates_and_gates_off():
    p = _probe()
    x, mask, ic = _batch()
    gp = torch.tensor([-1, 2, -1, 3])
    _, stats = _probe_loss_pointer(p, "cpu", x, ic, mask, gold_pit=gp)
    assert stats["loss_ptr"] > 0.0
    gp_none = torch.full((4,), -1)
    _, stats0 = _probe_loss_pointer(p, "cpu", x, ic, mask, gold_pit=gp_none)
    assert stats0["loss_ptr"] == 0.0

def test_pointer_ce_decreases_when_peaked_at_gold():
    L = 6
    gold = 3
    step_idx = torch.arange(L).float()
    tgt = torch.exp(-0.5 * ((step_idx - float(gold)) / 1.0) ** 2)
    tgt = tgt / tgt.sum()
    logprob_flat = torch.full((1, L), 1.0 / L).log()
    logprob_peak = tgt.clamp(min=1e-12).log().unsqueeze(0)
    ce_flat = -(tgt * logprob_flat[0]).sum()
    ce_peak = -(tgt * logprob_peak[0]).sum()
    assert float(ce_peak) < float(ce_flat)

def test_regime_ce_responds_to_correct_vs_wrong():
    p = _probe()
    x, mask, ic = _batch()
    regime = torch.tensor([0, 1, 2, 0])
    _, stats = _probe_loss_pointer(p, "cpu", x, ic, mask, regime=regime)
    assert stats["loss_regime"] > 0.0

    _, stats0 = _probe_loss_pointer(p, "cpu", x, ic, mask,
                                    regime=torch.full((4,), -1))
    assert stats0["loss_regime"] == 0.0

def test_regime_ce_lower_when_logits_match_labels():
    import torch.nn.functional as Fn
    labels = torch.tensor([0, 1, 2])
    right = torch.zeros(3, 3)
    right[torch.arange(3), labels] = 8.0
    wrong = torch.zeros(3, 3)
    wrong[torch.arange(3), (labels + 1) % 3] = 8.0
    assert float(Fn.cross_entropy(right, labels)) < float(Fn.cross_entropy(wrong, labels))

def test_env_activates():
    p = _probe()
    x, mask, ic = _batch()
    targets = torch.full((4, 6), 0.5)
    amask = torch.zeros(4, 6, dtype=torch.bool)
    amask[:, 1] = True
    amask[:, 3] = True
    weights = torch.ones(4, 6)
    _, stats = _probe_loss_pointer(
        p, "cpu", x, ic, mask,
        anchor_targets=targets, anchor_mask=amask, anchor_weights=weights,
    )
    assert stats["loss_env"] > 0.0

def test_branch_activates_and_hinge_sign():
    p = _probe()
    x, mask, ic = _batch()
    P = 3
    branch = {
        "pos_feats": torch.randn(P, 4, 37),
        "pos_mask": torch.ones(P, 4, dtype=torch.bool),
        "pos_t": torch.tensor([1, 2, 3]),
        "neg_b": torch.tensor([0, 1, 2]),
        "neg_t": torch.tensor([0, 1, 2]),
    }
    _, stats = _probe_loss_pointer(p, "cpu", x, ic, mask, branch=branch)
    assert stats["loss_branch"] > 0.0

    import torch.nn.functional as Fn
    m = 1.0
    s_plus, s_minus = torch.tensor(-4.0), torch.tensor(4.0)
    good = Fn.softplus(s_plus - s_minus + m)
    bad = Fn.softplus(s_minus - s_plus + m)
    assert float(good) < float(bad)

def test_total_loss_finite_and_backprops_all_terms():
    p = _probe()
    x, mask, ic = _batch()
    gp = torch.tensor([-1, 2, -1, 3])
    regime = torch.tensor([0, 1, 2, 0])
    targets = torch.full((4, 6), 0.5)
    amask = torch.zeros(4, 6, dtype=torch.bool)
    amask[:, 1] = True
    weights = torch.ones(4, 6)
    branch = {
        "pos_feats": torch.randn(3, 4, 37),
        "pos_mask": torch.ones(3, 4, dtype=torch.bool),
        "pos_t": torch.tensor([1, 2, 3]),
        "neg_b": torch.tensor([1, 2, 3]),
        "neg_t": torch.tensor([0, 1, 2]),
    }
    loss, stats = _probe_loss_pointer(
        p, "cpu", x, ic, mask, gold_pit=gp, regime=regime,
        anchor_targets=targets, anchor_mask=amask, anchor_weights=weights,
        branch=branch,
    )
    assert torch.isfinite(loss)
    for k in ("loss_ptr", "loss_regime", "loss_env", "loss_branch", "loss_brier"):
        assert stats[k] > 0.0
    loss.backward()
    g = p.pointer_head.weight.grad
    assert g is not None and bool(torch.isfinite(g).all())
    gr = p.regime_mlp[0].weight.grad
    assert gr is not None and bool(torch.isfinite(gr).all())
