from __future__ import annotations

import math

import torch

from sst.probe import token_features as tf

def test_decision_features_shape_and_names():
    logits = torch.randn(3, 7, 128)
    feats = tf.decision_token_features(logits)
    assert feats.shape == (3, 7, tf.N_LOGIT_FEATURES)
    assert feats.dtype == torch.float32
    assert tf.N_LOGIT_FEATURES == len(tf.LOGIT_FEATURE_NAMES) == 5

def test_entropy_extremes():
    V = 1000

    peaked = torch.full((1, V), -30.0)
    peaked[0, 0] = 30.0
    f = tf.decision_token_features(peaked)[0]
    names = tf.LOGIT_FEATURE_NAMES
    assert f[names.index("entropy")] < 1e-3
    assert f[names.index("max_prob")] > 0.999
    assert f[names.index("prob_margin")] > 0.999
    assert f[names.index("top5_mass")] > 0.999

    uniform = torch.zeros(1, V)
    g = tf.decision_token_features(uniform)[0]
    assert abs(g[names.index("entropy")].item() - math.log(V)) < 1e-3
    assert abs(g[names.index("max_prob")].item() - 1.0 / V) < 1e-5
    assert g[names.index("prob_margin")] < 1e-4

def test_concat_hidden_and_logit():
    h = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    lf = torch.randn(2, 5, tf.N_LOGIT_FEATURES)
    cat = tf.concat_hidden_and_logit(h, lf)
    assert cat.shape == (2, 5, 16 + tf.N_LOGIT_FEATURES)
    assert cat.dtype == torch.bfloat16

def test_concat_rejects_step_mismatch():
    h = torch.randn(2, 5, 16)
    lf = torch.randn(2, 6, tf.N_LOGIT_FEATURES)
    try:
        tf.concat_hidden_and_logit(h, lf)
    except ValueError:
        return
    raise AssertionError("expected ValueError on step-axis mismatch")
