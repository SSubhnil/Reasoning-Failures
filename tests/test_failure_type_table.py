from __future__ import annotations

import numpy as np
import pytest

from analysis.failure_type_classifier_table import paired_interval, summarize


def test_paired_interval_brackets_auroc_gap():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(200), 2)
    y = np.tile([0, 1], 200)
    strong = y + rng.normal(0, 0.5, size=y.size)
    weak = y + rng.normal(0, 2.0, size=y.size)
    lo, hi = paired_interval(y, strong, weak, groups, n_boot=200)
    assert 0 < lo < hi
    lo_same, hi_same = paired_interval(y, strong, strong, groups, n_boot=50)
    assert lo_same == hi_same == 0


def row(margin: float, lo: float, hi: float) -> dict:
    return {"transcoder_auroc": 0.70, "hidden_auroc": 0.62, "strongest_baseline_auroc": 0.70 - margin,
            "transcoder_minus_strongest": margin, "transcoder_minus_strongest_ci95": [lo, hi],
            "transcoder_minus_hidden": 0.08, "transcoder_minus_hidden_ci95": [0.05, 0.11]}


def test_summarize_margin_rule_and_counts():
    meets = summarize([row(0.06, 0.03, 0.09)] * 5 + [row(0.05, -0.01, 0.10)])
    assert meets["meets_margin"] and meets["intervals_excluding_zero"] == 5
    assert meets["intervals_containing_margin"] == 6 and meets["partitions_at_or_above_margin"] == 6
    assert meets["transcoder_auroc_mean"] == pytest.approx(0.70)
    misses = summarize([row(0.04, 0.01, 0.07)] * 6)
    assert not misses["meets_margin"] and misses["partitions_at_or_above_margin"] == 0
