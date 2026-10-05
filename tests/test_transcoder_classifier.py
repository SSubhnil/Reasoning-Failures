from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from analysis.failure_type_classifier_fit import binarize, column_percentile, first_trace_per_problem, presence_scores
from analysis.transcoder_presence_extract import encode, load_failed_traces


def test_column_percentile_counts_implicit_zeros():
    x = sp.csr_matrix(np.array([[0.0, 4.0], [0.0, 2.0], [3.0, 0.0], [0.0, 1.0]]))
    tau = column_percentile(x, np.arange(4), 50.0, block=1)
    assert np.allclose(tau, [0.0, 1.5])


def test_binarize_matches_dense_threshold():
    rng = np.random.default_rng(0)
    dense = np.where(rng.random((6, 5)) < 0.5, rng.random((6, 5)), 0.0)
    tau = np.array([0.0, 0.2, 0.5, 0.9, 0.0])
    assert np.array_equal(binarize(sp.csr_matrix(dense), tau).toarray(), (dense > tau).astype(np.float32))


def test_first_trace_per_problem_keeps_first_seen():
    traces = [{"problem_id": "b"}, {"problem_id": "a"}, {"problem_id": "b"}, {"problem_id": "c"}]
    assert first_trace_per_problem(traces) == [0, 1, 3]


def test_presence_scores_separate_planted_signal():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 40)
    x = rng.random((80, 30)) * (rng.random((80, 30)) < 0.3)
    x[y == 1, :5] += 1.0
    scores = presence_scores(sp.csr_matrix(x), y, np.arange(80), 50.0, seed=0)
    assert scores[y == 1].mean() > scores[y == 0].mean()


class _ReluTranscoder:
    _threshold = None

    def enc(self, x):
        return x - 0.5


def test_encode_relu_and_threshold():
    x = torch.tensor([[0.2, 0.7, 1.5]])
    assert torch.allclose(encode(_ReluTranscoder(), x), torch.tensor([[0.0, 0.2, 1.0]]))
    jump = _ReluTranscoder()
    jump._threshold = torch.tensor(0.6)
    assert torch.allclose(encode(jump, x), torch.tensor([[0.0, 0.0, 1.0]]))


def test_load_failed_traces_orders_by_problem(tmp_path):
    rows = [{"trace_id": "p10__b", "problem_id": "p10", "question": "q", "trace": "t", "is_correct": False},
            {"trace_id": "p1__a", "problem_id": "p1", "question": "q", "trace": "t", "is_correct": False},
            {"trace_id": "p2__c", "problem_id": "p2", "question": "q", "trace": "t", "is_correct": True}]
    torch.save(rows, tmp_path / "shard_000000.pt")
    got = load_failed_traces(tmp_path, {"p10__b", "p1__a"}, {"p1": "pit", "p10": "capability"})
    assert [r[0] for r in got] == ["p1", "p10"]
