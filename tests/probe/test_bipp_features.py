from __future__ import annotations

import torch

from sst.probe.bipp_features import (
    N_LOGIT_FEATURES,
    _BLOCK,
    bipp_assemble_input,
    bipp_d_input,
)

def _synthetic_shard(L: int = 7) -> dict:
    torch.manual_seed(0)
    return {
        "span_feats": [torch.randn(L, _BLOCK).to(torch.bfloat16)],
        "final_feats": [torch.randn(L, _BLOCK).to(torch.bfloat16)],
        "answer_feats": [torch.randn(_BLOCK).to(torch.bfloat16)],
        "logit_features": [torch.randn(L, N_LOGIT_FEATURES)],
    }

def test_d_input_value():
    assert bipp_d_input() == 3 * _BLOCK + N_LOGIT_FEATURES == 30725

def test_assemble_shape_and_dtype():
    L = 7
    shard = _synthetic_shard(L)
    u = bipp_assemble_input(shard, 0)
    assert u.shape == (L, bipp_d_input())
    assert u.dtype == torch.float32

def test_answer_block_is_broadcast():
    L = 7
    shard = _synthetic_shard(L)
    u = bipp_assemble_input(shard, 0)
    ans_block = u[:, 2 * _BLOCK:3 * _BLOCK]
    for t in range(1, L):
        assert torch.equal(ans_block[0], ans_block[t])

    assert torch.equal(ans_block[0], shard["answer_feats"][0].float())

def test_step_count_mismatch_raises():
    shard = _synthetic_shard(7)
    shard["logit_features"] = [torch.randn(6, N_LOGIT_FEATURES)]
    try:
        bipp_assemble_input(shard, 0)
    except ValueError as e:
        assert "step-count mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError on step-count mismatch")
