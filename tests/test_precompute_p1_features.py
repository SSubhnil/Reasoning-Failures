from __future__ import annotations

import torch

from scripts.precompute_p1_features import has_features, shard_features


def test_shard_features_slices_to_canonical_length_in_bfloat16():
    hs = torch.randn(2, 5, 6)
    out = shard_features(hs, [3, 5])
    assert [tuple(t.shape) for t in out] == [(3, 6), (5, 6)]
    assert all(t.dtype == torch.bfloat16 for t in out)
    assert torch.equal(out[0], hs[0, :3].to(torch.bfloat16))


def test_has_features(tmp_path):
    p = tmp_path / "shard_0000.pt"
    assert not has_features(p)
    torch.save({"traces": ["a"]}, p)
    assert not has_features(p)
    torch.save({"traces": ["a"], "features": [torch.zeros(1, 2)]}, p)
    assert has_features(p)
