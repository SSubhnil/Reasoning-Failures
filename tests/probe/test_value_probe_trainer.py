from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "train_value_probe", ROOT / "scripts" / "train_value_probe.py"
)
train_value_probe = importlib.util.module_from_spec(_spec)
sys.modules["train_value_probe"] = train_value_probe
_spec.loader.exec_module(train_value_probe)

def _make_synthetic(tmp_path: Path, d_input: int = 32, n_problems: int = 8):
    shard_dir = tmp_path / "shards"
    camp_dir = tmp_path / "campaign" / "slice_0_99"
    shard_dir.mkdir(parents=True)
    camp_dir.mkdir(parents=True)

    traces, md5s, pids, lengths, feats = [], [], [], [], []
    steps_lines = []
    for prob in range(n_problems):
        for kind, curve in (("recov", [0.9, 0.6, 0.2, 0.05]), ("doom", [0.0, 0.0, 0.0])):
            text = f"problem-{prob}-{kind}-trace-body"
            md5 = hashlib.md5(text.encode()).hexdigest()
            L = len(curve)
            traces.append(text)
            md5s.append(md5)
            pids.append(f"prob_{prob}")
            lengths.append(L)
            feats.append(torch.randn(L, d_input).to(torch.bfloat16))
            steps = {str(s): {"p_correct_mc": curve[s], "n_rollouts": 16 if s in (1, L - 1) else 8}
                     for s in range(L)}
            steps_lines.append(json.dumps(
                {"trace_id": f"prob_{prob}_{kind}", "md5": md5,
                 "is_correct": False, "T": L, "steps": steps}))

    torch.save({
        "traces": traces, "md5": md5s, "problem_ids": pids,
        "canonical_lengths": lengths, "is_correct": [False] * len(traces),
        "features": feats,
    }, shard_dir / "shard_00000000.pt")
    (camp_dir.parent / "slice_0_99" / "steps.jsonl").write_text("\n".join(steps_lines))
    return shard_dir, camp_dir.parent

def test_trainer_runs_end_to_end_on_cpu(tmp_path):
    shard_dir, campaign_dir = _make_synthetic(tmp_path)
    out = tmp_path / "out"
    rc = train_value_probe.main([
        "--shard-dir", str(shard_dir),
        "--campaign-dir", str(campaign_dir),
        "--feature-mode", "hidden",
        "--out", str(out),
        "--device", "cpu",
        "--epochs", "2", "--batch", "4",
        "--d-proj", "32", "--n-heads", "4", "--n-layers", "2", "--max-steps", "16",
        "--val-frac", "0.25", "--seed", "0",
    ])
    assert rc == 0
    ckpt = out / "value_probe.pt"
    assert ckpt.exists()
    d = torch.load(ckpt, weights_only=False, map_location="cpu")
    assert d["value_head"] == "value"
    assert d["causal_mask"] is False
    assert d["with_conf_head"] is False
    assert "val_mae" in d

def test_trainer_trains_only_recoverable_stratum(tmp_path):
    shard_dir, campaign_dir = _make_synthetic(tmp_path)
    items = train_value_probe.load_value_dataset(
        shard_dir, campaign_dir, feature_mode="hidden", theta_doom=0.1, exclude=set())
    assert len(items) == 8
    assert all(it.problem_id.startswith("prob_") for it in items)
