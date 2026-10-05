#!/usr/bin/env python
'Write per-step hidden-state features into packed trace shards.'
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sst.shared.extractors import SkyworkExtractor
from sst.student.model import StudentModel

ROOT = Path(__file__).resolve().parent.parent


def has_features(path: Path) -> bool:
    return path.exists() and "features" in torch.load(path, weights_only=False, map_location="cpu")


def shard_features(hidden_states: torch.Tensor, canonical_lengths: list[int]) -> list[torch.Tensor]:
    return [hidden_states[i, :length].detach().cpu().to(torch.bfloat16) for i, length in enumerate(canonical_lengths)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--target-layers", type=int, nargs="+", default=[23, 24])
    ap.add_argument("--student-model", default=str(ROOT / "models/student/Qwen3-4B"))
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.shard_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(args.shard_dir.glob("shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shards in {args.shard_dir}")
    end = len(shard_files) if args.shard_end < 0 else args.shard_end
    my_files = shard_files[args.shard_start:end]
    if not my_files:
        raise ValueError(f"Empty slice [{args.shard_start},{end}) of {len(shard_files)} shards")
    pending = [sf for sf in my_files if not has_features(out_dir / sf.name)]
    if not pending:
        return 0

    student = StudentModel(args.student_model, SkyworkExtractor(option="back_line_k"),
                           target_layers=list(args.target_layers), vllm_client=None)
    student.model = student.model.to("cuda")
    assert student.model.device.type == "cuda"
    student.model.eval()

    for sf in pending:
        shard = torch.load(sf, weights_only=False, map_location="cpu")
        shard.pop("features", None)
        gen = student.reencode_traces(prompts=shard["prompts"], traces=shard["traces"], step_spans=shard["step_spans"])
        shard["features"] = shard_features(gen.hidden_states, shard["canonical_lengths"])
        torch.save(shard, out_dir / sf.name)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
