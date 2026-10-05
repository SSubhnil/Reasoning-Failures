from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sst.probe.heads import TemporalProbe

from . import train_conf_probe as _tcp
stratified_split = _tcp.stratified_split
grouped_split = _tcp.grouped_split

def load_corpus_full(shard_dir: Path):
    shard_files = sorted(glob.glob(str(shard_dir / "shard_*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shards in {shard_dir}")
    feats: list[torch.Tensor] = []
    ic: list[float] = []
    pid: list[str] = []
    md5: list[str] = []
    spans: list[list] = []
    steps: list[list] = []
    pre: list[str] = []
    tr: list[str] = []
    for sf in shard_files:
        shard = torch.load(sf, weights_only=False, map_location="cpu")
        sf_feats = shard["features"]
        sf_len = shard["canonical_lengths"]
        for i, L in enumerate(sf_len):
            if L <= 0:
                continue
            f = sf_feats[i]
            if f.layout != torch.strided:
                f = f.to_dense()
            f = f[:L].to(torch.bfloat16)
            feats.append(f)
            ic.append(1.0 if bool(shard["is_correct"][i]) else 0.0)
            pid.append(str(shard["problem_ids"][i]))
            md5.append(str(shard["md5"][i]))
            spans.append(shard["step_spans"][i])
            steps.append(shard["extracted_steps"][i])
            pre.append(shard["prompts"][i])
            tr.append(shard["traces"][i])
    return feats, ic, pid, md5, spans, steps, pre, tr

def load_conf_probe(ckpt: Path, device: str) -> TemporalProbe:
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    if ck.get("value_head") != "confidence":
        raise ValueError(f"{ckpt} value_head={ck.get('value_head')!r}, expected 'confidence'")
    c = ck["config"]
    probe = TemporalProbe(
        d_input=ck["d_input"], d_proj=c["d_proj"], n_heads=c["n_heads"],
        n_transformer_layers=c["n_layers"], max_steps=c["max_steps"],
        dropout=c.get("dropout", 0.1), causal_mask=True, value_head="confidence",
        with_conf_head=True,
    )
    probe.load_state_dict(ck["state_dict"])
    return probe.to(device).eval()

@torch.no_grad()
def confidence_curves(probe: TemporalProbe, feats: list[torch.Tensor],
                      d_input: int, device: str, batch: int) -> list[list[float]]:

    out: list[list[float]] = [None] * len(feats)
    for s in range(0, len(feats), batch):
        idx = list(range(s, min(s + batch, len(feats))))
        lens = [feats[i].shape[0] for i in idx]
        Lmax = max(lens)
        hs = torch.zeros(len(idx), Lmax, d_input, dtype=torch.bfloat16, device=device)
        mask = torch.zeros(len(idx), Lmax, dtype=torch.bool, device=device)
        for bi, i in enumerate(idx):
            hs[bi, : lens[bi]] = feats[i].to(device)
            mask[bi, : lens[bi]] = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            conf = probe(hs, length_mask=mask)
        conf = conf.float().cpu().numpy()
        for bi, i in enumerate(idx):
            out[i] = conf[bi, : lens[bi]].tolist()
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--conf-probe", type=Path, required=True)
    ap.add_argument("--campaign-dir", type=Path, required=True,
                    help="scale-matched MC campaign dir holding gold_pits.json")
    ap.add_argument("--model", choices=["4b", "8b", "14b"], required=True)
    ap.add_argument("--loss", choices=["bce", "brier"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split-by", choices=["trace", "problem"], default="trace",
                    help="must match the ckpt's training split (checked against its "
                         "'split_by' field): 'trace' = stratified_split, 'problem' = "
                         "grouped_split, both reproduced verbatim from the trainer.")
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--preanswer-frac", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    device = "cuda"
    feats, ic, pid, md5, spans, steps, pre, tr = load_corpus_full(args.shard_dir)
    N = len(feats)
    d_input = feats[0].shape[-1]

    ck_split = torch.load(args.conf_probe, weights_only=False,
                          map_location="cpu").get("split_by", "trace")
    if ck_split != args.split_by:
        raise ValueError(f"[scoreB] ckpt split_by={ck_split!r} != --split-by {args.split_by!r}")
    if args.split_by == "problem":
        _, val_idx = grouped_split(ic, pid, args.val_frac, args.seed)
    else:
        _, val_idx = stratified_split(ic, args.val_frac, args.seed)
    split = np.array(["train"] * N, dtype=object)
    split[np.array(val_idx, dtype=int)] = "val"

    gold = json.load(open(args.campaign_dir / "gold_pits.json"))
    regime_by_tid = {tid: g.get("regime") for tid, g in gold.items()}

    probe = load_conf_probe(args.conf_probe, device)
    curves = confidence_curves(probe, feats, d_input, device, args.batch)

    rows = []
    for i in range(N):
        c = curves[i]
        L = len(c)
        pa = min(L - 1, max(0, int(round(args.preanswer_frac * (L - 1)))))
        tid = f"{pid[i]}__{md5[i][:12]}"
        rows.append({
            "trace_id": tid,
            "problem_id": pid[i],
            "Y": int(ic[i]),
            "c_t": c,
            "c_T": float(c[-1]),
            "c_pre_answer_mean": float(np.mean(c[: pa + 1])),
            "regime": regime_by_tid.get(tid),
            "split": split[i],
        })
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
