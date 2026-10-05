from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from sst.probe.heads import TemporalProbe
from sst.probe.value_targets import load_mc_step_targets

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 137, 271, 314, 577]
KAPPA = 0.5
VAL_FRAC = 0.3

from scripts import score_confidence_probe as scb

def load_mc_probe(ckpt: Path, device: str) -> TemporalProbe:
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    if ck.get("value_head") != "value":
        raise ValueError(f"{ckpt} value_head={ck.get('value_head')!r}, expected 'value'")
    c = ck["config"]
    probe = TemporalProbe(
        d_input=ck["d_input"], d_proj=c["d_proj"], n_heads=c["n_heads"],
        n_transformer_layers=c["n_layers"], max_steps=c["max_steps"],
        causal_mask=False, value_head="value", with_conf_head=False,
    )
    probe.load_state_dict(ck["state_dict"])
    return probe.to(device).eval()

def mc_item_pids(shard_dir: Path, campaign_dir: Path, pid_order, md5_order, ic_order) -> list[str]:
    targets = load_mc_step_targets(campaign_dir)
    out = []
    for pid, md5, ic in zip(pid_order, md5_order, ic_order):
        if ic >= 0.5:
            out.append(pid)
        elif targets.get(md5) or targets.get(f"{pid}__{md5[:12]}"):
            out.append(pid)
    return out

def val_problem_set(item_pids: list[str], seed: int) -> set:
    probs = sorted(set(item_pids))
    rng = np.random.default_rng(seed)
    rng.shuffle(probs)
    n_val = int(round(len(probs) * VAL_FRAC)) if len(probs) > 1 else 0
    return set(probs[:n_val])

def _metrics(df: pd.DataFrame, col: str) -> dict:
    sub = df.dropna(subset=[col])
    cT = sub[col].to_numpy(); Y = sub["Y"].to_numpy()
    out = {"n": int(len(sub)), "detection_auroc": float(roc_auc_score(Y, cT))}
    m0 = sub["Y"] == 0
    out["CER_all_wrong"] = float((sub.loc[m0, col] >= KAPPA).mean())
    for reg in ["capability", "pit", "reaches"]:
        mask = (sub["regime"] == reg) & (sub["Y"] == 0)
        out[f"CER_{reg}"] = float((sub.loc[mask, col] >= KAPPA).mean()) if int(mask.sum()) else None
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["4b", "8b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)
    m = args.model
    shard_dir = ROOT / f"runs/hf_pull/probe_corpus_16k/qwen3-{m}/_p1_packed"
    camp_dir = ROOT / f"runs/hf_pull/mc_rollouts_locus_k16/qwen3-{m}/campaign"

    feats, ic, pid, md5, *_ = scb.load_corpus_full(shard_dir)
    tids = [f"{pid[i]}__{md5[i][:12]}" for i in range(len(pid))]
    d_input = int(feats[0].shape[-1])

    item_pids = mc_item_pids(shard_dir, camp_dir, pid, md5, ic)
    val_sets = {s: val_problem_set(item_pids, s) for s in SEEDS}

    per_seed = {}
    for s in SEEDS:
        ck = ROOT / f"runs/mc_conf_probe/qwen3-{m}/seed{s}/value_probe.pt"
        if not ck.exists():
            print(f"[mc-vs-bce] MISSING {ck}", flush=True); continue
        probe = load_mc_probe(ck, args.device)
        curves = scb.confidence_curves(probe, feats, d_input, args.device, args.batch)
        per_seed[s] = np.array([c[-1] for c in curves], dtype=np.float64)
        del probe
        if args.device == "cuda":
            torch.cuda.empty_cache()
    if not per_seed:
        raise RuntimeError("no MC probes found")

    n = len(feats)
    mc_full = np.mean([per_seed[s] for s in per_seed], axis=0)

    held_sum = np.zeros(n); held_cnt = np.zeros(n)
    for s, cT in per_seed.items():
        inval = np.array([pid[i] in val_sets[s] for i in range(n)])
        held_sum[inval] += cT[inval]; held_cnt[inval] += 1
    mc_held = np.where(held_cnt > 0, held_sum / np.maximum(held_cnt, 1), np.nan)

    mc = pd.DataFrame({"trace_id": tids, "is_correct": ic,
                       "mc_full": mc_full, "mc_held": mc_held})

    bce_files = sorted(glob.glob(str(ROOT / f"runs/conf_probe/blindspot_scores/{m}_bce_s*.parquet")))
    parts = [pd.read_parquet(f, columns=["trace_id", "Y", "c_T", "regime", "split"]) for f in bce_files]
    allb = pd.concat(parts)
    bce_full = allb.groupby("trace_id").agg(Y=("Y", "first"), regime=("regime", "first"),
                                            bce_full=("c_T", "mean")).reset_index()
    valb = allb[allb["split"] == "val"]
    bce_held = valb.groupby("trace_id").agg(bce_held=("c_T", "mean")).reset_index()
    bce = bce_full.merge(bce_held, on="trace_id", how="left")

    df = mc.merge(bce, on="trace_id", how="inner")

    res = {"model": m, "n_joined": len(df), "n_mc_seeds": len(per_seed),
           "FULL_SET": {"note": "train-inclusive ensemble (fair for delta, optimistic absolute)",
                        "MC": _metrics(df, "mc_full"), "BCE": _metrics(df, "bce_full")},
           "HELD_OUT": {"note": "each trace scored only by seeds that held it out",
                        "MC": _metrics(df, "mc_held"), "BCE": _metrics(df, "bce_held")}}
    (ROOT / f"runs/mc_conf_probe/mc_vs_bce_{m}.json").write_text(json.dumps(res, indent=2))
    df.to_parquet(ROOT / f"runs/mc_conf_probe/mc_vs_bce_{m}_perseed.parquet")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
