#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from .blindness_common import ROOT, cluster_bootstrap, write_json

CAMPAIGN = {
    "4b": ROOT / "runs" / "mc_gold" / "campaign_final" / "gold_pits.json",
    "8b": ROOT / "runs" / "mc_gold" / "locus_8b_k16" / "gold_pits.json",
}
RULER = {"4b": "dense (campaign_final, 818 pits)", "8b": "production K=16 locus (9508 pits)"}
RNG = np.random.default_rng(20260627)
POS_WIN = 0.15

def load_gold_pits(model: str) -> dict:
    g = json.loads(CAMPAIGN[model].read_text())
    return {k: v["gold_pit"] for k, v in g.items()
            if v.get("regime") == "pit" and v.get("gold_pit") is not None}

def mean_ct(model: str, ids: set) -> dict:
    fs = sorted(glob.glob(str(ROOT / f"runs/conf_probe_psplit/blindspot_scores/{model}_bce_s*.parquet")))
    if not fs:
        raise FileNotFoundError(f"no blindspot_scores parquets for {model}")
    acc: dict = {}
    for f in fs:
        d = pd.read_parquet(f, columns=["trace_id", "c_t", "split"])
        d = d[d["trace_id"].isin(ids) & (d["split"] == "val")]
        for tid, ct in zip(d["trace_id"], d["c_t"]):
            acc.setdefault(tid, []).append(np.asarray(ct, dtype=float))
    return {tid: np.mean(np.stack(arrs), axis=0) for tid, arrs in acc.items()}

def build_table(model: str) -> pd.DataFrame:
    gp = load_gold_pits(model)
    ct = mean_ct(model, set(gp))
    rows = []
    n_oob = 0
    for tid, p in gp.items():
        if tid not in ct:
            n_oob += 1
            continue
        c = ct[tid]
        L = len(c)
        if not (1 <= p < L - 1):
            n_oob += 1
            continue
        dc = np.abs(np.diff(c))
        dc_pit = float(dc[p - 1])
        other = np.delete(dc, p - 1)
        dc_rand_mean = float(other.mean()) if other.size else float("nan")
        ridx = int(RNG.integers(0, other.size)) if other.size else -1
        dc_rand_single = float(other[ridx]) if ridx >= 0 else float("nan")

        other_lvl = np.delete(c, p)
        level_pit = float(c[p])
        level_other_mean = float(other_lvl.mean())

        d_pit = p / (L - 1)
        step_idx = np.arange(L)
        d_all = step_idx / (L - 1)
        band = (np.abs(d_all - d_pit) <= POS_WIN) & (step_idx != p) & (step_idx >= 1)
        band_dc = dc[step_idx[band] - 1]
        dc_rand_posmatched = float(band_dc.mean()) if band_dc.size else float("nan")
        band_lvl = c[step_idx[(np.abs(d_all - d_pit) <= POS_WIN) & (step_idx != p)]]
        level_other_posmatched = float(band_lvl.mean()) if band_lvl.size else float("nan")
        rows.append({
            "trace_id": tid,
            "problem_id": tid.split("__")[0],
            "gold_pit": p, "length": L, "depth": d_pit,
            "dc_pit": dc_pit,
            "dc_rand_mean": dc_rand_mean,
            "dc_rand_single": dc_rand_single,
            "dc_rand_posmatched": dc_rand_posmatched,
            "n_band": int(band.sum()),
            "diff_pit_minus_rand": dc_pit - dc_rand_mean,
            "diff_pit_minus_posmatched": dc_pit - dc_rand_posmatched,
            "signed_dc_pit": float(c[p] - c[p - 1]),
            "level_pit": level_pit,
            "level_other_mean": level_other_mean,
            "level_pit_minus_other": level_pit - level_other_mean,
            "level_pit_minus_posmatched": level_pit - level_other_posmatched,
            "pre": float(c[p - 1]), "at": float(c[p]), "post": float(c[p + 1]),
        })
    df = pd.DataFrame(rows)
    df.attrs["n_excluded_oob"] = n_oob
    return df

def _stat(sub) -> dict:
    return {
        "mean_dc_pit": float(sub["dc_pit"].mean()),
        "mean_dc_rand": float(sub["dc_rand_mean"].mean()),
        "mean_dc_rand_single": float(sub["dc_rand_single"].mean()),
        "mean_dc_rand_posmatched": float(sub["dc_rand_posmatched"].mean()),
        "mean_diff_pit_minus_rand": float(sub["diff_pit_minus_rand"].mean()),
        "mean_diff_pit_minus_posmatched": float(sub["diff_pit_minus_posmatched"].mean()),
        "mean_signed_dc_pit": float(sub["signed_dc_pit"].mean()),
        "mean_level_pit": float(sub["level_pit"].mean()),
        "mean_level_other": float(sub["level_other_mean"].mean()),
        "mean_level_pit_minus_other": float(sub["level_pit_minus_other"].mean()),
        "mean_level_pit_minus_posmatched": float(sub["level_pit_minus_posmatched"].mean()),
        "mean_pre": float(sub["pre"].mean()),
        "mean_at": float(sub["at"].mean()),
        "mean_post": float(sub["post"].mean()),
    }

def run_model(model: str) -> dict:
    df = build_table(model)
    boot = cluster_bootstrap(df, _stat)
    diff = boot["mean_diff_pit_minus_rand"]

    not_larger = diff["ci_lo"] is not None and diff["ci_lo"] <= 0

    diff_pm = boot["mean_diff_pit_minus_posmatched"]
    not_larger_pm = diff_pm["ci_lo"] is not None and diff_pm["ci_lo"] <= 0

    lvl = boot["mean_level_pit_minus_other"]
    pit_not_lower = lvl["ci_hi"] is not None and lvl["ci_hi"] >= 0
    return {
        "model": model, "n_traces": int(len(df)),
        "n_excluded_oob": df.attrs["n_excluded_oob"],
        "median_pit_depth": float(df["depth"].median()),
        "stats": boot,
        "pit_dc_not_distinctively_larger": bool(not_larger),
        "pit_dc_not_larger_positionmatched": bool(not_larger_pm),
        "pit_level_not_lower_than_other": bool(pit_not_lower),
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="4b", choices=sorted(CAMPAIGN))
    ap.add_argument("--out", default=None, help="output filename under runs/conf_probe/conceal")
    args = ap.parse_args(argv)

    r = run_model(args.model)
    payload = {
        "task": "T6_spatial_concealment_control",
        "claim": ("no distinctive confidence drop at the known pit: "
                  "|Delta c| at pit <= |Delta c| at a matched random step (within CI)"),
        "method": (f"gold pits from the {RULER[args.model]} ruler; mean c_t over the seeds that held the problem out (problem split); "
                   "|Delta c| at pit vs mean |Delta c| over the trace's other interior "
                   "steps; cluster-bootstrap over problem_id"),
        "gold_pit_source": str(CAMPAIGN[args.model]),
        "per_model": {args.model: r},
    }
    write_json(payload, args.out or f"t6_spatial_control_{args.model}.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
