#!/usr/bin/env python
'Run value probe vs length rule 8b.'
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .length_rule_baseline import clustered_ci, fit_argmax, hit, predict

ROOT = Path(__file__).resolve().parent.parent
CAMP = ROOT / "runs/hf_pull/mc_rollouts_locus_k16/qwen3-8b/campaign"
LABELS = ROOT / "runs/resp/bipp_locus_8b/gold_pits_by_md5.pt"
THETA = 0.1
VAL_FRAC = 0.3

def split_val_problems(problem_ids: list[str], seed: int) -> set[str]:
    probs = sorted(set(problem_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(probs)
    n_val = int(round(len(probs) * VAL_FRAC)) if len(probs) > 1 else 0
    return set(probs[:n_val])

def build_rows() -> list[dict]:
    mc_gt = torch.load(CAMP / "mc_ground_truth.pt", weights_only=False)
    gold = json.load(open(CAMP / "gold_pits.json"))
    labels = torch.load(LABELS, weights_only=False)
    by_tid = {v["trace_id"]: v for v in labels.values()}

    rows = []
    for tid, gt in mc_gt.items():
        lab = by_tid.get(tid)
        if lab is None:
            continue
        L = int(lab["n_steps"])
        tgt, vis = np.zeros(L), np.zeros(L, bool)
        for s, sv in gt.items():
            s = int(s)
            if 0 <= s < L:
                tgt[s] = float(sv["p_correct_mc"])
                vis[s] = True
        if not vis.any() or tgt[vis].max() <= THETA:
            continue
        rows.append({"problem_id": lab["problem_id"], "regime": gold.get(tid, {}).get("regime"),
                     "gold_pit": int(lab["gold_pit"]), "n_steps": L})
    return rows

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readout", type=Path, default=ROOT / "runs/value_probe/qwen3-8b/readout.json")
    ap.add_argument("--arm", default="hidden", help="label for the probe arm being compared")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = build_rows()
    pids = [r["problem_id"] for r in rows]
    banked = {r["seed"]: r for r in json.load(open(args.readout))["per_seed"]}

    gate_rows = []
    for seed in sorted(banked):
        vs = split_val_problems(pids, seed)
        val = [r for r in rows if r["problem_id"] in vs]
        n_val = len(val)
        pit_n = sum(1 for r in val if r["regime"] == "pit" and r["gold_pit"] >= 0)
        b = banked[seed]
        ok = n_val == b["n_val_traces"] and pit_n == b["pit_n"]
        gate_rows.append({"seed": seed, "n_val": n_val, "n_val_banked": b["n_val_traces"],
                          "pit_n": pit_n, "pit_n_banked": b["pit_n"], "match": ok})
    if not all(g["match"] for g in gate_rows):
        raise RuntimeError("[a1] split reconstruction does not reproduce the banked sizes - "
                           "the matched comparison is not valid; do not report it")

    per_seed = []
    for seed in sorted(banked):
        vs = split_val_problems(pids, seed)
        pit = [r for r in rows if r["regime"] == "pit" and r["gold_pit"] >= 0]
        tr = [r for r in pit if r["problem_id"] not in vs]
        te = [r for r in pit if r["problem_id"] in vs]
        g_tr = np.array([r["gold_pit"] for r in tr], float)
        L_tr = np.array([r["n_steps"] for r in tr], float)
        g_te = np.array([r["gold_pit"] for r in te], float)
        L_te = np.array([r["n_steps"] for r in te], float)
        p_te = np.array([r["problem_id"] for r in te], object)

        f = fit_argmax(g_tr, L_tr)
        pred = predict(f, L_te)
        h = hit(g_te, pred)
        lo, hi = clustered_ci(g_te, pred, p_te, args.n_boot, args.boot_seed)
        probe = float(banked[seed]["implied_locus_hit_pm2"])
        per_seed.append({"seed": seed, "n_pits": len(te), "f": f, "rule_hit": h,
                         "rule_ci95": [lo, hi], "probe_hit": probe, "probe_minus_rule": probe - h})

    d = np.array([r["probe_minus_rule"] for r in per_seed])
    rule_m = float(np.mean([r["rule_hit"] for r in per_seed]))
    rule_s = float(np.std([r["rule_hit"] for r in per_seed]))
    probe_m = float(np.mean([r["probe_hit"] for r in per_seed]))
    probe_s = float(np.std([r["probe_hit"] for r in per_seed]))
    n_beat = int((d > 0).sum())
    print(f"\n[a1] MATCHED, 5 seeds, identical pit sets per seed:")
    print(f"       rule  Hit@±2 = {rule_m:.4f} ± {rule_s:.4f}")
    print(f"       probe Hit@±2 = {probe_m:.4f} ± {probe_s:.4f}")
    print(f"       paired probe-minus-rule = {d.mean():+.4f} ± {d.std():.4f} "
          f"(range {d.min():+.4f} to {d.max():+.4f}); {n_beat}/5 seeds beat the rule")

    out = {"arm": args.arm, "n_recoverable_items": len(rows),
           "n_problems": len(set(pids)), "theta_doom": THETA, "val_frac": VAL_FRAC,
           "fitter": "argmax", "n_boot": args.n_boot, "boot_seed": args.boot_seed,
           "gate": gate_rows, "per_seed": per_seed,
           "mean": {"rule_hit": rule_m, "rule_sd": rule_s, "probe_hit": probe_m,
                    "probe_sd": probe_s, "probe_minus_rule": float(d.mean()),
                    "probe_minus_rule_sd": float(d.std()), "seeds_probe_beats_rule": n_beat},
           "caveats": ["seed-level pairing, n=5; readout.json has no per-pit predictions and no "
                       "probe checkpoints survive, so a per-pit paired bootstrap is not possible",
                       "scored against the production K=8/16 locus ruler, not a gold set "
                       "(no dense reference set is supplied at 8B)",
                       "population is the recoverable-then-failed stratum the probe was scored on, "
                       "not the 9,508-pit bipp_locus_8b set behind the published 0.5003"]}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
