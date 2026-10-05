#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .blindness_common import (CLASSES, MODELS, PRIMARY_LOSS, ROOT, SCORES, SEEDS, build_master,
                            cluster_bootstrap)
from .residualized_miss_rate_effect import cell_stratified_gap, cluster_bootstrap_fit, make_design
from .miss_rate_by_failure_class import _stat as t3_stat

PROTOCOL = ROOT / "protocols/label_sensitivity.md"
ARMS = ["S", "A", "B", "C"]
SPLIT_DIRS = {"trace": ROOT / "runs/conf_probe/blindspot_scores",
              "problem": ROOT / "runs/conf_probe_psplit/blindspot_scores"}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_corrections(model: str, score_ids: set[str]) -> dict:
    pc = pd.read_parquet(ROOT / f"runs/paper/labels/packed_classes_{model}.parquet")
    bd = pd.read_parquet(ROOT / f"runs/paper/labels/binary_label_defect_{model}.parquet",
                         columns=["trace_key", "problem_id", "gold_class"])
    lg = pd.read_parquet(ROOT / f"runs/paper/labels/letter_gold/letter_gold_traces_{model}.parquet")
    uv = pd.read_parquet(ROOT / f"runs/paper/correction_flags/{model}/unverified_flags.parquet")

    for name, keys in [("packed_classes", pc["trace_key"]), ("binary_label_defect", bd["trace_key"]),
                       ("letter_gold", lg["trace_id"]), ("unverified_flags", uv["trace_key"])]:
        n_match = len(set(keys) & score_ids)
        if n_match == 0 or n_match != keys.nunique():
            raise ValueError(f"{model}: {name} matches {n_match} of {keys.nunique()} score ids")

    flip_a = set(pc.loc[(pc["cls_pre"] != "correct") & (pc["cls_post"] == "correct"), "trace_key"])
    flip_c = set(lg.loc[lg["flip_to_correct"], "trace_id"])
    per_problem = bd.groupby("problem_id")["gold_class"].nunique()
    if (per_problem > 1).any():
        raise ValueError(f"{model}: gold_class varies within a problem")
    binary_problems = set(bd.loc[bd["gold_class"] == "binary", "problem_id"])
    letter_problems = set(lg["problem_id"])
    unverified = set(uv.loc[uv["unverified"], "trace_key"])
    for name, s in [("flip_a", flip_a), ("flip_c", flip_c), ("binary_problems", binary_problems),
                    ("letter_problems", letter_problems), ("unverified", unverified)]:
        if not s:
            raise ValueError(f"{model}: correction set {name} is empty")
    return {
        "flip_a": flip_a, "flip_c": flip_c, "flips": flip_a | flip_c,
        "drop_problems": binary_problems | letter_problems, "unverified": unverified,
        "counts": {"flip_115a": len(flip_a), "flip_115c": len(flip_c),
                   "flip_overlap": len(flip_a & flip_c),
                   "binary_problems": len(binary_problems), "letter_problems": len(letter_problems),
                   "unverified_flagged": len(unverified)},
    }

def arm_mask(df: pd.DataFrame, arm: str, corr: dict, id_col: str, failure_only: bool) -> np.ndarray:
    keep = np.ones(len(df), dtype=bool)
    if arm == "S":
        return keep
    ids = df[id_col].to_numpy()
    if failure_only:
        keep &= ~np.isin(ids, list(corr["flips"]))
    if arm == "B":
        keep &= ~df["problem_id"].isin(corr["drop_problems"]).to_numpy()
    if arm == "C":
        keep &= ~np.isin(ids, list(corr["unverified"]))
    return keep

def taxonomy(model: str, corr: dict) -> dict:
    g = json.loads((ROOT / f"runs/mc_gold/locus_{model}_k16/gold_pits.json").read_text())
    df = pd.DataFrame({"trace_id": list(g), "regime": [r["regime"] for r in g.values()]})
    df["problem_id"] = df["trace_id"].str.split("__").str[0]
    out = {}
    for arm in ARMS:
        sub = df[arm_mask(df, arm, corr, "trace_id", failure_only=True)]
        n = len(sub)
        out[arm] = {"n_incorrect": n, "n_problems": int(sub["problem_id"].nunique()),
                    "counts": {c: int((sub["regime"] == c).sum()) for c in CLASSES},
                    "shares": {c: float((sub["regime"] == c).mean()) for c in CLASSES}}
    return out

def detection_auroc(model: str, corr: dict) -> dict:
    out = {}
    frames = {s: pd.read_parquet(SCORES / f"{model}_{PRIMARY_LOSS}_s{s}.parquet",
                                 columns=["trace_id", "problem_id", "Y", "c_T", "split"])
              for s in SEEDS}
    for arm in ARMS:
        per_seed = []
        for s in SEEDS:
            d = frames[s]
            d = d[d["split"] == "val"]
            y = d["Y"].to_numpy().astype(int)
            if arm != "S":
                y = np.where(d["trace_id"].isin(corr["flips"]).to_numpy(), 1, y)
            keep = arm_mask(d, arm, corr, "trace_id", failure_only=False)
            per_seed.append(float(roc_auc_score(y[keep], d["c_T"].to_numpy()[keep])))
        out[arm] = {"mean": float(np.mean(per_seed)), "sd_ddof1": float(np.std(per_seed, ddof=1)),
                    "per_seed": per_seed}
    return out

def blindness(model: str, corr: dict, n_boot_t2: int) -> dict:
    m3 = build_master(model, PRIMARY_LOSS)
    m2 = build_master(model, PRIMARY_LOSS, with_answer_pos=True)
    out = {}
    for arm in ARMS:
        s3 = m3[arm_mask(m3, arm, corr, "trace_id", failure_only=True)].reset_index(drop=True)
        boot = cluster_bootstrap(s3, t3_stat)
        s2 = m2[arm_mask(m2, arm, corr, "trace_id", failure_only=True)].reset_index(drop=True)
        _, scaler = make_design(s2, None)
        stats = cluster_bootstrap_fit(s2, scaler, n_boot_t2)
        cells = cell_stratified_gap(s2, "pit")
        lo = stats["pit_logodds"]
        gate = bool((lo["ci_lo"] > 0 or lo["ci_hi"] < 0) and stats["pit_ame"]["point"] >= 0.08
                    and cells["all_same_sign"] and cells["all_positive"])
        out[arm] = {
            "t3_n": int(len(s3)),
            "t3_class_counts": {c: int((s3["regime"] == c).sum()) for c in CLASSES},
            "miss_rate": {c: boot[f"{c}_cer"] for c in CLASSES},
            "pit_minus_near_miss": boot["pit_minus_reaches"],
            "cap_minus_near_miss": boot["cap_minus_reaches"],
            "pit_minus_cap": boot["pit_minus_cap"],
            "t2_n": int(len(s2)),
            "t2": stats,
            "t2_cells_all_positive": cells["all_positive"],
            "t2_cells_populated": cells["n_populated"],
            "t2_gate_pass": gate,
        }
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLIT_DIRS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot-t2", type=int, default=1000)
    args = ap.parse_args()
    if SCORES.resolve() != SPLIT_DIRS[args.split].resolve():
        raise ValueError(f"--split {args.split} but SST_CONCEAL_SCORES resolves to {SCORES}")

    payload = {"prereg": str(PROTOCOL.relative_to(ROOT)), "prereg_sha256": sha256(PROTOCOL),
               "driver_sha256": sha256(Path(__file__)), "split": args.split,
               "scores_dir": str(SCORES), "per_model": {}}
    for model in MODELS:
        score_ids = set(pd.read_parquet(SCORES / f"{model}_{PRIMARY_LOSS}_s0.parquet",
                                        columns=["trace_id"])["trace_id"])
        corr = load_corrections(model, score_ids)
        res = {"corrections": corr["counts"], "taxonomy": taxonomy(model, corr),
               "auroc": detection_auroc(model, corr),
               "blindness": blindness(model, corr, args.n_boot_t2)}
        payload["per_model"][model] = res
        print(f"[{model}] {json.dumps(res['corrections'])}", flush=True)
        for arm in ARMS:
            b = res["blindness"][arm]
            print(f"[{model}/{arm}] n_inc={res['taxonomy'][arm]['n_incorrect']} "
                  f"shares={ {c: round(v, 4) for c, v in res['taxonomy'][arm]['shares'].items()} } "
                  f"auroc={res['auroc'][arm]['mean']:.4f} "
                  f"miss={ {c: round(b['miss_rate'][c]['point'], 4) for c in CLASSES} } "
                  f"pit-nm={b['pit_minus_near_miss']['point']:+.4f} "
                  f"ame={b['t2']['pit_ame']['point']:+.4f} "
                  f"[{b['t2']['pit_ame']['ci_lo']:+.4f},{b['t2']['pit_ame']['ci_hi']:+.4f}] "
                  f"gate={b['t2_gate_pass']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
