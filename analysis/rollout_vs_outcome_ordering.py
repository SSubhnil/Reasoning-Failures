from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

PERSEED_DIR = Path(os.environ.get("SST_CALIB_PERSEED_DIR", ROOT / "runs/conf_probe_psplit"))
OUT_JSON = Path(os.environ.get("SST_CALIB_OUT",
                               ROOT / "outputs/calibration_matched.json"))
NREP = 1000

def auroc_ci(df: pd.DataFrame, score_col: str, pos: str, neg: str) -> dict | None:
    sub = df[df["regime"].isin([pos, neg])].dropna(subset=[score_col]).copy()
    sub["lab"] = (sub["regime"] == pos).astype(int)
    if sub["lab"].nunique() < 2:
        return None
    point = float(roc_auc_score(sub["lab"], sub[score_col]))
    groups = [(g["lab"].to_numpy(), g[score_col].to_numpy())
              for _, g in sub.groupby("problem_id")]
    rng = np.random.default_rng(0)
    ng = len(groups)
    boots = []
    for _ in range(NREP):
        idx = rng.integers(0, ng, ng)
        labs = np.concatenate([groups[i][0] for i in idx])
        scs = np.concatenate([groups[i][1] for i in idx])
        if labs.min() == labs.max():
            continue
        boots.append(roc_auc_score(labs, scs))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_pos": int(sub["lab"].sum()), "n_neg": int((1 - sub["lab"]).sum())}

def matched_cer(df: pd.DataFrame, score_col: str, target_recall: float) -> dict:
    wrong = df[(df["Y"] == 0)].dropna(subset=[score_col])
    thr = float(np.quantile(wrong[score_col], target_recall))
    out = {"threshold": thr}
    for reg in ["capability", "pit", "reaches"]:
        m = wrong["regime"] == reg
        out[f"CER_{reg}"] = float((wrong.loc[m, score_col] >= thr).mean()) if int(m.sum()) else None
    out["pit_minus_reaches"] = (out["CER_pit"] - out["CER_reaches"]
                                if out["CER_pit"] is not None and out["CER_reaches"] is not None else None)
    return out

def main() -> int:
    res = {}
    for m in ["4b", "8b"]:
        df = pd.read_parquet(PERSEED_DIR / f"mc_vs_bce_{m}_perseed.parquet")
        df["problem_id"] = df["trace_id"].str.rsplit("__", n=1).str[0]
        scale = {"n_traces": len(df)}
        for probe in ["bce_held", "mc_held"]:
            scale[probe] = {
                "pit_vs_reaches": auroc_ci(df, probe, "pit", "reaches"),
                "cap_vs_reaches": auroc_ci(df, probe, "capability", "reaches"),
            }

        bce_recall = float((df[(df["Y"] == 0)].dropna(subset=["bce_held"])["bce_held"] < 0.5).mean())
        scale["matched_op"] = {
            "target_overall_error_recall": bce_recall,
            "bce_held": matched_cer(df, "bce_held", bce_recall),
            "mc_held": matched_cer(df, "mc_held", bce_recall),
        }
        res[m] = scale

    def verdict(scale: dict) -> str:
        b = scale["bce_held"]["pit_vs_reaches"]; mc = scale["mc_held"]["pit_vs_reaches"]
        overlap = (mc["ci_hi"] >= b["ci_lo"]) and (b["ci_hi"] >= mc["ci_lo"])
        mc_above_half = mc["ci_lo"] > 0.5
        if overlap or mc_above_half:
            return "SUPERVISION-ROBUST (stronger framing: states encode the differential)"
        if mc["ci_hi"] <= 0.55:
            return "BCE-SPECIFIC (narrow framing: deployed-detector pathology)"
        return "AMBIGUOUS — report both, read branch jointly"
    res["verdict"] = {m: verdict(res[m]) for m in ["4b", "8b"]}

    out = OUT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
