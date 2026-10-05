#!/usr/bin/env python

from __future__ import annotations

import argparse

from .blindness_common import (CLASSES, MODELS, PRIMARY_LOSS, build_master,
                            cluster_bootstrap, write_json)

def _stat(sub) -> dict:
    out = {}
    conf = sub["conf"].to_numpy(dtype=float)
    regime = sub["regime"].to_numpy()
    cer = {}
    for cls in CLASSES:
        msk = regime == cls
        cer[cls] = float(conf[msk].mean()) if msk.any() else float("nan")
        out[f"{cls}_cer"] = cer[cls]
    out["pit_minus_reaches"] = cer["pit"] - cer["reaches"]
    out["cap_minus_reaches"] = cer["capability"] - cer["reaches"]
    out["pit_minus_cap"] = cer["pit"] - cer["capability"]
    return out

def run_model(model: str, loss: str) -> dict:
    m = build_master(model, loss)
    boot = cluster_bootstrap(m, _stat)
    counts = {cls: int((m["regime"] == cls).sum()) for cls in CLASSES}
    pit_above_reaches = (boot["pit_cer"]["ci_lo"] is not None
                         and boot["reaches_cer"]["ci_hi"] is not None
                         and boot["pit_cer"]["ci_lo"] > boot["reaches_cer"]["ci_hi"])
    return {
        "model": model, "loss": loss, "n_traces": int(len(m)),
        "class_counts": counts,
        "cer": {cls: boot[f"{cls}_cer"] for cls in CLASSES},
        "pit_minus_reaches": boot["pit_minus_reaches"],
        "cap_minus_reaches": boot["cap_minus_reaches"],
        "pit_minus_cap": boot["pit_minus_cap"],
        "pit_ci_strictly_above_reaches": bool(pit_above_reaches),
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default=PRIMARY_LOSS)
    args = ap.parse_args(argv)

    results = {m: run_model(m, args.loss) for m in MODELS}

    parity = {}
    for cls in CLASSES:
        a, b = results["4b"]["cer"][cls], results["8b"]["cer"][cls]
        overlap = not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])
        parity[cls] = {"4b": [a["ci_lo"], a["ci_hi"]], "8b": [b["ci_lo"], b["ci_hi"]],
                       "ci_overlap": bool(overlap)}

    payload = {
        "task": "T3_per_class_cer",
        "method": "5-seed held-out-union c_T (BCE primary); CER=P(c_T>=0.5|Y=0); cluster-bootstrap over problem_id",
        "per_model": results,
        "cross_scale_parity": parity,
    }
    write_json(payload, "t3_per_class_cer.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
