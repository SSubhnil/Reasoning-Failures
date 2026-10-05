#!/usr/bin/env python

from __future__ import annotations

import argparse

import numpy as np

from .blindness_common import DELTA_RAW, MODELS, PRIMARY_LOSS, VSTAR_EDGES, build_master, cluster_bootstrap, vstar_bin, write_json

N_BINS = len(VSTAR_EDGES) - 1
BIN_CENTERS = [(VSTAR_EDGES[i] + VSTAR_EDGES[i + 1]) / 2 for i in range(N_BINS)]

def _stat(sub) -> dict:
    regime = sub["regime"].to_numpy()
    conf = sub["conf"].to_numpy(dtype=float)
    vbin = sub["vbin"].to_numpy()
    out: dict[str, float] = {}

    cap = regime == "capability"
    out["cap_cer"] = float(conf[cap].mean()) if cap.any() else np.nan

    pit_cer = np.full(N_BINS, np.nan)
    rch_cer = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        pm = (regime == "pit") & (vbin == b)
        rm = (regime == "reaches") & (vbin == b)
        pc = float(conf[pm].mean()) if pm.any() else np.nan
        rc = float(conf[rm].mean()) if rm.any() else np.nan
        pit_cer[b], rch_cer[b] = pc, rc
        out[f"pit_b{b}_cer"] = pc
        out[f"reaches_b{b}_cer"] = rc
        out[f"gap_b{b}"] = pc - rc

    out["reaches_trend"] = rch_cer[N_BINS - 1] - rch_cer[0]
    out["pit_trend"] = pit_cer[N_BINS - 1] - pit_cer[0]
    return out

def run_model(model: str, loss: str) -> dict:
    m = build_master(model, loss)
    m["vbin"] = vstar_bin(m["vstar0"].to_numpy())

    counts = {b: {"pit": int(((m.regime == "pit") & (m.vbin == b)).sum()),
                  "reaches": int(((m.regime == "reaches") & (m.vbin == b)).sum())}
              for b in range(N_BINS)}
    thin = [b for b in range(N_BINS) if counts[b]["reaches"] < 30]

    boot = cluster_bootstrap(m, _stat)

    bins = []
    gate_bins = []
    for b in range(N_BINS):
        gap = boot[f"gap_b{b}"]
        gap_excl0 = (gap["ci_lo"] is not None and gap["ci_hi"] is not None
                     and (gap["ci_lo"] > 0 or gap["ci_hi"] < 0))
        gap_big = gap["point"] is not None and abs(gap["point"]) >= DELTA_RAW
        bins.append({
            "bin": b,
            "vstar_range": [VSTAR_EDGES[b], VSTAR_EDGES[b + 1]],
            "n_pit": counts[b]["pit"],
            "n_reaches": counts[b]["reaches"],
            "pit_cer": boot[f"pit_b{b}_cer"],
            "reaches_cer": boot[f"reaches_b{b}_cer"],
            "gap": gap,
            "gap_excludes_0": bool(gap_excl0),
            "gap_ge_delta": bool(gap_big),
        })
        if counts[b]["reaches"] >= 30:
            gate_bins.append(gap_excl0 and gap_big)

    model_pass = bool(gate_bins) and all(gate_bins)
    return {
        "model": model,
        "loss": loss,
        "n_traces": int(len(m)),
        "n_dropped_no_locus": m.attrs["n_dropped_no_locus"],
        "capability_cer": boot["cap_cer"],
        "capability_vstar_mean": float(m.loc[m.regime == "capability", "vstar0"].mean()),
        "reaches_trend": boot["reaches_trend"],
        "pit_trend": boot["pit_trend"],
        "thin_reaches_bins(<30)": thin,
        "bins": bins,
        "model_pass": model_pass,
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default=PRIMARY_LOSS)
    args = ap.parse_args(argv)

    results = {m: run_model(m, args.loss) for m in MODELS}
    both_pass = all(results[m]["model_pass"] for m in MODELS)

    fig = {}
    for m in MODELS:
        r = results[m]
        fig[m] = {
            "vstar_bin_centers": BIN_CENTERS,
            "capability": {"vstar": r["capability_vstar_mean"],
                           "cer": r["capability_cer"]["point"],
                           "ci": [r["capability_cer"]["ci_lo"], r["capability_cer"]["ci_hi"]]},
            "pit": {"cer": [b["pit_cer"]["point"] for b in r["bins"]],
                    "ci_lo": [b["pit_cer"]["ci_lo"] for b in r["bins"]],
                    "ci_hi": [b["pit_cer"]["ci_hi"] for b in r["bins"]]},
            "reaches": {"cer": [b["reaches_cer"]["point"] for b in r["bins"]],
                        "ci_lo": [b["reaches_cer"]["ci_lo"] for b in r["bins"]],
                        "ci_hi": [b["reaches_cer"]["ci_hi"] for b in r["bins"]]},
            "gap": {"point": [b["gap"]["point"] for b in r["bins"]],
                    "ci_lo": [b["gap"]["ci_lo"] for b in r["bins"]],
                    "ci_hi": [b["gap"]["ci_hi"] for b in r["bins"]]},
        }

    payload = {
        "task": "T1_value_control_gate_visual",
        "claim": ("matched-V*_0 pit-vs-reaches CER gap survives -> concealment is "
                  "escapability-differential, not a V^pi relabel"),
        "method": ("5-seed held-out-union c_T (BCE primary); CER=P(c_T>=0.5|Y=0); "
                   "fixed V*_0 bins on pit u reaches; cluster-bootstrap over problem_id"),
        "delta_raw": DELTA_RAW,
        "per_model": results,
        "gate_visual_pass_both_scales": both_pass,
        "figure_inputs": fig,
    }
    write_json(payload, "t1_value_control.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
