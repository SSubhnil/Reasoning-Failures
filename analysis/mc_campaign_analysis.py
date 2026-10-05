from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

THETA = 0.1
MIN_AFTER = 1

def first_passage(curve: list[float], threshold: float, min_after: int) -> dict:
    T = len(curve)
    if T < 2:
        return {"gold_pit": None, "regime": "short"}
    suf = [0.0] * T
    suf[T - 1] = curve[T - 1]
    for t in range(T - 2, -1, -1):
        suf[t] = curve[t] if curve[t] > suf[t + 1] else suf[t + 1]
    pit = next((t for t in range(T) if suf[t] <= threshold), None)
    if pit is None:
        return {"gold_pit": None, "regime": "reaches"}
    if pit == 0:
        return {"gold_pit": None, "regime": "capability"}
    if T - pit < min_after:
        return {"gold_pit": None, "regime": "late_edge"}
    return {"gold_pit": int(pit), "regime": "pit",
            "pit_value": float(curve[pit]), "pre_pit_value": float(curve[pit - 1])}

def _pct(vals: list[float]) -> dict:
    if not vals:
        return {k: None for k in ("min", "p25", "p50", "p75", "p90", "max", "mean")}
    s = sorted(vals)
    n = len(s)

    def q(p: float) -> float:
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return float(s[i])

    return {"min": float(s[0]), "p25": q(0.25), "p50": q(0.50), "p75": q(0.75),
            "p90": q(0.90), "max": float(s[-1]), "mean": sum(s) / n}

def scan_steps(campaign_dir: Path) -> dict:
    slices = sorted(campaign_dir.glob("slice_*"))
    if not slices:
        raise FileNotFoundError(f"No slice_* dirs in {campaign_dir}")

    regimes: Counter = Counter()
    n_traces = 0
    n_wrong = 0

    pit_records: dict[str, dict] = {}
    rel_depth: list[float] = []
    abs_pit: list[float] = []
    Ts_pit: list[float] = []
    pre_vals: list[float] = []
    pit_vals: list[float] = []
    seen: set[str] = set()

    for sd in slices:
        sp = sd / "steps.jsonl"
        if not sp.exists():
            continue
        with open(sp) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                tid = r["trace_id"]
                if tid in seen:
                    continue
                seen.add(tid)
                n_traces += 1
                if not r["is_correct"]:
                    n_wrong += 1
                steps = {int(s): v["p_correct_mc"] for s, v in r["steps"].items()}
                ev = sorted(steps)
                curve = [steps[s] for s in ev]
                lab = first_passage(curve, THETA, MIN_AFTER)
                regimes[lab["regime"]] += 1
                if lab["regime"] != "pit":
                    continue
                idx = lab["gold_pit"]
                real_pit = ev[idx]
                T = int(r["T"])
                pit_records[tid] = {
                    "real_pit": real_pit,
                    "T": T,
                    "pre_pit_value": lab["pre_pit_value"],
                    "pit_value": lab["pit_value"],
                    "pit_minus1_evaluated": (real_pit - 1) in steps,
                }
                rel_depth.append(real_pit / T if T else 0.0)
                abs_pit.append(float(real_pit))
                Ts_pit.append(float(T))
                pre_vals.append(lab["pre_pit_value"])
                pit_vals.append(lab["pit_value"])

    return {
        "n_traces": n_traces,
        "n_wrong": n_wrong,
        "regimes": dict(regimes),
        "n_pit": len(pit_records),
        "pit_records": pit_records,
        "dist": {
            "pit_relative_depth": _pct(rel_depth),
            "pit_abs_step": _pct(abs_pit),
            "pit_trace_T": _pct(Ts_pit),
            "pre_pit_value": _pct(pre_vals),
            "pit_value": _pct(pit_vals),
        },
    }

def crosscheck_gold_pits(campaign_dir: Path, pit_records: dict) -> dict:
    gp_path = campaign_dir / "gold_pits.json"
    if not gp_path.exists():
        return {"present": False}
    with open(gp_path) as f:
        gold = json.load(f)
    merged_pits = {tid for tid, v in gold.items() if v.get("regime") == "pit"}
    mine = set(pit_records)
    return {
        "present": True,
        "merged_n_pit": len(merged_pits),
        "recomputed_n_pit": len(mine),
        "agree": len(merged_pits & mine),
        "only_merged": len(merged_pits - mine),
        "only_recomputed": len(mine - merged_pits),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump-csv", type=Path, default=None,
                    help="Write per-pit-trace rows (rel_depth, pre_pit_value, ...) for plotting.")
    args = ap.parse_args()

    s = scan_steps(args.campaign_dir)
    pit_records = s.pop("pit_records")

    n_pit = len(pit_records)
    n_pm1_eval = sum(1 for v in pit_records.values() if v["pit_minus1_evaluated"])

    if args.dump_csv:
        import csv
        with open(args.dump_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "rel_depth", "pre_pit_value", "pit_value", "abs_pit", "T"])
            for v in pit_records.values():
                T = v["T"]
                w.writerow([args.model, (v["real_pit"] / T) if T else 0.0,
                            v["pre_pit_value"], v["pit_value"], v["real_pit"], T])

    out = {
        "model": args.model,
        "campaign_dir": str(args.campaign_dir),
        "n_traces": s["n_traces"],
        "n_wrong": s["n_wrong"],
        "regimes": s["regimes"],
        "n_pit": n_pit,
        "pit_frac_of_wrong": (n_pit / s["n_wrong"]) if s["n_wrong"] else 0.0,
        "gate1_pit_minus1_evaluated": {
            "n_pit": n_pit,
            "n_evaluated": n_pm1_eval,
            "frac": (n_pm1_eval / n_pit) if n_pit else 0.0,
        },
        "distributions": s["dist"],
        "gold_pits_crosscheck": crosscheck_gold_pits(args.campaign_dir, pit_records),
    }

    print(json.dumps(out, indent=2))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
