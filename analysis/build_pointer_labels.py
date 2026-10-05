#!/usr/bin/env python
'Run build bipp locus labels.'
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

REGIME_CLASSES = ["pit", "capability", "reaches"]
THETA = 0.1
MIN_AFTER = 1

def suffix_max(values: list[float]) -> list[float]:
    out = list(values)
    for t in range(len(out) - 2, -1, -1):
        if out[t + 1] > out[t]:
            out[t] = out[t + 1]
    return out

def first_passage(curve: list[float]) -> tuple[int | None, str]:
    T = len(curve)
    if T < 2:
        return None, "short"
    suf = suffix_max(curve)
    pit = next((t for t in range(T) if suf[t] <= THETA), None)
    if pit is None:
        return None, "reaches"
    if pit == 0:
        return None, "capability"
    if T - pit < MIN_AFTER:
        return None, "late_edge"
    return pit, "pit"

def load_campaign_corpus(corpus_dir: Path) -> dict[str, dict]:
    files = sorted(glob.glob(str(corpus_dir / "shard_*.pt")))
    if not files:
        raise FileNotFoundError(f"no shard_*.pt under {corpus_dir}")
    out: dict[str, dict] = {}
    for sf in files:
        for r in torch.load(sf, weights_only=False, map_location="cpu"):
            md5 = hashlib.md5(r["trace"].encode()).hexdigest()
            tid = f"{r['problem_id']}__{md5[:12]}"
            if tid != r["trace_id"]:
                raise ValueError(
                    f"{sf}: reconstructed trace_id {tid} != stored {r['trace_id']}"
                )
            if md5 in out:
                raise ValueError(f"duplicate md5 {md5} in {sf} - join key not injective")
            out[md5] = {
                "trace_id": r["trace_id"],
                "problem_id": str(r["problem_id"]),
                "n_steps": int(r["canonical_length"]),
                "is_correct": bool(r["is_correct"]),
            }
    return out

def reverify(gold: dict, mc: dict) -> dict[str, int]:
    d = Counter()
    for tid, rec in gold.items():
        ev = sorted(int(s) for s in mc[tid])
        curve = [float(mc[tid][s]["p_correct_mc"]) for s in ev]
        idx, regime = first_passage(curve)
        d["n"] += 1
        if regime != rec["regime"]:
            d["regime_disagree"] += 1
        stored = rec["gold_pit"]
        if regime == "pit":
            if idx >= len(ev):
                d["pit_index_out_of_curve"] += 1
            elif stored != ev[idx]:
                d["gold_pit_disagree"] += 1
            if stored is not None and stored not in set(ev):
                d["gold_pit_not_an_evaluated_step"] += 1
        elif stored is not None:
            d["nonpit_with_gold_pit"] += 1
        if curve != [float(x) for x in rec["curve"]]:
            d["curve_mismatch"] += 1
    return dict(d)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["qwen3-4b", "qwen3-8b"])
    ap.add_argument("--gold", type=Path, required=True, help="campaign gold_pits.json")
    ap.add_argument("--rollouts-root", type=Path,
                    default=ROOT / "runs/mc_rollouts_locus_k16")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    base = args.rollouts_root / args.model
    gold = json.loads(args.gold.read_text())
    mc = torch.load(base / "campaign/mc_ground_truth.pt", weights_only=False,
                    map_location="cpu")
    if set(gold) != set(mc):
        raise ValueError(
            f"gold_pits.json ({len(gold)}) and mc_ground_truth.pt ({len(mc)}) "
            f"key sets differ by {len(set(gold) ^ set(mc))}"
        )

    identity = load_campaign_corpus(base / "corpus")
    tid_to_md5 = {v["trace_id"]: md5 for md5, v in identity.items()}
    missing = sorted(set(gold) - set(tid_to_md5))
    if missing:
        raise KeyError(f"{len(missing)} campaign trace_ids absent from the corpus, "
                       f"e.g. {missing[:3]}")
    print(f"[locus] {args.model}: corpus {len(identity)} traces, campaign {len(gold)}, "
          f"joined {len(gold)}/{len(gold)}")

    print(f"[locus] independent first-passage re-verification (theta={THETA}, "
          f"min_after={MIN_AFTER}) from raw per-step p_correct_mc:")
    for k, v in sorted(reverify(gold, mc).items()):
        print(f"[locus]   {k:34s} {v}")

    pits: dict[str, dict] = {}
    regime: dict[str, int] = {}
    for tid, rec in gold.items():
        md5 = tid_to_md5[tid]
        idn = identity[md5]
        gp = rec["gold_pit"]
        if gp is not None and not 0 <= int(gp) < idn["n_steps"]:
            raise ValueError(f"{tid}: gold_pit {gp} outside [0,{idn['n_steps']})")
        pits[md5] = {
            "gold_pit": int(gp) if gp is not None else -1,
            "trace_id": tid,
            "problem_id": idn["problem_id"],
            "n_steps": idn["n_steps"],
        }
        regime[md5] = REGIME_CLASSES.index(rec["regime"])

    k_max = max(int(v["n_rollouts"]) for st in mc.values() for v in st.values())
    anchors: dict[str, dict] = {}
    n_anchored = 0
    k_hist: Counter = Counter()
    for tid, steps in mc.items():
        md5 = tid_to_md5[tid]
        T = identity[md5]["n_steps"]
        ev = sorted(int(s) for s in steps)
        p_raw = [float(steps[s]["p_correct_mc"]) for s in ev]
        v_env = suffix_max(p_raw)
        tgt = torch.zeros(T)
        tgt_raw = torch.zeros(T)
        w = torch.zeros(T)
        kt = torch.zeros(T)
        m = torch.zeros(T, dtype=torch.bool)
        for s, pr, ve in zip(ev, p_raw, v_env):
            if not 0 <= s < T:
                raise ValueError(f"{tid}: MC step {s} outside [0,{T})")
            K = int(steps[s]["n_rollouts"])
            tgt[s], tgt_raw[s], kt[s], w[s], m[s] = ve, pr, float(K), K / k_max, True
            k_hist[K] += 1
        n_anchored += int(m.sum())
        anchors[md5] = {"anchor_targets": tgt, "anchor_targets_raw": tgt_raw,
                        "anchor_mask": m, "anchor_weights": w,
                        "anchor_n_rollouts": kt}

    by_problem: dict[str, list[str]] = {}
    for md5, rec in pits.items():
        by_problem.setdefault(rec["problem_id"], []).append(md5)
    with_pit = sorted(p for p, ms in by_problem.items()
                      if any(pits[m]["gold_pit"] >= 0 for m in ms))
    without = sorted(set(by_problem) - set(with_pit))
    rng = random.Random(args.split_seed)
    val: list[str] = []
    for group in (with_pit, without):
        g = list(group)
        rng.shuffle(g)
        val.extend(g[:round(args.val_frac * len(g))])
    val_set = set(val)

    def counts(problems: set[str]) -> tuple[int, int, int]:
        ms = [m for p in problems for m in by_problem[p]]
        return (len(problems), len(ms),
                sum(1 for m in ms if pits[m]["gold_pit"] >= 0))

    v_prob, v_tr, v_pit = counts(val_set)
    t_prob, t_tr, t_pit = counts(set(by_problem) - val_set)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(pits, args.out / "gold_pits_by_md5.pt")
    torch.save({"classes": REGIME_CLASSES, "regime": regime},
               args.out / "regime_by_md5.pt")
    torch.save({"k_max": k_max, "weight_rule": "K_t/K_max (global K_max)",
                "traces": anchors}, args.out / "anchors.pt")
    (args.out / "val_problems.json").write_text(json.dumps(sorted(val), indent=1))

    counts_by_class = Counter(REGIME_CLASSES[r] for r in regime.values())
    n = sum(counts_by_class.values())
    inv = {c: n / (len(REGIME_CLASSES) * counts_by_class[c]) for c in REGIME_CLASSES}
    print(f"[locus] regimes {dict(counts_by_class)}")
    print(f"[locus] inverse-frequency class weights "
          + ", ".join(f"{c}={inv[c]:.3f}" for c in REGIME_CLASSES))
    print(f"[locus] anchors: {len(anchors)} traces, {n_anchored} anchored steps, "
          f"K_max={k_max}, per-step K histogram {dict(sorted(k_hist.items()))}")
    print(f"[locus] split seed={args.split_seed} frac={args.val_frac}")
    print(f"[locus]   train problems={t_prob} traces={t_tr} pits={t_pit}")
    print(f"[locus]   val   problems={v_prob} traces={v_tr} pits={v_pit}")
    print(f"[locus] wrote 4 artifacts to {args.out}")

if __name__ == "__main__":
    main()
