from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .extract_pointer_features import extract_records, resolve_records
from sst.probe.bipp_features import BIPP_LAYERS, BIPP_MODES
from sst.probe.heads import TemporalProbe
from .train_pointer_probe import (
    REGIME_INDEX, collate, load_bipp_dataset, load_gold_pits, load_regimes,
)

PIT, CAPABILITY = REGIME_INDEX["pit"], REGIME_INDEX["capability"]

def hit_at_2(gold: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gold) <= 2))

def auroc(label: np.ndarray, score: np.ndarray) -> float:
    n1, n0 = int(label.sum()), int((1 - label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def ece(label: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    e = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = (pred >= lo) & ((pred < hi) if i < n_bins - 1 else (pred <= hi))
        if m.any():
            e += m.sum() / len(pred) * abs(float(pred[m].mean()) - float(label[m].mean()))
    return float(e)

def clustered_ci(
    label: np.ndarray, score: np.ndarray, groups: np.ndarray, fn,
    *, seed: int = 0, n_boot: int = 1000,
) -> tuple[float, float, float, int]:

    if len(label) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = fn(label, score)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    g2i = {g: np.where(groups == g)[0] for g in uniq}
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([g2i[g] for g in samp])
        v = fn(label[idx], score[idx])
        if np.isfinite(v):
            boots.append(v)
    if not boots:
        return point, float("nan"), float("nan"), len(label)
    return (point, float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)), len(label))

@torch.no_grad()
def forward_seed(ckpt: Path, items: list, d_input: int, device: str, batch: int,
                 mode: str = "span") -> dict:

    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    if ck["d_input"] != d_input:
        raise ValueError(f"[bipp-score] {ckpt} trained at d_input={ck['d_input']}, "
                         f"eval features are {d_input}")
    c = ck["config"]
    probe = TemporalProbe(
        d_input=d_input, d_proj=c["d_proj"], n_heads=c["n_heads"],
        n_transformer_layers=c["n_layers"], max_steps=c["max_steps"],
        causal_mask=ck["causal_mask"], value_head="pointer",
    ).to(device)
    probe.load_state_dict(ck["state_dict"])
    probe.eval()

    ptr_probs: list[np.ndarray] = []
    route = np.zeros(len(items))
    r_final = np.zeros(len(items))
    for start in range(0, len(items), batch):
        blk = items[start:start + batch]
        b = collate(blk, d_input, device, mode)
        pl, rl, R_hat, _ = probe(
            b["hidden_states"], length_mask=b["length_mask"], return_pointer=True,
        )
        p = torch.softmax(pl.float().masked_fill(~b["length_mask"], float("-inf")), dim=1)
        rp = torch.softmax(rl.float(), dim=1)
        for j, it in enumerate(blk):
            ptr_probs.append(p[j, :it.L].cpu().numpy())
            route[start + j] = float(rp[j, CAPABILITY] / (rp[j, CAPABILITY] + rp[j, PIT]))
            r_final[start + j] = float(R_hat[j, it.L - 1])
    return {"seed": int(ck["seed"]), "ptr_probs": ptr_probs,
            "route": route, "r_final": r_final}

def metrics_for(pred: dict, items: list, n_boot: int, seed: int) -> dict:
    groups = np.array([it.problem_id for it in items])
    argmax = np.array([int(np.argmax(p)) for p in pred["ptr_probs"]])

    loc = np.array([i for i, it in enumerate(items)
                    if it.regime == PIT and 0 <= it.gold_pit < it.L], dtype=int)
    if len(loc):
        g_loc = clustered_ci(
            np.array([items[i].gold_pit for i in loc], float),
            argmax[loc].astype(float), groups[loc], hit_at_2, seed=seed, n_boot=n_boot)
    else:
        g_loc = (float("nan"), float("nan"), float("nan"), 0)

    rte = np.array([i for i, it in enumerate(items)
                    if it.regime in (PIT, CAPABILITY)], dtype=int)
    if len(rte):
        g_route = clustered_ci(
            np.array([1.0 if items[i].regime == CAPABILITY else 0.0 for i in rte]),
            pred["route"][rte], groups[rte], auroc, seed=seed, n_boot=n_boot)
    else:
        g_route = (float("nan"), float("nan"), float("nan"), 0)

    g_cal = clustered_ci(
        np.array([it.is_correct for it in items]), pred["r_final"], groups,
        ece, seed=seed, n_boot=n_boot)

    return {
        "pit_hit_pm2": g_loc[0], "pit_hit_pm2_lo": g_loc[1], "pit_hit_pm2_hi": g_loc[2],
        "n_pit": g_loc[3],
        "route_auroc": g_route[0], "route_auroc_lo": g_route[1], "route_auroc_hi": g_route[2],
        "n_route": g_route[3],
        "r_hat_ece": g_cal[0], "r_hat_ece_lo": g_cal[1], "r_hat_ece_hi": g_cal[2],
        "n_cal": g_cal[3],
    }

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resp-dir", type=Path, required=True,
                    help="Dir holding {tag}_s{seed}/bipp_probe.pt.")
    ap.add_argument("--tag", required=True, help="Arm name, e.g. full / b1_causal.")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--student-model", default=None,
                    help="Only loaded if the eval feature cache is incomplete; "
                         "required unless --no-extract.")
    ap.add_argument("--feature-mode", choices=sorted(BIPP_MODES), default="span",
                    help="Must match the trained checkpoints. 'lasttok' reads the "
                         "cached {23,24} shards and cannot be extracted here.")
    ap.add_argument("--restrict-to-labels", action="store_true",
                    help="Read only the md5s in --gold-pit-file (large shard dirs).")
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="Eval trace dir: corpus/ + _packed_bipp/.")
    ap.add_argument("--out", type=Path, required=True, help="CSV path; JSON is written beside it.")
    ap.add_argument("--gold-pit-file", type=Path, required=True,
                    help="gold_pits_by_md5.pt (analysis/build_pointer_labels.py).")
    ap.add_argument("--regime-file", type=Path, required=True,
                    help="regime_by_md5.pt, or the campaign gold_pits.json.")
    ap.add_argument("--val-problem-file", type=Path, default=None,
                    help="JSON list of held-out problem_ids; score ONLY these rows.")
    ap.add_argument("--eval-corpus", type=Path, default=None)
    ap.add_argument("--corpus-index", type=Path, default=None,
                    help="Extract eval features from corpus_index.json instead of a "
                         "BenchmarkTrace corpus dir (identity = --gold-pit-file).")
    ap.add_argument("--eval-cache-dir", type=Path, default=None)
    ap.add_argument("--no-extract", action="store_true",
                    help="Use the existing feature cache; never load the student.")
    ap.add_argument("--target-layers", type=int, nargs="+", default=list(BIPP_LAYERS))
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda|cpu (auto when omitted).")
    args = ap.parse_args(argv)

    corpus_dir = args.eval_corpus or args.eval_dir / "corpus"
    cache_dir = args.eval_cache_dir or args.eval_dir / "_packed_bipp"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not args.no_extract:
        if args.feature_mode != "span":
            raise ValueError(f"[bipp-score] --feature-mode={args.feature_mode} has no "
                             f"extractor; pass --no-extract with a prebuilt cache.")
        if args.student_model is None:
            raise ValueError("[bipp-score] --student-model is required unless --no-extract.")
        records = resolve_records(
            None if args.corpus_index else corpus_dir, args.corpus_index,
            args.gold_pit_file,
        )
        extract_records(records, cache_dir, args.student_model,
                        target_layers=args.target_layers, shard_size=args.shard_size)
    gold_pits = load_gold_pits(args.gold_pit_file)
    items = load_bipp_dataset(
        cache_dir, gold_pits=gold_pits,
        regimes=load_regimes(args.regime_file), anchors=None, exclude=set(),
        mode=args.feature_mode,
        keep_md5=set(gold_pits) if args.restrict_to_labels else None,
    )
    if args.val_problem_file is not None:
        keep = {str(p) for p in json.loads(args.val_problem_file.read_text())}
        items = [it for it in items if it.problem_id in keep]
    if not items:
        raise RuntimeError(f"[bipp-score] no eval rows under {cache_dir}")
    d_input = int(BIPP_MODES[args.feature_mode][0](items[0].rec, 0).shape[-1])

    ckpts = []
    for s in args.seeds:
        p = args.resp_dir / f"{args.tag}_s{s}" / "bipp_probe.pt"
        if not p.exists():
            raise FileNotFoundError(f"[bipp-score] missing checkpoint {p}")
        ckpts.append(p)

    preds = [forward_seed(c, items, d_input, device, args.batch, args.feature_mode)
             for c in ckpts]
    rows = [{"arm": args.tag, "seed": str(p["seed"]),
             **metrics_for(p, items, args.n_boot, args.boot_seed)}
            for p in preds]

    pooled = {
        "ptr_probs": [np.mean([p["ptr_probs"][i] for p in preds], axis=0)
                      for i in range(len(items))],
        "route": np.mean([p["route"] for p in preds], axis=0),
        "r_final": np.mean([p["r_final"] for p in preds], axis=0),
    }
    rows.append({"arm": args.tag, "seed": "pooled",
                 **metrics_for(pooled, items, args.n_boot, args.boot_seed)})

    args.out.parent.mkdir(parents=True, exist_ok=True)

    pred_path = args.out.with_name(args.out.stem + "_predictions.csv")
    with open(pred_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["md5", "problem_id", "n_steps", "gold_pit", "pred_step", "hit", "seed"])
        for p, lbl in [(q, str(q["seed"])) for q in preds] + [(pooled, "pooled")]:
            for i, it in enumerate(items):
                if it.gold_pit < 0:
                    continue
                ps = int(np.argmax(p["ptr_probs"][i]))
                w.writerow([it.md5, it.problem_id, it.L, it.gold_pit, ps,
                            int(abs(ps - it.gold_pit) <= 2), lbl])
    print(f"Wrote {pred_path}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(
        {"arm": args.tag, "seeds": args.seeds, "n_eval_traces": len(items),
         "eval_dir": str(args.eval_dir), "n_boot": args.n_boot, "rows": rows}, indent=2))

    print(f"\n=== BiPP gate - arm={args.tag}, {len(preds)} seeds, {len(items)} eval traces ===")
    for r in rows:
        print(f"  {r['seed']:>7}: pit_hit±2={r['pit_hit_pm2']:.3f} "
              f"[{r['pit_hit_pm2_lo']:.3f}, {r['pit_hit_pm2_hi']:.3f}] (n={r['n_pit']}) | "
              f"route_AUROC={r['route_auroc']:.3f} "
              f"[{r['route_auroc_lo']:.3f}, {r['route_auroc_hi']:.3f}] (n={r['n_route']}) | "
              f"R̂_T ECE={r['r_hat_ece']:.3f} "
              f"[{r['r_hat_ece_lo']:.3f}, {r['r_hat_ece_hi']:.3f}] (n={r['n_cal']})")
    print(f"\nWrote {args.out} and {json_path}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
