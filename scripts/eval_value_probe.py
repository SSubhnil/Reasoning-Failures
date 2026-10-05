from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sst.probe.heads import TemporalProbe
from sst.probe.q_utils import first_passage_pit
from sst.probe.token_features import concat_hidden_and_logit
from .train_value_probe import split_by_problem

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[1]
CONF_DETECTOR_AUROC = 0.817
K16_K32_HIT = 0.754
THETA = 0.1

@dataclass(slots=True)
class Item:
    md5: str
    problem_id: str
    feats: torch.Tensor
    logit: torch.Tensor | None
    target: np.ndarray
    k: np.ndarray
    visited: np.ndarray
    regime: str | None
    gold_pit: int

    def model_input(self, feature_mode: str) -> torch.Tensor:
        if feature_mode == "hidden_logit":
            if self.logit is None:
                raise ValueError("feature_mode=hidden_logit but no logit_features cached")
            return concat_hidden_and_logit(self.feats.unsqueeze(0).float(),
                                           self.logit.unsqueeze(0).float())[0]
        return self.feats.float()

_MAD_CACHE: dict = {}
def binom_mad(K: int, p: float) -> float:
    key = (int(K), round(float(p), 4))
    if key in _MAD_CACHE:
        return _MAD_CACHE[key]
    K, p = key
    if K <= 0 or p <= 0 or p >= 1:
        _MAD_CACHE[key] = 0.0
        return 0.0
    lg = math.lgamma
    lp, l1 = math.log(p), math.log1p(-p)
    v = sum(math.exp(lg(K + 1) - lg(x + 1) - lg(K - x + 1) + x * lp + (K - x) * l1)
            * abs(x / K - p) for x in range(K + 1))
    _MAD_CACHE[key] = v
    return v

def pava_decreasing(y: np.ndarray) -> np.ndarray:
    g = (-np.asarray(y, float))
    sv: list[float] = []
    sw: list[float] = []
    for val in g:
        v, w = float(val), 1.0
        while sv and sv[-1] > v:
            pv, pw = sv.pop(), sw.pop()
            v = (pv * pw + v * w) / (pw + w); w += pw
        sv.append(v); sw.append(w)
    res: list[float] = []
    for v, w in zip(sv, sw):
        res.extend([v] * int(round(w)))
    return -np.asarray(res[:len(y)])

def _auroc(scores: list[float], labels: list[int]) -> float:
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def build_items(shard_dir: Path, mc_gt: dict, regime_by_tid: dict, gp_by_md5: dict,
                exclude: set[str], theta_doom: float) -> list[Item]:

    items: list[Item] = []
    n_seen = n_nogt = 0
    for sf in sorted(glob.glob(str(Path(shard_dir) / "shard_*.pt"))):
        sh = torch.load(sf, weights_only=False, map_location="cpu")
        feats, md5s, pids, cls = sh["features"], sh["md5"], sh["problem_ids"], sh["canonical_lengths"]
        logits = sh.get("logit_features")
        for i in range(len(cls)):
            n_seen += 1
            md5 = md5s[i]
            if md5 in exclude:
                continue
            tid = f"{pids[i]}__{md5[:12]}"
            gt = mc_gt.get(tid)
            if gt is None:
                n_nogt += 1
                continue
            L = int(cls[i])
            tgt = np.zeros(L); kk = np.zeros(L); vis = np.zeros(L, bool)
            for s, sv in gt.items():
                s = int(s)
                if 0 <= s < L:
                    tgt[s] = float(sv["p_correct_mc"]); kk[s] = int(sv["n_rollouts"]); vis[s] = True
            if not vis.any() or tgt[vis].max() <= theta_doom:
                continue
            lf = logits[i].float() if logits is not None else None
            items.append(Item(md5, str(pids[i]), feats[i].float(), lf, tgt, kk, vis,
                              regime_by_tid.get(tid), int(gp_by_md5.get(md5, -1))))
    print(f"[eval] build_items: {len(items)} recoverable / {n_seen} shard rows "
          f"({n_nogt} had no mc_ground_truth entry)", flush=True)
    return items

def eval_seed(ckpt: Path, items: list[Item]) -> dict:
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    c = ck["config"]
    probe = TemporalProbe(
        d_input=ck["d_input"], d_proj=c["d_proj"], n_heads=c["n_heads"],
        n_transformer_layers=c["n_layers"], max_steps=c["max_steps"],
        causal_mask=ck.get("causal_mask", False), value_head="value", with_conf_head=False)
    probe.load_state_dict(ck["state_dict"]); probe.eval()
    seed = int(ck["seed"]); fm = ck.get("feature_mode", "hidden")
    _, val = split_by_problem(items, 0.3, seed)

    ae, ae0, aem, aemono, floor = [], [], [], [], []
    pbin = {0.0: [], 0.25: [], 0.5: [], 0.75: []}
    cal_v, cal_p = [], []
    pit_hits = pit_n = 0
    rp_scores, rp_labels = [], []

    Vs: list = []
    B = 64
    for st in range(0, len(val), B):
        batch = val[st:st + B]
        xis = [it.model_input(fm) for it in batch]
        Lmax = max(x.shape[0] for x in xis)
        D = xis[0].shape[1]
        hs = torch.zeros(len(batch), Lmax, D)
        lm = torch.zeros(len(batch), Lmax, dtype=torch.bool)
        for bi, x in enumerate(xis):
            L = x.shape[0]; hs[bi, :L] = x; lm[bi, :L] = True
        with torch.no_grad():
            Vb = probe(hs, length_mask=lm).cpu().numpy()
        for bi, it in enumerate(batch):
            Vs.append(Vb[bi, :it.feats.shape[0]])
    for it, V in zip(val, Vs):
        vis = it.visited
        if not vis.any():
            continue
        Vv, pv, kv = V[vis], it.target[vis], it.k[vis]
        tmean = float(pv.mean()); mono = pava_decreasing(pv)
        for vi, pi, ki, mi in zip(Vv, pv, kv, mono):
            ae.append(abs(vi - pi)); ae0.append(pi); aem.append(abs(pi - tmean))
            aemono.append(abs(mi - pi)); floor.append(binom_mad(int(ki), float(pi)))
            b = 0.0 if pi < 0.125 else 0.25 if pi < 0.375 else 0.5 if pi < 0.625 else 0.75
            pbin[b].append(abs(vi - pi)); cal_v.append(vi); cal_p.append(pi)
        env = np.maximum.accumulate(Vv[::-1])[::-1]
        if it.regime in ("pit", "reaches"):
            rp_scores.append(float(env.min())); rp_labels.append(1 if it.regime == "reaches" else 0)
        if it.regime == "pit" and it.gold_pit >= 0:
            tau = first_passage_pit(env.tolist(), theta=THETA)
            vis_steps = np.where(vis)[0]
            tau_step = int(vis_steps[tau]) if (tau is not None and 0 <= tau < len(vis_steps)) else -1
            pit_n += 1
            if tau_step >= 0 and abs(tau_step - it.gold_pit) <= 2:
                pit_hits += 1

    def ece(vs, ps, nb=10):
        vs, ps = np.asarray(vs), np.asarray(ps)
        if not len(vs):
            return float("nan")
        e = 0.0
        for i in range(nb):
            lo, hi = i / nb, (i + 1) / nb
            m = (vs >= lo) & ((vs < hi) if i < nb - 1 else (vs <= hi))
            if m.sum():
                e += m.sum() / len(vs) * abs(vs[m].mean() - ps[m].mean())
        return float(e)

    return {
        "seed": seed, "n_val_traces": len(val), "n_steps": len(ae),
        "mae": float(np.mean(ae)), "floor": float(np.mean(floor)),
        "mae_zero": float(np.mean(ae0)), "mae_tmean": float(np.mean(aem)),
        "mae_monotone": float(np.mean(aemono)), "ece": ece(cal_v, cal_p),
        "mae_by_pbin": {f"p{b}": (round(float(np.mean(v)), 4) if v else None) for b, v in pbin.items()},
        "reaches_vs_pit_auroc": round(_auroc(rp_scores, rp_labels), 4),
        "implied_locus_hit_pm2": round(pit_hits / max(pit_n, 1), 4), "pit_n": pit_n,
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-dir", type=Path, default=ROOT / "runs/value_probe/qwen3-8b")
    ap.add_argument("--shard-dir", type=Path, default=ROOT / "runs/resp/locus_8b_k16/_packed")
    ap.add_argument("--campaign-dir", type=Path, default=ROOT / "runs/resp/locus_8b_k16/campaign")
    ap.add_argument("--exclude-md5-file", type=Path, default=ROOT / "runs/localizer/ruler_md5.txt")
    ap.add_argument("--theta-doom", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exclude = {ln.strip() for ln in args.exclude_md5_file.read_text().splitlines() if ln.strip()}
    mc_gt = torch.load(args.campaign_dir / "mc_ground_truth.pt", weights_only=False)
    gold = json.load(open(args.campaign_dir / "gold_pits.json"))
    regime_by_tid = {tid: g.get("regime") for tid, g in gold.items()}
    gp_by_md5 = torch.load(args.campaign_dir / "gold_pits_by_md5.pt", weights_only=False)

    ckpts = sorted(glob.glob(str(args.value_dir / "s*" / "value_probe.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No value_probe.pt under {args.value_dir}/s*/")
    items = build_items(args.shard_dir, mc_gt, regime_by_tid, gp_by_md5, exclude, args.theta_doom)
    if not items:
        raise RuntimeError("[eval] 0 recoverable items - check the shard<->mc_ground_truth join.")
    rows = [eval_seed(Path(c), items) for c in ckpts]

    def agg(key):
        vs = [r[key] for r in rows if isinstance(r[key], (int, float)) and not math.isnan(r[key])]
        return (round(float(np.mean(vs)), 4), round(float(np.std(vs)), 4)) if vs else (float("nan"), 0.0)

    print(f"\n=== Value-probe held-out readout ({len(rows)} seeds) ===")
    for r in rows:
        print(f"  s{r['seed']}: MAE={r['mae']:.4f} (floor={r['floor']:.4f} zero={r['mae_zero']:.4f} "
              f"tmean={r['mae_tmean']:.4f} mono={r['mae_monotone']:.4f}) ECE={r['ece']:.4f} "
              f"reach-pit-AUROC={r['reaches_vs_pit_auroc']} locusHit±2={r['implied_locus_hit_pm2']} "
              f"(val={r['n_val_traces']} pit_n={r['pit_n']})")
    mae_m, mae_s = agg("mae"); fl_m, _ = agg("floor"); mono_m, _ = agg("mae_monotone")
    z_m, _ = agg("mae_zero"); tm_m, _ = agg("mae_tmean")
    au_m, au_s = agg("reaches_vs_pit_auroc"); hit_m, hit_s = agg("implied_locus_hit_pm2")
    print(f"\n  MEAN: MAE={mae_m}±{mae_s}  floor={fl_m}  zero={z_m}  tmean={tm_m}  monotone={mono_m}")
    print(f"        reach-vs-pit-AUROC={au_m}±{au_s}  implied-locus-Hit±2={hit_m}±{hit_s}")
    print("\n  Decision thresholds:")
    print(f"    [{'PASS' if mae_m < mono_m else 'FAIL'}] MAE {mae_m} < monotone baseline {mono_m}")
    print(f"    [{'PASS' if au_m > CONF_DETECTOR_AUROC else 'FAIL'}] near-miss-vs-pit AUROC {au_m} > conf {CONF_DETECTOR_AUROC}")
    print(f"    [{'PASS' if hit_m >= K16_K32_HIT else 'FAIL'}] implied-locus Hit@±2 {hit_m} >= K16/K32 {K16_K32_HIT}")
    if args.out:
        args.out.write_text(json.dumps({"per_seed": rows, "mean": {
            "mae": mae_m, "floor": fl_m, "zero": z_m, "tmean": tm_m, "monotone": mono_m,
            "reaches_vs_pit_auroc": au_m, "implied_locus_hit_pm2": hit_m}}, indent=2))
        print(f"\nWrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
