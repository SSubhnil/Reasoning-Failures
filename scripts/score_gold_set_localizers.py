from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sst.probe.heads import TemporalProbe
from sst.probe.q_utils import compute_blindspot_confidence, first_passage_pit
from sst.probe.token_features import concat_hidden_and_logit

ROOT = Path(__file__).resolve().parents[1]
THETA = 0.1
SEEDS = (42, 137, 271, 314, 577)
MODES = ("hidden", "hidden_logit")
PIT_SIGNAL = "advantage"
ADV_BASELINE = "prev"

@dataclass(slots=True)
class Item:
    trace_id: str
    md5: str
    feats: torch.Tensor
    logit: torch.Tensor
    is_correct: bool
    visited: np.ndarray
    gold_pit: int

    def model_input(self, feature_mode: str) -> torch.Tensor:
        if feature_mode == "hidden_logit":
            return concat_hidden_and_logit(self.feats.unsqueeze(0).float(),
                                           self.logit.unsqueeze(0).float())[0]
        return self.feats.float()

def build_items(shard_dir: Path, mc_gt: dict, regime_by_tid: dict, gp_by_md5: dict) -> list[Item]:
    items: list[Item] = []
    for sf in sorted(glob.glob(str(Path(shard_dir) / "shard_*.pt"))):
        sh = torch.load(sf, weights_only=False, map_location="cpu")
        feats, md5s, pids, cls = sh["features"], sh["md5"], sh["problem_ids"], sh["canonical_lengths"]
        logits, isc = sh["logit_features"], sh["is_correct"]
        for i in range(len(cls)):
            md5 = md5s[i]
            if md5 not in gp_by_md5:
                continue
            tid = f"{pids[i]}__{md5[:12]}"
            gt = mc_gt[tid]
            regime = regime_by_tid[tid]
            if regime != "pit":
                raise ValueError(f"{tid} in gold_pits_by_md5 but regime={regime!r} (expected pit)")
            L = int(cls[i])
            vis = np.zeros(L, bool)
            for s in gt:
                s = int(s)
                if 0 <= s < L:
                    vis[s] = True
            items.append(Item(tid, md5, feats[i].float(), logits[i].float(),
                              bool(isc[i]), vis, int(gp_by_md5[md5])))
    return items

def _load_probe(ckpt: Path, value_head: str, with_conf_head: bool, causal_mask: bool) -> TemporalProbe:
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    c = ck["config"]
    probe = TemporalProbe(
        d_input=ck["d_input"], d_proj=c["d_proj"], n_heads=c["n_heads"],
        n_transformer_layers=c["n_layers"], max_steps=c["max_steps"],
        causal_mask=causal_mask, value_head=value_head, with_conf_head=with_conf_head)
    probe.load_state_dict(ck["state_dict"])
    return probe.eval()

def _forward_curves(probe: TemporalProbe, items: list[Item], feature_mode: str,
                    value_head: str) -> list[np.ndarray]:

    curves: list[np.ndarray] = []
    B = 64
    for st in range(0, len(items), B):
        batch = items[st:st + B]
        xis = [it.model_input(feature_mode) for it in batch]
        Lmax = max(x.shape[0] for x in xis)
        D = xis[0].shape[1]
        hs = torch.zeros(len(batch), Lmax, D)
        lm = torch.zeros(len(batch), Lmax, dtype=torch.bool)
        for bi, x in enumerate(xis):
            L = x.shape[0]; hs[bi, :L] = x; lm[bi, :L] = True
        with torch.no_grad():
            out = probe(hs, length_mask=lm)
        cb = (out[0] if value_head == "td_q" else out).cpu().numpy()
        for bi, it in enumerate(batch):
            curves.append(cb[bi, :it.feats.shape[0]])
    return curves

def score_treatment(items: list[Item], curve_by_seed_mode: dict) -> list[dict]:
    rows: list[dict] = []
    for mode in MODES:
        for seed in SEEDS:
            curves = curve_by_seed_mode[("treatment", mode, seed)]
            for it, V in zip(items, curves):
                vis = it.visited
                vis_steps = np.where(vis)[0]
                Vv = V[vis]
                env = np.maximum.accumulate(Vv[::-1])[::-1]
                tau = first_passage_pit(env.tolist(), theta=THETA)
                tau_step = int(vis_steps[tau]) if (tau is not None and 0 <= tau < len(vis_steps)) else -1
                hit = int(tau_step >= 0 and abs(tau_step - it.gold_pit) <= 2)
                rows.append({"trace_id": it.trace_id, "gold_pit": it.gold_pit,
                             "arm": "treatment", "feature_mode": mode, "seed": seed,
                             "implied_pit_step": tau_step, "hit_pm2": hit})
    return rows

def score_control(items: list[Item], curve_by_seed_mode: dict) -> list[dict]:
    rows: list[dict] = []
    for mode in MODES:
        for seed in SEEDS:
            curves = curve_by_seed_mode[("control", mode, seed)]
            for it, q in zip(items, curves):
                fp = compute_blindspot_confidence(
                    q.tolist(), it.is_correct,
                    pit_signal=PIT_SIGNAL, advantage_baseline=ADV_BASELINE)["pit_step"]
                hit = int(fp >= 0 and abs(int(fp) - it.gold_pit) <= 2)
                rows.append({"trace_id": it.trace_id, "gold_pit": it.gold_pit,
                             "arm": "control", "feature_mode": mode, "seed": seed,
                             "implied_pit_step": int(fp), "hit_pm2": hit})
    return rows

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, default=ROOT / "runs/value_probe/ask_a_818/_packed")
    ap.add_argument("--campaign-dir", type=Path, default=ROOT / "runs/resp/campaign_final")
    ap.add_argument("--treatment-hidden-dir", type=Path,
                    default=ROOT / "runs/value_probe/qwen3-4b")
    ap.add_argument("--treatment-hidlogit-dir", type=Path,
                    default=ROOT / "runs/value_probe/qwen3-4b-hidlogit")
    ap.add_argument("--control-dir", type=Path, default=ROOT / "runs/localizer/qwen3-4b")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/value_probe/ask_a_818_pertrace.parquet")
    args = ap.parse_args()

    mc_gt = torch.load(args.campaign_dir / "mc_ground_truth.pt", weights_only=False)
    gold = json.load(open(args.campaign_dir / "gold_pits.json"))
    regime_by_tid = {tid: g.get("regime") for tid, g in gold.items()}
    gp_by_md5 = torch.load(args.campaign_dir / "gold_pits_by_md5.pt", weights_only=False)

    items = build_items(args.shard_dir, mc_gt, regime_by_tid, gp_by_md5)
    if len(items) != len(gp_by_md5):
        raise RuntimeError(f"[ask-a] joined {len(items)} != {len(gp_by_md5)} target 818 - "
                           "shard<->campaign join incomplete.")

    curve_by_seed_mode: dict = {}
    treatment_dir = {"hidden": args.treatment_hidden_dir, "hidden_logit": args.treatment_hidlogit_dir}
    for mode in MODES:
        for seed in SEEDS:
            tck = treatment_dir[mode] / f"s{seed}" / "value_probe.pt"
            cck = args.control_dir / f"{mode}_s{seed}" / "localizer_probe.pt"
            if not tck.exists():
                raise FileNotFoundError(f"treatment ckpt missing: {tck}")
            if not cck.exists():
                raise FileNotFoundError(f"control ckpt missing: {cck}")

            tp = _load_probe(tck, value_head="value", with_conf_head=False, causal_mask=False)
            cp = _load_probe(cck, value_head="td_q", with_conf_head=True, causal_mask=True)
            curve_by_seed_mode[("treatment", mode, seed)] = _forward_curves(tp, items, mode, "value")
            curve_by_seed_mode[("control", mode, seed)] = _forward_curves(cp, items, mode, "td_q")

    rows = score_treatment(items, curve_by_seed_mode) + score_control(items, curve_by_seed_mode)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
