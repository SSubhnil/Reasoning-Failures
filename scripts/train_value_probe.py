from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sst.probe import q_utils
from sst.probe.heads import TemporalProbe
from sst.probe.token_features import concat_hidden_and_logit
from sst.probe.value_targets import build_target_row, is_recoverable, load_mc_step_targets

@dataclass(frozen=True, slots=True)
class ValueExample:
    md5: str
    problem_id: str
    feats: torch.Tensor
    target: torch.Tensor
    k: torch.Tensor
    visited: torch.Tensor

def _build_input(hid: torch.Tensor, logit: torch.Tensor | None, mode: str) -> torch.Tensor:
    if mode == "hidden":
        return hid
    if mode == "hidden_logit":
        if logit is None:
            raise ValueError("[value] feature-mode=hidden_logit but no logit_features cached.")
        return concat_hidden_and_logit(hid.unsqueeze(0).float(), logit.unsqueeze(0).float())[0]
    raise ValueError(f"[value] unknown feature-mode={mode!r}.")

def load_value_dataset(
    shard_dir: Path | str,
    campaign_dir: Path | str,
    *,
    feature_mode: str,
    theta_doom: float,
    exclude: set[str],
    max_traces: int | None = None,
    label_mode: str = "recoverable",
) -> list[ValueExample]:

    targets = load_mc_step_targets(campaign_dir)
    shard_files = sorted(glob.glob(str(Path(shard_dir) / "shard_*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shards in {shard_dir}")
    items: list[ValueExample] = []
    for sf in shard_files:
        shard = torch.load(sf, weights_only=False, map_location="cpu")
        feats = shard["features"]
        logits = shard.get("logit_features")
        md5s = shard["md5"]
        pids = shard["problem_ids"]
        is_correct = shard["is_correct"] if label_mode == "mc_hybrid" else None
        for i, _ in enumerate(shard["canonical_lengths"]):
            md5 = md5s[i]
            pid = str(pids[i])
            if md5 in exclude:
                continue
            lf = logits[i] if logits is not None else None
            x = _build_input(feats[i], lf, feature_mode)
            L = int(x.shape[0])
            if label_mode == "mc_hybrid" and bool(is_correct[i]):

                tgt = torch.ones(L)
                k = torch.ones(L)
                vis = torch.ones(L, dtype=torch.bool)
            else:

                t = targets.get(md5) or targets.get(f"{pid}__{md5[:12]}")
                if t is None:
                    continue
                if label_mode == "recoverable" and not is_recoverable(t, theta_doom):
                    continue
                tgt, k, vis = build_target_row(t, L)
            items.append(ValueExample(md5, pid, x, tgt, k, vis))
            if max_traces and len(items) >= max_traces:
                return items
    return items

def split_by_problem(
    items: list[ValueExample], val_frac: float, seed: int
) -> tuple[list[ValueExample], list[ValueExample]]:

    probs = sorted({it.problem_id for it in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(probs)
    n_val = int(round(len(probs) * val_frac)) if len(probs) > 1 else 0
    val_set = set(probs[:n_val])
    train = [it for it in items if it.problem_id not in val_set]
    val = [it for it in items if it.problem_id in val_set]
    return train, val

def _collate(batch: list[ValueExample], d_input: int, device: str):
    lens = [it.feats.shape[0] for it in batch]
    Lmax, B = max(lens), len(batch)
    hs = torch.zeros(B, Lmax, d_input, device=device)
    lm = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
    tgt = torch.zeros(B, Lmax, device=device)
    kk = torch.zeros(B, Lmax, device=device)
    vis = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
    for bi, it in enumerate(batch):
        L = lens[bi]
        hs[bi, :L] = it.feats.to(device)
        lm[bi, :L] = True
        tgt[bi, :L] = it.target.to(device)
        kk[bi, :L] = it.k.to(device)
        vis[bi, :L] = it.visited.to(device)
    return hs, lm, tgt, kk, vis

@torch.no_grad()
def evaluate_mae(probe, items, d_input, device, batch) -> float:
    if not items:
        return float("nan")
    probe.eval()
    abs_err, n = 0.0, 0
    for start in range(0, len(items), batch):
        hs, lm, tgt, _, vis = _collate(items[start:start + batch], d_input, device)
        V = probe(hs, length_mask=lm)
        m = vis & lm
        if m.any():
            abs_err += float((V[m] - tgt[m]).abs().sum())
            n += int(m.sum())
    probe.train()
    return abs_err / max(n, 1)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--campaign-dir", type=Path, required=True,
                    help="Campaign dir of slice_*/steps.jsonl (raw MC p̂).")
    ap.add_argument("--feature-mode", choices=["hidden", "hidden_logit"], default="hidden")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None, help="cuda|cpu (auto when omitted).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d-proj", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=48)
    ap.add_argument("--theta-doom", type=float, default=0.1,
                    help="Recoverable iff max visited p̂ > theta (else doomed).")
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--max-traces", type=int, default=0, help="0 = all.")
    ap.add_argument("--label-mode", choices=["recoverable", "mc_hybrid"], default="recoverable",
                    help="recoverable (default): recoverable incorrect traces, rollout-rate targets (value probe). "
                         "mc_hybrid: full 144K - correct=terminal Y=1, all incorrect=MC "
                         "(combined probe).")
    ap.add_argument("--exclude-md5-file", type=Path, default=None,
                    help="Newline file of md5(trace) to drop (ruler leakage guard).")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exclude: set[str] = set()
    if args.exclude_md5_file is not None:
        exclude = {ln.strip() for ln in args.exclude_md5_file.read_text().splitlines() if ln.strip()}

    items = load_value_dataset(
        args.shard_dir, args.campaign_dir, feature_mode=args.feature_mode,
        theta_doom=args.theta_doom, exclude=exclude,
        max_traces=args.max_traces or None, label_mode=args.label_mode)
    if not items:
        raise RuntimeError("[value] no traces with features + labels (check shard-dir/campaign-dir/label-mode).")
    d_input = int(items[0].feats.shape[-1])
    train, val = split_by_problem(items, args.val_frac, args.seed)

    probe = TemporalProbe(
        d_input=d_input, d_proj=args.d_proj, n_heads=args.n_heads,
        n_transformer_layers=args.n_layers, max_steps=args.max_steps,
        causal_mask=False, value_head="value", with_conf_head=False,
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    order = np.arange(len(train))
    val_mae = float("nan")
    for epoch in range(args.epochs):
        np.random.shuffle(order)
        probe.train()
        running, nb = 0.0, 0
        for start in range(0, len(order), args.batch):
            batch = [train[i] for i in order[start:start + args.batch].tolist()]
            hs, lm, tgt, kk, vis = _collate(batch, d_input, device)
            loss, stats = q_utils._probe_loss_value(probe, device, hs, tgt, vis, kk, lm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += stats["loss_value"]
            nb += 1
        val_mae = evaluate_mae(probe, val, d_input, device, args.batch)
        print(f"[value] epoch {epoch+1}/{args.epochs} train_loss={running/max(nb,1):.4f} "
              f"val_mae={val_mae:.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "value_probe.pt"
    torch.save(
        {"state_dict": probe.state_dict(), "d_input": d_input,
         "feature_mode": args.feature_mode, "seed": args.seed,
         "value_head": "value", "causal_mask": False, "with_conf_head": False,
         "theta_doom": args.theta_doom, "val_mae": val_mae,
         "config": {"d_proj": args.d_proj, "n_heads": args.n_heads,
                    "n_layers": args.n_layers, "max_steps": args.max_steps}},
        ckpt,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
