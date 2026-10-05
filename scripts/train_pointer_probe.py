from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sst.probe import q_utils
from sst.probe.bipp_features import BIPP_MODES
from sst.probe.heads import TemporalProbe

REGIME_CLASSES = ["pit", "capability", "reaches"]
REGIME_INDEX = {"pit": 0, "capability": 1, "reach": 2, "reaches": 2,
                "late_edge": -1, "short": -1}

@dataclass(frozen=True, slots=True)
class BippExample:
    md5: str
    problem_id: str
    trace_id: str
    rec: dict
    L: int
    is_correct: float
    gold_pit: int
    regime: int
    anchor_target: torch.Tensor | None
    anchor_weight: torch.Tensor | None
    anchor_mask: torch.Tensor | None

def load_gold_pits(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in torch.load(path, weights_only=False, map_location="cpu").items():
        if isinstance(v, dict):
            out[str(k)] = int(v["gold_pit"])
        elif isinstance(v, (int, float)):
            out[str(k)] = int(v)
        else:
            raise ValueError(f"[bipp] {path}: unsupported gold-pit value {type(v)} for {k}")
    return out

def load_regimes(path: Path) -> tuple[dict[str, int], str]:
    path = Path(path)
    if path.suffix == ".pt":
        raw = torch.load(path, weights_only=False, map_location="cpu")
        if list(raw["classes"]) != REGIME_CLASSES:
            raise ValueError(f"[bipp] {path} class order {raw['classes']} != {REGIME_CLASSES}")
        return {str(k): int(v) for k, v in raw["regime"].items()}, "md5"
    if path.suffix == ".json":
        out: dict[str, int] = {}
        for tid, rec in json.loads(path.read_text()).items():
            name = rec["regime"]
            if name not in REGIME_INDEX:
                raise ValueError(f"[bipp] unknown regime {name!r} for {tid} in {path}")
            out[str(tid)] = REGIME_INDEX[name]
        return out, "trace_id"
    raise ValueError(f"[bipp] --regime-file must be .pt or .json, got {path}")

def load_anchors(path: Path, target_key: str = "anchor_targets") -> dict[str, dict]:
    raw = torch.load(path, weights_only=False, map_location="cpu")
    if "traces" not in raw:
        raise ValueError(f"[bipp] {path} has no 'traces' key - expected the "
                         f"analysis/build_pointer_labels.py anchors.pt schema")
    return {str(md5): {"target": rec[target_key], "mask": rec["anchor_mask"],
                       "weight": rec["anchor_weights"]}
            for md5, rec in raw["traces"].items()}

def _anchor_row(entry: dict, L: int, md5: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t, m, w = entry["target"], entry["mask"], entry["weight"]
    if t.shape[0] != L:
        raise ValueError(f"[bipp] anchor length {t.shape[0]} != feature step count {L} "
                         f"for md5={md5} - the anchors and the shards were built from "
                         f"different step segmentations")
    return t.float(), w.float(), m.bool()

def load_bipp_dataset(
    shard_dir: Path | str,
    *,
    gold_pits: dict[str, int] | None,
    regimes: tuple[dict[str, int], str] | None,
    anchors: dict[str, dict] | None,
    exclude: set[str],
    mode: str = "span",
    keep_md5: set[str] | None = None,
) -> list[BippExample]:

    shard_files = sorted(glob.glob(str(Path(shard_dir) / "shard_*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"[bipp] no shard_*.pt in {shard_dir}")
    if mode not in BIPP_MODES:
        raise ValueError(f"[bipp] unknown feature mode {mode!r}; have {sorted(BIPP_MODES)}")
    rec_keys = BIPP_MODES[mode][2]
    regime_map, regime_key = regimes if regimes is not None else ({}, "md5")
    items: list[BippExample] = []
    for sf in shard_files:
        shard = torch.load(sf, weights_only=False, map_location="cpu")
        for i, md5 in enumerate(shard["md5"]):
            if md5 in exclude:
                continue
            if keep_md5 is not None and md5 not in keep_md5:
                continue
            pid = str(shard["problem_ids"][i])
            tid = str(shard["trace_ids"][i]) if mode == "span" else str(md5)
            rec = {k: [shard[k][i]] for k in rec_keys}
            L = int(shard[rec_keys[0]][i].shape[0])
            at = aw = am = None
            if anchors is not None and md5 in anchors:
                at, aw, am = _anchor_row(anchors[md5], L, md5)
            items.append(BippExample(
                md5=md5, problem_id=pid, trace_id=tid, rec=rec, L=L,
                is_correct=float(bool(shard["is_correct"][i])),
                gold_pit=int(gold_pits.get(md5, -1)) if gold_pits is not None else -1,
                regime=int(regime_map.get(md5 if regime_key == "md5" else tid, -1)),
                anchor_target=at, anchor_weight=aw, anchor_mask=am,
            ))
    return items

def collate(batch: list[BippExample], d_input: int, device: str, mode: str = "span") -> dict:
    assemble = BIPP_MODES[mode][0]
    B = len(batch)
    Lmax = max(it.L for it in batch)
    hs = torch.zeros(B, Lmax, d_input, device=device)
    lm = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
    at = torch.zeros(B, Lmax, device=device)
    aw = torch.zeros(B, Lmax, device=device)
    am = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
    for bi, it in enumerate(batch):
        hs[bi, :it.L] = assemble(it.rec, 0).to(device)
        lm[bi, :it.L] = True
        if it.anchor_mask is not None:
            at[bi, :it.L] = it.anchor_target.to(device)
            aw[bi, :it.L] = it.anchor_weight.to(device)
            am[bi, :it.L] = it.anchor_mask.to(device)
    return {
        "hidden_states": hs,
        "length_mask": lm,
        "is_correct": torch.tensor([it.is_correct for it in batch], device=device),
        "gold_pit": torch.tensor([it.gold_pit for it in batch], dtype=torch.long, device=device),
        "regime": torch.tensor([it.regime for it in batch], dtype=torch.long, device=device),
        "anchor_targets": at, "anchor_weights": aw, "anchor_mask": am,
    }

def regime_class_weights(items: list[BippExample], device: str) -> torch.Tensor:
    counts = torch.zeros(3)
    for it in items:
        if it.regime >= 0:
            counts[it.regime] += 1
    n = float(counts.sum())
    return torch.tensor([n / (3.0 * max(float(c), 1.0)) for c in counts], device=device)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42, help="Init + shuffle only; NOT a data partition.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--gold-pit-file", type=Path, default=None, help="Omit -> L_ptr = 0.")
    ap.add_argument("--regime-file", type=Path, default=None, help="Omit -> L_regime = 0.")
    ap.add_argument("--anchor-file", type=Path, default=None, help="Omit -> L_env = 0.")
    ap.add_argument("--anchor-target", choices=["envelope", "raw"], default="envelope",
                    help="L_env target: suffix-max MC envelope (default) or raw p̂.")
    ap.add_argument("--exclude-md5-file", type=Path, default=None,
                    help="Newline file of md5(trace) to drop (eval leakage guard).")
    ap.add_argument("--val-problem-file", type=Path, default=None,
                    help="JSON list of held-out problem_ids; matching train rows are dropped.")
    ap.add_argument("--feature-mode", choices=sorted(BIPP_MODES), default="span",
                    help="span = 30725-d span-pooled shards; lasttok = cached {23,24} "
                         "last-token shards (10240-d, paper feature standard).")
    ap.add_argument("--restrict-to-labels", action="store_true",
                    help="Read only the md5s present in the supplied label files "
                         "(for a shard dir far larger than the labelled set).")
    ap.add_argument("--uniform-class-weights", action="store_true",
                    help="Router CE without inverse-frequency weights (sensitivity arm).")
    ap.add_argument("--causal-mask", action="store_true",
                    help="Ablation B1: causal trunk. Default (absent) = bidirectional.")
    ap.add_argument("--device", default=None, help="cuda|cpu (auto when omitted).")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d-proj", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=48)
    ap.add_argument("--lambda-ptr", type=float, default=1.0)
    ap.add_argument("--lambda-regime", type=float, default=1.0)
    ap.add_argument("--lambda-env", type=float, default=0.25)
    ap.add_argument("--beta-conf", type=float, default=0.5)
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    exclude: set[str] = set()
    if args.exclude_md5_file is not None:
        exclude = {ln.strip() for ln in args.exclude_md5_file.read_text().splitlines() if ln.strip()}
    gold_pits = load_gold_pits(args.gold_pit_file) if args.gold_pit_file else None
    regimes = load_regimes(args.regime_file) if args.regime_file else None
    anchors = load_anchors(
        args.anchor_file,
        "anchor_targets" if args.anchor_target == "envelope" else "anchor_targets_raw",
    ) if args.anchor_file else None

    keep_md5: set[str] | None = None
    if args.restrict_to_labels:
        keep_md5 = set()
        for m in (gold_pits, anchors,
                  regimes[0] if regimes is not None and regimes[1] == "md5" else None):
            if m is not None:
                keep_md5 |= set(m)
        if not keep_md5:
            raise RuntimeError("[bipp] --restrict-to-labels needs at least one "
                               "md5-keyed label file to restrict by.")

    items = load_bipp_dataset(
        args.shard_dir, gold_pits=gold_pits, regimes=regimes,
        anchors=anchors, exclude=exclude, mode=args.feature_mode,
        keep_md5=keep_md5,
    )
    if not items:
        raise RuntimeError(f"[bipp] no usable rows in {args.shard_dir} after the md5 exclusion.")
    val_problems: set[str] = set()
    if args.val_problem_file is not None:
        val_problems = {str(p) for p in json.loads(args.val_problem_file.read_text())}
    train = [it for it in items if it.problem_id not in val_problems]
    if not train:
        raise RuntimeError("[bipp] every row fell in --val-problem-file; nothing to train on.")

    assemble, d_input_fn, _ = BIPP_MODES[args.feature_mode]
    d_input = int(assemble(train[0].rec, 0).shape[-1])
    if d_input != d_input_fn():
        raise ValueError(f"[bipp] shard assembles to d_input={d_input}, "
                         f"mode={args.feature_mode} contract says {d_input_fn()}")

    cw = (None if args.uniform_class_weights
          else regime_class_weights(train, device)) if regimes is not None else None
    n_pit = sum(1 for it in train if it.gold_pit >= 0)
    n_reg = sum(1 for it in train if it.regime >= 0)
    n_anc = sum(1 for it in train if it.anchor_mask is not None and bool(it.anchor_mask.any()))

    for path, name, n in ((args.gold_pit_file, "--gold-pit-file", n_pit),
                          (args.regime_file, "--regime-file", n_reg),
                          (args.anchor_file, "--anchor-file", n_anc)):
        if path is not None and n == 0:
            raise RuntimeError(
                f"[bipp] {name}={path} joined 0 of {len(train)} training rows - "
                f"the md5 join is broken. Omit the flag to run without that term.")

    probe = TemporalProbe(
        d_input=d_input, d_proj=args.d_proj, n_heads=args.n_heads,
        n_transformer_layers=args.n_layers, max_steps=args.max_steps,
        causal_mask=args.causal_mask, value_head="pointer",
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    history: list[dict] = []
    order = np.arange(len(train))
    for epoch in range(args.epochs):
        rng.shuffle(order)
        probe.train()
        acc = {k: 0.0 for k in ("loss_ptr", "loss_regime", "loss_env", "loss_branch", "loss_brier")}
        total, nb = 0.0, 0
        for start in range(0, len(order), args.batch):
            batch = [train[i] for i in order[start:start + args.batch].tolist()]
            b = collate(batch, d_input, device, args.feature_mode)
            loss, stats = q_utils._probe_loss_pointer(
                probe, device, b["hidden_states"], b["is_correct"], b["length_mask"],
                gold_pit=b["gold_pit"] if gold_pits is not None else None,
                regime=b["regime"] if regimes is not None else None,
                anchor_targets=b["anchor_targets"] if anchors is not None else None,
                anchor_mask=b["anchor_mask"] if anchors is not None else None,
                anchor_weights=b["anchor_weights"] if anchors is not None else None,
                lambda_ptr=args.lambda_ptr, lambda_regime=args.lambda_regime,
                lambda_env=args.lambda_env, beta_conf=args.beta_conf,
                regime_class_weights=cw, smooth_sigma=args.smooth_sigma,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k in acc:
                acc[k] += stats[k]
            total += float(loss)
            nb += 1
        row = {"epoch": epoch + 1, "loss_total": total / max(nb, 1),
               **{k: v / max(nb, 1) for k, v in acc.items()}}
        history.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "bipp_probe.pt"
    torch.save({
        "state_dict": probe.state_dict(), "d_input": d_input, "seed": args.seed,
        "value_head": "pointer", "causal_mask": args.causal_mask,
        "feature_mode": args.feature_mode,
        "config": {"d_proj": args.d_proj, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "max_steps": args.max_steps},
        "terms": {"ptr": gold_pits is not None, "regime": regimes is not None,
                  "env": anchors is not None, "anchor_target": args.anchor_target},
        "n_train": len(train), "n_val_problems": len(val_problems),
    }, ckpt)
    (args.out / "loss_history.json").write_text(json.dumps(history, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
