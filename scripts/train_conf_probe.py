from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from sst.probe import q_utils
from sst.probe.heads import TemporalProbe
from sst.probe import conf_eval
from sst.shared.rescue import is_rescued_trace

def load_corpus(shard_dir: Path, max_traces: int):
    shard_files = sorted(glob.glob(str(shard_dir / "shard_*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shards in {shard_dir}")

    feats: list[torch.Tensor] = []
    is_correct: list[float] = []
    strict: list[float] = []
    traces: list[str] = []
    lengths: list[int] = []
    keys: list[str] = []
    for sf in shard_files:
        shard = torch.load(sf, weights_only=False, map_location="cpu")
        sf_feats = shard["features"]
        sf_ic = shard["is_correct"]
        sf_tr = shard["traces"]
        sf_len = shard["canonical_lengths"]
        sf_pid = shard["problem_ids"]
        sf_md5 = shard["md5"]
        for i, L in enumerate(sf_len):
            if L <= 0:
                continue
            f = sf_feats[i]
            if f.layout != torch.strided:
                f = f.to_dense()
            f = f[:L].to(torch.bfloat16)
            ic = 1.0 if bool(sf_ic[i]) else 0.0
            tr = sf_tr[i]
            feats.append(f)
            is_correct.append(ic)
            strict.append(1.0 if (ic == 1.0 and not is_rescued_trace(tr)) else 0.0)
            traces.append(tr)
            lengths.append(int(f.shape[0]))
            keys.append(f"{sf_pid[i]}__{str(sf_md5[i])[:12]}")
            if max_traces and len(feats) >= max_traces:
                break
        if max_traces and len(feats) >= max_traces:
            break
    return feats, is_correct, strict, traces, lengths, keys

def load_vstar0(mc_targets: Path, keys: list[str]) -> tuple[list[int], list[float]]:
    gt = torch.load(mc_targets, weights_only=False, map_location="cpu")
    covered_idx: list[int] = []
    vstar0: list[float] = []
    for i, k in enumerate(keys):
        if k not in gt:
            continue
        per_step = gt[k]
        s0 = min(per_step)
        v = per_step[s0]
        p = float(v["p_correct_mc"] if isinstance(v, dict) else v)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"[conf] V*_0 out of [0,1] for {k}: {p}")
        covered_idx.append(i)
        vstar0.append(p)
    if not covered_idx:
        raise RuntimeError(
            f"[conf] 0 of {len(keys)} corpus traces joined {mc_targets}. "
            "Check that the shard dir and the campaign are the same model/scale."
        )
    return covered_idx, vstar0

def grouped_split(is_correct: list[float], problem_ids: list[str], val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, pid in enumerate(problem_ids):
        groups.setdefault(pid, []).append(i)
    strata: dict[float, list[str]] = {}
    for pid, idx in groups.items():
        key = round(4 * float(np.mean([is_correct[i] for i in idx]))) / 4
        strata.setdefault(key, []).append(pid)
    train, val = [], []
    for pids in strata.values():
        rng.shuffle(pids)
        n_val_target = int(round(sum(len(groups[p]) for p in pids) * val_frac))
        n_val = 0
        for p in pids:
            if n_val < n_val_target:
                val.extend(groups[p])
                n_val += len(groups[p])
            else:
                train.extend(groups[p])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val

def stratified_split(is_correct: list[float], val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx_c = np.array([i for i, y in enumerate(is_correct) if y == 1.0])
    idx_w = np.array([i for i, y in enumerate(is_correct) if y == 0.0])
    rng.shuffle(idx_c)
    rng.shuffle(idx_w)
    train, val = [], []
    for idx in (idx_c, idx_w):
        n_val = int(round(len(idx) * val_frac))
        val.extend(idx[:n_val].tolist())
        train.extend(idx[n_val:].tolist())
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val

def make_batch(feats, ic, idx, d_input, device):
    lens = [feats[i].shape[0] for i in idx]
    Lmax, B = max(lens), len(idx)
    hs = torch.zeros(B, Lmax, d_input, dtype=torch.bfloat16, device=device)
    mask = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
    for bi, i in enumerate(idx):
        hs[bi, : lens[bi]] = feats[i].to(device)
        mask[bi, : lens[bi]] = True
    y = torch.tensor([ic[i] for i in idx], dtype=torch.float32, device=device)
    return hs, mask, y, lens

@torch.no_grad()
def eval_loss(probe, feats, ic, idx, d_input, device, batch, loss, label_smooth):
    probe.eval()
    total, n = 0.0, 0
    for s in range(0, len(idx), batch):
        b = idx[s : s + batch]
        hs, mask, y, _ = make_batch(feats, ic, b, d_input, device)
        l, _ = q_utils._probe_loss_confidence(
            probe, device, hs, y, mask, loss=loss, label_smooth=label_smooth)
        total += float(l) * len(b)
        n += len(b)
    return total / max(n, 1)

@torch.no_grad()
def collect_step_probs(probe, feats, idx, d_input, device, batch):
    probe.eval()
    out = [None] * len(idx)
    pos = {i: k for k, i in enumerate(idx)}
    for s in range(0, len(idx), batch):
        b = idx[s : s + batch]
        hs, mask, _, lens = make_batch(feats, [0.0] * len(feats), b, d_input, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            conf = probe(hs, length_mask=mask)
        conf = conf.float().cpu().numpy()
        for bi, i in enumerate(b):
            out[pos[i]] = conf[bi, : lens[bi]].tolist()
    return out

def readout_indices(step_probs, lengths, idx, preanswer_frac):
    term, pre = [], []
    for k, i in enumerate(idx):
        L = len(step_probs[k])
        term.append(L - 1)
        pre.append(min(L - 1, max(0, int(round(preanswer_frac * (L - 1))))))
    return term, pre

def evaluate(step_probs, val_idx, is_correct, strict, preanswer_frac):
    term_idx, pre_idx = readout_indices(step_probs, None, val_idx, preanswer_frac)
    aw = np.array([is_correct[i] for i in val_idx], dtype=np.float64)
    st = np.array([strict[i] for i in val_idx], dtype=np.float64)
    term = np.array([step_probs[k][term_idx[k]] for k in range(len(val_idx))], dtype=np.float64)
    pre = np.array([step_probs[k][pre_idx[k]] for k in range(len(val_idx))], dtype=np.float64)

    regimes = {
        "as_written": conf_eval.summarize_trace_level(term, aw),
        "strict": conf_eval.summarize_trace_level(term, st),
        "pre_answer": conf_eval.summarize_trace_level(pre, aw),
    }
    per_step = conf_eval.per_step_calibration(step_probs, [is_correct[i] for i in val_idx])
    sel = conf_eval.selective_accuracy(term, aw)
    return regimes, per_step, {"coverage": sel["coverage"], "accuracy": sel["accuracy"]}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--model", choices=["4b", "8b", "14b"], required=True)
    ap.add_argument("--loss", choices=["bce", "brier"], default="bce")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--preanswer-frac", type=float, default=0.9)
    ap.add_argument("--max-traces", type=int, default=0, help="0 = all 144K")

    ap.add_argument("--mc-targets", type=Path, default=None,
                    help="campaign mc_ground_truth.pt. RESTRICTS the pool to MC-covered traces. "
                         "Required by --label vstar0, and must ALSO be passed to the terminal arm "
                         "so both arms train on the identical pool and split (that matching IS the "
                         "experiment).")
    ap.add_argument("--label", choices=["terminal", "vstar0"], default="terminal",
                    help="which per-trace scalar is broadcast to every step. 'terminal' = verified "
                         "final-answer correctness in {0,1} (the deployed probe). 'vstar0' = MC "
                         "recovery chance at the first evaluated step, in [0,1]. Mask, pool, split "
                         "granularity, loss and architecture are identical either way - the label is "
                         "the only difference, which is what makes the reversal attributable to it.")
    ap.add_argument("--split-by", choices=["trace", "problem"], default="trace",
                    help="'trace' = original per-trace stratified split (leaks sibling traces "
                         "of one problem across the split). 'problem' keeps all siblings on one "
                         "side, stratified by problem-level mean correctness.")
    ap.add_argument("--d-proj", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feats, is_correct, strict, traces, lengths, keys = load_corpus(args.shard_dir, args.max_traces)
    del traces

    if args.label == "vstar0" and args.mc_targets is None:
        raise ValueError("[conf] --label vstar0 requires --mc-targets")

    target = list(is_correct)
    if args.mc_targets is not None:
        covered_idx, vstar0 = load_vstar0(args.mc_targets, keys)
        n_before = len(feats)
        feats = [feats[i] for i in covered_idx]
        is_correct = [is_correct[i] for i in covered_idx]
        strict = [strict[i] for i in covered_idx]
        lengths = [lengths[i] for i in covered_idx]
        keys = [keys[i] for i in covered_idx]
        target = vstar0 if args.label == "vstar0" else list(is_correct)
        print(f"[conf] MC join: {len(covered_idx)}/{n_before} traces covered "
              f"({len(covered_idx)/n_before:.1%}) by {args.mc_targets}", flush=True)

    N = len(feats)
    d_input = feats[0].shape[-1]
    base_rate = float(np.mean(is_correct))
    if args.split_by == "problem":
        pids = [k.rsplit("__", 1)[0] for k in keys]
        train_idx, val_idx = grouped_split(is_correct, pids, args.val_frac, args.seed)
    else:
        train_idx, val_idx = stratified_split(is_correct, args.val_frac, args.seed)

    probe = TemporalProbe(
        d_input=d_input, d_proj=args.d_proj, n_heads=args.n_heads,
        n_transformer_layers=args.n_layers, max_steps=args.max_steps,
        dropout=args.dropout, value_head="confidence",
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.max_epochs):
        probe.train()
        order = np.array(train_idx)
        rng.shuffle(order)
        order = order.tolist()
        running, nb = 0.0, 0
        for s in range(0, len(order), args.batch):
            b = order[s : s + args.batch]
            if len(b) < 2:
                continue
            hs, mask, y, _ = make_batch(feats, target, b, d_input, device)
            loss, _ = q_utils._probe_loss_confidence(
                probe, device, hs, y, mask, loss=args.loss, label_smooth=args.label_smooth)
            if not torch.isfinite(loss):
                raise RuntimeError(f"[conf] non-finite loss at epoch {epoch+1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            nb += 1

        val = eval_loss(probe, feats, target, val_idx, d_input, device,
                        args.batch, args.loss, args.label_smooth)
        print(f"[conf] epoch {epoch+1}/{args.max_epochs} "
              f"train={running/max(nb,1):.4f} val={val:.4f}"
              + ("  *best*" if val < best_val else ""), flush=True)
        if val < best_val - 1e-5:
            best_val = val
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[conf] early stop at epoch {epoch+1} (best={best_epoch}, val={best_val:.4f})", flush=True)
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    step_probs = collect_step_probs(probe, feats, val_idx, d_input, device, args.batch)
    regimes, per_step, risk_cov = evaluate(step_probs, val_idx, is_correct, strict, args.preanswer_frac)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": probe.state_dict(), "d_input": d_input,
         "value_head": "confidence", "model": args.model, "loss": args.loss,
         "seed": args.seed, "label": args.label, "split_by": args.split_by,
         "mc_targets": str(args.mc_targets) if args.mc_targets else None,
         "config": {"d_proj": args.d_proj, "n_heads": args.n_heads,
                    "n_layers": args.n_layers, "max_steps": args.max_steps,
                    "dropout": args.dropout}},
        args.out / "conf_probe.pt",
    )
    metrics = {
        "model": args.model, "loss": args.loss, "seed": args.seed,
        "label": args.label, "split_by": args.split_by,
        "mc_targets": str(args.mc_targets) if args.mc_targets else None,
        "target_mean": float(np.mean(target)),
        "n_traces": N, "d_input": d_input, "base_rate": base_rate,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "best_epoch": best_epoch, "best_val_loss": best_val,
        "regimes": regimes,
        "per_step_calibration": per_step,
        "risk_coverage": risk_cov,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
