#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_MIN_SENT_CHARS = 3

_PLANNING = re.compile(
    r"\b(let'?s|let me|i (?:will|'ll|should|need to|can)|we (?:can|should|need to|will)|"
    r"first,|next,|to (?:solve|find|compute|show|prove|evaluate)|"
    r"the (?:plan|approach|idea|strategy) is|start(?:ing)? by|begin(?:ning)? by|"
    r"consider|recall that)\b", re.I)
_DERIVATION = re.compile(
    r"(=|\\frac|\\sum|\\int|\\lim|\b(?:therefore|thus|hence|gives|equals|"
    r"substitut\w*|plug\w*|simplif\w*|comput\w*|calculat\w*|expand\w*|factor\w*|"
    r"deriv\w*|integrat\w*)\b)", re.I)
_VERIFICATION = re.compile(
    r"\b(check\w*|verif\w*|confirm\w*|make sure|sanity|double[- ]check\w*|"
    r"does (?:this|it) (?:work|hold|satisfy)|is (?:this|that) (?:correct|right)|"
    r"indeed|consistent with|match(?:es)?)\b", re.I)
_BACKTRACKING = re.compile(
    r"\b(wait|hmm+|hold on|actually|nope|"
    r"that'?s (?:wrong|not right|incorrect)|mistake|error|"
    r"let me (?:redo|re-?do|reconsider|rethink|try again|go back|start over)|"
    r"scratch that|on second thought|instead)\b|(?<![A-Za-z])no,", re.I)
_FINALIZATION = re.compile(
    r"(\\boxed|\bfinal answer\b|\bthe answer is\b|\bhence the answer\b|\banswer:)", re.I)

ROLES = ["backtracking", "finalization", "verification", "planning", "derivation", "other"]
_PATTERNS = [
    ("backtracking", _BACKTRACKING),
    ("finalization", _FINALIZATION),
    ("verification", _VERIFICATION),
    ("planning", _PLANNING),
    ("derivation", _DERIVATION),
]
_ROLE_IDX = {r: i for i, r in enumerate(ROLES)}
_K = len(ROLES)

DENSITY_ROLES = ["planning", "derivation", "verification", "backtracking", "finalization"]
FEATURES = ([f"dens_{r}" for r in DENSITY_ROLES]
            + ["cycle_rate", "weighted_self_transition", "transition_entropy",
               "finalization_timing", "log_n_sentences", "log_mean_sent_chars"])

def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= _MIN_SENT_CHARS]

def label_roles(sents: list[str]) -> np.ndarray:
    out = np.empty(len(sents), dtype=np.int64)
    for i, s in enumerate(sents):
        code = _ROLE_IDX["other"]
        for name, pat in _PATTERNS:
            if pat.search(s):
                code = _ROLE_IDX[name]
                break
        out[i] = code
    return out

def dynamics(roles: np.ndarray, sent_chars: np.ndarray) -> dict[str, float]:
    T = len(roles)
    counts = np.bincount(roles, minlength=_K).astype(float)
    dens = counts / T

    P = np.zeros((_K, _K), dtype=float)
    if T >= 2:
        np.add.at(P, (roles[:-1], roles[1:]), 1.0)
    row_tot = P.sum(axis=1)
    emitting = row_tot > 0
    pi = np.zeros(_K, dtype=float)
    if emitting.any():
        pi[emitting] = row_tot[emitting] / row_tot.sum()
    Pn = np.zeros_like(P)
    Pn[emitting] = P[emitting] / row_tot[emitting, None]

    self_trans = float((pi * np.diagonal(Pn)).sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(Pn > 0, np.log(Pn), 0.0)
    row_H = -(Pn * logp).sum(axis=1)
    trans_ent = float((pi * row_H).sum() / np.log(_K))

    if T >= 3:
        hits = 0
        for t in range(2, T):
            if roles[t] != roles[t - 1] and (
                    roles[t] == roles[t - 2] or (t >= 3 and roles[t] == roles[t - 3])):
                hits += 1
        cycle = hits / (T - 2)
    else:
        cycle = 0.0

    fin_idx = np.flatnonzero(roles == _ROLE_IDX["finalization"])
    fin_time = 1.0 if fin_idx.size == 0 else (
        float(fin_idx[0]) / (T - 1) if T > 1 else 1.0)

    out = {f"dens_{r}": float(dens[_ROLE_IDX[r]]) for r in DENSITY_ROLES}
    out.update({
        "cycle_rate": float(cycle),
        "weighted_self_transition": self_trans,
        "transition_entropy": trans_ent,
        "finalization_timing": fin_time,
        "log_n_sentences": float(np.log1p(T)),
        "log_mean_sent_chars": float(np.log1p(sent_chars.mean())),
    })
    return out

def run(shard_dir: Path, out: Path) -> int:
    shard_files = sorted(glob.glob(str(shard_dir / "shard_*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shards in {shard_dir}")
    rows: list[dict] = []
    for n, sf in enumerate(shard_files):
        shard = torch.load(sf, weights_only=False, map_location="cpu")
        lens = shard["canonical_lengths"]
        for i, L in enumerate(lens):
            if L <= 0:
                continue
            pid = str(shard["problem_ids"][i])
            tid = f"{pid}__{str(shard['md5'][i])[:12]}"
            sents = split_sentences(shard["traces"][i])
            if not sents:
                raise ValueError(f"{sf}: trace {tid} segmented to zero sentences")
            roles = label_roles(sents)
            chars = np.array([len(s) for s in sents], dtype=float)
            rec = {"trace_id": tid, "problem_id": pid, "n_sentences": len(sents)}
            rec.update(dynamics(roles, chars))
            rows.append(rec)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(shard_files)} shards, {len(rows)} traces", flush=True)

    df = pd.DataFrame(rows)
    if df["trace_id"].duplicated().any():
        d = df[df["trace_id"].duplicated()]["trace_id"].head(5).tolist()
        raise ValueError(f"duplicate trace_ids: {d}")
    for c in FEATURES:
        if not np.isfinite(df[c].to_numpy()).all():
            raise ValueError(f"non-finite values in {c}")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}  rows={len(df)}", flush=True)
    print(df[FEATURES].describe().to_string(), flush=True)
    return 0

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return run(args.shard_dir, args.out)

if __name__ == "__main__":
    raise SystemExit(main())
