#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .blindness_common import OUT, ROOT, steps_cache_path

BOXED_KEY = "\\boxed"
_WS = re.compile(r"\s+")
_SPACING = re.compile(r"\\(left|right|!|,|;|:|quad|qquad|displaystyle|,| )")

def extract_boxed(text: str) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(text)
    while True:
        j = text.find(BOXED_KEY, i)
        if j < 0:
            break
        k = j + len(BOXED_KEY)
        while k < n and text[k] in " \t":
            k += 1
        if k >= n or text[k] != "{":
            i = j + len(BOXED_KEY)
            continue
        depth = 0
        start = k
        while k < n:
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start + 1:k])
                    break
            k += 1
        i = k + 1
    return out

def normalize_answer(a: str) -> str:
    s = a.strip()
    s = _SPACING.sub("", s)
    s = _WS.sub("", s)
    s = s.strip("$").rstrip(".")
    return s

def first_boxed_step(steps: list[str]) -> int:
    for j, st in enumerate(steps):
        if BOXED_KEY in st:
            return j
    return -1

def process_model(model: str, force: bool) -> Path:
    out_path = steps_cache_path(model)
    if out_path.exists() and not force:
        print(f"[{model}] cache exists: {out_path} (use --force to rebuild)", flush=True)
        return out_path

    shard_glob = str(ROOT / "runs" / "hf_pull" / "probe_corpus_16k" / f"qwen3-{model}" / "_p1_packed" / "shard_*.pt")
    shards = sorted(glob.glob(shard_glob))
    if not shards:
        raise FileNotFoundError(f"no shards at {shard_glob}")

    rows = []
    n_no_box = 0
    for si, sp in enumerate(shards):
        s = torch.load(sp, map_location="cpu", weights_only=False)
        es = s["extracted_steps"]
        pid = s["problem_ids"]
        md5 = s["md5"]
        ic = s["is_correct"]
        cl = s["canonical_lengths"]
        for i in range(len(es)):
            steps = es[i]
            L = len(steps)
            if L != cl[i]:
                raise ValueError(f"{sp} trace {i}: len(extracted_steps)={L} != canonical_length={cl[i]}")
            tid = f"{pid[i]}__{md5[i][:12]}"
            fb = first_boxed_step(steps)
            if fb < 0:
                n_no_box += 1
                apos = np.nan
                fa = None
                fan = None
            else:
                apos = 1.0 if L <= 1 else fb / (L - 1)
                boxes = extract_boxed(steps[-1]) or extract_boxed("".join(steps))
                fa = boxes[-1] if boxes else None
                fan = normalize_answer(fa) if fa is not None else None
            rows.append({
                "trace_id": tid,
                "problem_id": pid[i],
                "is_correct": bool(ic[i]),
                "canonical_length": int(L),
                "n_box_steps": int(sum(1 for st in steps if BOXED_KEY in st)),
                "answer_position_norm": apos,
                "final_answer": fa,
                "final_answer_norm": fan,
            })
        if (si + 1) % 200 == 0:
            print(f"[{model}] {si + 1}/{len(shards)} shards", flush=True)

    df = pd.DataFrame(rows)
    dup = df["trace_id"].duplicated().sum()
    if dup:
        raise ValueError(f"[{model}] {dup} duplicate trace_ids across shards")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{model}] wrote {out_path}  n_traces={len(df)}  "
          f"no_box={n_no_box}  median_answer_pos={df['answer_position_norm'].median():.3f}",
          flush=True)
    return out_path

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["4b", "8b"], action="append")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    models = args.model or ["4b", "8b"]
    for m in models:
        process_model(m, args.force)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
