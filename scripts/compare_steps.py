"""Check that get_trace(view="steps") splits real traces exactly as the paper does.

    python scripts/compare_steps.py --per-collection 30 --out step_comparison.json

The server calls the paper's own `merged_step_char_spans`; only the tokenizer differs. The paper passes
`AutoTokenizer.from_pretrained(<model>)` (transformers), the server a `tokenizers.Tokenizer` for Qwen/Qwen3-4B
behind `sst_mcp.steps.TokenizerAdapter`. This script checks, on full-length traces from shard 0000 of each
collection (the longest traces and a random sample):

- that tokenizer.json is the same file for Qwen3-1.7B, 4B, 8B and 14B,
- that the adapter and the transformers tokenizers of Qwen3-4B and Qwen3-8B give the same token ids,
- that the step spans are identical at step_tokens 400 (the paper's) and 200.

Needs the paper's base environment (transformers, pyarrow, huggingface_hub). HF_TOKEN is optional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from sst.shared.extractors import merged_step_char_spans  # noqa: E402
from sst_mcp import steps  # noqa: E402

COLLECTIONS = (
    "eval_aime",
    "eval_competition_math",
    "eval_gsm_plus",
    "eval_math500",
    "eval_putnam_axiom",
    "eval_math_perturb",
    "eval_math_test",
)
MODELS = ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B")
PAPER_MODELS = ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B")


def sample_traces(per_collection: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for collection in COLLECTIONS:
        path = hf_hub_download(
            "ssubhnil/Reasoning-Failures",
            f"{collection}/train/0000.parquet",
            repo_type="dataset",
            revision="refs/convert/parquet",
        )
        traces = pq.read_table(path, columns=["trace"]).column("trace").to_pylist()
        idx = sorted(range(len(traces)), key=lambda i: len(traces[i] or ""), reverse=True)
        longest = idx[: per_collection // 2]
        rest = rng.sample(idx[per_collection // 2 :], per_collection - len(longest))
        for i in longest + rest:
            if traces[i]:
                out.append({"row_id": f"{collection}/train/0000.parquet#{i}", "trace": traces[i]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-collection", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--out", default="step_comparison.json")
    args = parser.parse_args()

    hashes = {
        m: hashlib.sha256(Path(hf_hub_download(m, "tokenizer.json")).read_bytes()).hexdigest()[:16] for m in MODELS
    }
    adapter = steps.get_adapter()
    refs = {m: AutoTokenizer.from_pretrained(m) for m in PAPER_MODELS}
    rows = sample_traces(args.per_collection, args.seed)

    mismatches = []
    n_steps = {400: 0, 200: 0}
    for row in rows:
        trace = row["trace"]
        ids = adapter(trace)["input_ids"]
        for model, ref in refs.items():
            if ref(trace)["input_ids"] != ids:
                mismatches.append({"row_id": row["row_id"], "what": f"token ids vs {model}"})
            for k in (400, 200):
                ours = steps.step_spans(trace, k, adapter)
                paper = [(int(s), int(e)) for s, e in merged_step_char_spans(trace, k=k, tokenizer=ref)]
                if ours != paper:
                    mismatches.append({"row_id": row["row_id"], "what": f"spans k={k} vs {model}"})
                elif model == PAPER_MODELS[0]:
                    n_steps[k] += len(ours)

    summary = {
        "traces": len(rows),
        "max_trace_chars": max(len(r["trace"]) for r in rows),
        "max_trace_tokens": max(len(adapter(r["trace"])["input_ids"]) for r in rows),
        "tokenizer_json_sha256": hashes,
        "tokenizer_json_identical": len(set(hashes.values())) == 1,
        "steps_compared": n_steps,
        "mismatches": len(mismatches),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "mismatches": mismatches}, indent=1))
    print(json.dumps(summary, indent=1))
    for m in mismatches[:20]:
        print("MISMATCH", m)


if __name__ == "__main__":
    main()
