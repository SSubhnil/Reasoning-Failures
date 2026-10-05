from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from sst.shared.extractors import merged_step_char_spans
from sst.shared.verify_pool import batch_check_hard_timeout


@dataclass(frozen=True, slots=True)
class StepPrefix:
    trace_id: str
    problem_id: str
    step: int
    prompt: str
    trace_prefix: str
    ground_truth: str


TRACE_FIELDS = ("trace_id", "problem_id", "question", "trace", "canonical_length", "ground_truth")


def load_traces(corpus_dir: Path, trace_ids: set[str]) -> dict[str, dict]:
    """Read the requested traces from the corpus shards."""
    traces: dict[str, dict] = {}
    shard_paths = sorted(corpus_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.pt in {corpus_dir}")
    for path in shard_paths:
        for row in torch.load(path, map_location="cpu", weights_only=False):
            if row["trace_id"] in trace_ids:
                traces[row["trace_id"]] = {field: row[field] for field in TRACE_FIELDS}
    missing = trace_ids - traces.keys()
    if missing:
        raise KeyError(f"{len(missing)} traces absent from {corpus_dir}: {sorted(missing)[:5]}")
    return traces


def step_prefixes(traces: dict[str, dict], step_k: int, tokenizer) -> list[StepPrefix]:
    """Question plus the trace up to the end of each merged step."""
    prefixes = []
    for trace_id, row in traces.items():
        spans = merged_step_char_spans(row["trace"], step_k, tokenizer)
        if len(spans) != int(row["canonical_length"]):
            raise ValueError(f"{trace_id}: {len(spans)} merged steps against canonical_length {row['canonical_length']}")
        for step, (_, end) in enumerate(spans):
            trace_prefix = row["trace"][:end]
            prefixes.append(StepPrefix(trace_id, row["problem_id"], step, row["question"] + trace_prefix,
                                       trace_prefix, row["ground_truth"]))
    return prefixes


def graded_texts(prefixes: list[StepPrefix], completions: list[list[str]]) -> list[str]:
    """Trace prefix plus completion, without the prompt, as the corpus grader sees a trace."""
    return [prefix.trace_prefix + completion for prefix, group in zip(prefixes, completions, strict=True)
            for completion in group]


def success_rates(prefixes: list[StepPrefix], flags: list[bool], k: int) -> list[dict]:
    """Per-step verified success counts; the first continuation of each prefix is the greedy one."""
    if len(flags) != k * len(prefixes):
        raise ValueError(f"{len(flags)} verdicts for {len(prefixes)} prefixes at k={k}")
    rows = []
    for i, prefix in enumerate(prefixes):
        verdicts = [int(flag) for flag in flags[i * k:(i + 1) * k]]
        rows.append({"trace_id": prefix.trace_id, "problem_id": prefix.problem_id, "step": prefix.step,
                     "k": k, "successes": sum(verdicts), "greedy_success": verdicts[0],
                     "p_correct": sum(verdicts) / k})
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample continuations from every step prefix and grade them.")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--trace-ids", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--prefix-batch", type=int, required=True)
    parser.add_argument("--step-k", type=int, required=True)
    parser.add_argument("--verify-timeout-s", type=float, required=True)
    parser.add_argument("--verify-workers", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.k < 2:
        raise ValueError("--k counts the greedy continuation, so it must be at least 2")
    if not torch.cuda.is_available():
        raise RuntimeError("rollouts require a CUDA device")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    traces = load_traces(args.corpus_dir, set(json.loads(args.trace_ids.read_text())))
    prefixes = step_prefixes(traces, args.step_k, AutoTokenizer.from_pretrained(args.model))
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=True)
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for start in range(0, len(prefixes), args.prefix_batch):
            batch = prefixes[start:start + args.prefix_batch]
            sampled = [SamplingParams(n=args.k - 1, temperature=args.temperature, top_p=args.top_p,
                                      max_tokens=args.max_new_tokens, seed=args.seed + (start + i) * args.k)
                       for i in range(len(batch))]
            outputs = llm.generate([p.prompt for p in batch for _ in (0, 1)],
                                   [params for pair in zip([greedy] * len(batch), sampled) for params in pair])
            completions = [[c.text for c in g.outputs + s.outputs] for g, s in zip(outputs[0::2], outputs[1::2])]
            texts = graded_texts(batch, completions)
            golds = [prefix.ground_truth for prefix in batch for _ in range(args.k)]
            flags, _ = batch_check_hard_timeout(texts, golds, timeout_s=args.verify_timeout_s,
                                                max_workers=args.verify_workers)
            for row in success_rates(batch, flags, args.k):
                out.write(json.dumps(row) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
