"""Question and trace token log-probability statistics of one failed trace per problem (GPU)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from analysis.transcoder_presence_extract import MAX_TRACE_TOKENS, load_failed_traces
from sst.shared.extractors import SkyworkExtractor
from sst.student.model import StudentModel

GATHER_CHUNK = 2048


@torch.no_grad()
def trace_logprobs(student: StudentModel, prompt: str, trace: str) -> dict:
    """Teacher-forced log-probabilities of the question and trace tokens, summarized."""
    p_ids = student.tokenizer(prompt, add_special_tokens=False).input_ids
    t_ids = student.tokenizer(trace, add_special_tokens=False).input_ids
    truncated = len(t_ids) > MAX_TRACE_TOKENS
    t_ids = t_ids[:MAX_TRACE_TOKENS]
    if len(t_ids) < 1 or len(p_ids) < 2:
        raise ValueError(f"empty prompt or trace: {len(p_ids)} prompt tokens, {len(t_ids)} trace tokens")
    input_ids = torch.tensor([p_ids + t_ids], dtype=torch.long, device=student.model.device)
    logits = student.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[0]
    targets = input_ids[0, 1:]
    lp = torch.empty(logits.shape[0] - 1, dtype=torch.float32)
    for start in range(0, len(lp), GATHER_CHUNK):
        end = min(start + GATHER_CHUNK, len(lp))
        lp[start:end] = torch.log_softmax(logits[start:end].float(), dim=-1).gather(
            -1, targets[start:end, None]).squeeze(-1).cpu()
    q_lp, t_lp = lp[:len(p_ids) - 1], lp[len(p_ids) - 1:]
    return {"n_q_tok": len(p_ids), "n_t_tok": len(t_ids), "truncated": truncated,
            "q_mean_lp": float(q_lp.mean()), "t_mean_lp": float(t_lp.mean()),
            "t_std_lp": float(t_lp.std()) if t_lp.numel() > 1 else 0.0, "t_min_lp": float(t_lp.min()),
            "t_p10_lp": float(torch.quantile(t_lp, 0.10)), "empty": False}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--problem-regimes", type=Path, required=True)
    parser.add_argument("--trace-ids", type=Path, required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists")
    if not torch.cuda.is_available():
        raise RuntimeError("log-probability extraction requires a CUDA device")
    meta_in = json.loads(args.problem_regimes.read_text())
    regimes = {p: r for p, r in zip(meta_in["problem_ids"], meta_in["problem_regimes"]) if r in ("capability", "pit")}
    items = load_failed_traces(args.corpus_dir, set(json.loads(args.trace_ids.read_text())), regimes)
    student = StudentModel(args.student_model, SkyworkExtractor(option="original"), vllm_client=None)
    student.model = student.model.to("cuda").eval()
    assert next(student.model.parameters()).device.type == "cuda"
    traces = {}
    for problem_id, trace_id, question, trace in items:
        traces[trace_id] = {**trace_logprobs(student, question, trace), "problem_id": problem_id,
                            "regime": regimes[problem_id]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_traces": len(traces), "traces": traces}))


if __name__ == "__main__":
    main()
