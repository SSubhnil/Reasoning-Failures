from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sst.inference import VLLMClient
from sst.inference.pipelined_generation import generate_in_batches
from sst.inference.server import ServerSettings, running_server
from sst.shared.extractors import SkyworkExtractor
from sst.shared.rescue import is_rescued_trace
from sst.shared.rule_checker import RuleBasedChecker
from sst.shared.verify_pool import batch_check_hard_timeout
from sst.student.model import StudentModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Greedy evaluation of JSONL questions through a local vLLM server.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-python", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--pipeline-depth", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--startup-timeout-s", type=float, required=True)
    parser.add_argument("--verify-timeout-s", type=float, required=True)
    parser.add_argument("--verify-workers", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature; 0 is greedy")
    parser.add_argument("--seed", type=int, default=None, help="sampling seed sent to the server")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires a CUDA device for the vLLM server")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    checker = RuleBasedChecker()
    unparsed = [row["id"] for row in rows if not checker.extract_answer(row["ground_truth_solution"])]
    if unparsed:
        raise ValueError(f"{len(unparsed)} ground_truth_solution values have no \\boxed{{}} or $...$ answer: {unparsed[:10]}")
    settings = ServerSettings(args.model, args.server_python, args.port, args.max_model_len, args.max_num_seqs,
                              args.gpu_memory_utilization, args.tensor_parallel_size, args.startup_timeout_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with running_server(settings) as base_url, VLLMClient(base_url, args.model) as client, \
            args.out.open("w", buffering=1) as out:
        student = StudentModel(args.model, SkyworkExtractor(option="original"), vllm_client=client)
        questions = [row["question"] for row in rows]
        for start, generated in generate_in_batches(student, questions, args.max_num_seqs, args.max_new_tokens,
                                                    args.pipeline_depth, args.temperature, args.seed):
            batch = rows[start:start + args.max_num_seqs]
            correct, answer_steps = batch_check_hard_timeout(
                generated.traces, [row["ground_truth_solution"] for row in batch], generated.extracted_steps,
                timeout_s=args.verify_timeout_s, max_workers=args.verify_workers)
            for row, trace, is_correct, answer_step, finish_reason, token_count in zip(
                    batch, generated.traces, correct, answer_steps, generated.finish_reasons, generated.token_counts):
                out.write(json.dumps({"problem_id": row["id"], "question": row["question"],
                                      "ground_truth_solution": row["ground_truth_solution"], "trace": trace,
                                      "is_correct": is_correct, "answer_step_idx": answer_step,
                                      "finish_reason": finish_reason, "token_count": token_count,
                                      "rescued": is_rescued_trace(trace, finish_reason)}) + "\n")


if __name__ == "__main__":
    main()
