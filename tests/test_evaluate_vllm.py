from __future__ import annotations

import contextlib
import json
import sys

import pytest
import torch
from torch import nn
from transformers import BatchEncoding

import scripts.evaluate_vllm as ev
from sst.student.model import StudentModel


class ServerReached(Exception):
    pass


def set_argv(tmp_path, monkeypatch, golds: list[str], flags: list[str]) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps({"id": f"q{i}", "question": "Q", "ground_truth_solution": g}) + "\n"
                            for i, g in enumerate(golds)))
    monkeypatch.setattr(ev.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["evaluate_vllm", "--input", str(path), "--out", str(tmp_path / "out.jsonl"),
                                      "--model", "m", "--server-python", "p", "--max-new-tokens", "8",
                                      "--max-model-len", "64", "--max-num-seqs", "2", "--pipeline-depth", "1",
                                      "--gpu-memory-utilization", "0.5", "--tensor-parallel-size", "1", "--port", "1",
                                      "--startup-timeout-s", "1", "--verify-timeout-s", "1", "--verify-workers", "1"]
                        + flags)


def run_main(tmp_path, monkeypatch, golds: list[str]) -> None:
    def server(settings):
        raise ServerReached

    monkeypatch.setattr(ev, "running_server", server)
    set_argv(tmp_path, monkeypatch, golds, [])
    ev.main()


def test_bare_gold_stops_with_its_id(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"1 ground_truth_solution .*\['q1'\]"):
        run_main(tmp_path, monkeypatch, ["\\boxed{4}", "4", "$5$"])


def test_delimited_golds_pass_the_check(tmp_path, monkeypatch):
    with pytest.raises(ServerReached):
        run_main(tmp_path, monkeypatch, ["\\boxed{4}", "The answer is \\boxed{\\frac{1}{2}}.", "$5$"])


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[-1]["content"]

    def __call__(self, text, **kw):
        if isinstance(text, list):
            return BatchEncoding({"attention_mask": torch.ones(len(text), 2, dtype=torch.long)})
        return BatchEncoding({"input_ids": torch.ones(1, len(text.split()), dtype=torch.long)})


@pytest.mark.parametrize("flags,temperature,seed", [([], 0.0, None), (["--temperature", "0.7", "--seed", "1"], 0.7, 1)])
def test_decoding_flags_reach_the_request(tmp_path, monkeypatch, flags, temperature, seed):
    sent: list[dict] = []

    def post(self, path, payload):
        sent.append(payload)
        return {"choices": [{"index": i, "text": "</think> \\boxed{4}", "finish_reason": "stop", "token_ids": [1]}
                            for i in range(len(payload["prompt"]))]}

    def student(model, extractor, vllm_client):
        s = StudentModel.__new__(StudentModel)
        nn.Module.__init__(s)
        s.tokenizer, s.model, s.step_extractor, s.vllm_client = StubTokenizer(), nn.Linear(1, 1), extractor, vllm_client
        s.system_prompt, s.gen_kwargs = "", {"temperature": 0.7, "max_new_tokens": 256}
        return s

    monkeypatch.setattr(ev, "running_server", lambda settings: contextlib.nullcontext("http://localhost:0"))
    monkeypatch.setattr(ev.VLLMClient, "_post_with_retry", post)
    monkeypatch.setattr(ev, "StudentModel", student)
    monkeypatch.setattr(ev, "batch_check_hard_timeout",
                        lambda traces, *a, **kw: ([True] * len(traces), [None] * len(traces)))
    set_argv(tmp_path, monkeypatch, ["\\boxed{4}"], flags)
    ev.main()
    assert [(p["temperature"], p.get("seed")) for p in sent] == [(temperature, seed)]
