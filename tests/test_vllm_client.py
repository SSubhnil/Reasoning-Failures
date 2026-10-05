from __future__ import annotations

import pytest

from sst.inference.vllm_client import VLLMClient


def canned_client(choices: list[dict], sent: list[dict]) -> VLLMClient:
    client = VLLMClient("http://localhost:0", "m")

    def post(path: str, payload: dict) -> dict:
        sent.append(payload)
        return {"choices": choices, "usage": {"completion_tokens": sum(len(c.get("token_ids") or []) for c in choices)}}

    client._post_with_retry = post
    return client


def test_token_counts_are_generated_id_lengths():
    sent: list[dict] = []
    choices = [{"index": 0, "text": "a b", "finish_reason": "stop", "token_ids": [1, 2]},
               {"index": 1, "text": "c", "finish_reason": "length", "token_ids": [3, 4, 5]}]
    result = canned_client(choices, sent).generate_batch(["p0", "p1"], max_tokens=8)
    assert sent[0]["return_token_ids"] is True
    assert result.token_counts == [2, 3]
    assert result.finish_reasons == ["stop", "length"]


def test_choice_without_token_ids_raises():
    choices = [{"index": 0, "text": "a", "finish_reason": "stop", "token_ids": [1]},
               {"index": 1, "text": "b", "finish_reason": "stop"}]
    with pytest.raises(RuntimeError, match="choice 1 carries no token_ids"):
        canned_client(choices, []).generate_batch(["p0", "p1"], max_tokens=8)
