import importlib.util

import pytest
from fakes import WordTokenizer

from sst.shared.extractors import merged_step_char_spans
from sst_mcp import steps
from sst_mcp.server import _fit_steps

TOK = WordTokenizer()


def test_step_spans_match_repo_function():
    trace = "one two three\n\nfour\n\nfive six seven eight\n\nnine ten"
    assert steps.step_spans(trace, 3, TOK) == merged_step_char_spans(trace, k=3, tokenizer=TOK)


def test_empty_trace_has_no_steps():
    assert steps.step_spans("", 400, TOK) == []


def test_stopiteration_falls_back_to_one_step(monkeypatch):
    def broken(*a, **k):
        raise StopIteration

    monkeypatch.setattr(steps, "merged_step_char_spans", broken)
    assert steps.step_spans("abc", 400, TOK) == [(0, 3)]


def test_adapter_interface():
    class FakeEncoding:
        ids = [1, 2, 3]

    class FakeTok:
        def encode(self, text):
            return FakeEncoding()

    assert steps.TokenizerAdapter(FakeTok())("hi")["input_ids"] == [1, 2, 3]


def test_get_adapter_is_cached(monkeypatch):
    class FakeTokenizer:
        loads = 0

        @classmethod
        def from_pretrained(cls, repo):
            cls.loads += 1
            assert repo == "Qwen/Qwen3-4B"
            return cls()

    import tokenizers

    monkeypatch.setattr(tokenizers, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(steps, "_adapter", None)
    a = steps.get_adapter()
    assert steps.get_adapter() is a and FakeTokenizer.loads == 1
    monkeypatch.setattr(steps, "_adapter", None)


def test_fit_steps_keeps_first_two_and_last():
    trace = "\n\n".join(f"p{i} " + "x" * 98 for i in range(8))
    spans = steps.step_spans(trace, 1, TOK)
    assert len(spans) == 8
    kept, dropped, truncated = _fit_steps(trace, spans, 500)
    idx = [s.index for s in kept]
    assert idx[:2] == [0, 1] and idx[-1] == 7
    assert dropped == [2, 3, 4, 5] and truncated
    assert sum(len(s.text) for s in kept) <= 500


def test_fit_steps_clips_when_three_steps_too_long():
    trace = "\n\n".join("y" * 1000 for _ in range(3))
    kept, dropped, truncated = _fit_steps(trace, steps.step_spans(trace, 1, TOK), 600)
    assert dropped == [] and truncated
    assert kept[0].text.endswith("[...]") and kept[-1].text.startswith("[...]")


def test_fit_steps_no_cut():
    trace = "a\n\nb"
    kept, dropped, truncated = _fit_steps(trace, steps.step_spans(trace, 1, TOK), 500)
    assert [s.text for s in kept] == ["a", "b"] and not dropped and not truncated


@pytest.mark.network
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_adapter_matches_transformers_tokenizer():
    from transformers import AutoTokenizer

    try:
        ref = AutoTokenizer.from_pretrained(steps.TOKENIZER_REPO)
        adapter = steps.get_adapter()
    except Exception as exc:  # offline
        pytest.skip(f"tokenizer download failed: {exc}")
    samples = [
        "Let x = 3. Then x^2 = 9.",
        "<think>\nOkay, so I need to find \\frac{1}{2} of 10.\n\nThat is 5.</think>",
        "The answer is \\boxed{106}.",
        "Unicode: π ≈ 3.14159, √2, ∑_{k=1}^{n} k",
        "   leading spaces and\ttabs\n\n\nnewlines",
    ]
    for text in samples:
        assert adapter(text)["input_ids"] == ref(text)["input_ids"]
