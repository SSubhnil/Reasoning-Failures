from __future__ import annotations

import pytest

from sst.shared.extractors import (
    _merge_steps_by_min_tokens,
    merged_step_char_spans,
)

class WordTokenizer:
    def __call__(self, text, **kw):
        return {"input_ids": text.split()}

TOK = WordTokenizer()

def _spans_agree_with_merge(trace: str, k: int) -> bool:
    steps = _merge_steps_by_min_tokens(trace, k, TOK, delimiter="\n\n")
    spans = merged_step_char_spans(trace, k, TOK, delimiter="\n\n")
    if len(spans) != len(steps):
        return False
    for (s, e), step in zip(spans, steps):

        if " ".join(trace[s:e].split("\n\n")) != step:
            return False
    return True

@pytest.mark.parametrize("k", [1, 2, 3, 5, 100])
@pytest.mark.parametrize("trace", [
    "aaa\n\nbbb\n\nccc\n\nddd",
    "one two three\n\nfour\n\nfive six seven eight",
    "a b c d e f g\n\nh",
    "x\n\ny\n\nz",
    "solve\n\nstep one here\n\nstep two\n\nanswer 42",
])
def test_spans_delimit_the_merged_chunks(trace, k):
    assert _spans_agree_with_merge(trace, k)

@pytest.mark.parametrize("k", [1, 2, 3, 5])
@pytest.mark.parametrize("trace", [
    "aa bb\n\ncc dd\n\n",
    "aa bb\n\n\n\n\n\ncc dd",
    "\n\naa bb\n\ncc",
    "aa bb cc\n\n\n\n",
    "\n\n\n\n",
])
def test_empty_chunks_do_not_add_a_step(trace, k):
    steps = _merge_steps_by_min_tokens(trace, k, TOK, delimiter="\n\n")
    spans = merged_step_char_spans(trace, k, TOK, delimiter="\n\n")
    assert len(spans) == len(steps)

def test_span_count_matches_merge_on_random_traces():
    import random
    rng = random.Random(0)
    words = ["a", "bb", "ccc", ""]
    for _ in range(2000):
        n = rng.randint(1, 8)
        trace = "\n\n".join(
            " ".join(rng.choice(words) for _ in range(rng.randint(0, 4))) for _ in range(n))
        k = rng.randint(1, 6)
        steps = _merge_steps_by_min_tokens(trace, k, TOK, delimiter="\n\n")
        spans = merged_step_char_spans(trace, k, TOK, delimiter="\n\n")
        assert len(spans) == len(steps), (trace, k, steps, spans)
        assert all(0 <= s <= e <= len(trace) for s, e in spans), (trace, k, spans)

def test_spans_are_ordered_and_within_the_trace():
    trace = "one two three\n\nfour\n\nfive six seven eight\n\nnine"
    spans = merged_step_char_spans(trace, 2, TOK, delimiter="\n\n")
    assert spans == sorted(spans)
    assert all(0 <= s < e <= len(trace) for s, e in spans)

def test_slice_is_the_original_text_not_the_rejoin():
    trace = "aa\n\nbb\n\ncc\n\ndd"
    steps = _merge_steps_by_min_tokens(trace, 2, TOK, delimiter="\n\n")
    spans = merged_step_char_spans(trace, 2, TOK, delimiter="\n\n")
    merged_two = [i for i, st in enumerate(steps) if " " in st]
    assert merged_two, "expected at least one step built from two raw chunks"
    i = merged_two[0]
    s, e = spans[i]
    assert "\n\n" in trace[s:e]
    assert "\n\n" not in steps[i]
    assert trace[s:e] != steps[i]
