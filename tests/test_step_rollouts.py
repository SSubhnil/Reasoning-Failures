from __future__ import annotations

import pytest
import torch

from scripts.run_step_rollouts import (StepPrefix, build_parser, graded_texts, load_traces, step_prefixes,
                                       success_rates)
from sst.shared.rule_checker import RuleBasedChecker


def whitespace_tokenizer(text: str) -> dict:
    return {"input_ids": text.split()}


def trace_row(trace_id: str, canonical_length: int) -> dict:
    return {"trace_id": trace_id, "problem_id": "p1", "question": "Q: add 2 and 2.\n",
            "trace": "first step here\n\nsecond step\n\nso the answer is \\boxed{4}",
            "ground_truth": "4", "canonical_length": canonical_length}


def test_step_prefixes_are_question_plus_trace_up_to_each_step_end():
    prefixes = step_prefixes({"t1": trace_row("t1", 3)}, step_k=1, tokenizer=whitespace_tokenizer)
    assert [p.step for p in prefixes] == [0, 1, 2]
    assert prefixes[0].prompt == "Q: add 2 and 2.\nfirst step here"
    assert prefixes[0].trace_prefix == "first step here"
    assert prefixes[1].prompt == "Q: add 2 and 2.\nfirst step here\n\nsecond step"
    assert prefixes[2].trace_prefix.endswith("\\boxed{4}")
    assert all(p.ground_truth == "4" for p in prefixes)


def test_step_prefixes_reject_a_segmentation_that_disagrees_with_the_corpus():
    with pytest.raises(ValueError):
        step_prefixes({"t1": trace_row("t1", 5)}, step_k=1, tokenizer=whitespace_tokenizer)


def test_success_rates_group_verdicts_by_prefix_with_greedy_first():
    prefixes = [StepPrefix("t1", "p1", 0, "Qa", "a", "4"), StepPrefix("t1", "p1", 1, "Qab", "ab", "4")]
    rows = success_rates(prefixes, [True, False, False, False, True, True, True, False], k=4)
    assert [(r["step"], r["successes"], r["greedy_success"], r["p_correct"]) for r in rows] == [
        (0, 1, 1, 0.25), (1, 3, 1, 0.75)]


def test_success_rates_reject_a_verdict_count_that_does_not_match_k():
    with pytest.raises(ValueError):
        success_rates([StepPrefix("t1", "p1", 0, "Qa", "a", "4")], [True, False, True], k=4)


def test_load_traces_reads_requested_rows_and_fails_on_missing(tmp_path):
    torch.save([trace_row("t1", 3), trace_row("t2", 3)], tmp_path / "shard_0000.pt")
    assert set(load_traces(tmp_path, {"t1"})) == {"t1"}
    with pytest.raises(KeyError):
        load_traces(tmp_path, {"t1", "t9"})


def test_sampling_settings_are_required_arguments():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--corpus-dir", "c", "--trace-ids", "t.json", "--model", "m", "--out", "o"])


def test_graded_text_excludes_the_prompt_so_a_boxed_answer_in_the_question_cannot_score():
    prefix = StepPrefix("t1", "p1", 0, "Show that \\boxed{4} is 2+2.\nfirst step", "first step", "4")
    texts = graded_texts([prefix], [[" and no final answer"]])
    assert texts == ["first step and no final answer"]
    checker = RuleBasedChecker()
    assert not checker.check(texts[0], "\\boxed{4}")[0]
    assert checker.check(prefix.prompt + " and no final answer", "\\boxed{4}")[0]
