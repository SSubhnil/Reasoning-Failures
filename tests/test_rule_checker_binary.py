from __future__ import annotations

import pytest

from sst.shared.rule_checker import RuleBasedChecker

CHECKER = RuleBasedChecker()

YES = ["Yes", "yes", "YES", "True", "true", "TRUE", "\\text{Yes}", "\\text{True}"]
NO = ["No", "no", "NO", "False", "false", "FALSE", "\\text{No}", "\\text{False}"]

def _correct(pred: str, gold: str) -> bool:
    ok, _ = CHECKER.check(f"So the answer is \\boxed{{{pred}}}.", f"\\boxed{{{gold}}}")
    return ok

@pytest.mark.parametrize("gold", YES)
@pytest.mark.parametrize("pred", YES)
def test_yes_true_equivalent(pred: str, gold: str) -> None:
    assert _correct(pred, gold)

@pytest.mark.parametrize("gold", NO)
@pytest.mark.parametrize("pred", NO)
def test_no_false_equivalent(pred: str, gold: str) -> None:
    assert _correct(pred, gold)

@pytest.mark.parametrize("gold", YES)
@pytest.mark.parametrize("pred", NO)
def test_opposite_binary_incorrect_pred_no(pred: str, gold: str) -> None:
    assert not _correct(pred, gold)

@pytest.mark.parametrize("gold", NO)
@pytest.mark.parametrize("pred", YES)
def test_opposite_binary_incorrect_pred_yes(pred: str, gold: str) -> None:
    assert not _correct(pred, gold)

@pytest.mark.parametrize(
    ("pred", "gold"),
    [("1", "True"), ("True", "1"), ("0", "False"), ("False", "0"),
     ("1", "Yes"), ("Yes", "1"), ("0", "No"), ("No", "0")],
)
def test_numeric_not_binary_synonym(pred: str, gold: str) -> None:
    assert not _correct(pred, gold)

def test_numeric_answers_unchanged() -> None:
    assert _correct("42", "42")
    assert not _correct("41", "42")

def test_binary_within_answer_not_rewritten() -> None:
    assert not _correct("\\text{Yes, it converges}", "True")

def test_step_scan_finds_binary_synonym() -> None:
    steps = ["We check convergence. ", "It converges, so \\boxed{Yes}."]
    ok, idx = CHECKER.check("".join(steps), "\\boxed{True}", extracted_steps=steps)
    assert ok
    assert idx == 1
