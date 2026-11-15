import sys
from pathlib import Path

import pytest

from sst_mcp.checker import AnswerChecker, _score, normalise_gold, summarise, sympy_gold_to_latex

sys.path.insert(0, str(Path(__file__).parent))


@pytest.mark.parametrize(
    "gt, expected",
    [
        ("[106]", ["\\boxed{106}"]),
        ("['\\\\frac{1}{2}', '0.5']", ["\\boxed{\\frac{1}{2}}", "\\boxed{0.5}"]),
        ('["$x=2$"]', ["$x=2$"]),
        ("\\boxed{3}", ["\\boxed{3}"]),
        ("42", ["\\boxed{42}"]),
        ("[1, 2", ["\\boxed{[1, 2}"]),
        ("[]", []),
        ("[2*pi + 18]", ["\\boxed{2*pi + 18}"]),
        ("[Interval(1, 100)]", ["\\boxed{Interval(1, 100)}"]),
    ],
)
def test_normalise_gold(gt, expected):
    assert normalise_gold(gt) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2*pi + 18", "2 \\pi + 18"),
        ("(16*pi)/3", "\\frac{16 \\pi}{3}"),
        ("Interval.open(0, 1)", "\\left(0, 1\\right)"),
        ("Eq(x, 2)", "x = 2"),
        ("a_1 + b_2", "a_{1} + b_{2}"),
    ],
)
def test_sympy_gold_to_latex(text, expected):
    assert sympy_gold_to_latex(text) == expected


@pytest.mark.parametrize(
    "text", ["().__class__", "x.subs(1, 2)", "Symbol.mro()", "'a'", "x;y", "\\frac{1}{2}", "1" * 301, "lambda: 1"]
)
def test_sympy_gold_to_latex_rejects(text):
    assert sympy_gold_to_latex(text) is None


def test_sympy_gold_to_latex_has_no_builtins():
    # Unknown names, builtins included, become undefined sympy functions, never Python calls.
    assert sympy_gold_to_latex("open(1)") == "\\operatorname{open}{\\left(1 \\right)}"


def test_score_sympy_printed_gold():
    # Without the LaTeX form, '2*pi + 18' is read as 2*i*p + 18 and a correct answer fails.
    assert _score("so \\boxed{18 + 2\\pi}", "\\boxed{2*pi + 18}")[0] is True
    assert _score("so \\boxed{[1, 100]}", "\\boxed{Interval(1, 100)}")[0] is True
    assert _score("so \\boxed{17}", "\\boxed{(16*pi)/3}")[0] is False


def test_score_in_process():
    assert _score("so \\boxed{106}", "\\boxed{106}") == (True, {"reason": "ok", "extracted_answer": "106"})
    ok, extra = _score("so \\boxed{107}", "\\boxed{106}")
    assert not ok and extra["reason"] == "ok"
    assert _score("no box here", "\\boxed{1}") == (False, {"reason": "no_boxed_answer", "extracted_answer": None})
    ok, extra = _score("\\boxed{}", "\\boxed{1}")
    assert not ok and extra["reason"] == "unparseable"


def test_summarise_priorities():
    assert summarise([False, True], [{"reason": "ok", "extracted_answer": "2"}] * 2)["correct"]
    assert summarise([False], [None])["reason"] == "timeout"
    assert summarise([False], [{"reason": "no_boxed_answer", "extracted_answer": None}])["reason"] == "no_boxed_answer"
    assert summarise([False], [{"reason": "unparseable", "extracted_answer": None}])["reason"] == "unparseable"
    assert summarise([], [])["reason"] == "unparseable"


@pytest.fixture(scope="module")
def checker():
    return AnswerChecker()


def test_pool_correct_and_wrong(checker):
    right = checker.check_sync("The answer is \\boxed{106}.", normalise_gold("[106]"), 10)
    assert right == {"correct": True, "extracted_answer": "106", "reason": "ok"}
    wrong = checker.check_sync("The answer is \\boxed{107}.", normalise_gold("[106]"), 10)
    assert wrong == {"correct": False, "extracted_answer": "107", "reason": "ok"}


def test_pool_any_gold_element_counts(checker):
    out = checker.check_sync("\\boxed{0.5}", normalise_gold("['\\\\frac{1}{3}', '\\\\frac{1}{2}']"), 10)
    assert out["correct"]


def test_pool_no_boxed(checker):
    out = checker.check_sync("I think it is 106", normalise_gold("[106]"), 10)
    assert out == {"correct": False, "extracted_answer": None, "reason": "no_boxed_answer"}


def test_hard_timeout():
    from slow_scorer import slow_score

    slow = AnswerChecker(score_fn=slow_score)
    out = slow.check_sync("\\boxed{1}", ["\\boxed{1}"], 1)
    assert out["reason"] == "timeout" and out["correct"] is False
    assert slow._warm is False


def test_worker_exception_reports_unparseable(monkeypatch):
    c = AnswerChecker()
    c._warm = True

    def boom(*a, **k):
        raise RuntimeError("worker died")

    monkeypatch.setattr(c, "_run", boom)
    assert c.check_sync("\\boxed{1}", ["\\boxed{1}"], 1)["reason"] == "unparseable"
