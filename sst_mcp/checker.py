"""check_answer: the repo's deterministic math checker behind a hard process timeout."""

from __future__ import annotations

import ast
import json
import re
import threading
from collections.abc import Callable
from typing import Any

_WORKER: dict[str, Any] = {}


def _score(trace: str, gt: str, steps: Any = None) -> tuple[bool, dict[str, Any]]:
    """Mirror RuleBasedChecker.check and also report the extracted answer. Runs in a spawn worker."""
    if "\\boxed{" not in trace:
        return False, {"reason": "no_boxed_answer", "extracted_answer": None}
    if "checker" not in _WORKER:
        from sst.shared.rule_checker import RuleBasedChecker

        _WORKER["checker"] = RuleBasedChecker()
    import sympy
    from math_verify import parse, verify

    from sst.shared.rule_checker import _canon_binary_boxed

    checker = _WORKER["checker"]
    gold = checker.extract_answer(gt)
    if gt.startswith("\\boxed{") and gt.endswith("}"):
        latex = sympy_gold_to_latex(gt[len("\\boxed{") : -1])
        if latex is not None:
            gold = list(gold) + list(checker.extract_answer("\\boxed{" + latex + "}"))
    pred = parse(
        _canon_binary_boxed(trace),
        extraction_config=checker._boxed_cfg,
        parsing_timeout=checker.timeout,
        fallback_mode="no_fallback",
    )
    if not pred:
        return False, {"reason": "unparseable", "extracted_answer": None}
    correct = bool(verify(gold, pred, timeout_seconds=checker.timeout))
    try:
        extracted = sympy.sstr(pred[0])
    except Exception:
        extracted = str(pred[0])
    return correct, {"reason": "ok", "extracted_answer": extracted}


_SYMPY_NAMES = (
    "sqrt pi oo E I Interval Union Intersection Complement FiniteSet EmptySet Eq Ne Lt Le Gt Ge Matrix "
    "factorial binomial log exp sin cos tan cot sec csc asin acos atan acot sinh cosh tanh Abs floor ceiling "
    "gamma Sum Product Rational Integer Float Symbol Function Mul Add Pow"
).split()
_SYMPY_GOLD_RE = re.compile(r"[0-9A-Za-z_ +\-*/().,{}\[\]<>=]{1,300}")
_ATTR_RE = re.compile(r"\.(?=[A-Za-z_])")
_INTERVAL_ATTR_RE = re.compile(r"\bInterval\.(?:open|Lopen|Ropen)\b")


def sympy_gold_to_latex(text: str) -> str | None:
    """LaTeX for a gold answer stored as a sympy-printed string (`2*pi + 18`, `Interval.open(0, 1)`), or None.

    The dataset stores most gold answers as `str()` of sympy objects, which the LaTeX parser misreads
    (`2*pi + 18` comes out as 2*i*p + 18). `parse_expr` evaluates Python, and the text may come from the caller,
    so only short text over a fixed character set is accepted, without `__` or attribute access other than
    `Interval.open`/`Lopen`/`Ropen`, and it is evaluated in a namespace of sympy names only, without builtins.
    Unknown names become symbols."""
    text = re.sub(r"\s+", " ", text.strip())  # sympy prints matrices over several lines
    if not _SYMPY_GOLD_RE.fullmatch(text) or "__" in text:
        return None
    if _ATTR_RE.search(_INTERVAL_ATTR_RE.sub("Interval", text)):
        return None
    import sympy
    from sympy.parsing.sympy_parser import parse_expr

    namespace: dict[str, Any] = {"__builtins__": {}}
    namespace.update({name: getattr(sympy, name) for name in _SYMPY_NAMES})
    try:
        return sympy.latex(parse_expr(text, local_dict={}, global_dict=namespace, evaluate=False))
    except Exception:
        return None


def _warm(trace: str, gt: str, steps: Any = None) -> tuple[bool, dict[str, Any]]:
    return _score("\\boxed{1}", "\\boxed{1}")


def normalise_gold(gt_answer: str) -> list[str]:
    """Turn a gold answer such as '[106]' into ['\\boxed{106}'] so the checker can read it."""
    text = gt_answer.strip()
    elems: list[Any] | None = None
    if text.startswith("[") and text.endswith("]"):
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(text)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if isinstance(value, list):
                elems = value
                break
    if elems is None:
        # Not a JSON or Python literal, e.g. '[2*pi + 18]' (a sympy-printed gold): drop the brackets.
        inner = text[1:-1].strip() if text.startswith("[") and text.endswith("]") else text
        elems = [inner]
    out = []
    for elem in elems:
        s = str(elem).strip()
        if not s:
            continue
        out.append(s if ("\\boxed{" in s or "$" in s) else "\\boxed{" + s + "}")
    return out


class AnswerChecker:
    """Runs _score in sst.shared.verify_pool's process pool with a hard per-answer timeout."""

    def __init__(self, score_fn: Callable = _score, max_workers: int = 2, warm_timeout_s: float = 60.0) -> None:
        self.score_fn = score_fn
        self.max_workers = max_workers
        self.warm_timeout_s = warm_timeout_s
        self._warm = False
        self._lock = threading.Lock()

    def _run(self, traces: list[str], golds: list[str], timeout_s: float, fn: Callable):
        from sst.shared.verify_pool import batch_check_hard_timeout

        return batch_check_hard_timeout(traces, golds, timeout_s=timeout_s, score_fn=fn, max_workers=self.max_workers)

    def check_sync(self, response: str, golds: list[str], timeout_s: float) -> dict[str, Any]:
        with self._lock:
            if not self._warm:
                # Cold spawn workers import sympy and math_verify inside the timed task, so warm them first.
                self._run(["w"] * self.max_workers, ["w"] * self.max_workers, self.warm_timeout_s, _warm)
                self._warm = True
            try:
                flags, extras = self._run([response] * len(golds), golds, timeout_s, self.score_fn)
            except Exception:  # includes pebble.ProcessExpired
                self._warm = False
                return {"correct": False, "extracted_answer": None, "reason": "unparseable"}
            if any(e is None for e in extras):
                self._warm = False  # a timed-out worker is replaced by a cold one
        return summarise(flags, extras)


def summarise(flags: list[bool], extras: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Correct if any gold verifies; otherwise report the most informative reason."""
    extracted = next((e.get("extracted_answer") for e in extras if e and e.get("extracted_answer")), None)
    if any(flags):
        return {"correct": True, "extracted_answer": extracted, "reason": "ok"}
    reasons = [e.get("reason", "unparseable") if e else "timeout" for e in extras]
    for reason in ("ok", "timeout", "no_boxed_answer", "unparseable"):
        if reason in reasons:
            return {"correct": False, "extracted_answer": extracted, "reason": reason}
    return {"correct": False, "extracted_answer": extracted, "reason": "unparseable"}
