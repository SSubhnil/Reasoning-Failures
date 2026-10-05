from __future__ import annotations

import re

import sympy
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

_BINARY_BOXED_RE = re.compile(
    r"\\boxed\{\s*(?:\\(?:text|textbf|mathrm)\{\s*(yes|no|true|false)\s*\}|(yes|no|true|false))\s*\}",
    re.IGNORECASE,
)
_BINARY_CANON = {"yes": "yes", "true": "yes", "no": "no", "false": "no"}

def _canon_binary_boxed(text: str) -> str:
    return _BINARY_BOXED_RE.sub(
        lambda m: "\\boxed{\\text{%s}}" % _BINARY_CANON[(m.group(1) or m.group(2)).lower()], text)

class RuleBasedChecker:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

        self.mv_cfg = [LatexExtractionConfig(), ExprExtractionConfig(try_extract_without_anchor=False)]

        self._boxed_cfg = [LatexExtractionConfig(try_extract_without_anchor=False)]

    def extract_answer(self, trace):
        parsed = parse(_canon_binary_boxed(trace), extraction_config=self.mv_cfg, parsing_timeout=self.timeout,
                       fallback_mode="no_fallback")
        return parsed

    @staticmethod
    def _parsed_str(parsed) -> str:
        if not parsed:
            return ""
        try:
            return sympy.sstr(parsed[0], order='none')
        except Exception:
            return ""

    def check(self, generated_trace: str, gt_solution: str, extracted_steps=None):
        if "\\boxed{" not in generated_trace:
            return False, None
        gold_parsed = self.extract_answer(gt_solution)

        full_pred = parse(_canon_binary_boxed(generated_trace), extraction_config=self._boxed_cfg,
                          parsing_timeout=self.timeout, fallback_mode="no_fallback")
        if not full_pred:
            return False, None
        full_correct = verify(gold_parsed, full_pred, timeout_seconds=self.timeout)

        if not extracted_steps:
            return bool(full_correct), None

        if full_correct:

            gold_str = self._parsed_str(full_pred)
            earliest_correct_idx = None
            if gold_str:
                for idx in range(len(extracted_steps) - 1, -1, -1):
                    prefix = "".join(extracted_steps[: idx + 1])
                    try:
                        pred_parsed = self.extract_answer(prefix)
                        if self._parsed_str(pred_parsed) == gold_str:
                            earliest_correct_idx = idx
                    except Exception:
                        continue
            return True, earliest_correct_idx
        else:

            earliest_parseable_idx = None
            for idx in range(len(extracted_steps) - 1, -1, -1):
                prefix = "".join(extracted_steps[: idx + 1])
                try:
                    pred_parsed = self.extract_answer(prefix)
                    if pred_parsed:
                        earliest_parseable_idx = idx
                except Exception:
                    continue
            return False, earliest_parseable_idx
