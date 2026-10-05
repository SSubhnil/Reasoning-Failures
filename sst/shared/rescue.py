from __future__ import annotations

FORCE_FINAL_SUFFIX = "</think> \n\n Final answer in \\boxed{"

def is_rescued_trace(trace: str, finish_reason: str | None = None) -> bool:
    return FORCE_FINAL_SUFFIX in trace or finish_reason == "missing_end_think"
